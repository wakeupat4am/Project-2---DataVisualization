from __future__ import annotations

import argparse
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


# Edit these defaults for the label you want to fine-tune.
PROCESSED_DATA_PATH = Path("processed_data/classification_ready.csv")
TARGET_LABEL_COLUMN = "mit_risk_domain"
OUTPUT_BASE_DIR = Path("model_outputs")

# Data split defaults. Existing processed data usually already has train/test.
USE_EXISTING_SPLIT_COLUMN = True
SPLIT_COLUMN = "split"
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42
MIN_TEXT_CHARS = 40

# DistilBERT fine-tuning defaults.
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 3.0
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LOGGING_STEPS = 50

# If a chosen target has multiple labels joined by " | ", choose:
# "as_label", "first", or "drop".
MULTILABEL_SEPARATOR = " | "
MULTILABEL_STRATEGY = "as_label"


@dataclass(frozen=True)
class DistilBertConfig:
    data_path: Path
    target_label_column: str
    output_dir: Path
    use_existing_split_column: bool = USE_EXISTING_SPLIT_COLUMN
    split_column: str = SPLIT_COLUMN
    train_size: float = TRAIN_SIZE
    validation_size: float = VALIDATION_SIZE
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    min_text_chars: int = MIN_TEXT_CHARS
    model_name: str = MODEL_NAME
    max_length: int = MAX_LENGTH
    learning_rate: float = LEARNING_RATE
    train_batch_size: int = TRAIN_BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    num_epochs: float = NUM_EPOCHS
    weight_decay: float = WEIGHT_DECAY
    warmup_ratio: float = WARMUP_RATIO
    logging_steps: int = LOGGING_STEPS
    multilabel_strategy: str = MULTILABEL_STRATEGY
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "target"


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def candidate_label_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "input_text",
        "incident_id",
        "report_id",
        "split",
        "report_title",
        "report_text",
        "report_url",
        "source_domain",
        "incident_title",
        "incident_description",
        "incident_date",
        "incident_year",
    }
    candidates = []
    for column in frame.columns:
        if column in excluded:
            continue
        if column.startswith(("mit_", "gmf_", "cset")) or frame[column].nunique(dropna=True) <= 100:
            candidates.append(column)
    return candidates


def normalize_target_label(value: object, separator: str, strategy: str) -> str:
    label = normalize_text(value)
    if not label or label.lower() in {"nan", "none", "null", "n/a", "na"}:
        return ""
    if separator not in label:
        return label
    if strategy == "as_label":
        return label
    if strategy == "first":
        return label.split(separator)[0].strip()
    if strategy == "drop":
        return ""
    raise ValueError("multilabel_strategy must be one of: as_label, first, drop")


def load_dataset(config: DistilBertConfig) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {config.data_path}")

    frame = pd.read_csv(config.data_path, low_memory=False)
    print(f"Loaded {config.data_path} with {len(frame):,} rows and {len(frame.columns):,} columns.")
    print("Available columns:")
    print(", ".join(frame.columns))

    required = {"input_text", "incident_id", config.target_label_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        print("\nCandidate label columns:")
        print(", ".join(candidate_label_columns(frame)))
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    return frame


def prepare_modeling_frame(frame: pd.DataFrame, config: DistilBertConfig) -> pd.DataFrame:
    output = frame.copy()
    output["input_text"] = output["input_text"].map(normalize_text)
    output["incident_id"] = output["incident_id"].map(normalize_text)
    output["target_label"] = output[config.target_label_column].map(
        lambda value: normalize_target_label(value, MULTILABEL_SEPARATOR, config.multilabel_strategy)
    )

    before = len(output)
    output = output[
        output["input_text"].str.len().ge(config.min_text_chars)
        & output["incident_id"].ne("")
        & output["target_label"].ne("")
    ].copy()
    output = output.drop_duplicates(subset=["incident_id", "input_text", "target_label"])

    print(f"\nRows after cleaning target/text: {len(output):,} (removed {before - len(output):,}).")
    print(f"Target column: {config.target_label_column}")
    print(f"Classes: {output['target_label'].nunique():,}")
    print("\nOverall class distribution:")
    print(output["target_label"].value_counts().to_string())

    if output["target_label"].nunique() < 2:
        raise ValueError("Need at least two target classes after cleaning to fine-tune DistilBERT.")
    return output


def encode_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    labels = sorted(frame["target_label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    output = frame.copy()
    output["label"] = output["target_label"].map(label2id).astype(int)
    return output, label2id, id2label


def build_split_groups(frame: pd.DataFrame) -> pd.Series:
    """Group incidents that share report IDs so duplicated source text stays in one split."""
    incident_ids = sorted(frame["incident_id"].dropna().astype(str).loc[lambda s: s.ne("")].unique().tolist())
    parent = {incident_id: incident_id for incident_id in incident_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    shared_report_count = 0
    if "report_id" in frame.columns:
        links = frame[["incident_id", "report_id"]].dropna().drop_duplicates()
        links["incident_id"] = links["incident_id"].astype(str)
        for _, group in links.groupby("report_id"):
            linked_incidents = [incident_id for incident_id in group["incident_id"].unique() if incident_id in parent]
            if len(linked_incidents) <= 1:
                continue
            shared_report_count += 1
            first = linked_incidents[0]
            for incident_id in linked_incidents[1:]:
                union(first, incident_id)

    group_map = {incident_id: find(incident_id) for incident_id in incident_ids}
    print(
        "\nSplit grouping: "
        f"{len(incident_ids):,} incidents collapsed into {len(set(group_map.values())):,} components; "
        f"{shared_report_count:,} report IDs link multiple incidents."
    )
    return pd.Series(group_map, name="split_group")


def split_groups(
    groups: list[str],
    train_size: float,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> dict[str, str]:
    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + validation_size + test_size must sum to 1.0.")
    if len(groups) < 3:
        raise ValueError("Need at least three incident groups to create train/validation/test splits.")

    train_groups, holdout_groups = train_test_split(
        groups,
        train_size=train_size,
        test_size=validation_size + test_size,
        random_state=random_state,
        shuffle=True,
    )
    relative_test_size = test_size / (validation_size + test_size)
    validation_groups, test_groups = train_test_split(
        holdout_groups,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True,
    )

    split_map = {group: "train" for group in train_groups}
    split_map.update({group: "val" for group in validation_groups})
    split_map.update({group: "test" for group in test_groups})
    return split_map


def split_train_subset_for_validation(
    frame: pd.DataFrame,
    split_groups_series: pd.Series,
    config: DistilBertConfig,
) -> pd.DataFrame:
    output = frame.copy()
    train_incidents = sorted(output.loc[output["split"] == "train", "incident_id"].unique().tolist())
    train_groups = sorted({split_groups_series.loc[incident_id] for incident_id in train_incidents})
    if len(train_groups) < 2:
        raise ValueError("Existing train split is too small to carve out a validation split.")

    val_fraction = min(0.5, max(config.validation_size, 0.01))
    kept_train_groups, val_groups = train_test_split(
        train_groups,
        test_size=val_fraction,
        random_state=config.random_state,
        shuffle=True,
    )
    group_to_split = {group: "train" for group in kept_train_groups}
    group_to_split.update({group: "val" for group in val_groups})

    train_mask = output["split"] == "train"
    output.loc[train_mask, "split"] = output.loc[train_mask, "incident_id"].map(split_groups_series).map(group_to_split)
    return output


def assign_splits(frame: pd.DataFrame, config: DistilBertConfig) -> pd.DataFrame:
    output = frame.copy()
    split_groups_series = build_split_groups(output)
    output["_split_group"] = output["incident_id"].map(split_groups_series)

    if config.use_existing_split_column and config.split_column in output.columns:
        output["split"] = output[config.split_column].astype(str).str.lower().replace({"validation": "val"})
        has_train = (output["split"] == "train").any()
        has_test = (output["split"] == "test").any()
        has_val = (output["split"] == "val").any()
        if has_train and has_test and has_val:
            print(f"\nUsing existing `{config.split_column}` column for train/val/test.")
            return output
        if has_train and has_test:
            print(f"\nUsing existing `{config.split_column}` train/test and carving validation from train incidents.")
            return split_train_subset_for_validation(output, split_groups_series, config)
        print(f"\nExisting `{config.split_column}` column was incomplete. Creating a fresh grouped train/val/test split.")

    groups = sorted(output["_split_group"].dropna().unique().tolist())
    group_to_split = split_groups(groups, config.train_size, config.validation_size, config.test_size, config.random_state)
    output["split"] = output["_split_group"].map(group_to_split)
    print("\nCreated fresh grouped train/val/test split.")
    return output


def validate_splits(frame: pd.DataFrame) -> None:
    for group_column in ["incident_id", "report_id"]:
        if group_column not in frame.columns:
            continue
        split_counts = frame.dropna(subset=[group_column]).groupby(group_column)["split"].nunique()
        leaking = split_counts[split_counts > 1]
        if not leaking.empty:
            examples = ", ".join(leaking.head(10).index.astype(str))
            raise ValueError(f"Data leakage detected: {group_column} values appear in multiple splits: {examples}")
        print(f"Leakage check passed: every {group_column} belongs to exactly one split.")

    split_counts = frame["split"].value_counts()
    print("\nRows by split:")
    print(split_counts.to_string())
    print("\nIncidents by split:")
    print(frame.groupby("split")["incident_id"].nunique().to_string())

    required = {"train", "val", "test"}
    missing = required - set(split_counts.index)
    if missing:
        raise ValueError(f"Missing required split(s): {', '.join(sorted(missing))}")


def maybe_limit_samples(frame: pd.DataFrame, split_name: str, max_samples: int | None, random_state: int) -> pd.DataFrame:
    subset = frame[frame["split"] == split_name].copy()
    if max_samples is not None and len(subset) > max_samples:
        subset = subset.sample(n=max_samples, random_state=random_state)
    return subset.reset_index(drop=True)


def make_torch_dataset(frame: pd.DataFrame, tokenizer: Any, max_length: int) -> Any:
    import torch

    class TextClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, data: pd.DataFrame) -> None:
            self.encodings = tokenizer(
                data["input_text"].tolist(),
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            self.labels = data["label"].astype(int).tolist()

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    return TextClassificationDataset(frame)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
    labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def make_training_arguments(config: DistilBertConfig) -> Any:
    from transformers import TrainingArguments

    args = {
        "output_dir": str(config.output_dir / "checkpoints"),
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "num_train_epochs": config.num_epochs,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": config.random_state,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        args["eval_strategy"] = "epoch"
    else:
        args["evaluation_strategy"] = "epoch"
    return TrainingArguments(**args)


def make_trainer(
    model: Any,
    tokenizer: Any,
    training_args: Any,
    train_dataset: Any,
    val_dataset: Any,
) -> Any:
    from transformers import Trainer

    trainer_args = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
    }
    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_args["processing_class"] = tokenizer
    else:
        trainer_args["tokenizer"] = tokenizer
    return Trainer(**trainer_args)


def save_label_mapping(
    output_dir: Path,
    label2id: dict[str, int],
    id2label: dict[int, str],
    target_label_column: str,
) -> None:
    mapping = {
        "target_label_column": target_label_column,
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
    }
    (output_dir / "label_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def evaluate_test_set(
    trainer: Any,
    test_dataset: Any,
    test_frame: pd.DataFrame,
    id2label: dict[int, str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    output = trainer.predict(test_dataset)
    logits = output.predictions
    y_true = test_frame["label"].to_numpy()
    y_pred = np.argmax(logits, axis=-1)

    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted_logits) / np.exp(shifted_logits).sum(axis=1, keepdims=True)

    labels = sorted(id2label.keys())
    target_names = [id2label[idx] for idx in labels]
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    predictions = test_frame.copy()
    predictions["true_label"] = predictions["label"].map(id2label)
    predictions["predicted_label_id"] = y_pred
    predictions["predicted_label"] = [id2label[int(idx)] for idx in y_pred]
    predictions["predicted_confidence"] = probabilities.max(axis=1)
    predictions["top_3_predictions"] = [
        " | ".join(
            f"{id2label[int(label_idx)]}: {probabilities[row_idx, label_idx]:.4f}"
            for label_idx in np.argsort(probabilities[row_idx])[::-1][:3]
        )
        for row_idx in range(len(predictions))
    ]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "labels": target_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "test_class_distribution": predictions["true_label"].value_counts().to_dict(),
        "prediction_distribution": predictions["predicted_label"].value_counts().to_dict(),
    }
    return metrics, predictions, report_frame


def save_confusion_matrix_png(metrics: dict[str, Any], output_path: Path) -> None:
    labels = metrics["labels"]
    cm = np.asarray(metrics["confusion_matrix"])
    size = max(8, min(24, 0.75 * len(labels) + 4))

    fig, ax = plt.subplots(figsize=(size, size))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Purples", colorbar=True, values_format="d")
    ax.set_title("DistilBERT Confusion Matrix")
    ax.tick_params(axis="x", labelrotation=45 if len(labels) <= 12 else 90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_outputs(
    config: DistilBertConfig,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    report_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    metrics_path = config.output_dir / "evaluation_metrics.json"
    predictions_path = config.output_dir / "test_predictions.csv"
    report_path = config.output_dir / "classification_report.csv"
    confusion_path = config.output_dir / "confusion_matrix.png"

    predictions.to_csv(predictions_path, index=False)
    report_frame.to_csv(report_path)
    save_confusion_matrix_png(metrics, confusion_path)

    serializable_metrics = {
        **metrics,
        "target_label_column": config.target_label_column,
        "data_path": str(config.data_path),
        "model_name": config.model_name,
        "max_length": config.max_length,
        "learning_rate": config.learning_rate,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "num_epochs": config.num_epochs,
        "weight_decay": config.weight_decay,
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
        "train_incidents": int(train_frame["incident_id"].nunique()),
        "val_incidents": int(val_frame["incident_id"].nunique()),
        "test_incidents": int(test_frame["incident_id"].nunique()),
        "train_class_distribution": train_frame["target_label"].value_counts().to_dict(),
        "val_class_distribution": val_frame["target_label"].value_counts().to_dict(),
    }
    metrics_path.write_text(json.dumps(make_json_safe(serializable_metrics), indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    print(config.output_dir / "model")
    print(config.output_dir / "model" / "tokenizer files")
    print(config.output_dir / "label_mapping.json")
    print(metrics_path)
    print(report_path)
    print(predictions_path)
    print(confusion_path)


def train_distilbert(config: DistilBertConfig) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(config)
    frame = prepare_modeling_frame(raw, config)
    frame, label2id, id2label = encode_labels(frame)
    frame = assign_splits(frame, config)
    validate_splits(frame)

    train_frame = maybe_limit_samples(frame, "train", config.max_train_samples, config.random_state)
    val_frame = maybe_limit_samples(frame, "val", config.max_val_samples, config.random_state)
    test_frame = maybe_limit_samples(frame, "test", config.max_test_samples, config.random_state)

    print("\nTraining class distribution:")
    print(train_frame["target_label"].value_counts().to_string())
    print("\nValidation class distribution:")
    print(val_frame["target_label"].value_counts().to_string())
    print("\nTest class distribution:")
    print(test_frame["target_label"].value_counts().to_string())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        id2label={int(key): value for key, value in id2label.items()},
        label2id=label2id,
    )

    train_dataset = make_torch_dataset(train_frame, tokenizer, config.max_length)
    val_dataset = make_torch_dataset(val_frame, tokenizer, config.max_length)
    test_dataset = make_torch_dataset(test_frame, tokenizer, config.max_length)

    training_args = make_training_arguments(config)
    trainer = make_trainer(model, tokenizer, training_args, train_dataset, val_dataset)

    print("\nFine-tuning DistilBERT...")
    trainer.train()

    model_dir = config.output_dir / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    save_label_mapping(config.output_dir, label2id, id2label, config.target_label_column)

    metrics, predictions, report_frame = evaluate_test_set(trainer, test_dataset, test_frame, id2label)
    print("\nTest evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nClassification report:")
    print(report_frame.to_string())

    save_outputs(config, metrics, predictions, report_frame, train_frame, val_frame, test_frame)
    return metrics


def load_inference_artifacts(model_dir: Path) -> tuple[Any, Any, dict[int, str], str]:
    import torch
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    model_dir = Path(model_dir)
    mapping_path = model_dir.parent / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping not found: {mapping_path}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    id2label = {int(key): value for key, value in mapping["id2label"].items()}
    target_label_column = mapping["target_label_column"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, id2label, target_label_column


def predict_new_report_text(
    report_text: str,
    model_dir: Path = OUTPUT_BASE_DIR / f"distilbert_{slugify(TARGET_LABEL_COLUMN)}" / "model",
    top_k: int = 3,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Predict a label for one new report text with the fine-tuned DistilBERT model."""
    import torch

    cleaned = normalize_text(report_text)
    if not cleaned:
        raise ValueError("report_text must not be empty.")

    model, tokenizer, id2label, target_label_column = load_inference_artifacts(model_dir)
    device = next(model.parameters()).device
    encoded = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probabilities)[::-1][:top_k]
    predicted_idx = int(top_indices[0])
    return {
        "target_label_column": target_label_column,
        "predicted_label": id2label[predicted_idx],
        "confidence": float(probabilities[predicted_idx]),
        "top_predictions": [
            {"label": id2label[int(idx)], "confidence": float(probabilities[int(idx)])}
            for idx in top_indices
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for AI incident text classification.")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--target", default=TARGET_LABEL_COLUMN, help="Target label column to predict.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder. Defaults to model_outputs/distilbert_<target>.")
    parser.add_argument("--ignore-existing-split", action="store_true", help="Create a fresh grouped train/val/test split.")
    parser.add_argument("--train-size", type=float, default=TRAIN_SIZE)
    parser.add_argument("--validation-size", type=float, default=VALIDATION_SIZE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--min-text-chars", type=int, default=MIN_TEXT_CHARS)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--num-epochs", type=float, default=NUM_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--logging-steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--multilabel-strategy", choices=["as_label", "first", "drop"], default=MULTILABEL_STRATEGY)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional smoke-test sample cap.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional smoke-test sample cap.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional smoke-test sample cap.")
    return parser


def config_from_args(argv: list[str] | None = None) -> DistilBertConfig:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir or OUTPUT_BASE_DIR / f"distilbert_{slugify(args.target)}"
    return DistilBertConfig(
        data_path=args.data_path,
        target_label_column=args.target,
        output_dir=output_dir,
        use_existing_split_column=not args.ignore_existing_split,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
        min_text_chars=args.min_text_chars,
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        multilabel_strategy=args.multilabel_strategy,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return train_distilbert(config_from_args(argv))


if __name__ == "__main__":
    main()
