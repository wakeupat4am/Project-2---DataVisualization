from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Edit these defaults for the label you want to model.
PROCESSED_DATA_PATH = Path("processed_data/classification_ready.csv")
TARGET_LABEL_COLUMN = "mit_risk_domain"
OUTPUT_BASE_DIR = Path("model_outputs")

# Split and preprocessing defaults.
USE_EXISTING_SPLIT_COLUMN = True
SPLIT_COLUMN = "split"
TEST_SIZE = 0.20
RANDOM_STATE = 42
MIN_TEXT_CHARS = 40

# TF-IDF / Logistic Regression defaults.
MAX_FEATURES = 50_000
MIN_DF = 2
MAX_DF = 0.95
NGRAM_RANGE = (1, 2)
LOGREG_MAX_ITER = 2_000

# Classification-ready labels are usually single-label. If a chosen target has
# multiple labels joined by " | ", choose: "as_label", "first", or "drop".
MULTILABEL_SEPARATOR = " | "
MULTILABEL_STRATEGY = "as_label"


@dataclass(frozen=True)
class BaselineConfig:
    data_path: Path
    target_label_column: str
    output_dir: Path
    use_existing_split_column: bool = USE_EXISTING_SPLIT_COLUMN
    split_column: str = SPLIT_COLUMN
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    min_text_chars: int = MIN_TEXT_CHARS
    max_features: int = MAX_FEATURES
    min_df: int = MIN_DF
    max_df: float = MAX_DF
    ngram_range: tuple[int, int] = NGRAM_RANGE
    multilabel_strategy: str = MULTILABEL_STRATEGY


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "target"


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


def load_dataset(config: BaselineConfig) -> pd.DataFrame:
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


def prepare_modeling_frame(frame: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
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
        raise ValueError("Need at least two target classes after cleaning to train Logistic Regression.")
    return output


def split_by_incident_id(frame: pd.DataFrame, config: BaselineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.use_existing_split_column and config.split_column in frame.columns:
        split_values = frame[config.split_column].astype(str).str.lower()
        train = frame[split_values == "train"].copy()
        test = frame[split_values == "test"].copy()
        if not train.empty and not test.empty:
            print(f"\nUsing existing `{config.split_column}` column for train/test split.")
            return train, test
        print(f"\nExisting `{config.split_column}` column found, but train/test rows were incomplete. Falling back to grouped split.")

    incident_ids = frame["incident_id"].dropna().astype(str).loc[lambda s: s.ne("")].unique()
    train_ids, test_ids = train_test_split(
        incident_ids,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )
    train_id_set = set(train_ids)
    test_id_set = set(test_ids)
    train = frame[frame["incident_id"].isin(train_id_set)].copy()
    test = frame[frame["incident_id"].isin(test_id_set)].copy()
    print(f"\nCreated grouped train/test split by incident_id with test_size={config.test_size}.")
    return train, test


def validate_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_incidents = set(train["incident_id"].astype(str))
    test_incidents = set(test["incident_id"].astype(str))
    overlap = train_incidents & test_incidents
    if overlap:
        examples = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"Data leakage detected: incident_id values appear in both train and test: {examples}")

    print("\nSplit validation passed: no incident_id appears in both train and test.")
    print(f"Train rows: {len(train):,}; train incidents: {len(train_incidents):,}")
    print(f"Test rows: {len(test):,}; test incidents: {len(test_incidents):,}")

    if "report_id" in train.columns and "report_id" in test.columns:
        train_reports = set(train["report_id"].dropna().astype(str))
        test_reports = set(test["report_id"].dropna().astype(str))
        report_overlap = train_reports & test_reports
        if report_overlap:
            examples = ", ".join(sorted(report_overlap)[:10])
            raise ValueError(f"Report-level leakage detected: report_id values appear in both train and test: {examples}")
        print("Extra leakage check passed: no report_id appears in both train and test.")


def build_pipeline(config: BaselineConfig) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                    max_df=config.max_df,
                    max_features=config.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=LOGREG_MAX_ITER,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, test: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_true = test["target_label"]
    y_pred = model.predict(test["input_text"])
    labels = sorted(set(y_true) | set(y_pred))

    probabilities = model.predict_proba(test["input_text"])
    classes = list(model.named_steps["logreg"].classes_)
    max_probabilities = probabilities.max(axis=1)

    predictions = test.copy()
    predictions["predicted_label"] = y_pred
    predictions["predicted_confidence"] = max_probabilities

    top3 = []
    for row_probs in probabilities:
        ranked = sorted(zip(classes, row_probs), key=lambda item: item[1], reverse=True)[:3]
        top3.append(" | ".join(f"{label}: {probability:.4f}" for label, probability in ranked))
    predictions["top_3_predictions"] = top3

    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_frame = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "test_class_distribution": y_true.value_counts().to_dict(),
        "prediction_distribution": pd.Series(y_pred).value_counts().to_dict(),
    }
    return metrics, predictions, report_frame


def save_confusion_matrix_png(metrics: dict[str, Any], output_path: Path) -> None:
    labels = metrics["labels"]
    cm = np.asarray(metrics["confusion_matrix"])
    size = max(8, min(24, 0.75 * len(labels) + 4))

    fig, ax = plt.subplots(figsize=(size, size))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("TF-IDF + Logistic Regression Confusion Matrix")
    ax.tick_params(axis="x", labelrotation=45 if len(labels) <= 12 else 90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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


def save_outputs(
    model: Pipeline,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    classification_report_frame: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: BaselineConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = config.output_dir / "tfidf_logreg_model.joblib"
    metrics_path = config.output_dir / "evaluation_metrics.json"
    report_path = config.output_dir / "classification_report.csv"
    predictions_path = config.output_dir / "test_predictions.csv"
    confusion_path = config.output_dir / "confusion_matrix.png"

    joblib.dump(model, model_path)
    predictions.to_csv(predictions_path, index=False)
    classification_report_frame.to_csv(report_path)
    save_confusion_matrix_png(metrics, confusion_path)

    serializable_metrics = {
        **metrics,
        "target_label_column": config.target_label_column,
        "data_path": str(config.data_path),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_incidents": int(train["incident_id"].nunique()),
        "test_incidents": int(test["incident_id"].nunique()),
        "train_class_distribution": train["target_label"].value_counts().to_dict(),
    }
    metrics_path.write_text(json.dumps(make_json_safe(serializable_metrics), indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    print(model_path)
    print(metrics_path)
    print(report_path)
    print(predictions_path)
    print(confusion_path)


def predict_new_report_text(model: Pipeline, report_text: str) -> dict[str, Any]:
    """Predict one new report text with the trained scikit-learn pipeline."""
    cleaned = normalize_text(report_text)
    if not cleaned:
        raise ValueError("report_text must not be empty.")

    predicted_label = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    classes = list(model.named_steps["logreg"].classes_)
    confidence = float(probabilities.max())

    return {
        "predicted_label": str(predicted_label),
        "confidence": confidence,
        "probability": confidence,
        "target_label_column": getattr(model, "target_label_column_", TARGET_LABEL_COLUMN),
        "top_probabilities": {
            str(label): float(probability)
            for label, probability in sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:5]
        },
    }


def load_trained_model(model_path: Path) -> Pipeline:
    return joblib.load(model_path)


def train_baseline(config: BaselineConfig) -> tuple[Pipeline, dict[str, Any]]:
    raw = load_dataset(config)
    frame = prepare_modeling_frame(raw, config)
    train, test = split_by_incident_id(frame, config)
    validate_split(train, test)

    train_classes = set(train["target_label"])
    test_classes = set(test["target_label"])
    unseen_test_classes = sorted(test_classes - train_classes)
    if unseen_test_classes:
        print("\nWarning: these test classes are absent from training and cannot be predicted:")
        print(", ".join(unseen_test_classes))

    print("\nTrain class distribution:")
    print(train["target_label"].value_counts().to_string())
    print("\nTest class distribution:")
    print(test["target_label"].value_counts().to_string())

    model = build_pipeline(config)
    print("\nTraining TF-IDF + Logistic Regression baseline...")
    model.fit(train["input_text"], train["target_label"])
    model.target_label_column_ = config.target_label_column

    metrics, predictions, report_frame = evaluate_model(model, test)
    print("\nEvaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(test["target_label"], predictions["predicted_label"], zero_division=0))

    save_outputs(model, metrics, predictions, report_frame, train, test, config)
    return model, metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TF-IDF + Logistic Regression baseline classifier.")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--target", default=TARGET_LABEL_COLUMN, help="Target label column to predict.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder. Defaults to model_outputs/tfidf_logreg_<target>.")
    parser.add_argument("--ignore-existing-split", action="store_true", help="Create a fresh grouped incident_id train/test split.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--min-text-chars", type=int, default=MIN_TEXT_CHARS)
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES)
    parser.add_argument("--multilabel-strategy", choices=["as_label", "first", "drop"], default=MULTILABEL_STRATEGY)
    return parser


def config_from_args(argv: list[str] | None = None) -> BaselineConfig:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir or OUTPUT_BASE_DIR / f"tfidf_logreg_{slugify(args.target)}"
    return BaselineConfig(
        data_path=args.data_path,
        target_label_column=args.target,
        output_dir=output_dir,
        use_existing_split_column=not args.ignore_existing_split,
        test_size=args.test_size,
        random_state=args.random_state,
        min_text_chars=args.min_text_chars,
        max_features=args.max_features,
        multilabel_strategy=args.multilabel_strategy,
    )


def main(argv: list[str] | None = None) -> tuple[Pipeline, dict[str, Any]]:
    return train_baseline(config_from_args(argv))


if __name__ == "__main__":
    main()
