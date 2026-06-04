from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize


"""
Unsupervised NLP pipeline for AI Incident Database reports.

Dashboard use:
- `clustered_records.csv` can drive an interactive 2D scatter plot where each
  point is a report or incident, colored by cluster and filtered by metadata.
- `cluster_summary.csv` provides cluster titles, top keywords, examples, and
  metadata profiles for side panels or dashboard cards.
- `cluster_metadata_distribution.csv` can power small multiples or drilldowns
  showing how discovered themes vary by year, risk taxonomy, sector, or country.
"""


# Edit these defaults for the dataset you want to cluster.
PROCESSED_DATA_PATH = Path("processed_data/report_level_processed.csv")
OUTPUT_DIR = Path("cluster_outputs/sbert_report_clusters")
TEXT_COLUMN = "input_text"
ID_COLUMN = "incident_id"

# Production default: Sentence-BERT. Use `--embedding-backend tfidf-svd` for a
# quick smoke test that avoids downloading a transformer model.
EMBEDDING_BACKEND = "sentence-transformer"
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
MAX_RECORDS: int | None = None
MIN_TEXT_CHARS = 80

# Visualization and clustering defaults.
REDUCTION_METHOD = "umap"  # "umap" or "pca"; falls back to PCA if UMAP is unavailable.
CLUSTER_METHOD = "kmeans"  # "kmeans" or "hdbscan"; falls back to KMeans if HDBSCAN is unavailable.
N_CLUSTERS = 12
HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_MIN_SAMPLES: int | None = None
RANDOM_STATE = 42

# Keyword extraction defaults.
TOP_KEYWORDS = 12
TOP_EXAMPLES = 5
KEYWORD_MAX_FEATURES = 10_000
KEYWORD_NGRAM_RANGE = (1, 2)


@dataclass(frozen=True)
class ClusterConfig:
    data_path: Path = PROCESSED_DATA_PATH
    output_dir: Path = OUTPUT_DIR
    text_column: str = TEXT_COLUMN
    id_column: str = ID_COLUMN
    embedding_backend: str = EMBEDDING_BACKEND
    sentence_model_name: str = SENTENCE_MODEL_NAME
    batch_size: int = BATCH_SIZE
    max_records: int | None = MAX_RECORDS
    min_text_chars: int = MIN_TEXT_CHARS
    reduction_method: str = REDUCTION_METHOD
    cluster_method: str = CLUSTER_METHOD
    n_clusters: int = N_CLUSTERS
    hdbscan_min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE
    hdbscan_min_samples: int | None = HDBSCAN_MIN_SAMPLES
    random_state: int = RANDOM_STATE
    top_keywords: int = TOP_KEYWORDS
    top_examples: int = TOP_EXAMPLES
    keyword_max_features: int = KEYWORD_MAX_FEATURES


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "value"


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


def load_and_clean_data(config: ClusterConfig) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {config.data_path}")

    frame = pd.read_csv(config.data_path, low_memory=False)
    print(f"Loaded {config.data_path} with {len(frame):,} rows and {len(frame.columns):,} columns.")
    print("Available columns:")
    print(", ".join(frame.columns))

    missing = [column for column in [config.text_column, config.id_column] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    output = frame.copy()
    output[config.text_column] = output[config.text_column].map(normalize_text)
    output[config.id_column] = output[config.id_column].map(normalize_text)

    before = len(output)
    output = output[
        output[config.text_column].str.len().ge(config.min_text_chars)
        & output[config.id_column].ne("")
    ].copy()
    output = output.drop_duplicates(subset=[config.id_column, config.text_column], keep="first")

    if config.max_records is not None and len(output) > config.max_records:
        output = output.sample(n=config.max_records, random_state=config.random_state).sort_index()

    output = output.reset_index(drop=True)
    print(f"\nRows after text cleaning/deduplication: {len(output):,} (removed {before - len(output):,}).")
    return output


def generate_sentence_transformer_embeddings(frame: pd.DataFrame, config: ClusterConfig) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for the default embedding backend. "
            "Install requirements.txt or run with `--embedding-backend tfidf-svd` for a lightweight smoke test."
        ) from exc

    print(f"\nLoading Sentence-BERT model: {config.sentence_model_name}")
    model = SentenceTransformer(config.sentence_model_name)
    texts = frame[config.text_column].tolist()
    embeddings = model.encode(
        texts,
        batch_size=config.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def generate_tfidf_svd_embeddings(frame: pd.DataFrame, config: ClusterConfig) -> np.ndarray:
    print("\nUsing lightweight TF-IDF + SVD embeddings for smoke testing.")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2 if len(frame) >= 50 else 1,
        max_df=0.95,
        max_features=20_000,
    )
    matrix = vectorizer.fit_transform(frame[config.text_column])
    n_components = min(384, max(2, matrix.shape[0] - 1), max(2, matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=config.random_state)
    embeddings = svd.fit_transform(matrix)
    return normalize(embeddings).astype("float32")


def generate_embeddings(frame: pd.DataFrame, config: ClusterConfig) -> np.ndarray:
    if config.embedding_backend == "sentence-transformer":
        return generate_sentence_transformer_embeddings(frame, config)
    if config.embedding_backend == "tfidf-svd":
        return generate_tfidf_svd_embeddings(frame, config)
    raise ValueError("embedding_backend must be `sentence-transformer` or `tfidf-svd`.")


def reduce_embeddings(embeddings: np.ndarray, config: ClusterConfig) -> tuple[np.ndarray, str]:
    if config.reduction_method == "umap":
        try:
            import umap

            print("\nReducing embeddings to 2D with UMAP.")
            reducer = umap.UMAP(
                n_neighbors=min(30, max(5, len(embeddings) // 20)),
                min_dist=0.08,
                n_components=2,
                metric="cosine",
                random_state=config.random_state,
            )
            return reducer.fit_transform(embeddings), "umap"
        except ImportError:
            print("\nUMAP is not installed; falling back to PCA.")

    print("\nReducing embeddings to 2D with PCA.")
    reducer = PCA(n_components=2, random_state=config.random_state)
    return reducer.fit_transform(embeddings), "pca"


def cluster_embeddings(embeddings: np.ndarray, config: ClusterConfig) -> tuple[np.ndarray, str]:
    if config.cluster_method == "hdbscan":
        try:
            import hdbscan

            print("\nClustering embeddings with HDBSCAN.")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=config.hdbscan_min_cluster_size,
                min_samples=config.hdbscan_min_samples,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(embeddings)
            if len(set(labels)) <= 1:
                print("HDBSCAN produced one or fewer clusters; falling back to KMeans.")
            else:
                return labels.astype(int), "hdbscan"
        except ImportError:
            print("\nHDBSCAN is not installed; falling back to KMeans.")

    cluster_count = min(config.n_clusters, len(embeddings))
    if cluster_count < 2:
        raise ValueError("Need at least two rows to create clusters.")

    print(f"\nClustering embeddings with KMeans (k={cluster_count}).")
    clusterer = KMeans(n_clusters=cluster_count, random_state=config.random_state, n_init="auto")
    return clusterer.fit_predict(embeddings).astype(int), "kmeans"


def infer_metadata_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    columns = list(frame.columns)

    def first_existing(candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        return None

    sector_candidates = [column for column in columns if "sector" in column and frame[column].notna().any()]
    risk_candidates = [
        column
        for column in columns
        if ("risk" in column or "harm_type" in column or "failure" in column)
        and frame[column].notna().any()
    ]
    country_candidates = [
        column
        for column in columns
        if ("country" in column or column in {"country_label", "location_region"})
        and frame[column].notna().any()
    ]

    return {
        "year": first_existing(["incident_year", "report_year", "year"]),
        "country": country_candidates[0] if country_candidates else None,
        "sector": sector_candidates[0] if sector_candidates else None,
        "risk_category": first_existing(["mit_risk_domain", "risk_category"]) or (risk_candidates[0] if risk_candidates else None),
        "failure_type": next((column for column in risk_candidates if "failure" in column), None),
    }


def split_multivalue_series(series: pd.Series) -> pd.Series:
    values: list[str] = []
    for value in series.dropna():
        text = normalize_text(value)
        if not text:
            continue
        values.extend(part.strip() for part in text.split(" | ") if part.strip())
    return pd.Series(values, dtype="object")


def extract_cluster_keywords(
    frame: pd.DataFrame,
    config: ClusterConfig,
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    rows: list[dict[str, Any]] = []
    keywords_by_cluster: dict[int, list[str]] = {}
    min_df = 2 if len(frame) >= 50 else 1
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=KEYWORD_NGRAM_RANGE,
        min_df=min_df,
        max_df=0.9,
        max_features=config.keyword_max_features,
    )

    try:
        matrix = vectorizer.fit_transform(frame[config.text_column])
    except ValueError:
        return pd.DataFrame(columns=["cluster_label", "keyword", "score", "rank"]), {}

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    for cluster_label, indices in frame.groupby("cluster_label").groups.items():
        cluster_matrix = matrix[list(indices)]
        scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        top_indices = scores.argsort()[::-1][: config.top_keywords]
        keywords = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
        keywords_by_cluster[int(cluster_label)] = keywords
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                continue
            rows.append(
                {
                    "cluster_label": int(cluster_label),
                    "keyword": feature_names[idx],
                    "score": float(scores[idx]),
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows), keywords_by_cluster


def representative_examples(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    config: ClusterConfig,
) -> dict[int, pd.DataFrame]:
    examples: dict[int, pd.DataFrame] = {}
    working = frame.reset_index(drop=True)
    for cluster_label, subset in working.groupby("cluster_label"):
        indices = subset.index.to_numpy()
        cluster_embeddings = embeddings[indices]
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        _, distances = pairwise_distances_argmin_min(cluster_embeddings, centroid)
        ranked = subset.assign(_distance_to_centroid=distances).sort_values("_distance_to_centroid")
        examples[int(cluster_label)] = ranked.head(config.top_examples)
    return examples


def summarize_metadata_for_cluster(
    subset: pd.DataFrame,
    metadata_columns: dict[str, str | None],
) -> dict[str, str]:
    summary: dict[str, str] = {}
    for metadata_name, column in metadata_columns.items():
        if column is None or column not in subset.columns:
            continue
        values = split_multivalue_series(subset[column])
        if values.empty:
            continue
        top_values = values.value_counts().head(3)
        summary[f"top_{metadata_name}"] = " | ".join(f"{idx} ({count})" for idx, count in top_values.items())
    return summary


def build_cluster_summary(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    keywords_by_cluster: dict[int, list[str]],
    metadata_columns: dict[str, str | None],
    config: ClusterConfig,
) -> pd.DataFrame:
    examples_by_cluster = representative_examples(frame, embeddings, config)
    title_columns = [column for column in ["incident_title", "report_title", "title"] if column in frame.columns]
    rows: list[dict[str, Any]] = []

    for cluster_label, subset in frame.groupby("cluster_label"):
        example_subset = examples_by_cluster[int(cluster_label)]
        example_ids = example_subset[config.id_column].astype(str).head(config.top_examples).tolist()
        example_titles: list[str] = []
        for _, row in example_subset.iterrows():
            title = next((normalize_text(row[column]) for column in title_columns if normalize_text(row[column])), "")
            if title:
                example_titles.append(title)

        row = {
            "cluster_label": int(cluster_label),
            "cluster_name": "Noise / outliers" if int(cluster_label) == -1 else f"Cluster {int(cluster_label)}",
            "size": int(len(subset)),
            "unique_incidents": int(subset[config.id_column].nunique()),
            "top_keywords": ", ".join(keywords_by_cluster.get(int(cluster_label), [])),
            "example_incident_ids": " | ".join(example_ids),
            "example_titles": " | ".join(example_titles[: config.top_examples]),
        }
        row.update(summarize_metadata_for_cluster(subset, metadata_columns))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("size", ascending=False)


def build_metadata_distribution(
    frame: pd.DataFrame,
    metadata_columns: dict[str, str | None],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metadata_name, column in metadata_columns.items():
        if column is None or column not in frame.columns:
            continue
        for cluster_label, subset in frame.groupby("cluster_label"):
            values = split_multivalue_series(subset[column])
            if values.empty:
                continue
            counts = values.value_counts()
            total = counts.sum()
            for value, count in counts.items():
                rows.append(
                    {
                        "metadata_type": metadata_name,
                        "metadata_column": column,
                        "cluster_label": int(cluster_label),
                        "value": value,
                        "count": int(count),
                        "share_within_cluster": float(count / total) if total else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def plot_scatter(frame: pd.DataFrame, output_path: Path) -> None:
    labels = sorted(frame["cluster_label"].unique())
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.get_cmap("tab20")
    for idx, label in enumerate(labels):
        subset = frame[frame["cluster_label"] == label]
        color = "#9ca3af" if label == -1 else cmap(idx % 20)
        ax.scatter(subset["x"], subset["y"], s=18, alpha=0.75, color=color, label=str(label))

    ax.set_title("AI Incident Text Clusters")
    ax.set_xlabel("Embedding dimension 1")
    ax.set_ylabel("Embedding dimension 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cluster_sizes(summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = summary.sort_values("size", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_data))))
    ax.barh(plot_data["cluster_name"], plot_data["size"], color="#4f46e5")
    ax.set_title("Cluster Sizes")
    ax.set_xlabel("Records")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cluster_timeline(frame: pd.DataFrame, year_column: str, output_path: Path) -> pd.DataFrame:
    working = frame.copy()
    working[year_column] = pd.to_numeric(working[year_column], errors="coerce")
    working = working.dropna(subset=[year_column])
    working = working[working[year_column].between(1900, 2100)]
    if working.empty:
        return pd.DataFrame(columns=["year", "cluster_label", "count"])

    working["year"] = working[year_column].astype(int)
    counts = working.groupby(["year", "cluster_label"], as_index=False).size().rename(columns={"size": "count"})
    top_clusters = (
        counts.groupby("cluster_label")["count"].sum().sort_values(ascending=False).head(8).index.tolist()
    )
    plot_data = counts[counts["cluster_label"].isin(top_clusters)]
    pivot = plot_data.pivot(index="year", columns="cluster_label", values="count").fillna(0)

    fig, ax = plt.subplots(figsize=(11, 6))
    for cluster_label in pivot.columns:
        ax.plot(pivot.index, pivot[cluster_label], marker="o", linewidth=2, label=f"Cluster {cluster_label}")
    ax.set_title("Cluster Frequency Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Records")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return counts


def save_outputs(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    keywords: pd.DataFrame,
    metadata_distribution: pd.DataFrame,
    embeddings: np.ndarray,
    metadata_columns: dict[str, str | None],
    reduction_used: str,
    cluster_method_used: str,
    config: ClusterConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    clustered_path = config.output_dir / "clustered_records.csv"
    summary_path = config.output_dir / "cluster_summary.csv"
    keywords_path = config.output_dir / "cluster_keywords.csv"
    metadata_path = config.output_dir / "cluster_metadata_distribution.csv"
    embeddings_path = config.output_dir / "embeddings.npy"
    scatter_path = config.output_dir / "cluster_scatter.png"
    sizes_path = config.output_dir / "cluster_sizes.png"
    timeline_path = config.output_dir / "cluster_timeline.png"
    year_distribution_path = config.output_dir / "cluster_year_distribution.csv"
    config_path = config.output_dir / "cluster_config.json"

    frame.to_csv(clustered_path, index=False)
    summary.to_csv(summary_path, index=False)
    keywords.to_csv(keywords_path, index=False)
    metadata_distribution.to_csv(metadata_path, index=False)
    np.save(embeddings_path, embeddings)

    plot_scatter(frame, scatter_path)
    plot_cluster_sizes(summary, sizes_path)
    if metadata_columns.get("year"):
        year_counts = plot_cluster_timeline(frame, metadata_columns["year"], timeline_path)
        year_counts.to_csv(year_distribution_path, index=False)

    config_payload = {
        **config.__dict__,
        "data_path": str(config.data_path),
        "output_dir": str(config.output_dir),
        "metadata_columns": metadata_columns,
        "reduction_used": reduction_used,
        "cluster_method_used": cluster_method_used,
    }
    config_path.write_text(json.dumps(make_json_safe(config_payload), indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    for path in [
        clustered_path,
        summary_path,
        keywords_path,
        metadata_path,
        embeddings_path,
        scatter_path,
        sizes_path,
        timeline_path if metadata_columns.get("year") else None,
        year_distribution_path if metadata_columns.get("year") else None,
        config_path,
    ]:
        if path is not None:
            print(path)


def run_pipeline(config: ClusterConfig) -> dict[str, Any]:
    frame = load_and_clean_data(config)
    embeddings = generate_embeddings(frame, config)
    coords, reduction_used = reduce_embeddings(embeddings, config)
    cluster_labels, cluster_method_used = cluster_embeddings(embeddings, config)

    output = frame.copy()
    output["x"] = coords[:, 0]
    output["y"] = coords[:, 1]
    output["cluster_label"] = cluster_labels

    metadata_columns = infer_metadata_columns(output)
    print("\nDetected metadata columns:")
    for name, column in metadata_columns.items():
        print(f"{name}: {column or 'not found'}")

    keywords, keywords_by_cluster = extract_cluster_keywords(output, config)
    summary = build_cluster_summary(output, embeddings, keywords_by_cluster, metadata_columns, config)
    metadata_distribution = build_metadata_distribution(output, metadata_columns)

    print("\nCluster summary:")
    print(summary[["cluster_label", "size", "unique_incidents", "top_keywords"]].to_string(index=False))

    save_outputs(
        output,
        summary,
        keywords,
        metadata_distribution,
        embeddings,
        metadata_columns,
        reduction_used,
        cluster_method_used,
        config,
    )
    return {
        "rows": len(output),
        "clusters": int(output["cluster_label"].nunique()),
        "reduction_used": reduction_used,
        "cluster_method_used": cluster_method_used,
        "metadata_columns": metadata_columns,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover hidden themes in AI incident reports with unsupervised NLP clustering.")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--text-column", default=TEXT_COLUMN)
    parser.add_argument("--id-column", default=ID_COLUMN)
    parser.add_argument("--embedding-backend", choices=["sentence-transformer", "tfidf-svd"], default=EMBEDDING_BACKEND)
    parser.add_argument("--sentence-model-name", default=SENTENCE_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS)
    parser.add_argument("--min-text-chars", type=int, default=MIN_TEXT_CHARS)
    parser.add_argument("--reduction-method", choices=["umap", "pca"], default=REDUCTION_METHOD)
    parser.add_argument("--cluster-method", choices=["kmeans", "hdbscan"], default=CLUSTER_METHOD)
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=HDBSCAN_MIN_CLUSTER_SIZE)
    parser.add_argument("--hdbscan-min-samples", type=int, default=HDBSCAN_MIN_SAMPLES)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--top-keywords", type=int, default=TOP_KEYWORDS)
    parser.add_argument("--top-examples", type=int, default=TOP_EXAMPLES)
    parser.add_argument("--keyword-max-features", type=int, default=KEYWORD_MAX_FEATURES)
    return parser


def config_from_args(argv: list[str] | None = None) -> ClusterConfig:
    args = build_arg_parser().parse_args(argv)
    return ClusterConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        text_column=args.text_column,
        id_column=args.id_column,
        embedding_backend=args.embedding_backend,
        sentence_model_name=args.sentence_model_name,
        batch_size=args.batch_size,
        max_records=args.max_records,
        min_text_chars=args.min_text_chars,
        reduction_method=args.reduction_method,
        cluster_method=args.cluster_method,
        n_clusters=args.n_clusters,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        random_state=args.random_state,
        top_keywords=args.top_keywords,
        top_examples=args.top_examples,
        keyword_max_features=args.keyword_max_features,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run_pipeline(config_from_args(argv))


if __name__ == "__main__":
    main()
