from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from .visualizations import PALETTE, empty_figure, short_label


CLUSTER_OUTPUT_DIR = Path("cluster_outputs/sbert_report_clusters")


@dataclass
class ClusterOutputs:
    records: pd.DataFrame
    summary: pd.DataFrame
    metadata_distribution: pd.DataFrame
    year_distribution: pd.DataFrame
    notes: list[str]


@dataclass
class ClusterTextClassifier:
    vectorizer: TfidfVectorizer
    example_matrix: Any
    records: pd.DataFrame
    risk_model: LogisticRegression | None
    problem_model: LogisticRegression | None
    summary: pd.DataFrame


def _read_optional_csv(path: Path, notes: list[str]) -> pd.DataFrame:
    if not path.exists():
        notes.append(f"`{path}` was not found.")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        notes.append(f"`{path}` could not be loaded: {exc}")
        return pd.DataFrame()


def _normalize_cluster_label(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "cluster_label" not in frame.columns:
        return frame
    output = frame.copy()
    output["cluster_label"] = pd.to_numeric(output["cluster_label"], errors="coerce").astype("Int64")
    output = output.dropna(subset=["cluster_label"])
    output["cluster_label"] = output["cluster_label"].astype(int)
    return output


def load_cluster_outputs(output_dir: str | Path = CLUSTER_OUTPUT_DIR) -> ClusterOutputs:
    output_path = Path(output_dir)
    notes: list[str] = []
    records = _read_optional_csv(output_path / "clustered_records.csv", notes)
    summary = _read_optional_csv(output_path / "cluster_summary.csv", notes)
    metadata = _read_optional_csv(output_path / "cluster_metadata_distribution.csv", notes)
    years = _read_optional_csv(output_path / "cluster_year_distribution.csv", notes)

    records = _normalize_cluster_label(records)
    summary = _normalize_cluster_label(summary)
    metadata = _normalize_cluster_label(metadata)
    years = _normalize_cluster_label(years)

    if not records.empty and "cluster_name" not in records.columns:
        names = cluster_name_map(summary)
        records["cluster_name"] = records["cluster_label"].map(names).fillna(
            records["cluster_label"].map(lambda value: f"Cluster {value}")
        )

    return ClusterOutputs(records=records, summary=summary, metadata_distribution=metadata, year_distribution=years, notes=notes)


def build_cluster_text_classifier(outputs: ClusterOutputs) -> ClusterTextClassifier | None:
    records = outputs.records.copy()
    if records.empty or "input_text" not in records.columns:
        return None

    records = records.dropna(subset=["input_text"]).copy()
    if records.empty:
        return None

    texts = records["input_text"].astype(str).str.strip()
    records = records[texts.ne("")].copy()
    if records.empty:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=18000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(records["input_text"].astype(str))

    risk_model = None
    if "mit_risk_domain" in records.columns:
        risk_mask = records["mit_risk_domain"].notna() & records["mit_risk_domain"].astype(str).str.strip().ne("")
        if int(risk_mask.sum()) >= 20 and records.loc[risk_mask, "mit_risk_domain"].nunique() >= 2:
            risk_model = LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                C=3.0,
            )
            risk_model.fit(matrix[risk_mask.to_numpy()], records.loc[risk_mask, "mit_risk_domain"].astype(str))

    problem_model = None
    problem_labels = records["gmf_known_ai_technical_failure"].apply(_primary_problem_label) if "gmf_known_ai_technical_failure" in records.columns else pd.Series(index=records.index, dtype=object)
    if not problem_labels.empty:
        problem_counts = problem_labels.dropna().value_counts()
        keep_problems = problem_counts[problem_counts >= 3].index
        problem_mask = problem_labels.isin(keep_problems)
        if int(problem_mask.sum()) >= 20 and problem_labels[problem_mask].nunique() >= 2:
            problem_model = LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                C=2.5,
            )
            problem_model.fit(matrix[problem_mask.to_numpy()], problem_labels[problem_mask].astype(str))

    return ClusterTextClassifier(
        vectorizer=vectorizer,
        example_matrix=matrix,
        records=records.reset_index(drop=True),
        risk_model=risk_model,
        problem_model=problem_model,
        summary=outputs.summary.copy(),
    )


def cluster_name_map(summary: pd.DataFrame) -> dict[int, str]:
    if summary.empty or not {"cluster_label", "cluster_name"}.issubset(summary.columns):
        return {}
    return {
        int(row.cluster_label): str(row.cluster_name)
        for row in summary[["cluster_label", "cluster_name"]].dropna().itertuples()
    }


def cluster_choice_map(outputs: ClusterOutputs) -> dict[str, str]:
    choices = {"__all__": "All clusters"}
    summary = outputs.summary.copy()
    if summary.empty and not outputs.records.empty:
        summary = (
            outputs.records.groupby("cluster_label", as_index=False)
            .size()
            .rename(columns={"size": "size"})
        )
        summary["top_keywords"] = ""
        summary["cluster_name"] = summary["cluster_label"].map(lambda value: f"Cluster {value}")

    if summary.empty:
        return choices

    sort_col = "size" if "size" in summary.columns else "cluster_label"
    for row in summary.sort_values(sort_col, ascending=False).itertuples():
        label = int(row.cluster_label)
        name = getattr(row, "cluster_name", f"Cluster {label}")
        keywords = str(getattr(row, "top_keywords", "") or "")
        size = getattr(row, "size", None)
        suffix = f" ({int(size):,})" if pd.notna(size) else ""
        keyword_part = f": {short_label(keywords, 68)}" if keywords else ""
        choices[str(label)] = f"{name}{suffix}{keyword_part}"
    return choices


def classify_new_report(
    classifier: ClusterTextClassifier | None,
    title: str,
    body: str,
    source_domain: str = "",
    top_k: int = 5,
) -> dict[str, Any]:
    if classifier is None:
        return {"error": "The NLP classifier is unavailable because the trained corpus could not be loaded."}

    parts = []
    title = (title or "").strip()
    body = (body or "").strip()
    source_domain = (source_domain or "").strip()
    if title:
        parts.append(f"Report title: {title}")
    if body:
        parts.append(f"Report text: {body}")
    if source_domain:
        parts.append(f"Source domain: {source_domain}")
    text = " ".join(parts).strip()
    if not text:
        return {"error": "Enter a report title or report body to run NLP classification."}

    query_vector = classifier.vectorizer.transform([text])

    risk_predictions: list[dict[str, Any]] = []
    predicted_risk = "Not available"
    risk_confidence = None
    if classifier.risk_model is not None:
        risk_probs = classifier.risk_model.predict_proba(query_vector)[0]
        ranked_idx = np.argsort(risk_probs)[::-1][:top_k]
        for idx in ranked_idx:
            risk_predictions.append(
                {
                    "label": str(classifier.risk_model.classes_[idx]),
                    "score": float(risk_probs[idx]),
                    "type": "risk_category",
                }
            )
        if risk_predictions:
            predicted_risk = risk_predictions[0]["label"]
            risk_confidence = risk_predictions[0]["score"]

    problem_predictions: list[dict[str, Any]] = []
    predicted_problem = "Not available"
    problem_confidence = None
    if classifier.problem_model is not None:
        problem_probs = classifier.problem_model.predict_proba(query_vector)[0]
        ranked_idx = np.argsort(problem_probs)[::-1][:top_k]
        for idx in ranked_idx:
            problem_predictions.append(
                {
                    "label": str(classifier.problem_model.classes_[idx]),
                    "score": float(problem_probs[idx]),
                    "type": "problem_type",
                }
            )
        if problem_predictions:
            predicted_problem = problem_predictions[0]["label"]
            problem_confidence = problem_predictions[0]["score"]

    record_scores = cosine_similarity(query_vector, classifier.example_matrix).ravel()
    nearest_idx = np.argsort(record_scores)[::-1]
    if predicted_risk != "Not available" and "mit_risk_domain" in classifier.records.columns:
        same_risk = classifier.records["mit_risk_domain"].astype(str) == predicted_risk
        same_risk_idx = [idx for idx in nearest_idx if same_risk.iloc[idx]]
        if same_risk_idx:
            nearest_idx = np.array(same_risk_idx + [idx for idx in nearest_idx if not same_risk.iloc[idx]])
    nearest_idx = nearest_idx[:8]
    examples = classifier.records.iloc[nearest_idx].copy()
    examples["similarity_score"] = record_scores[nearest_idx]
    examples["similarity_weight"] = examples["similarity_score"].clip(lower=0.0)
    top_example_cluster = None
    if not examples.empty and "cluster_label" in examples.columns:
        try:
            top_example_cluster = int(examples.iloc[0]["cluster_label"])
        except Exception:
            top_example_cluster = None

    def weighted_top(column: str) -> str:
        if column not in examples.columns:
            return "Not available"
        subset = examples[[column, "similarity_weight"]].dropna()
        if subset.empty:
            return "Not available"
        exploded_rows: list[dict[str, Any]] = []
        for row in subset.itertuples():
            raw_value = str(getattr(row, column))
            parts = [part.strip() for part in re.split(r"\||,", raw_value) if part.strip()]
            if not parts:
                parts = [raw_value.strip()]
            for part in parts:
                if part and part.lower() != "nan":
                    exploded_rows.append({"label": part, "weight": float(row.similarity_weight)})
        if not exploded_rows:
            return "Not available"
        frame = pd.DataFrame(exploded_rows)
        scores = frame.groupby("label", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
        return str(scores.iloc[0]["label"])

    if predicted_risk == "Not available":
        predicted_risk = weighted_top("mit_risk_domain")

    nearest_problem = weighted_top("gmf_known_ai_technical_failure")
    if predicted_problem == "Not available" or (problem_confidence is not None and problem_confidence < 0.18):
        predicted_problem = nearest_problem

    predicted_sector = weighted_top("csetv1_sector_of_deployment")

    if predicted_problem == "Not available" and top_example_cluster is not None and not classifier.summary.empty and {"cluster_label", "top_failure_type"}.issubset(classifier.summary.columns):
        cluster_row = classifier.summary[classifier.summary["cluster_label"] == top_example_cluster]
        if not cluster_row.empty:
            predicted_problem = _summary_problem_label(cluster_row.iloc[0]["top_failure_type"], predicted_risk) or "Not available"

    keep = [column for column in ["cluster_name", "incident_title", "report_title", "source_domain", "incident_year", "mit_risk_domain", "gmf_known_ai_technical_failure", "similarity_score"] if column in examples.columns]
    examples = examples[keep]
    examples = examples.rename(
        columns={
            "cluster_name": "Theme",
            "incident_title": "Incident title",
            "report_title": "Report title",
            "source_domain": "Source domain",
            "incident_year": "Year",
            "mit_risk_domain": "Risk category",
            "gmf_known_ai_technical_failure": "Related problem",
            "similarity_score": "Similarity",
        }
    )
    if "Similarity" in examples.columns:
        examples["Similarity"] = examples["Similarity"].map(lambda value: f"{value:.3f}")

    return {
        "text": text,
        "predictions": risk_predictions,
        "problem_predictions": problem_predictions,
        "predicted_risk": predicted_risk,
        "predicted_problem": predicted_problem,
        "predicted_sector": predicted_sector,
        "risk_confidence": risk_confidence,
        "problem_confidence": problem_confidence,
        "examples": examples.reset_index(drop=True),
    }


def prediction_bar_figure(predictions: list[dict[str, Any]]) -> go.Figure:
    if not predictions:
        return empty_figure("Run a prediction to see the top incident categories.")
    frame = pd.DataFrame(predictions).copy()
    frame["label_short"] = frame["label"].apply(lambda value: short_label(value, 28))
    fig = px.bar(
        frame.sort_values("score"),
        x="score",
        y="label_short",
        orientation="h",
        color="score",
        color_continuous_scale=["#dceef0", "#79c7c5", "#2b2d42"],
        custom_data=["label"],
    )
    fig.update_traces(hovertemplate="Incident category %{customdata[0]}<br>Predicted probability %{x:.3f}<extra></extra>")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_showscale=False,
        xaxis_title="Predicted probability",
        yaxis_title="Incident category",
    )
    return fig


def _split_problem_labels(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in re.split(r"\||,", text) if part.strip()]


def _primary_problem_label(value: Any) -> str | None:
    parts = _split_problem_labels(value)
    return parts[0] if parts else None


def _summary_problem_label(value: Any, predicted_risk: str | None = None) -> str | None:
    if value is None or pd.isna(value):
        return None
    segments = [segment.strip() for segment in str(value).split("|") if segment.strip()]
    if not segments:
        return None
    if predicted_risk:
        risk_text = str(predicted_risk).lower()
        preferred_terms: list[str] = []
        if "misinformation" in risk_text:
            preferred_terms = ["misinformation", "hallucination", "fabrication"]
        elif "misuse" in risk_text:
            preferred_terms = ["misuse", "unsafe", "security", "privacy", "exposure"]
        elif "discrimination" in risk_text or "toxicity" in risk_text:
            preferred_terms = ["bias", "toxicity", "discrimination"]
        elif "privacy" in risk_text or "security" in risk_text:
            preferred_terms = ["privacy", "security", "exposure"]
        elif "safety" in risk_text or "limitations" in risk_text:
            preferred_terms = ["failure", "bug", "capability", "generalization", "hardware", "latency"]
        for term in preferred_terms:
            for segment in segments:
                if term in segment.lower():
                    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", segment).strip()
                    return cleaned or None
    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", segments[0]).strip()
    return cleaned or None


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def metadata_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"year": None, "risk_category": None, "sector": None, "source_domain": None, "failure_type": None}

    sector = next((column for column in frame.columns if "sector" in column and frame[column].notna().any()), None)
    failure = next((column for column in frame.columns if "failure" in column and frame[column].notna().any()), None)
    return {
        "year": _first_existing(frame, ["incident_year", "report_year", "year"]),
        "risk_category": _first_existing(frame, ["mit_risk_domain", "risk_category"]),
        "sector": sector,
        "source_domain": _first_existing(frame, ["source_domain", "domain"]),
        "failure_type": failure,
    }


def _contains_any(series: pd.Series, values: list[str]) -> pd.Series:
    if not values:
        return pd.Series([True] * len(series), index=series.index)
    text = series.fillna("").astype(str)
    return text.apply(lambda item: any(value in item.split(" | ") or value == item for value in values))


def filter_cluster_records(
    records: pd.DataFrame,
    year_range: tuple[int, int] | None,
    selected_risks: list[str] | None,
    selected_domains: list[str] | None,
    keyword: str | None,
    selected_cluster: str | None,
) -> pd.DataFrame:
    if records.empty:
        return records.copy()

    frame = records.copy()
    cols = metadata_columns(frame)

    year_col = cols["year"]
    if year_range and year_col in frame.columns:
        low, high = year_range
        years = pd.to_numeric(frame[year_col], errors="coerce")
        frame = frame[years.between(low, high)]

    risk_col = cols["risk_category"]
    if selected_risks and risk_col in frame.columns:
        frame = frame[_contains_any(frame[risk_col], selected_risks)]

    domain_col = cols["source_domain"]
    if selected_domains and domain_col in frame.columns:
        frame = frame[frame[domain_col].isin(selected_domains)]

    if keyword:
        pattern = str(keyword).strip().lower()
        if pattern:
            search_cols = [column for column in ["input_text", "incident_title", "report_title"] if column in frame.columns]
            if search_cols:
                search_text = frame[search_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                frame = frame[search_text.str.contains(pattern, na=False, regex=False)]

    if selected_cluster and selected_cluster != "__all__":
        try:
            cluster_value = int(selected_cluster)
            frame = frame[frame["cluster_label"] == cluster_value]
        except ValueError:
            pass

    return frame.reset_index(drop=True)


def summarize_visible_clusters(summary: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()

    counts = (
        records.groupby("cluster_label", as_index=False)
        .agg(visible_records=("cluster_label", "size"), visible_incidents=("incident_id", "nunique"))
    )
    if summary.empty:
        output = counts.copy()
        output["cluster_name"] = output["cluster_label"].map(lambda value: f"Cluster {value}")
        output["top_keywords"] = ""
        return output.sort_values("visible_records", ascending=False)

    output = summary.merge(counts, on="cluster_label", how="inner")
    if "cluster_name" not in output.columns:
        output["cluster_name"] = output["cluster_label"].map(lambda value: f"Cluster {value}")
    if "top_keywords" not in output.columns:
        output["top_keywords"] = ""
    return output.sort_values("visible_records", ascending=False)


def cluster_metric_summary(records: pd.DataFrame, summary: pd.DataFrame) -> dict[str, int | str]:
    if records.empty:
        return {"records": 0, "incidents": 0, "clusters": 0, "largest_cluster": "N/A"}
    visible = summarize_visible_clusters(summary, records)
    largest = visible.iloc[0]["cluster_name"] if not visible.empty and "cluster_name" in visible.columns else "N/A"
    return {
        "records": int(len(records)),
        "incidents": int(records["incident_id"].nunique()) if "incident_id" in records.columns else 0,
        "clusters": int(records["cluster_label"].nunique()) if "cluster_label" in records.columns else 0,
        "largest_cluster": str(largest),
    }


def cluster_scatter_figure(records: pd.DataFrame) -> go.Figure:
    required = {"x", "y", "cluster_label"}
    if records.empty or not required.issubset(records.columns):
        return empty_figure("Run the clustering pipeline to create the NLP theme scatter plot.")

    frame = records.copy()
    if "cluster_name" not in frame.columns:
        frame["cluster_name"] = frame["cluster_label"].map(lambda value: f"Cluster {value}")
    hover_cols = [column for column in ["incident_title", "report_title", "source_domain", "mit_risk_domain", "incident_year"] if column in frame.columns]
    fig = px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster_name",
        hover_data=hover_cols,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.76, line=dict(width=0.6, color="#ffffff")))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=90),
        xaxis_title="Embedding dimension 1",
        yaxis_title="Embedding dimension 2",
        legend=dict(orientation="h", y=-0.22, x=0, title=None),
    )
    return fig


def cluster_size_figure(summary: pd.DataFrame, records: pd.DataFrame) -> go.Figure:
    visible = summarize_visible_clusters(summary, records)
    if visible.empty:
        return empty_figure("No cluster-size data is available for the current filters.")
    frame = visible.sort_values("visible_records", ascending=True).tail(15)
    fig = px.bar(
        frame,
        x="visible_records",
        y="cluster_name",
        orientation="h",
        color="visible_records",
        color_continuous_scale=["#dceef0", "#79c7c5", "#2b2d42"],
        hover_data=["visible_incidents", "top_keywords"] if "top_keywords" in frame.columns else ["visible_incidents"],
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Visible records",
        yaxis_title="Theme cluster",
        coloraxis_showscale=False,
    )
    return fig


def cluster_timeline_figure(records: pd.DataFrame) -> go.Figure:
    if records.empty:
        return empty_figure("No timeline data is available for the current filters.")
    year_col = metadata_columns(records)["year"]
    if not year_col or year_col not in records.columns:
        return empty_figure("Cluster timeline requires year metadata.")

    frame = records.copy()
    frame["year"] = pd.to_numeric(frame[year_col], errors="coerce")
    frame = frame.dropna(subset=["year"])
    frame = frame[frame["year"].between(1900, 2100)]
    if frame.empty:
        return empty_figure("Cluster timeline has no valid years after filtering.")

    frame["year"] = frame["year"].astype(int)
    if "cluster_name" not in frame.columns:
        frame["cluster_name"] = frame["cluster_label"].map(lambda value: f"Cluster {value}")
    counts = frame.groupby(["year", "cluster_name"], as_index=False).size().rename(columns={"size": "count"})
    top_clusters = counts.groupby("cluster_name")["count"].sum().sort_values(ascending=False).head(8).index
    counts = counts[counts["cluster_name"].isin(top_clusters)]
    fig = px.line(
        counts,
        x="year",
        y="count",
        color="cluster_name",
        markers=True,
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=90),
        xaxis_title="Year",
        yaxis_title="Records",
        legend=dict(orientation="h", y=-0.24, x=0, title=None),
    )
    return fig


def cluster_metadata_figure(records: pd.DataFrame) -> go.Figure:
    if records.empty:
        return empty_figure("No metadata profile is available for the current filters.")
    cols = metadata_columns(records)
    rows: list[dict[str, Any]] = []
    for label, column in [
        ("Risk category", cols["risk_category"]),
        ("Sector", cols["sector"]),
        ("Failure type", cols["failure_type"]),
        ("Source domain", cols["source_domain"]),
    ]:
        if not column or column not in records.columns:
            continue
        values = records[column].dropna().astype(str)
        if values.empty:
            continue
        split_values: list[str] = []
        for value in values:
            split_values.extend(part.strip() for part in value.split(" | ") if part.strip())
        if not split_values:
            continue
        counts = pd.Series(split_values).value_counts().head(8)
        for value, count in counts.items():
            rows.append({"metadata": label, "value": short_label(value, 42), "count": int(count)})

    if not rows:
        return empty_figure("No risk, sector, failure, or source metadata is available for these records.")
    frame = pd.DataFrame(rows)
    fig = px.bar(
        frame.sort_values("count", ascending=True),
        x="count",
        y="value",
        color="metadata",
        orientation="h",
        facet_row="metadata",
        color_discrete_sequence=PALETTE,
    )
    fig.update_yaxes(matches=None, title=None)
    fig.update_xaxes(title="Records")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title=None,
        showlegend=False,
    )
    return fig


def cluster_summary_table(summary: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    visible = summarize_visible_clusters(summary, records)
    if visible.empty:
        return pd.DataFrame({"message": ["Run the clustering pipeline or adjust filters to see discovered themes."]})
    columns = {
        "cluster_name": "Theme",
        "visible_records": "Visible records",
        "visible_incidents": "Visible incidents",
        "top_keywords": "Top keywords",
        "example_titles": "Example titles",
        "top_risk_category": "Top risk category",
        "top_sector": "Top sector",
    }
    keep = [column for column in columns if column in visible.columns]
    return visible[keep].rename(columns=columns).head(30)


def cluster_examples_table(records: pd.DataFrame, limit: int = 60) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame({"message": ["No clustered records match the current filters."]})
    frame = records.copy()
    if "cluster_name" not in frame.columns:
        frame["cluster_name"] = frame["cluster_label"].map(lambda value: f"Cluster {value}")
    title_col = _first_existing(frame, ["incident_title", "report_title"])
    columns = ["cluster_name", "incident_id"]
    if title_col:
        columns.append(title_col)
    for column in ["incident_year", "source_domain", "mit_risk_domain", "top_keywords"]:
        if column in frame.columns:
            columns.append(column)
    if "input_text" in frame.columns:
        frame["Text preview"] = frame["input_text"].fillna("").astype(str).str.slice(0, 240)
        columns.append("Text preview")
    output = frame[columns].head(limit).copy()
    rename = {
        "cluster_name": "Theme",
        "incident_id": "Incident ID",
        "incident_title": "Incident title",
        "report_title": "Report title",
        "incident_year": "Year",
        "source_domain": "Source domain",
        "mit_risk_domain": "Risk category",
    }
    return output.rename(columns=rename)
