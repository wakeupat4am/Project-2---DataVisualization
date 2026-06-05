from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .visualizations import PALETTE, empty_figure, short_label


CLUSTER_OUTPUT_DIR = Path("cluster_outputs/sbert_report_clusters")


@dataclass
class ClusterOutputs:
    records: pd.DataFrame
    summary: pd.DataFrame
    metadata_distribution: pd.DataFrame
    year_distribution: pd.DataFrame
    notes: list[str]


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
