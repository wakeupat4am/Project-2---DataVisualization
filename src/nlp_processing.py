from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import load_csv_files
from .utils import first_matching_column, normalize_name


STRICT_INCIDENT_ID_CANDIDATES = [
    "incident_id",
    "incident_number",
    "incidentid",
    "incident",
]
REPORT_ID_CANDIDATES = [
    "report_number",
    "ref_number",
    "report_id",
    "reportid",
    "reference_number",
]
REPORT_LINK_CANDIDATES = [
    "reports",
    "report_ids",
    "linked_reports",
    "report_numbers",
    "report_number",
    "ref_numbers",
]
DATE_CANDIDATES = [
    "date",
    "date_published",
    "incident_date",
    "date_of_incident_year",
    "published",
]
TITLE_CANDIDATES = ["title", "name", "headline"]
DESCRIPTION_CANDIDATES = ["description", "summary", "full_description", "short_description", "text"]
TEXT_CANDIDATES = ["text", "description", "summary", "full_text", "article_text"]
URL_CANDIDATES = ["url", "source_url", "link"]
DOMAIN_CANDIDATES = ["source_domain", "domain", "publisher_domain"]

LABEL_INCLUDE_TOKENS = (
    "risk",
    "harm",
    "failure",
    "sector",
    "intent",
    "goal",
    "technology",
    "technique",
    "application",
    "task",
    "function",
    "domain",
    "subdomain",
    "severity",
    "near_miss",
    "deployed",
    "autonomy",
    "public_sector",
    "physical_system",
    "problem_nature",
    "infrastructure",
    "location_region",
    "protected_characteristic",
    "rights_violation",
    "detrimental_content",
    "minor",
    "critical_services",
)
LABEL_EXCLUDE_TOKENS = (
    "id",
    "namespace",
    "published",
    "date",
    "month",
    "day",
    "year",
    "annotator",
    "reviewer",
    "quality",
    "status",
    "snippet",
    "discussion",
    "notes",
    "description",
    "text",
    "title",
    "url",
    "source_domain",
    "source_url",
    "location_city",
    "location_state",
    "location_country",
    "entities",
    "lives_lost",
    "injuries",
    "financial_cost",
    "quantity",
    "quantities",
    "image",
    "downloaded",
    "modified",
    "submitted",
    "mongodb",
)


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("processed_data")
    split: tuple[float, float, float] = (0.8, 0.0, 0.2)
    random_state: int = 42
    min_text_chars: int = 40
    max_class_values_to_print: int = 12


@dataclass
class PipelineResult:
    report_level: pd.DataFrame
    incident_level: pd.DataFrame
    classification_ready: pd.DataFrame
    label_columns: list[str]
    split_map: dict[str, str]


def normalize_text(value: object) -> str:
    """Return a safe, whitespace-normalized string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(item) for item in value if item is not None)
    else:
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
    text = str(value).replace("\x00", " ").strip()
    return re.sub(r"\s+", " ", text)


def normalize_key(value: object) -> str:
    return normalize_text(value)


def string_column(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if not column or column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype="object")
    return frame[column].map(normalize_text)


def parse_list_like(value: object) -> list[str]:
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = normalize_text(value)
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [normalize_text(item) for item in parsed if normalize_text(item)]
    except (SyntaxError, ValueError):
        pass

    stripped = text.strip("[]")
    if not stripped:
        return []
    return [item.strip().strip('"').strip("'") for item in stripped.split(",") if item.strip()]


def extract_domain(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"https://{text}")
    return parsed.netloc.lower().removeprefix("www.")


def extract_year(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    years = parsed.dt.year
    if years.notna().any():
        return years.astype("Int64")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where((numeric >= 1900) & (numeric <= 2100))
    return numeric.astype("Int64")


def require_column(
    columns: pd.Index,
    candidates: list[str],
    table_name: str,
    semantic_name: str,
) -> str:
    detected = first_matching_column(columns, candidates)
    if detected is None:
        available = ", ".join(columns)
        expected = ", ".join(candidates)
        raise ValueError(
            f"Could not detect {semantic_name} for `{table_name}`. "
            f"Tried candidates: {expected}. Available columns: {available}"
        )
    return detected


def detect_optional_column(columns: pd.Index, candidates: list[str]) -> str | None:
    return first_matching_column(columns, candidates)


def print_available_columns(datasets: dict[str, pd.DataFrame]) -> None:
    print("\n=== Available CSV Columns ===")
    for name, frame in sorted(datasets.items()):
        print(f"\n{name}.csv ({len(frame):,} rows, {len(frame.columns):,} columns)")
        print(", ".join(frame.columns))


def prepare_incidents(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    incident_col = require_column(frame.columns, STRICT_INCIDENT_ID_CANDIDATES, "incidents.csv", "incident ID")
    title_col = detect_optional_column(frame.columns, TITLE_CANDIDATES)
    desc_col = detect_optional_column(frame.columns, DESCRIPTION_CANDIDATES)
    date_col = detect_optional_column(frame.columns, DATE_CANDIDATES)
    report_links_col = detect_optional_column(frame.columns, REPORT_LINK_CANDIDATES)

    incidents = pd.DataFrame(index=frame.index)
    incidents["incident_id"] = frame[incident_col].map(normalize_key)
    incidents["incident_title"] = string_column(frame, title_col)
    incidents["incident_description"] = string_column(frame, desc_col)
    incidents["incident_date"] = pd.to_datetime(frame[date_col], errors="coerce", utc=True) if date_col else pd.NaT
    incidents["incident_year"] = extract_year(frame[date_col]) if date_col else pd.Series([pd.NA] * len(frame), dtype="Int64")
    incidents["incident_report_ids"] = frame[report_links_col].apply(parse_list_like) if report_links_col else [[] for _ in range(len(frame))]

    for source_col, output_col in [
        ("alleged_deployer_of_ai_system", "ai_deployer"),
        ("alleged_developer_of_ai_system", "ai_developer"),
        ("alleged_harmed_or_nearly_harmed_parties", "harmed_parties"),
    ]:
        incidents[output_col] = string_column(frame, source_col if source_col in frame.columns else None)

    incidents = incidents[incidents["incident_id"].ne("")]
    incidents = incidents.drop_duplicates(subset=["incident_id"], keep="first")
    detected = {
        "incident_id": incident_col,
        "title": title_col,
        "description": desc_col,
        "date": date_col,
        "report_links": report_links_col,
    }
    return incidents.reset_index(drop=True), detected


def prepare_reports(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    report_col = detect_optional_column(frame.columns, REPORT_ID_CANDIDATES)
    incident_col = detect_optional_column(frame.columns, STRICT_INCIDENT_ID_CANDIDATES)
    title_col = detect_optional_column(frame.columns, TITLE_CANDIDATES)
    text_col = detect_optional_column(frame.columns, TEXT_CANDIDATES)
    date_col = detect_optional_column(frame.columns, DATE_CANDIDATES)
    url_col = detect_optional_column(frame.columns, URL_CANDIDATES)
    domain_col = detect_optional_column(frame.columns, DOMAIN_CANDIDATES)

    if report_col is None and incident_col is None:
        available = ", ".join(frame.columns)
        raise ValueError(
            "Could not detect either a report ID or an incident ID for `reports.csv`. "
            f"Available columns: {available}"
        )

    reports = pd.DataFrame(index=frame.index)
    reports["report_id"] = frame[report_col].map(normalize_key) if report_col else frame.index.map(lambda idx: f"generated_report_{idx}")
    reports["direct_incident_id"] = frame[incident_col].map(normalize_key) if incident_col else ""
    reports["report_title"] = string_column(frame, title_col)
    reports["report_text"] = string_column(frame, text_col)
    reports["report_url"] = string_column(frame, url_col)
    reports["source_domain"] = string_column(frame, domain_col) if domain_col else reports["report_url"].map(extract_domain)
    reports["source_domain"] = reports["source_domain"].replace("", pd.NA).fillna("Unknown")
    reports["report_date"] = pd.to_datetime(frame[date_col], errors="coerce", utc=True) if date_col else pd.NaT
    reports["report_year"] = extract_year(frame[date_col]) if date_col else pd.Series([pd.NA] * len(frame), dtype="Int64")
    reports = reports[reports["report_id"].ne("")]

    detected = {
        "report_id": report_col,
        "incident_id": incident_col,
        "title": title_col,
        "text": text_col,
        "date": date_col,
        "url": url_col,
        "domain": domain_col,
    }
    return reports.reset_index(drop=True), detected


def join_reports_to_incidents(
    reports: pd.DataFrame,
    incidents: pd.DataFrame,
    incident_detection: dict[str, str | None],
    report_detection: dict[str, str | None],
) -> pd.DataFrame:
    incident_meta_cols = [
        "incident_id",
        "incident_title",
        "incident_description",
        "incident_date",
        "incident_year",
        "ai_deployer",
        "ai_developer",
        "harmed_parties",
    ]

    direct_matches = 0
    if report_detection.get("incident_id"):
        direct_matches = reports["direct_incident_id"].isin(incidents["incident_id"]).sum()

    if direct_matches > 0:
        print(f"\nJoin strategy: direct `incident_id` in reports.csv ({direct_matches:,} matching rows).")
        joined = reports.rename(columns={"direct_incident_id": "incident_id"}).merge(
            incidents[incident_meta_cols], on="incident_id", how="inner"
        )
        return joined

    if not incident_detection.get("report_links"):
        raise ValueError(
            "Could not join reports to incidents. `reports.csv` has no usable incident ID, "
            "and `incidents.csv` has no detectable report-link column."
        )

    links = incidents[["incident_id", "incident_report_ids"]].explode("incident_report_ids")
    links = links.rename(columns={"incident_report_ids": "report_id"})
    links["report_id"] = links["report_id"].map(normalize_key)
    links = links[links["report_id"].ne("")]
    if links.empty:
        raise ValueError("Incident report-link column was detected, but no usable report IDs were parsed from it.")

    print(f"\nJoin strategy: exploded incident report links ({len(links):,} incident-report links).")
    joined = reports.drop(columns=["direct_incident_id"]).merge(links, on="report_id", how="inner")
    joined = joined.merge(incidents[incident_meta_cols], on="incident_id", how="left")
    return joined


def label_source_name(dataset_name: str) -> str:
    text = dataset_name.replace("classifications_", "")
    return normalize_name(text) or normalize_name(dataset_name)


def infer_label_columns(frame: pd.DataFrame, key_col: str) -> list[str]:
    label_columns: list[str] = []
    for column in frame.columns:
        normalized = normalize_name(column)
        if column == key_col:
            continue
        if any(token == normalized or token in normalized for token in LABEL_EXCLUDE_TOKENS):
            continue
        if not any(token == normalized or token in normalized for token in LABEL_INCLUDE_TOKENS):
            continue

        values = frame[column].map(normalize_text)
        values = values[values.ne("")]
        if values.empty:
            continue

        unique_count = values.nunique(dropna=True)
        unique_ratio = unique_count / max(len(values), 1)
        avg_length = values.str.len().mean()
        if unique_count > 600 or unique_ratio > 0.85 or avg_length > 160:
            continue
        label_columns.append(column)
    return label_columns


def clean_label_value(value: object) -> str:
    text = normalize_text(value)
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na", "[]"}:
        return ""
    return text


def collapse_unique_values(series: pd.Series) -> str:
    values: list[str] = []
    for value in series:
        text = clean_label_value(value)
        if not text:
            continue
        parsed_items = parse_list_like(text) if text.startswith("[") and text.endswith("]") else [text]
        for item in parsed_items:
            item = clean_label_value(item)
            if item:
                values.append(item)

    deduped = sorted(set(values), key=lambda item: item.lower())
    return " | ".join(deduped)


def aggregate_label_frame(frame: pd.DataFrame, key: str, label_columns: list[str]) -> pd.DataFrame:
    if frame.empty or not label_columns:
        return pd.DataFrame(columns=[key])
    aggregation = {column: collapse_unique_values for column in label_columns}
    grouped = frame.groupby(key, as_index=False).agg(aggregation)
    for column in label_columns:
        grouped[column] = grouped[column].replace("", pd.NA)
    return grouped


def build_classification_labels(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    incident_label_frames: list[pd.DataFrame] = []
    report_label_frames: list[pd.DataFrame] = []
    label_columns: list[str] = []

    print("\n=== Classification Label Detection ===")
    for name, frame in sorted(datasets.items()):
        if not name.startswith("classifications_") or frame.empty or "load_error" in frame.columns:
            continue

        incident_key = detect_optional_column(frame.columns, STRICT_INCIDENT_ID_CANDIDATES)
        report_key = detect_optional_column(frame.columns, REPORT_ID_CANDIDATES)
        key_col = incident_key or report_key
        key_name = "incident_id" if incident_key else "report_id"
        if key_col is None:
            print(f"{name}.csv: skipped; no incident/report key detected.")
            continue

        inferred = infer_label_columns(frame, key_col)
        if not inferred:
            print(f"{name}.csv: skipped; no categorical label columns inferred.")
            continue

        source = label_source_name(name)
        renamed = pd.DataFrame()
        renamed[key_name] = frame[key_col].map(normalize_key)
        output_columns = []
        for column in inferred:
            output_col = f"{source}_{normalize_name(column)}"
            output_columns.append(output_col)
            renamed[output_col] = frame[column].map(clean_label_value)

        renamed = renamed[renamed[key_name].ne("")]
        aggregated = aggregate_label_frame(renamed, key_name, output_columns)
        label_columns.extend(output_columns)
        if key_name == "incident_id":
            incident_label_frames.append(aggregated)
        else:
            report_label_frames.append(aggregated)

        print(f"{name}.csv: key={key_name} ({key_col}); labels={', '.join(output_columns)}")

    incident_labels = merge_label_frames(incident_label_frames, "incident_id")
    report_labels = merge_label_frames(report_label_frames, "report_id")
    return incident_labels, report_labels, sorted(dict.fromkeys(label_columns))


def merge_label_frames(frames: list[pd.DataFrame], key: str) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty and key in frame.columns]
    if not frames:
        return pd.DataFrame(columns=[key])
    return reduce(lambda left, right: left.merge(right, on=key, how="outer"), frames)


def combine_text_fields(row: pd.Series, field_specs: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for label, column in field_specs:
        value = normalize_text(row.get(column, ""))
        if not value:
            continue
        normalized_value = value.lower()
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        parts.append(f"{label}: {value}")
    return normalize_text(" ".join(parts))


def add_report_input_text(report_level: pd.DataFrame) -> pd.DataFrame:
    frame = report_level.copy()
    text_fields = [
        ("Report title", "report_title"),
        ("Incident title", "incident_title"),
        ("Incident description", "incident_description"),
        ("Report text", "report_text"),
        ("Source domain", "source_domain"),
        ("Source URL", "report_url"),
    ]
    frame["input_text"] = frame.apply(lambda row: combine_text_fields(row, text_fields), axis=1)
    return frame


def build_incident_level(
    incidents: pd.DataFrame,
    report_level: pd.DataFrame,
    incident_labels: pd.DataFrame,
) -> pd.DataFrame:
    if report_level.empty:
        report_agg = pd.DataFrame(columns=["incident_id", "joined_report_count", "source_domains", "linked_report_titles"])
    else:
        report_agg = (
            report_level.groupby("incident_id", as_index=False)
            .agg(
                joined_report_count=("report_id", "nunique"),
                source_domains=("source_domain", collapse_unique_values),
                linked_report_titles=("report_title", collapse_unique_values),
            )
        )

    incident_level = incidents.drop(columns=["incident_report_ids"], errors="ignore").merge(
        report_agg, on="incident_id", how="left"
    )
    incident_level["joined_report_count"] = incident_level["joined_report_count"].fillna(0).astype(int)
    for column in ["source_domains", "linked_report_titles"]:
        incident_level[column] = incident_level[column].fillna("")

    if not incident_labels.empty:
        incident_level = incident_level.merge(incident_labels, on="incident_id", how="left")

    text_fields = [
        ("Incident title", "incident_title"),
        ("Incident description", "incident_description"),
        ("Report titles", "linked_report_titles"),
        ("Source domains", "source_domains"),
    ]
    incident_level["input_text"] = incident_level.apply(lambda row: combine_text_fields(row, text_fields), axis=1)
    return incident_level


def parse_split_spec(split: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in split.split("/") if part.strip()]
    if len(parts) not in {2, 3}:
        raise ValueError("Split must look like `80/20`, `0.8/0.2`, `70/15/15`, or `0.7/0.15/0.15`.")

    values = [float(part) for part in parts]
    if any(value <= 0 for value in values):
        raise ValueError("Split values must be positive.")
    if sum(values) > 1.5:
        values = [value / 100 for value in values]

    if len(values) == 2:
        train_size, test_size = values
        val_size = 0.0
    else:
        train_size, val_size, test_size = values

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split values must sum to 1.0 or 100. Received total={total:.4f}.")
    return train_size, val_size, test_size


def build_incident_split_groups(incident_ids: pd.Series, report_level: pd.DataFrame) -> pd.Series:
    """Group incidents that share report IDs so exact report text cannot cross splits."""
    ids = sorted(incident_ids.dropna().astype(str).loc[lambda s: s.ne("")].unique().tolist())
    parent = {incident_id: incident_id for incident_id in ids}

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
    if not report_level.empty and {"incident_id", "report_id"}.issubset(report_level.columns):
        report_links = report_level[["incident_id", "report_id"]].dropna().drop_duplicates()
        report_links["incident_id"] = report_links["incident_id"].astype(str)
        for _, group in report_links.groupby("report_id"):
            linked_incidents = [incident_id for incident_id in group["incident_id"].unique().tolist() if incident_id in parent]
            if len(linked_incidents) <= 1:
                continue
            shared_report_count += 1
            first = linked_incidents[0]
            for incident_id in linked_incidents[1:]:
                union(first, incident_id)

    group_map = {incident_id: find(incident_id) for incident_id in ids}
    component_count = len(set(group_map.values()))
    print(
        "\nSplit grouping: "
        f"{len(ids):,} incidents collapsed into {component_count:,} incident/report components; "
        f"{shared_report_count:,} report IDs link multiple incidents."
    )
    return pd.Series(group_map, name="split_group")


def create_group_split(
    incident_ids: pd.Series,
    split: tuple[float, float, float],
    random_state: int,
    group_ids: pd.Series | None = None,
) -> dict[str, str]:
    ids = sorted(incident_ids.dropna().astype(str).loc[lambda s: s.ne("")].unique().tolist())
    if not ids:
        raise ValueError("Cannot create train/test split because no incident IDs are available.")
    if len(ids) == 1:
        return {ids[0]: "train"}

    train_size, val_size, test_size = split
    if group_ids is None:
        item_to_group = {incident_id: incident_id for incident_id in ids}
    else:
        item_to_group = {
            str(incident_id): str(group_ids.loc[str(incident_id)])
            for incident_id in ids
            if str(incident_id) in group_ids.index
        }
    groups = sorted(set(item_to_group.values()))

    if len(groups) == 1:
        return {incident_id: "train" for incident_id in ids}

    train_ids, holdout_ids = train_test_split(
        groups,
        train_size=train_size,
        test_size=val_size + test_size,
        random_state=random_state,
        shuffle=True,
    )

    group_split_map = {group_id: "train" for group_id in train_ids}
    if val_size == 0:
        group_split_map.update({group_id: "test" for group_id in holdout_ids})
        return {incident_id: group_split_map[item_to_group[incident_id]] for incident_id in ids}

    relative_test_size = test_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True,
    )
    group_split_map.update({group_id: "val" for group_id in val_ids})
    group_split_map.update({group_id: "test" for group_id in test_ids})
    return {incident_id: group_split_map[item_to_group[incident_id]] for incident_id in ids}


def apply_split(frame: pd.DataFrame, split_map: dict[str, str]) -> pd.DataFrame:
    output = frame.copy()
    output["incident_id"] = output["incident_id"].astype(str)
    output["split"] = output["incident_id"].map(split_map)
    return output


def remove_short_and_duplicate_examples(
    frame: pd.DataFrame,
    min_text_chars: int,
    label_columns: list[str] | None = None,
    require_label: bool = False,
) -> pd.DataFrame:
    output = frame.copy()
    output["input_text"] = output["input_text"].map(normalize_text)
    output = output[output["input_text"].str.len() >= min_text_chars]

    if require_label:
        if not label_columns:
            raise ValueError("No label columns were detected, so a classification-ready dataset cannot be created.")
        has_label = output[label_columns].notna().any(axis=1)
        output = output[has_label]

    dedupe_columns = ["incident_id", "report_id", "input_text"] if "report_id" in output.columns else ["incident_id", "input_text"]
    output = output.drop_duplicates(subset=dedupe_columns, keep="first")
    return output.reset_index(drop=True)


def verify_no_group_leakage(frame: pd.DataFrame, dataset_name: str) -> None:
    if frame.empty:
        return
    split_counts = frame.groupby("incident_id")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        examples = ", ".join(leaking.head(10).index.astype(str))
        raise ValueError(f"Data leakage detected in {dataset_name}: incidents appear in multiple splits: {examples}")
    print(f"Leakage check passed for {dataset_name}: every incident_id belongs to exactly one split.")


def verify_no_report_leakage(frame: pd.DataFrame, dataset_name: str) -> None:
    if frame.empty or "report_id" not in frame.columns:
        return
    split_counts = frame.groupby("report_id")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        examples = ", ".join(leaking.head(10).index.astype(str))
        raise ValueError(f"Data leakage detected in {dataset_name}: report IDs appear in multiple splits: {examples}")
    print(f"Leakage check passed for {dataset_name}: every report_id belongs to exactly one split.")


def explode_label_values(series: pd.Series) -> pd.Series:
    values: list[str] = []
    for value in series.dropna():
        text = clean_label_value(value)
        if not text:
            continue
        values.extend([part.strip() for part in text.split(" | ") if part.strip()])
    return pd.Series(values, dtype="object")


def print_label_distribution(
    frame: pd.DataFrame,
    label_columns: list[str],
    title: str,
    max_values: int,
) -> None:
    print(f"\n=== Class Distribution: {title} ===")
    if frame.empty:
        print("No rows available.")
        return
    for column in label_columns:
        values = explode_label_values(frame[column]) if column in frame.columns else pd.Series(dtype="object")
        if values.empty:
            print(f"\n{column}: no labels")
            continue
        print(f"\n{column} ({len(values):,} assigned labels, {values.nunique():,} classes)")
        print(values.value_counts().head(max_values).to_string())


def print_summary(
    raw_incidents: pd.DataFrame,
    raw_reports: pd.DataFrame,
    report_level: pd.DataFrame,
    incident_level: pd.DataFrame,
    classification_ready: pd.DataFrame,
    label_columns: list[str],
    config: PipelineConfig,
) -> None:
    print("\n=== Processing Summary ===")
    print(f"Raw incidents: {len(raw_incidents):,}")
    print(f"Raw reports: {len(raw_reports):,}")
    print(f"Joined report-level rows: {len(report_level):,}")
    print(f"Incident-level rows: {len(incident_level):,}")
    print(f"Classification-ready rows: {len(classification_ready):,}")
    print(f"Detected label columns: {len(label_columns):,}")
    if label_columns:
        print(", ".join(label_columns))

    missing_columns = ["input_text", "split", *label_columns]
    missing_columns = [column for column in missing_columns if column in classification_ready.columns]
    print("\nMissing values in classification_ready:")
    print(classification_ready[missing_columns].isna().sum().sort_values(ascending=False).to_string())

    print("\nIncidents by split:")
    print(incident_level.groupby("split")["incident_id"].nunique().to_string())

    print("\nReport-level rows by split:")
    print(report_level["split"].value_counts().to_string())

    print("\nUnique report IDs by split:")
    print(report_level.groupby("split")["report_id"].nunique().to_string())

    print("\nClassification rows by split:")
    print(classification_ready["split"].value_counts().to_string())

    print_label_distribution(classification_ready, label_columns, "all classification rows", config.max_class_values_to_print)
    for split_name in ["train", "val", "test"]:
        subset = classification_ready[classification_ready["split"] == split_name]
        if not subset.empty:
            print_label_distribution(subset, label_columns, split_name, config.max_class_values_to_print)


def save_outputs(
    result: PipelineResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.report_level.to_csv(output_dir / "report_level_processed.csv", index=False)
    result.incident_level.to_csv(output_dir / "incident_level_processed.csv", index=False)
    result.classification_ready.to_csv(output_dir / "classification_ready.csv", index=False)
    result.classification_ready[result.classification_ready["split"] == "train"].to_csv(
        output_dir / "classification_train.csv", index=False
    )
    result.classification_ready[result.classification_ready["split"] == "test"].to_csv(
        output_dir / "classification_test.csv", index=False
    )
    if (result.classification_ready["split"] == "val").any():
        result.classification_ready[result.classification_ready["split"] == "val"].to_csv(
            output_dir / "classification_val.csv", index=False
        )


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    datasets = load_csv_files(config.data_dir)
    if not datasets:
        raise ValueError(f"No CSV files were found in `{config.data_dir}`.")
    if "incidents" not in datasets:
        raise ValueError("`incidents.csv` is required but was not found in the data folder.")
    if "reports" not in datasets:
        raise ValueError("`reports.csv` is required but was not found in the data folder.")

    print_available_columns(datasets)

    incidents, incident_detection = prepare_incidents(datasets["incidents"])
    reports, report_detection = prepare_reports(datasets["reports"])
    print("\n=== Detected Join Columns ===")
    print(f"incidents.csv: {incident_detection}")
    print(f"reports.csv: {report_detection}")

    report_level = join_reports_to_incidents(reports, incidents, incident_detection, report_detection)
    incident_labels, report_labels, label_columns = build_classification_labels(datasets)

    if not incident_labels.empty:
        report_level = report_level.merge(incident_labels, on="incident_id", how="left")
    if not report_labels.empty:
        report_level = report_level.merge(report_labels, on="report_id", how="left")

    report_level = add_report_input_text(report_level)
    report_level = remove_short_and_duplicate_examples(report_level, config.min_text_chars)

    incident_level = build_incident_level(incidents, report_level, incident_labels)
    split_groups = build_incident_split_groups(incident_level["incident_id"], report_level)
    split_map = create_group_split(incident_level["incident_id"], config.split, config.random_state, split_groups)
    report_level = apply_split(report_level, split_map)
    incident_level = apply_split(incident_level, split_map)

    classification_ready = remove_short_and_duplicate_examples(
        report_level,
        config.min_text_chars,
        label_columns=label_columns,
        require_label=True,
    )

    verify_no_group_leakage(report_level, "report_level_processed")
    verify_no_group_leakage(classification_ready, "classification_ready")
    verify_no_report_leakage(report_level, "report_level_processed")
    verify_no_report_leakage(classification_ready, "classification_ready")

    result = PipelineResult(
        report_level=report_level,
        incident_level=incident_level,
        classification_ready=classification_ready,
        label_columns=label_columns,
        split_map=split_map,
    )
    save_outputs(result, config.output_dir)
    print_summary(datasets["incidents"], datasets["reports"], report_level, incident_level, classification_ready, label_columns, config)

    print("\n=== Saved Outputs ===")
    for path in sorted(config.output_dir.glob("classification_*.csv")):
        print(path)
    print(config.output_dir / "incident_level_processed.csv")
    print(config.output_dir / "report_level_processed.csv")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build NLP-ready processed datasets from the AI Incident Database CSVs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder containing incidents.csv, reports.csv, and classification CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("processed_data"), help="Folder where processed CSV outputs will be written.")
    parser.add_argument("--split", default="80/20", help="Use `80/20` for train/test or `70/15/15` for train/val/test.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for incident-level splitting.")
    parser.add_argument("--min-text-chars", type=int, default=40, help="Drop examples with shorter input_text.")
    parser.add_argument("--max-class-values-to-print", type=int, default=12, help="Top class values printed per label column.")
    return parser


def main(argv: list[str] | None = None) -> PipelineResult:
    args = build_arg_parser().parse_args(argv)
    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=parse_split_spec(args.split),
        random_state=args.random_state,
        min_text_chars=args.min_text_chars,
        max_class_values_to_print=args.max_class_values_to_print,
    )
    return run_pipeline(config)


if __name__ == "__main__":
    main()
