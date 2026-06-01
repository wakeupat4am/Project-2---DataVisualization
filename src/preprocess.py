from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from .data_loader import load_csv_files
from .utils import coalesce_text, first_matching_column, normalize_name


INCIDENT_ID_CANDIDATES = [
    "incident_id",
    "incident_number",
    "incidentid",
    "id",
]
REPORT_ID_CANDIDATES = [
    "report_number",
    "ref_number",
    "report_id",
    "id",
]
DATE_CANDIDATES = [
    "date",
    "date_published",
    "incident_date",
    "date_of_incident_year",
]
TITLE_CANDIDATES = ["title", "name", "headline"]
DESCRIPTION_CANDIDATES = ["description", "text", "summary"]
URL_CANDIDATES = ["url", "source_url", "link"]
DOMAIN_CANDIDATES = ["source_domain", "domain", "publisher_domain"]
COUNTRY_CANDIDATES = [
    "location_country_two_letters",
    "country",
    "location_country",
    "location_region",
]
LOCATION_CANDIDATES = [
    "location_city",
    "location_state_province_two_letters",
    "location_country_two_letters",
    "location_region",
    "location",
]
RISK_CANDIDATES = [
    "risk_domain",
    "category",
    "risk_category",
    "harm_domain",
    "known_ai_technical_failure",
]


@dataclass
class PreparedData:
    datasets: dict[str, pd.DataFrame]
    incidents: pd.DataFrame
    reports: pd.DataFrame
    incident_reports: pd.DataFrame
    incident_risk_long: pd.DataFrame
    incident_location: pd.DataFrame
    metadata: dict[str, Any]
    notes: list[str]


def _safe_copy(frame: pd.DataFrame | None) -> pd.DataFrame:
    return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _as_string_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if not column or column not in frame.columns:
        return pd.Series(dtype="object")
    return frame[column].astype("string").fillna("")


def _parse_report_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    text = text.strip("[]")
    if not text:
        return []
    return [item.strip().strip('"').strip("'") for item in text.split(",") if item.strip()]


def _extract_domain(url: object) -> str | None:
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return None
    text = str(url).strip()
    if not text:
        return None
    parsed = urlparse(text)
    host = parsed.netloc.lower()
    return host or None


def _extract_year(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    year = parsed.dt.year
    if year.notna().any():
        return year.astype("Int64")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where((numeric >= 1900) & (numeric <= 2100))
    return numeric.astype("Int64")


def _detect_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    return {
        "incident_id": first_matching_column(frame.columns, INCIDENT_ID_CANDIDATES),
        "report_id": first_matching_column(frame.columns, REPORT_ID_CANDIDATES),
        "date": first_matching_column(frame.columns, DATE_CANDIDATES),
        "title": first_matching_column(frame.columns, TITLE_CANDIDATES),
        "description": first_matching_column(frame.columns, DESCRIPTION_CANDIDATES),
        "url": first_matching_column(frame.columns, URL_CANDIDATES),
        "domain": first_matching_column(frame.columns, DOMAIN_CANDIDATES),
        "country": first_matching_column(frame.columns, COUNTRY_CANDIDATES),
        "location": first_matching_column(frame.columns, LOCATION_CANDIDATES),
        "risk": first_matching_column(frame.columns, RISK_CANDIDATES),
    }


def _select_primary_table(datasets: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    for name in names:
        if name in datasets and "load_error" not in datasets[name].columns:
            return datasets[name].copy()
    return pd.DataFrame()


def _prepare_incidents(frame: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    if frame.empty:
        notes.append("`incidents.csv` was not found. The dashboard can only show partial views.")
        return pd.DataFrame(
            columns=[
                "incident_id",
                "incident_title",
                "incident_description",
                "incident_date",
                "incident_year",
                "report_ids",
                "report_count",
                "search_text",
            ]
        )

    detected = _detect_columns(frame)
    incident_col = detected["incident_id"]
    title_col = detected["title"]
    desc_col = detected["description"]
    date_col = detected["date"]
    report_links_col = first_matching_column(frame.columns, ["reports", "report_ids", "linked_reports"])

    incidents = frame.copy()
    if incident_col is None:
        incidents["incident_id"] = incidents.index.astype(str)
        notes.append("Incident ID column was not detected automatically. Row index is being used instead.")
    else:
        incidents["incident_id"] = incidents[incident_col].astype("string").fillna("").str.strip()

    incidents["incident_title"] = _as_string_series(incidents, title_col)
    incidents["incident_description"] = _as_string_series(incidents, desc_col)
    if date_col:
        incidents["incident_date"] = pd.to_datetime(incidents[date_col], errors="coerce", utc=True)
        incidents["incident_year"] = _extract_year(incidents[date_col])
    else:
        incidents["incident_date"] = pd.NaT
        incidents["incident_year"] = pd.Series([pd.NA] * len(incidents), dtype="Int64")
        notes.append("No incident date column was detected. Time-based views may be incomplete.")

    if report_links_col:
        incidents["report_ids"] = incidents[report_links_col].apply(_parse_report_list)
    else:
        incidents["report_ids"] = [[] for _ in range(len(incidents))]
        notes.append("Incident-to-report links were not found. Attention metrics may be limited.")

    incidents["report_count"] = incidents["report_ids"].apply(len)
    incidents["search_text"] = [
        coalesce_text(title, description)
        for title, description in zip(incidents["incident_title"], incidents["incident_description"])
    ]
    incidents = incidents[incidents["incident_id"].ne("")]
    return incidents


def _prepare_reports(frame: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    if frame.empty:
        notes.append("`reports.csv` was not found. Report-level attention views will be limited.")
        return pd.DataFrame(
            columns=["report_id", "report_title", "report_text", "report_url", "source_domain", "report_year"]
        )

    detected = _detect_columns(frame)
    report_col = detected["report_id"]
    title_col = detected["title"]
    desc_col = detected["description"]
    date_col = detected["date"]
    url_col = detected["url"]
    domain_col = detected["domain"]

    reports = frame.copy()
    if report_col is None:
        reports["report_id"] = reports.index.astype(str)
        notes.append("Report ID column was not detected automatically. Row index is being used instead.")
    else:
        reports["report_id"] = reports[report_col].astype("string").fillna("").str.strip()

    reports["report_title"] = _as_string_series(reports, title_col)
    reports["report_text"] = _as_string_series(reports, desc_col)
    reports["report_url"] = _as_string_series(reports, url_col)
    if domain_col:
        reports["source_domain"] = _as_string_series(reports, domain_col)
    else:
        reports["source_domain"] = reports["report_url"].apply(_extract_domain).astype("string")
    if date_col:
        reports["report_year"] = _extract_year(reports[date_col])
    else:
        reports["report_year"] = pd.Series([pd.NA] * len(reports), dtype="Int64")
    reports["source_domain"] = reports["source_domain"].replace("", pd.NA).fillna("Unknown")
    reports = reports[reports["report_id"].ne("")]
    return reports


def _build_incident_reports(incidents: pd.DataFrame, reports: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    links = incidents[["incident_id", "incident_title", "incident_description", "incident_year", "report_ids"]].explode(
        "report_ids"
    )
    links = links.rename(columns={"report_ids": "report_id"})
    links["report_id"] = links["report_id"].astype("string").fillna("").str.strip()
    links = links[links["report_id"].ne("")]

    if links.empty:
        if "incident_id" in reports.columns:
            fallback = reports.copy()
            fallback["incident_id"] = fallback["incident_id"].astype("string")
            return fallback
        notes.append("No usable incident-to-report links were created.")
        return pd.DataFrame(columns=["incident_id", "report_id", "source_domain", "report_year"])

    merged = links.merge(
        reports[["report_id", "report_title", "report_text", "report_url", "source_domain", "report_year"]],
        on="report_id",
        how="left",
    )
    return merged


def _classification_to_long(name: str, frame: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    if frame.empty or "load_error" in frame.columns:
        return pd.DataFrame(columns=["incident_id", "taxonomy_source", "risk_category", "secondary_label"])

    detected = _detect_columns(frame)
    incident_col = detected["incident_id"]
    if incident_col is None:
        notes.append(f"`{name}.csv` does not expose a detectable incident ID and was skipped for joins.")
        return pd.DataFrame(columns=["incident_id", "taxonomy_source", "risk_category", "secondary_label"])

    taxonomy = name.replace("classifications_", "").replace("_", " ")
    working = frame.copy()
    working["incident_id"] = working[incident_col].astype("string").fillna("").str.strip()
    working = working[working["incident_id"].ne("")]

    if name == "classifications_MIT":
        risk_col = first_matching_column(working.columns, ["risk_domain"])
        sub_col = first_matching_column(working.columns, ["risk_subdomain"])
        output = working[["incident_id"]].copy()
        output["taxonomy_source"] = "MIT"
        output["risk_category"] = _as_string_series(working, risk_col)
        output["secondary_label"] = _as_string_series(working, sub_col)
        return output[output["risk_category"].ne("")]

    if name == "classifications_GMF":
        risk_col = first_matching_column(working.columns, ["known_ai_technical_failure", "known_ai_goal"])
        secondary = first_matching_column(working.columns, ["known_ai_technology"])
        output = working[["incident_id"]].copy()
        output["taxonomy_source"] = "GMF"
        output["risk_category"] = _as_string_series(working, risk_col)
        output["secondary_label"] = _as_string_series(working, secondary)
        output = output[output["risk_category"].ne("")]
        output["risk_category"] = output["risk_category"].str.split(",")
        output = output.explode("risk_category")
        output["risk_category"] = output["risk_category"].astype("string").str.strip()
        return output[output["risk_category"].ne("")]

    if normalize_name(name).startswith("classifications_csetv1"):
        risk_col = first_matching_column(working.columns, ["ai_harm_level", "tangible_harm", "harm_domain"])
        secondary = first_matching_column(working.columns, ["sector_of_deployment", "location_region"])
        output = working[["incident_id"]].copy()
        output["taxonomy_source"] = "CSETv1"
        output["risk_category"] = _as_string_series(working, risk_col)
        output["secondary_label"] = _as_string_series(working, secondary)
        return output[output["risk_category"].ne("")]

    detected_risk = detected["risk"]
    if detected_risk:
        output = working[["incident_id"]].copy()
        output["taxonomy_source"] = taxonomy
        output["risk_category"] = _as_string_series(working, detected_risk)
        output["secondary_label"] = ""
        return output[output["risk_category"].ne("")]

    return pd.DataFrame(columns=["incident_id", "taxonomy_source", "risk_category", "secondary_label"])


def _prepare_locations(datasets: dict[str, pd.DataFrame], notes: list[str]) -> pd.DataFrame:
    location_sources = []
    for name, frame in datasets.items():
        if not name.startswith("classifications_") or frame.empty or "load_error" in frame.columns:
            continue
        detected = _detect_columns(frame)
        incident_col = detected["incident_id"]
        country_col = detected["country"]
        location_col = detected["location"]
        if incident_col is None or (country_col is None and location_col is None):
            continue
        subset = frame.copy()
        subset["incident_id"] = subset[incident_col].astype("string").fillna("").str.strip()
        subset["country"] = _as_string_series(subset, country_col).replace("", pd.NA)
        subset["location_label"] = _as_string_series(subset, location_col).replace("", pd.NA)
        subset["location_source"] = name
        subset = subset[["incident_id", "country", "location_label", "location_source"]]
        subset = subset[subset["incident_id"].ne("")]
        location_sources.append(subset)

    if not location_sources:
        notes.append("No reliable country/location metadata was detected for map views.")
        return pd.DataFrame(columns=["incident_id", "country", "location_label", "location_source"])

    merged = pd.concat(location_sources, ignore_index=True)
    merged = merged.dropna(subset=["country", "location_label"], how="all")
    merged = merged.drop_duplicates(subset=["incident_id", "country", "location_label"])
    return merged


def prepare_data(data_dir: str = "data") -> PreparedData:
    datasets = load_csv_files(data_dir)
    notes: list[str] = []

    incidents = _prepare_incidents(_select_primary_table(datasets, ["incidents"]), notes)
    reports = _prepare_reports(_select_primary_table(datasets, ["reports"]), notes)
    incident_reports = _build_incident_reports(incidents, reports, notes)

    risk_frames_by_name = {
        name: _classification_to_long(name, frame, notes)
        for name, frame in datasets.items()
        if name.startswith("classifications_")
    }
    primary_risk = risk_frames_by_name.get("classifications_MIT", pd.DataFrame())
    if not primary_risk.empty:
        incident_risk_long = primary_risk.copy()
    else:
        risk_frames = [frame for frame in risk_frames_by_name.values() if not frame.empty]
        incident_risk_long = pd.concat(risk_frames, ignore_index=True) if risk_frames else pd.DataFrame()
        if not incident_risk_long.empty:
            notes.append("MIT risk taxonomy was unavailable, so another classification file is being used as the risk fallback.")
    if incident_risk_long.empty:
        incident_risk_long = pd.DataFrame(columns=["incident_id", "taxonomy_source", "risk_category", "secondary_label"])
        notes.append("No classification labels could be joined. Risk Evolution views will show placeholders.")

    incident_location = _prepare_locations(datasets, notes)

    metadata: dict[str, Any] = {
        "files_loaded": sorted(datasets.keys()),
        "has_incidents": not incidents.empty,
        "has_reports": not reports.empty,
        "has_risk": not incident_risk_long.empty,
        "has_location": not incident_location.empty,
        "incident_year_min": int(incidents["incident_year"].dropna().min()) if incidents["incident_year"].dropna().any() else None,
        "incident_year_max": int(incidents["incident_year"].dropna().max()) if incidents["incident_year"].dropna().any() else None,
        "risk_categories": sorted(
            incident_risk_long["risk_category"].dropna().astype(str).loc[lambda s: s.ne("")].unique().tolist()
        ),
        "source_domains": sorted(reports["source_domain"].dropna().astype(str).unique().tolist()) if "source_domain" in reports.columns else [],
    }

    return PreparedData(
        datasets=datasets,
        incidents=incidents,
        reports=reports,
        incident_reports=incident_reports,
        incident_risk_long=incident_risk_long,
        incident_location=incident_location,
        metadata=metadata,
        notes=notes,
    )


def build_filtered_context(
    prepared: PreparedData,
    year_range: tuple[int, int] | None,
    selected_risks: list[str] | None,
    selected_domains: list[str] | None,
    keyword: str | None,
) -> dict[str, pd.DataFrame]:
    incidents = prepared.incidents.copy()
    incident_reports = prepared.incident_reports.copy()
    risk_long = prepared.incident_risk_long.copy()
    locations = prepared.incident_location.copy()

    if year_range and "incident_year" in incidents.columns:
        low, high = year_range
        incidents = incidents[incidents["incident_year"].fillna(-1).between(low, high)]

    if keyword:
        pattern = str(keyword).strip().lower()
        if pattern:
            incidents = incidents[
                incidents["search_text"].fillna("").str.lower().str.contains(pattern, na=False)
            ]

    if selected_risks:
        matching_ids = risk_long[risk_long["risk_category"].isin(selected_risks)]["incident_id"].unique()
        incidents = incidents[incidents["incident_id"].isin(matching_ids)]
        risk_long = risk_long[risk_long["incident_id"].isin(incidents["incident_id"])]

    if not risk_long.empty:
        risk_long = risk_long[risk_long["incident_id"].isin(incidents["incident_id"])]

    if not incident_reports.empty:
        incident_reports = incident_reports[incident_reports["incident_id"].isin(incidents["incident_id"])]
        if selected_domains:
            incident_reports = incident_reports[incident_reports["source_domain"].isin(selected_domains)]
            incidents = incidents[incidents["incident_id"].isin(incident_reports["incident_id"])]
            risk_long = risk_long[risk_long["incident_id"].isin(incidents["incident_id"])]

    if not locations.empty:
        locations = locations[locations["incident_id"].isin(incidents["incident_id"])]

    return {
        "incidents": incidents,
        "incident_reports": incident_reports,
        "risk_long": risk_long,
        "locations": locations,
    }
