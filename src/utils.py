from __future__ import annotations

import re
from typing import Iterable


def normalize_name(value: str) -> str:
    """Convert arbitrary column names to a consistent snake_case format."""
    value = value.strip().lower()
    value = re.sub(r"[^\w]+", "_", value)
    value = re.sub(r"__+", "_", value)
    return value.strip("_")


def unique_normalized_names(columns: Iterable[str]) -> list[str]:
    """Normalize names while preserving uniqueness for duplicated headers."""
    used: dict[str, int] = {}
    output: list[str] = []
    for column in columns:
        base = normalize_name(str(column))
        if not base:
            base = "column"
        count = used.get(base, 0)
        used[base] = count + 1
        output.append(base if count == 0 else f"{base}_{count + 1}")
    return output


def first_matching_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = list(columns)
    normalized = {normalize_name(col): col for col in column_set}
    for candidate in candidates:
        exact = normalized.get(normalize_name(candidate))
        if exact:
            return exact
    for candidate in candidates:
        token = normalize_name(candidate)
        for column in column_set:
            if token and token in normalize_name(column):
                return column
    return None


def coalesce_text(*values: object) -> str:
    parts = [str(value).strip() for value in values if value is not None and str(value).strip()]
    return " ".join(parts)
