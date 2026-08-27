"""Validation for binary tabular disease-detection datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when input data cannot safely be used by this MVP."""


@dataclass
class ValidationReport:
    """Auditable facts and decisions made while preparing a dataset."""

    source: str
    target_column: str
    sample_count: int
    feature_count: int
    class_distribution: dict[str, int]
    target_encoding: dict[str, int]
    removed_columns: list[dict[str, str]] = field(default_factory=list)
    missing_values: int = 0
    warnings: list[str] = field(default_factory=list)


TARGET_CANDIDATES = ("diagnosis", "target", "label", "class", "outcome")


def detect_target_column(frame: pd.DataFrame) -> str:
    """Detect a conventional binary-target column without guessing feature columns."""
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise DataValidationError(
        "Could not detect a target column. Expected one of: "
        + ", ".join(TARGET_CANDIDATES)
        + "."
    )


def encode_binary_target(target: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Return a documented 0/1 encoding for exactly two target classes."""
    if target.isna().any():
        raise DataValidationError("The target column contains missing values.")

    values = target.astype(str).str.strip()
    unique = list(pd.unique(values))
    if len(unique) != 2:
        raise DataValidationError(
            f"This MVP requires exactly two target classes; found {len(unique)}: {unique}."
        )

    if set(unique) == {"B", "M"}:
        mapping = {"B": 0, "M": 1}
    elif set(unique) == {"0", "1"}:
        mapping = {"0": 0, "1": 1}
    else:
        # Stable lexical encoding is explicit in the report rather than an implicit cast.
        ordered = sorted(unique)
        mapping = {ordered[0]: 0, ordered[1]: 1}

    return values.map(mapping).astype(int), mapping


def _is_obvious_non_feature_column(column: str) -> tuple[bool, str]:
    name = column.strip().lower()
    compact = name.replace(" ", "").replace("_", "")
    if not name or name.startswith("unnamed"):
        return True, "blank or unnamed index column"
    if compact in {"id", "patientid", "sampleid", "recordid"} or compact.endswith("id"):
        return True, "identifier column"
    if name == "index" or name.endswith("_index"):
        return True, "index column"
    return False, ""


def validate_feature_frame(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    source: str,
    target_column: str,
    target_encoding: dict[str, int],
) -> tuple[pd.DataFrame, pd.Series, ValidationReport]:
    """Drop only obvious non-feature columns and require numeric model features."""
    removed_columns: list[dict[str, str]] = []
    kept_columns: list[str] = []
    for column in features.columns:
        is_non_feature, reason = _is_obvious_non_feature_column(str(column))
        if is_non_feature:
            removed_columns.append({"column": str(column), "reason": reason})
        else:
            kept_columns.append(str(column))

    clean = features.loc[:, kept_columns].copy()
    if clean.empty:
        raise DataValidationError("No feature columns remain after removing ID/index columns.")

    non_numeric: list[str] = []
    for column in clean.columns:
        converted = pd.to_numeric(clean[column], errors="coerce")
        original_non_missing = clean[column].notna()
        newly_missing = original_non_missing & converted.isna()
        if newly_missing.any():
            non_numeric.append(str(column))
        clean[column] = converted

    if non_numeric:
        raise DataValidationError(
            "Unexpected non-numeric feature columns were not discarded: "
            + ", ".join(non_numeric)
            + ". Remove or encode them explicitly before use."
        )

    if clean.isna().all(axis=0).any():
        unusable = list(clean.columns[clean.isna().all(axis=0)])
        raise DataValidationError(
            "Feature columns contain no usable numeric values: " + ", ".join(unusable)
        )
    if np.isinf(clean.to_numpy(dtype=float)).any():
        raise DataValidationError("Feature matrix contains infinite values.")

    class_counts = target.value_counts().sort_index()
    report = ValidationReport(
        source=source,
        target_column=target_column,
        sample_count=len(clean),
        feature_count=clean.shape[1],
        class_distribution={str(label): int(count) for label, count in class_counts.items()},
        target_encoding=target_encoding,
        removed_columns=removed_columns,
        missing_values=int(clean.isna().sum().sum()),
    )
    return clean, target.reset_index(drop=True), report


def validate_csv_dataset(
    frame: pd.DataFrame, *, source: str
) -> tuple[pd.DataFrame, pd.Series, ValidationReport]:
    """Validate a CSV dataset after inspecting its schema."""
    if frame.empty:
        raise DataValidationError("The CSV contains no rows.")
    target_column = detect_target_column(frame)
    target, encoding = encode_binary_target(frame[target_column])
    features = frame.drop(columns=[target_column])
    return validate_feature_frame(
        features,
        target,
        source=source,
        target_column=target_column,
        target_encoding=encoding,
    )
