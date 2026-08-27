"""Load the supplied breast-cancer CSV, or a documented scikit-learn fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from .validation import ValidationReport, validate_csv_dataset, validate_feature_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"
CSV_SOURCE = "local CSV (data/breast_cancer.csv)"
SKLEARN_SOURCE = "scikit-learn built-in Breast Cancer Wisconsin dataset"


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    report: ValidationReport


def load_dataset(csv_path: str | Path | None = None) -> DatasetBundle:
    """Load a real local CSV when available, otherwise use sklearn's built-in data."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    if path.exists():
        frame = pd.read_csv(path)
        X, y, report = validate_csv_dataset(frame, source=CSV_SOURCE)
        return DatasetBundle(X=X, y=y, report=report)

    dataset = load_breast_cancer(as_frame=True)
    target = dataset.target.astype(int).reset_index(drop=True)
    X, y, report = validate_feature_frame(
        dataset.data,
        target,
        source=SKLEARN_SOURCE,
        target_column="target",
        target_encoding={"malignant": 0, "benign": 1},
    )
    return DatasetBundle(X=X, y=y, report=report)
