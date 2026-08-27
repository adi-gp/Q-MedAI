"""Metrics calculated from real predictions and continuous model scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


STATUS_NOT_TRAINED = "NOT TRAINED"
STATUS_TRAINING = "TRAINING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_TIMED_OUT = "TIMED OUT"


@dataclass
class ModelResult:
    name: str
    status: str = STATUS_NOT_TRAINED
    error: str | None = None
    training_seconds: float | None = None
    metrics: dict[str, float] | None = None
    confusion: np.ndarray | None = None
    scores: np.ndarray | None = None
    predictions: np.ndarray | None = None
    fpr: np.ndarray | None = None
    tpr: np.ndarray | None = None
    estimator: Any | None = None


def evaluate_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    continuous_scores: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate binary predictions, using continuous scores exclusively for ROC-AUC."""
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    scores = np.asarray(continuous_scores, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or scores.shape != actual.shape:
        raise ValueError("y_true, predictions, and continuous_scores must be same-length vectors.")
    if len(np.unique(actual)) != 2:
        raise ValueError("ROC-AUC requires both binary classes in y_true.")
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    metrics = {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall / Sensitivity": float(recall_score(actual, predicted, zero_division=0)),
        "Specificity": specificity,
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(actual, scores)),
    }
    fpr, tpr, _ = roc_curve(actual, scores)
    return metrics, matrix, fpr, tpr


def build_comparison_table(results: dict[str, ModelResult], model_names: tuple[str, ...]) -> pd.DataFrame:
    """Keep every required model visible, including omitted, failed, and timed-out ones."""
    rows: list[dict[str, Any]] = []
    for name in model_names:
        result = results.get(name, ModelResult(name=name))
        row: dict[str, Any] = {
            "Model": name,
            "Status": result.status,
            "Training time (s)": result.training_seconds,
            "Error": result.error,
        }
        for metric in ("Accuracy", "Precision", "Recall / Sensitivity", "Specificity", "F1", "ROC-AUC"):
            row[metric] = None if result.metrics is None else result.metrics.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)
