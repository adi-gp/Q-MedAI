"""Lightweight explanations for the models used by the MVP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import RANDOM_SEED
from src.models.quantum import VQCClassifier


def classical_permutation_importance(
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Rank transformed features with sklearn permutation importance."""
    result = permutation_importance(
        estimator,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=random_seed,
    )
    frame = pd.DataFrame(
        {
            "Feature / component": feature_names,
            "Mean importance": result.importances_mean,
            "Importance standard deviation": result.importances_std,
        }
    )
    return frame.sort_values("Mean importance", ascending=False, ignore_index=True)


def vqc_perturbation_sensitivity(
    model: VQCClassifier,
    X: np.ndarray,
    feature_names: list[str],
    *,
    perturbation: float = 0.10,
) -> pd.DataFrame:
    """Measure output change after deterministic plus/minus feature perturbations."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("Feature names must correspond to each column of X.")
    sensitivities: list[float] = []
    for column in range(values.shape[1]):
        plus = values.copy()
        minus = values.copy()
        plus[:, column] += perturbation
        minus[:, column] -= perturbation
        change = np.abs(model.predict_scores(plus) - model.predict_scores(minus))
        sensitivities.append(float(np.mean(change)))
    frame = pd.DataFrame(
        {
            "Feature / component": feature_names,
            "Sensitivity": sensitivities,
            "Perturbation": perturbation,
        }
    )
    return frame.sort_values("Sensitivity", ascending=False, ignore_index=True)
