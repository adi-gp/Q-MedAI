"""Comparable classical baseline models."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.config import RANDOM_SEED


CLASSICAL_MODEL_NAMES = ("Logistic Regression", "RBF SVM", "Random Forest")


def build_classical_model(name: str, random_seed: int = RANDOM_SEED):
    """Build one baseline with deterministic settings where sklearn supports them."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_seed)
    if name == "RBF SVM":
        return SVC(kernel="rbf", probability=True, random_state=random_seed)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=150,
            random_state=random_seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classical model: {name}")
