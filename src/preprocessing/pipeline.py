"""Train-only imputation, scaling, and feature reduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_TEST_SIZE, RANDOM_SEED


@dataclass
class PreparedSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: "LeakageSafePreprocessor"


@dataclass
class RawSplit:
    """The immutable stratified row split, before any fitted transformation."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class LeakageSafePreprocessor:
    """A stateful transformer deliberately fitted only on training data."""

    def __init__(self, method: str = "SelectKBest", n_features: int = 4, random_seed: int = RANDOM_SEED):
        if method not in {"PCA", "SelectKBest"}:
            raise ValueError("method must be 'PCA' or 'SelectKBest'.")
        self.method = method
        self.requested_n_features = n_features
        self.random_seed = random_seed
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.reducer: PCA | SelectKBest | None = None
        self.feature_names_in_: list[str] = []
        self.feature_names_out_: list[str] = []
        self.actual_n_features: int | None = None
        self._is_fitted = False

    def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> "LeakageSafePreprocessor":
        """Fit all preprocessing objects on training rows only."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_in_ = list(X_train.columns)
        else:
            self.feature_names_in_ = [f"feature_{index}" for index in range(np.asarray(X_train).shape[1])]
        values = np.asarray(X_train, dtype=float)
        y_values = np.asarray(y_train)
        if values.ndim != 2:
            raise ValueError("X_train must be a 2D matrix.")

        max_components = min(values.shape[1], values.shape[0])
        actual = min(self.requested_n_features, max_components)
        if actual < 1:
            raise ValueError("No valid reduced features can be created.")
        self.actual_n_features = actual

        imputed = self.imputer.fit_transform(values)
        scaled = self.scaler.fit_transform(imputed)
        if self.method == "PCA":
            self.reducer = PCA(n_components=actual, random_state=self.random_seed)
            self.feature_names_out_ = [f"PCA component {index + 1}" for index in range(actual)]
        else:
            # f_classif is explicit to keep supervised selection well-defined.
            self.reducer = SelectKBest(score_func=f_classif, k=actual)
        self.reducer.fit(scaled, y_values)
        if self.method == "SelectKBest":
            mask = self.reducer.get_support()
            self.feature_names_out_ = [name for name, selected in zip(self.feature_names_in_, mask) if selected]
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform rows with already-fitted training transformers; never refit."""
        if not self._is_fitted or self.reducer is None:
            raise RuntimeError("Preprocessor must be fitted on training data before transform.")
        values = np.asarray(X, dtype=float)
        return self.reducer.transform(self.scaler.transform(self.imputer.transform(values)))

    def fit_transform(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> np.ndarray:
        return self.fit(X_train, y_train).transform(X_train)


def split_raw_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> RawSplit:
    """Create the reproducible split before any preprocessing is fitted."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )
    return RawSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def prepare_from_raw_split(
    raw_split: RawSplit,
    *,
    method: str = "SelectKBest",
    n_features: int = 4,
    random_seed: int = RANDOM_SEED,
) -> PreparedSplit:
    """Fit one training-only preprocessing pipeline for a pre-existing row split."""
    preprocessor = LeakageSafePreprocessor(method=method, n_features=n_features, random_seed=random_seed)
    X_train_processed = preprocessor.fit_transform(raw_split.X_train, raw_split.y_train)
    X_test_processed = preprocessor.transform(raw_split.X_test)
    return PreparedSplit(
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=np.asarray(raw_split.y_train, dtype=int),
        y_test=np.asarray(raw_split.y_test, dtype=int),
        preprocessor=preprocessor,
    )


def split_and_preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    method: str = "SelectKBest",
    n_features: int = 4,
    test_size: float = DEFAULT_TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> PreparedSplit:
    """Stratify before fitting preprocessing to prevent test-set leakage."""
    raw_split = split_raw_data(X, y, test_size=test_size, random_seed=random_seed)
    return prepare_from_raw_split(
        raw_split,
        method=method,
        n_features=n_features,
        random_seed=random_seed,
    )
