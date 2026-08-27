"""Programmatic experiment runner shared by tests and the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data.loader import DatasetBundle, load_dataset
from src.evaluation.metrics import (
    ModelResult,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_TIMED_OUT,
    STATUS_TRAINING,
    build_comparison_table,
    evaluate_predictions,
)
from src.explainability.explanations import (
    classical_permutation_importance,
    vqc_perturbation_sensitivity,
)
from src.models.classical import CLASSICAL_MODEL_NAMES, build_classical_model
from src.models.quantum import QuantumKernelSVM, QuantumTimeoutError, VQCClassifier
from src.preprocessing.pipeline import PreparedSplit, split_and_preprocess


MODEL_NAMES = (
    "Logistic Regression",
    "RBF SVM",
    "Random Forest",
    "VQC",
    "Quantum Kernel SVM",
)


@dataclass
class ExperimentArtifacts:
    config: ExperimentConfig
    dataset: DatasetBundle
    prepared: PreparedSplit
    results: dict[str, ModelResult]
    explainability: dict[str, pd.DataFrame] = field(default_factory=dict)
    explanation_errors: dict[str, str] = field(default_factory=dict)
    kernel_subset_log: dict[str, object] | None = None

    @property
    def comparison_table(self) -> pd.DataFrame:
        return build_comparison_table(self.results, MODEL_NAMES)


def _complete_result(
    result: ModelResult,
    y_true: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    estimator,
) -> None:
    metrics, confusion, fpr, tpr = evaluate_predictions(y_true, predictions, scores)
    result.status = STATUS_COMPLETED
    result.metrics = metrics
    result.confusion = confusion
    result.scores = np.asarray(scores, dtype=float)
    result.predictions = np.asarray(predictions, dtype=int)
    result.fpr = fpr
    result.tpr = tpr
    result.estimator = estimator


def _record_failure(result: ModelResult, exc: Exception, training_seconds: float | None = None) -> None:
    result.status = STATUS_TIMED_OUT if isinstance(exc, QuantumTimeoutError) else STATUS_FAILED
    result.error = f"{type(exc).__name__}: {exc}"
    if training_seconds is not None:
        result.training_seconds = training_seconds


def _run_classical(name: str, prepared: PreparedSplit, config: ExperimentConfig) -> ModelResult:
    result = ModelResult(name=name, status=STATUS_TRAINING)
    model = build_classical_model(name, config.random_seed)
    started = perf_counter()
    try:
        model.fit(prepared.X_train, prepared.y_train)
        result.training_seconds = perf_counter() - started
        scores = model.predict_proba(prepared.X_test)[:, 1]
        predictions = model.predict(prepared.X_test)
        _complete_result(result, prepared.y_test, predictions, scores, model)
    except Exception as exc:
        _record_failure(result, exc, perf_counter() - started)
    return result


def _run_vqc(prepared: PreparedSplit, config: ExperimentConfig) -> ModelResult:
    result = ModelResult(name="VQC", status=STATUS_TRAINING)
    model = VQCClassifier(
        n_qubits=prepared.X_train.shape[1],
        layers=config.vqc_layers,
        iterations=config.vqc_iterations,
        timeout_seconds=config.quantum_timeout_seconds,
        random_seed=config.random_seed,
    )
    try:
        model.fit(prepared.X_train, prepared.y_train)
        result.training_seconds = model.training_seconds
        scores = model.predict_scores(prepared.X_test)
        predictions = model.predict(prepared.X_test, threshold=0.5)
        _complete_result(result, prepared.y_test, predictions, scores, model)
    except Exception as exc:
        _record_failure(result, exc, model.training_seconds)
    return result


def _run_quantum_kernel(prepared: PreparedSplit, config: ExperimentConfig) -> tuple[ModelResult, dict[str, object] | None]:
    result = ModelResult(name="Quantum Kernel SVM", status=STATUS_TRAINING)
    model = QuantumKernelSVM(
        n_qubits=prepared.X_train.shape[1],
        subset_size=config.kernel_subset_size,
        timeout_seconds=config.quantum_timeout_seconds,
        random_seed=config.random_seed,
    )
    subset_log: dict[str, object] | None = None
    try:
        model.fit(prepared.X_train, prepared.y_train)
        result.training_seconds = model.training_seconds
        # decision_function is intentionally the continuous ROC-AUC input.
        predictions, scores = model.predict_with_scores(prepared.X_test)
        _complete_result(result, prepared.y_test, predictions, scores, model)
        if model.subset_info is not None:
            subset_log = {
                "original training size": model.subset_info.original_training_size,
                "selected subset size": model.subset_info.selected_subset_size,
                "original class distribution": model.subset_info.original_class_distribution,
                "subset class distribution": model.subset_info.subset_class_distribution,
            }
    except Exception as exc:
        _record_failure(result, exc, model.training_seconds)
    return result, subset_log


def _add_explanations(artifacts: ExperimentArtifacts, selected_models: set[str]) -> None:
    names = artifacts.prepared.preprocessor.feature_names_out_
    for model_name in CLASSICAL_MODEL_NAMES:
        result = artifacts.results[model_name]
        if model_name in selected_models and result.status == STATUS_COMPLETED:
            try:
                artifacts.explainability[model_name] = classical_permutation_importance(
                    result.estimator,
                    artifacts.prepared.X_test,
                    artifacts.prepared.y_test,
                    names,
                    random_seed=artifacts.config.random_seed,
                )
            except Exception as exc:
                artifacts.explanation_errors[model_name] = f"{type(exc).__name__}: {exc}"

    vqc_result = artifacts.results["VQC"]
    if "VQC" in selected_models and vqc_result.status == STATUS_COMPLETED:
        try:
            artifacts.explainability["VQC"] = vqc_perturbation_sensitivity(
                vqc_result.estimator,
                artifacts.prepared.X_test,
                names,
            )
        except Exception as exc:
            artifacts.explanation_errors["VQC"] = f"{type(exc).__name__}: {exc}"


def run_experiment(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    csv_path: str | Path | None = None,
    selected_models: set[str] | None = None,
) -> ExperimentArtifacts:
    """Run one actual experiment. Unselected models remain explicitly NOT TRAINED."""
    selected = set(MODEL_NAMES if selected_models is None else selected_models)
    unknown = selected.difference(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown model selections: {sorted(unknown)}")

    dataset = load_dataset(csv_path)
    prepared = split_and_preprocess(
        dataset.X,
        dataset.y,
        method=config.reduction_method,
        n_features=config.n_features,
        test_size=config.test_size,
        random_seed=config.random_seed,
    )
    results = {name: ModelResult(name=name) for name in MODEL_NAMES}
    artifacts = ExperimentArtifacts(config=config, dataset=dataset, prepared=prepared, results=results)

    for name in CLASSICAL_MODEL_NAMES:
        if name in selected:
            artifacts.results[name] = _run_classical(name, prepared, config)
    if "VQC" in selected:
        artifacts.results["VQC"] = _run_vqc(prepared, config)
    if "Quantum Kernel SVM" in selected:
        result, subset_log = _run_quantum_kernel(prepared, config)
        artifacts.results["Quantum Kernel SVM"] = result
        artifacts.kernel_subset_log = subset_log

    _add_explanations(artifacts, selected)
    return artifacts
