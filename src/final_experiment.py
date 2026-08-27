"""One-shot SIH final experiment, artifact generation, and report rendering."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ExperimentConfig, RANDOM_SEED
from src.data.loader import load_dataset
from src.evaluation.metrics import ModelResult, STATUS_FAILED, STATUS_COMPLETED
from src.explainability.explanations import (
    classical_permutation_importance,
    vqc_perturbation_sensitivity,
)
from src.models.classical import CLASSICAL_MODEL_NAMES
from src.models.quantum import QuantumKernelSVM, stratified_subset_indices
from src.preprocessing.pipeline import RawSplit, prepare_from_raw_split, split_raw_data
from src.runner import MODEL_NAMES, _run_classical, _run_quantum_kernel, _run_vqc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
FINAL_RESULTS_PATH = RESULTS_DIR / "final_results.json"
FINAL_CSV_PATH = RESULTS_DIR / "final_results.csv"
FINAL_CONFIG_PATH = RESULTS_DIR / "final_config.json"
FINAL_REPORT_PATH = PROJECT_ROOT / "FINAL_EXPERIMENT_REPORT.md"
SIH_SUMMARY_PATH = PROJECT_ROOT / "SIH_FINAL_SUMMARY.md"
FINAL_RESULTS_README_PATH = RESULTS_DIR / "FINAL_RESULTS_README.md"

FINAL_CONFIG = ExperimentConfig(
    reduction_method="SelectKBest",
    n_features=4,
    vqc_layers=3,
    vqc_iterations=100,
    kernel_subset_size=150,
    quantum_timeout_seconds=600,
    random_seed=RANDOM_SEED,
)

CLASSICAL_TYPE = "Classical"
QUANTUM_TYPE = "Quantum"


def _model_type(name: str) -> str:
    return CLASSICAL_TYPE if name in CLASSICAL_MODEL_NAMES else QUANTUM_TYPE


def _distribution(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=int), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def _failed_quantum_kernel_result(reason: str) -> ModelResult:
    return ModelResult(name="Quantum Kernel SVM", status=STATUS_FAILED, error=reason)


def _result_record(
    result: ModelResult,
    *,
    condition: str,
    test_targets: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, Any]:
    metrics = result.metrics or {}
    record: dict[str, Any] = {
        "model": result.name,
        "type": _model_type(result.name),
        "experiment_condition": condition,
        "status": result.status,
        "error": result.error,
        "random_seed": config.random_seed,
        "configuration": {
            "preprocessing": config.reduction_method,
            "feature_count": config.n_features,
            "vqc_layers": config.vqc_layers if result.name == "VQC" else None,
            "vqc_iterations": config.vqc_iterations if result.name == "VQC" else None,
            "quantum_kernel_subset_size": config.kernel_subset_size
            if result.name == "Quantum Kernel SVM"
            else None,
        },
        "accuracy": metrics.get("Accuracy"),
        "precision": metrics.get("Precision"),
        "recall": metrics.get("Recall / Sensitivity"),
        "specificity": metrics.get("Specificity"),
        "f1": metrics.get("F1"),
        "roc_auc": metrics.get("ROC-AUC"),
        "training_time_seconds": result.training_seconds,
        "confusion_matrix": None if result.confusion is None else result.confusion.tolist(),
        "roc_curve": None
        if result.fpr is None or result.tpr is None
        else {"fpr": result.fpr.tolist(), "tpr": result.tpr.tolist()},
        "test_targets": test_targets.tolist() if result.scores is not None else None,
        "predictions": None if result.predictions is None else result.predictions.tolist(),
        "continuous_scores": None if result.scores is None else result.scores.tolist(),
    }
    if isinstance(result.estimator, QuantumKernelSVM):
        model = result.estimator
        record["kernel_details"] = {
            "original_training_size": model.subset_info.original_training_size
            if model.subset_info
            else None,
            "selected_subset_size": model.subset_info.selected_subset_size
            if model.subset_info
            else None,
            "original_class_distribution": model.subset_info.original_class_distribution
            if model.subset_info
            else None,
            "subset_class_distribution": model.subset_info.subset_class_distribution
            if model.subset_info
            else None,
            "train_kernel_seconds": model.train_kernel_seconds,
            "test_kernel_seconds": model.test_kernel_seconds,
            "svm_training_seconds": model.svm_training_seconds,
            "total_training_seconds": model.training_seconds,
        }
    return record


def _best_completed(records: list[dict[str, Any]], model_type: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in records
        if row["type"] == model_type
        and row["status"] == STATUS_COMPLETED
        and row["roc_auc"] is not None
    ]
    return max(candidates, key=lambda row: row["roc_auc"]) if candidates else None


def _observed_difference(quantum: dict[str, Any] | None, classical: dict[str, Any] | None) -> dict[str, Any]:
    if quantum is None or classical is None:
        return {"status": "NOT AVAILABLE"}
    return {
        "quantum_model": quantum["model"],
        "classical_reference": classical["model"],
        "roc_auc_difference": quantum["roc_auc"] - classical["roc_auc"],
        "accuracy_difference": quantum["accuracy"] - classical["accuracy"],
        "f1_difference": quantum["f1"] - classical["f1"],
        "label": "Observed performance difference",
    }


def _final_explainability(results: dict[str, ModelResult], prepared) -> dict[str, Any]:
    """Generate only the required, model-appropriate final explanations."""
    feature_names = prepared.preprocessor.feature_names_out_
    tables: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in CLASSICAL_MODEL_NAMES:
        result = results[name]
        if result.status == STATUS_COMPLETED:
            try:
                tables[name] = classical_permutation_importance(
                    result.estimator,
                    prepared.X_test,
                    prepared.y_test,
                    feature_names,
                    random_seed=RANDOM_SEED,
                ).to_dict(orient="records")
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
    vqc_result = results["VQC"]
    if vqc_result.status == STATUS_COMPLETED:
        try:
            tables["VQC"] = vqc_perturbation_sensitivity(
                vqc_result.estimator,
                prepared.X_test,
                feature_names,
            ).to_dict(orient="records")
        except Exception as exc:
            errors["VQC"] = f"{type(exc).__name__}: {exc}"
    return {"status": "COMPLETED", "tables": tables, "errors": errors}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(records: list[dict[str, Any]]) -> None:
    fields = [
        "experiment_condition",
        "model",
        "type",
        "status",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "training_time_seconds",
        "random_seed",
        "error",
    ]
    with FINAL_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def _artifact_plot_path(path: Path) -> str:
    """Keep local artifacts relative while allowing Kaggle working-directory output."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _save_roc_plot(records: list[dict[str, Any]], condition: str, filename: str) -> str | None:
    successful = [row for row in records if row["roc_curve"] is not None]
    if not successful:
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    for row in successful:
        curve = row["roc_curve"]
        axis.plot(curve["fpr"], curve["tpr"], label=f"{row['model']} ({row['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axis.set(
        title=f"{condition}: ROC curves",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(fontsize="small")
    figure.tight_layout()
    path = PLOTS_DIR / filename
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return _artifact_plot_path(path)


def _save_confusion_plot(records: list[dict[str, Any]], condition: str, filename: str) -> str | None:
    successful = [row for row in records if row["confusion_matrix"] is not None]
    if not successful:
        return None
    columns = min(3, len(successful))
    rows = math.ceil(len(successful) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False)
    for axis, record in zip(axes.ravel(), successful):
        matrix = np.asarray(record["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(record["model"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks([0, 1], ["B / 0", "M / 1"])
        axis.set_yticks([0, 1], ["B / 0", "M / 1"])
        for row_index in range(2):
            for column_index in range(2):
                axis.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.ravel()[len(successful) :]:
        axis.axis("off")
    figure.suptitle(f"{condition}: confusion matrices")
    figure.tight_layout()
    path = PLOTS_DIR / filename
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return _artifact_plot_path(path)


def _save_metric_plot(records: list[dict[str, Any]], metric: str, filename: str) -> str | None:
    successful = [row for row in records if row.get(metric) is not None]
    if not successful:
        return None
    labels = [f"{row['model']}\n({row['experiment_condition']})" for row in successful]
    values = [row[metric] for row in successful]
    colors = ["#2865A8" if row["type"] == CLASSICAL_TYPE else "#7D4EAD" for row in successful]
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    bars = axis.bar(range(len(values)), values, color=colors)
    axis.set_xticks(range(len(labels)), labels, rotation=22, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel(metric.replace("_", " ").upper())
    axis.set_title(f"{metric.replace('_', ' ').upper()} comparison")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    figure.tight_layout()
    path = PLOTS_DIR / filename
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return _artifact_plot_path(path)


def _save_plots(full_records: list[dict[str, Any]], matched_records: list[dict[str, Any]]) -> list[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = [
        _save_roc_plot(full_records, "Full-data final experiment", "full_data_roc_curves.png"),
        _save_roc_plot(matched_records, "Matched-data comparison", "matched_data_roc_curves.png"),
        _save_confusion_plot(full_records, "Full-data final experiment", "full_data_confusion_matrices.png"),
        _save_confusion_plot(matched_records, "Matched-data comparison", "matched_data_confusion_matrices.png"),
        _save_metric_plot(full_records + matched_records, "accuracy", "accuracy_comparison.png"),
        _save_metric_plot(full_records + matched_records, "f1", "f1_comparison.png"),
        _save_metric_plot(full_records + matched_records, "roc_auc", "roc_auc_comparison.png"),
    ]
    return [path for path in saved if path is not None]


def _markdown_table(records: list[dict[str, Any]]) -> str:
    header = "| Model | Type | Status | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Training time (s) |"
    separator = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, separator]
    for row in records:
        def value(key: str) -> str:
            raw = row.get(key)
            return "—" if raw is None else f"{raw:.4f}"

        duration = row.get("training_time_seconds")
        time_value = "—" if duration is None else f"{duration:.3f}"
        rows.append(
            "| {model} | {type} | {status} | {accuracy} | {precision} | {recall} | {specificity} | {f1} | {roc_auc} | {time} |".format(
                model=row["model"],
                type=row["type"],
                status=row["status"],
                accuracy=value("accuracy"),
                precision=value("precision"),
                recall=value("recall"),
                specificity=value("specificity"),
                f1=value("f1"),
                roc_auc=value("roc_auc"),
                time=time_value,
            )
        )
    return "\n".join(rows)


def _difference_text(difference: dict[str, Any]) -> str:
    if difference.get("status") == "NOT AVAILABLE":
        return "Not available because one required completed result is missing."
    return (
        f"{difference['quantum_model']} minus {difference['classical_reference']}: "
        f"ROC-AUC {difference['roc_auc_difference']:+.4f}; "
        f"accuracy {difference['accuracy_difference']:+.4f}; "
        f"F1 {difference['f1_difference']:+.4f}."
    )


def _write_reports_from_artifacts(payload: dict[str, Any]) -> None:
    """Render human-facing documents by reading the saved authoritative artifact."""
    full = payload["full_data_final_results"]
    matched = payload["matched_data_comparison"]
    analysis = payload["analysis"]
    config = payload["final_configuration"]
    report = f"""# Q-MedAI FINAL EXPERIMENT REPORT

## Objective

Q-MedAI experimentally compares classical and quantum machine-learning approaches under a controlled methodology. This FINAL EXPERIMENT is distinct from the prior **LOCAL BASELINE**, which used a 2-layer/35-iteration VQC and a 50-row quantum-kernel subset and was not rerun.

## Dataset, split, target, and leakage prevention

Dataset source: `{config['dataset']['source']}`. The supplied project files do not independently establish external provenance beyond the CSV schema, so provenance is reported as: **Source provenance not independently verified from supplied project files**. The dataset has {config['dataset']['sample_count']} rows and {config['dataset']['feature_count']} numeric features after documented removals. Target encoding is B = 0 (benign, negative) and M = 1 (malignant, positive). A single stratified {config['test_size']:.0%} train/test split used `random_state={config['random_seed']}`.

Imputation, scaling, and SelectKBest(`f_classif`) were fit only on training rows. The held-out test set was never used to fit preprocessing. The final configuration used {config['preprocessing']} with {config['feature_count']} selected features/qubits.

## Quantum configuration and backend

VQC: {config['vqc']['qubits']} qubits, {config['vqc']['layers']} layers, {config['vqc']['iterations']} optimization iterations. Quantum kernel: {config['quantum_kernel']['subset_size']} stratified training rows. Each quantum component had a cooperative {config['timeout_seconds']} second timeout. Backend: {config['backend']}.

## FINAL EXPERIMENT — full-data results

Classical models and the VQC used all {full['training_rows']} full-training rows. The quantum-kernel model was constrained to {config['quantum_kernel']['subset_size']} rows, so its full-data row is not an apples-to-apples training-size comparison with full-data classical models.

{_markdown_table(full['models'])}

## MATCHED-DATA COMPARISON — 150 training samples for all models

The exact same stratified 150 original training rows and unchanged held-out test set were used for all four models. A separate preprocessing pipeline was fit on those 150 rows only before transforming the test set.

{_markdown_table(matched['models'])}

## Observed performance difference

Full-data reference: {_difference_text(analysis['full_data_quantum_kernel_vs_best_classical'])}

Matched-data fair comparison: {_difference_text(analysis['matched_quantum_kernel_vs_best_classical'])}

These are observed differences from one split, not quantum advantage. Statistical confidence intervals were not computed for this run; results reflect a single train/test split.

## Computational cost, interpretation, and limitations

Quantum-kernel timing details are saved in `results/final_results.json`; VQC and kernel training times are shown above. This feasibility/prototype study on one supplied dataset is not evidence that quantum ML is generally superior or inferior for medical diagnosis. PennyLane `default.qubit` is a classical simulator, not physical quantum hardware. The study has one dataset, one split, limited qubit count, and no clinical validation.

## Conclusion

Q-MedAI provides a leakage-safe, controlled classical-plus-quantum comparison with a matched-data fairness condition. It does not establish quantum advantage merely by predictive accuracy.
"""
    FINAL_REPORT_PATH.write_text(report, encoding="utf-8")

    matched_difference = analysis["matched_quantum_kernel_vs_best_classical"]
    full_difference = analysis["full_data_quantum_kernel_vs_best_classical"]
    summary = f"""# Q-MedAI — SIH FINAL SUMMARY

## Problem and solution

Early disease-classification workflows can benefit from research ML assistance. Q-MedAI is a hybrid platform that compares strong classical baselines with quantum ML under one controlled, leakage-safe methodology. It is research software, not a medical diagnostic device.

## Architecture and methodology

Local CSV → validation (B=0, M=1) → stratified split (seed 42) → train-only imputation/scaling/SelectKBest(`f_classif`) → models → held-out metrics and plots. Classical models are Logistic Regression, RBF SVM, and Random Forest. Quantum models are a 4-qubit VQC and a fidelity quantum-kernel SVM, both simulated with PennyLane `default.qubit`.

## Why the matched-data comparison matters

The earlier **LOCAL BASELINE** used only 50 kernel-training rows while classical models used the complete training set, so it was not a fair head-to-head comparison. The **FINAL EXPERIMENT** uses 150 kernel rows, then tests all matched models on the exact same 150 rows with a separate training-only preprocessing pipeline and the same held-out test set.

## FINAL EXPERIMENT — full-data results

{_markdown_table(full['models'])}

## MATCHED-DATA COMPARISON — 150 rows for all models

{_markdown_table(matched['models'])}

## What quantum contributed

Full-data context: {_difference_text(full_difference)} The kernel used fewer training rows than full-data classical models, so this is contextual only.

Fair matched-data result: {_difference_text(matched_difference)} This is an observed performance difference, not quantum advantage.

## Limitations and future work

One supplied dataset and one train/test split are not clinical validation or general evidence. The quantum models run on a classical simulator and may be computationally slower. Future work: more datasets and splits, confidence intervals, larger carefully controlled circuits/kernels, appropriate GPU-accelerated simulation where supported, and eventual hardware studies.

## 30-second answer: Why quantum?

“We are not assuming quantum is better. Q-MedAI gives us a controlled way to compare quantum and classical approaches on the same held-out data. The project’s value is the fair methodology, including a matched 150-sample comparison for the quantum kernel, and a reusable platform for future experiments.”

## 20-second answer: Classical model has better accuracy—why use quantum?

“That is a valid outcome. This experiment tests the question rather than assuming a winner. On this split, we report the actual classical and quantum scores openly; the prototype demonstrates a controlled evaluation framework, not a claim of quantum superiority.”

## 20-second answer: Was the old 90% quantum-kernel result just less data?

“The old 50-sample kernel result had a training-size mismatch with full-data classical models, so it was not a fair head-to-head claim. The final study explicitly fixes that with the same 150 training rows, separate subset-only preprocessing, and the same held-out test set for all compared models.”

## Judge Q&A

- **Why only 4 qubits?** Four selected features map directly to four qubits and keep local simulation feasible.
- **Why this dataset?** It is the supplied binary breast-cancer CSV; its external provenance was not independently verified from project files.
- **Why only 150 kernel rows?** Fidelity-kernel matrices scale quadratically in training rows; 150 is the configured controlled limit.
- **Is that unfair?** It would be if compared directly to full-data models, which is why the matched-data comparison uses exactly those 150 rows for every model.
- **Is quantum slower?** Training times are reported from the run; `default.qubit` is classical simulation and not a speed claim.
- **Are you using a quantum computer?** No. PennyLane `default.qubit` simulates circuits classically.
- **Where is the quantum advantage?** This project makes no quantum-advantage claim. A higher score on one split would only be an observed performance difference.
- **How is leakage prevented?** The split happens first; preprocessing is fitted only on training rows, including a separate pipeline for the matched subset.
- **Is this clinically validated?** No. It is a research prototype and not a medical diagnostic device.
"""
    SIH_SUMMARY_PATH.write_text(summary, encoding="utf-8")

    FINAL_RESULTS_README_PATH.write_text(
        "# Final results artifacts\n\n"
        "These files correspond to the FINAL EXPERIMENT executed during this Codex run. "
        "`final_results.json` is the authoritative structured result, `final_results.csv` is the table-friendly export, "
        "`final_config.json` records the exact configuration, and `plots/` contains PNG figures generated from actual completed results.\n",
        encoding="utf-8",
    )


def load_final_artifacts(results_dir: Path = RESULTS_DIR) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load the final result/config pair for the Streamlit demo; never reconstruct defaults."""
    results_path = results_dir / "final_results.json"
    config_path = results_dir / "final_config.json"
    if not results_path.exists() or not config_path.exists():
        return None
    return (
        json.loads(results_path.read_text(encoding="utf-8")),
        json.loads(config_path.read_text(encoding="utf-8")),
    )


def run_final_experiment(
    *,
    csv_path: str | Path | None = None,
    results_dir: Path = RESULTS_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute the one mandatory final study and write its authoritative artifacts once."""
    global RESULTS_DIR, PLOTS_DIR, FINAL_RESULTS_PATH, FINAL_CSV_PATH, FINAL_CONFIG_PATH, FINAL_RESULTS_README_PATH
    global FINAL_REPORT_PATH, SIH_SUMMARY_PATH
    original_paths = (
        RESULTS_DIR,
        PLOTS_DIR,
        FINAL_RESULTS_PATH,
        FINAL_CSV_PATH,
        FINAL_CONFIG_PATH,
        FINAL_RESULTS_README_PATH,
        FINAL_REPORT_PATH,
        SIH_SUMMARY_PATH,
    )
    if results_dir != RESULTS_DIR:
        RESULTS_DIR = results_dir
        PLOTS_DIR = RESULTS_DIR / "plots"
        FINAL_RESULTS_PATH = RESULTS_DIR / "final_results.json"
        FINAL_CSV_PATH = RESULTS_DIR / "final_results.csv"
        FINAL_CONFIG_PATH = RESULTS_DIR / "final_config.json"
        FINAL_RESULTS_README_PATH = RESULTS_DIR / "FINAL_RESULTS_README.md"
        FINAL_REPORT_PATH = RESULTS_DIR.parent / "FINAL_EXPERIMENT_REPORT.md"
        SIH_SUMMARY_PATH = RESULTS_DIR.parent / "SIH_FINAL_SUMMARY.md"
    try:
        if FINAL_RESULTS_PATH.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite authoritative final artifact: {FINAL_RESULTS_PATH}."
            )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(csv_path)
        if dataset.report.target_encoding != {"B": 0, "M": 1}:
            raise ValueError(f"Final experiment requires B=0 and M=1; got {dataset.report.target_encoding}.")

        raw_split = split_raw_data(
            dataset.X,
            dataset.y,
            test_size=FINAL_CONFIG.test_size,
            random_seed=FINAL_CONFIG.random_seed,
        )
        full_prepared = prepare_from_raw_split(
            raw_split,
            method=FINAL_CONFIG.reduction_method,
            n_features=FINAL_CONFIG.n_features,
            random_seed=FINAL_CONFIG.random_seed,
        )

        full_results = {
            name: _run_classical(name, full_prepared, FINAL_CONFIG) for name in CLASSICAL_MODEL_NAMES
        }
        full_results["VQC"] = _run_vqc(full_prepared, FINAL_CONFIG)
        if len(raw_split.y_train) < FINAL_CONFIG.kernel_subset_size:
            full_results["Quantum Kernel SVM"] = _failed_quantum_kernel_result(
                f"Training split has {len(raw_split.y_train)} rows, fewer than required "
                f"{FINAL_CONFIG.kernel_subset_size} quantum-kernel rows."
            )
        else:
            full_results["Quantum Kernel SVM"], _ = _run_quantum_kernel(full_prepared, FINAL_CONFIG)

        subset_indices = stratified_subset_indices(
            np.asarray(raw_split.y_train, dtype=int),
            FINAL_CONFIG.kernel_subset_size,
            FINAL_CONFIG.random_seed,
        )
        matched_raw = RawSplit(
            X_train=raw_split.X_train.iloc[subset_indices],
            X_test=raw_split.X_test,
            y_train=raw_split.y_train.iloc[subset_indices],
            y_test=raw_split.y_test,
        )
        matched_prepared = prepare_from_raw_split(
            matched_raw,
            method=FINAL_CONFIG.reduction_method,
            n_features=FINAL_CONFIG.n_features,
            random_seed=FINAL_CONFIG.random_seed,
        )
        matched_results = {
            name: _run_classical(name, matched_prepared, FINAL_CONFIG) for name in CLASSICAL_MODEL_NAMES
        }
        matched_results["Quantum Kernel SVM"], _ = _run_quantum_kernel(matched_prepared, FINAL_CONFIG)

        full_records = [
            _result_record(full_results[name], condition="FULL_DATA_FINAL", test_targets=full_prepared.y_test, config=FINAL_CONFIG)
            for name in MODEL_NAMES
        ]
        matched_records = [
            _result_record(
                matched_results[name],
                condition="MATCHED_DATA_150",
                test_targets=matched_prepared.y_test,
                config=FINAL_CONFIG,
            )
            for name in (*CLASSICAL_MODEL_NAMES, "Quantum Kernel SVM")
        ]
        config_record = {
            "experiment_label": "FINAL EXPERIMENT",
            "dataset": {
                "path": str(csv_path) if csv_path is not None else "data/breast_cancer.csv",
                "source": dataset.report.source,
                "sample_count": dataset.report.sample_count,
                "feature_count": dataset.report.feature_count,
                "class_distribution": dataset.report.class_distribution,
                "removed_columns": dataset.report.removed_columns,
                "provenance_status": "Source provenance not independently verified from supplied project files",
            },
            "preprocessing": FINAL_CONFIG.reduction_method,
            "feature_count": FINAL_CONFIG.n_features,
            "selected_features_full_data": full_prepared.preprocessor.feature_names_out_,
            "selected_features_matched_data": matched_prepared.preprocessor.feature_names_out_,
            "random_seed": FINAL_CONFIG.random_seed,
            "test_size": FINAL_CONFIG.test_size,
            "train_test_split": "stratified train_test_split with random_state=42",
            "target_encoding": {"B": 0, "M": 1},
            "positive_class": "M = malignant = 1",
            "vqc": {
                "qubits": 4,
                "layers": FINAL_CONFIG.vqc_layers,
                "iterations": FINAL_CONFIG.vqc_iterations,
                "encoding": "RY",
                "ansatz": "RX/RY/RZ rotations with CNOT ring",
                "measurement": "PauliZ expectation converted to (1 + <Z>) / 2",
            },
            "quantum_kernel": {
                "subset_size": FINAL_CONFIG.kernel_subset_size,
                "kernel": "fidelity-based RY feature map",
                "evaluation_matrix": "test x train only",
            },
            "backend": "PennyLane default.qubit — classical quantum-circuit simulation",
            "timeout_seconds": FINAL_CONFIG.quantum_timeout_seconds,
        }
        payload = {
            "artifact_label": "FINAL EXPERIMENT",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "final_configuration": config_record,
            "full_data_final_results": {
                "training_rows": int(len(raw_split.y_train)),
                "test_rows": int(len(raw_split.y_test)),
                "models": full_records,
            },
            "matched_data_comparison": {
                "training_rows": int(len(matched_raw.y_train)),
                "test_rows": int(len(matched_raw.y_test)),
                "original_training_class_distribution": _distribution(raw_split.y_train),
                "matched_training_class_distribution": _distribution(matched_raw.y_train),
                "models": matched_records,
            },
            "explainability": _final_explainability(full_results, full_prepared),
            "analysis": {
                "full_data_quantum_kernel_vs_best_classical": _observed_difference(
                    next((row for row in full_records if row["model"] == "Quantum Kernel SVM"), None),
                    _best_completed(full_records, CLASSICAL_TYPE),
                ),
                "matched_quantum_kernel_vs_best_classical": _observed_difference(
                    next((row for row in matched_records if row["model"] == "Quantum Kernel SVM"), None),
                    _best_completed(matched_records, CLASSICAL_TYPE),
                ),
                "statistical_confidence": "Statistical confidence intervals were not computed for this run; results reflect a single train/test split.",
                "interpretation": "This is a feasibility/prototype study on one supplied dataset, not evidence that quantum ML is generally superior or inferior for medical diagnosis.",
            },
        }
        payload["plots"] = _save_plots(full_records, matched_records)
        _write_json(FINAL_CONFIG_PATH, config_record)
        _write_json(FINAL_RESULTS_PATH, payload)
        _write_csv(full_records + matched_records)
        saved_payload = json.loads(FINAL_RESULTS_PATH.read_text(encoding="utf-8"))
        _write_reports_from_artifacts(saved_payload)
        return saved_payload
    finally:
        if results_dir != original_paths[0]:
            (
                RESULTS_DIR,
                PLOTS_DIR,
                FINAL_RESULTS_PATH,
                FINAL_CSV_PATH,
                FINAL_CONFIG_PATH,
                FINAL_RESULTS_README_PATH,
                FINAL_REPORT_PATH,
                SIH_SUMMARY_PATH,
            ) = original_paths


if __name__ == "__main__":
    final = run_final_experiment()
    print(json.dumps({"status": "COMPLETED", "artifact": str(FINAL_RESULTS_PATH), "timestamp": final["timestamp_utc"]}))
