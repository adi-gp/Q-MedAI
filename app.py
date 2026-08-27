"""Presentation-ready Streamlit demo backed by saved Q-MedAI final artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MAX_KERNEL_SUBSET_SIZE, SUPPORTED_FEATURE_COUNTS, ExperimentConfig
from src.data.loader import load_dataset
from src.final_experiment import PROJECT_ROOT, load_final_artifacts
from src.runner import MODEL_NAMES, run_experiment


st.set_page_config(page_title="Q-MedAI", page_icon="🧬", layout="wide")


def _signature(config: ExperimentConfig, selected_models: set[str]) -> str:
    payload = {"config": asdict(config), "models": sorted(selected_models)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _metric_table(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    columns = [
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
    ]
    frame = frame.reindex(columns=columns)
    return frame.rename(
        columns={
            "model": "Model",
            "type": "Type",
            "status": "Status",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall / Sensitivity",
            "specificity": "Specificity",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
            "training_time_seconds": "Training time (s)",
        }
    )


def _dataset_view(config: dict | None) -> None:
    st.subheader("Dataset")
    try:
        dataset = load_dataset()
        report = dataset.report
        first, second, third = st.columns(3)
        first.metric("Samples", report.sample_count)
        second.metric("Numeric features", report.feature_count)
        third.metric("Missing feature values", report.missing_values)
        st.write("**Dataset source:**", report.source)
        st.write("**Class distribution:**", report.class_distribution)
        st.write("**Target encoding:** B = 0 (benign), M = 1 (malignant / positive class)")
        st.dataframe(report.removed_columns, use_container_width=True, hide_index=True)
        provenance = None if config is None else config["dataset"].get("provenance_status")
        if provenance:
            st.caption(provenance)
    except Exception as exc:
        st.error(f"Dataset validation failed: {type(exc).__name__}: {exc}")


def _download_artifacts() -> None:
    st.subheader("Download final artifacts")
    downloads = [
        ("Download final_results.csv", PROJECT_ROOT / "results" / "final_results.csv", "text/csv"),
        ("Download final_results.json", PROJECT_ROOT / "results" / "final_results.json", "application/json"),
        ("Download final_config.json", PROJECT_ROOT / "results" / "final_config.json", "application/json"),
        ("Download SIH final summary", PROJECT_ROOT / "SIH_FINAL_SUMMARY.md", "text/markdown"),
    ]
    for label, path, mime in downloads:
        if path.exists():
            st.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime)


def _overview(payload: dict | None) -> None:
    st.subheader("What Q-MedAI does")
    st.write(
        "Q-MedAI compares classical machine-learning models with quantum-machine-learning models "
        "for binary disease classification under the same held-out test methodology."
    )
    first, second = st.columns(2)
    first.markdown("**Classical ML**\n\nLogistic Regression, RBF SVM, and Random Forest establish strong baselines.")
    second.markdown(
        "**Quantum ML**\n\nA variational quantum classifier and fidelity quantum-kernel SVM are evaluated. "
        "PennyLane `default.qubit` simulates these circuits classically."
    )
    if payload is not None:
        st.success("DEMO / FINAL RESULTS mode is active. No training runs automatically.")
        st.info(
            "The fair comparison uses exactly 150 stratified training rows for every matched-data model. "
            "Predictive scores are observed results, not quantum advantage."
        )
    else:
        st.warning("No saved final artifacts were found. Use Run New Experiment to create a session-only exploratory result.")


def _final_results(payload: dict, config: dict) -> None:
    st.subheader("Experiment Status")
    statuses = {
        record["model"]: record["status"]
        for record in payload["full_data_final_results"]["models"]
    }
    st.json(statuses)
    st.subheader("FULL-DATA FINAL RESULTS")
    st.caption(
        "Classical models use all 455 training rows. The quantum kernel uses 150 rows here, "
        "so this table is context—not a matched training-size comparison."
    )
    st.dataframe(_metric_table(payload["full_data_final_results"]["models"]), use_container_width=True, hide_index=True)
    st.subheader("MATCHED-DATA COMPARISON — 150 training samples for all models")
    st.caption(
        "All four models use the exact same stratified 150 original training rows, a separately fitted "
        "training-only preprocessing pipeline, and the unchanged held-out test set."
    )
    st.dataframe(_metric_table(payload["matched_data_comparison"]["models"]), use_container_width=True, hide_index=True)
    difference = payload["analysis"]["matched_quantum_kernel_vs_best_classical"]
    if difference.get("status") != "NOT AVAILABLE":
        st.info(
            "Observed performance difference (quantum kernel minus best matched classical reference): "
            f"ROC-AUC {difference['roc_auc_difference']:+.4f}, "
            f"accuracy {difference['accuracy_difference']:+.4f}, F1 {difference['f1_difference']:+.4f}."
        )
    st.caption(payload["analysis"]["statistical_confidence"])
    _download_artifacts()


def _visualizations(payload: dict) -> None:
    st.subheader("Actual final plots")
    for relative_path in payload.get("plots", []):
        path = PROJECT_ROOT / relative_path
        if path.exists():
            st.image(str(path), caption=path.name, use_container_width=True)
            st.download_button(
                f"Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="image/png",
                key=f"download_{path.name}",
            )


def _explainability(payload: dict) -> None:
    st.subheader("Explainability")
    explainability = payload.get("explainability", {})
    for name, rows in explainability.get("tables", {}).items():
        label = "Permutation feature importance"
        st.markdown(f"**{name} — {label}**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    full_vqc = next(row for row in payload["full_data_final_results"]["models"] if row["model"] == "VQC")
    if full_vqc["status"] != "COMPLETED":
        st.warning(
            "VQC perturbation-based model sensitivity is unavailable because the final VQC did not complete "
            f"({full_vqc['status']}: {full_vqc['error']})."
        )
    for name, error in explainability.get("errors", {}).items():
        st.warning(f"{name} explanation failed: {error}")
    st.caption("Quantum explanation method: Perturbation-based model sensitivity. It is not SHAP.")


def _experiment_summary(payload: dict, config: dict) -> None:
    st.subheader("Saved final configuration")
    summary = {
        "Dataset": config["dataset"]["source"],
        "Preprocessing": config["preprocessing"],
        "Features / qubits": config["feature_count"],
        "VQC layers": config["vqc"]["layers"],
        "VQC iterations": config["vqc"]["iterations"],
        "Quantum-kernel subset": config["quantum_kernel"]["subset_size"],
        "Random seed": config["random_seed"],
        "Target encoding": config["target_encoding"],
        "Backend": config["backend"],
        "Final artifact timestamp": payload["timestamp_utc"],
    }
    st.json(summary)
    st.markdown("**SIH talking points**")
    st.write(
        "• Strong classical baselines were evaluated first.\n"
        "• The kernel’s constrained sample count is addressed through an exact matched-data comparison.\n"
        "• The simulator result is a feasibility finding, not quantum advantage or clinical validation."
    )


def _run_new_experiment() -> None:
    st.subheader("RUN NEW EXPERIMENT")
    st.caption("This is exploratory and session-only. It never overwrites FINAL EXPERIMENT artifacts.")
    method = st.radio("Feature reduction", ("SelectKBest", "PCA"), horizontal=True)
    feature_count = st.selectbox("Output features / quantum qubits", SUPPORTED_FEATURE_COUNTS, index=0)
    if method == "PCA":
        st.info(
            "The model is operating on PCA components, which are transformed combinations of the original "
            "biomarkers rather than individual biomarkers."
        )
    selected = {
        name
        for name in MODEL_NAMES
        if st.checkbox(name, value=True, key=f"explore_{name}")
    }
    first, second, third = st.columns(3)
    layers = first.number_input("VQC layers", min_value=1, max_value=3, value=3, step=1)
    iterations = second.number_input("VQC iterations", min_value=1, max_value=100, value=100, step=1)
    subset = third.number_input(
        "Quantum-kernel training subset", min_value=2, max_value=MAX_KERNEL_SUBSET_SIZE, value=150, step=1
    )
    config = ExperimentConfig(
        reduction_method=method,
        n_features=int(feature_count),
        vqc_layers=int(layers),
        vqc_iterations=int(iterations),
        kernel_subset_size=int(subset),
    )
    signature = _signature(config, selected)
    if st.button("🚀 Run Experiment", type="primary"):
        st.session_state["exploratory_status"] = "TRAINING"
        with st.spinner("Training selected exploratory models…"):
            st.session_state["exploratory_result"] = run_experiment(config, selected_models=selected)
            st.session_state["exploratory_signature"] = signature
            st.session_state["exploratory_status"] = "COMPLETED"
    status = st.session_state.get("exploratory_status", "NOT TRAINED")
    st.write(f"**Experiment Status: {status}**")
    result = st.session_state.get("exploratory_result")
    if result is not None:
        if st.session_state.get("exploratory_signature") != signature:
            st.warning("Configuration changed after this result. Press Run Experiment to replace stale session results.")
        else:
            st.dataframe(result.comparison_table, use_container_width=True, hide_index=True)


def main() -> None:
    loaded = load_final_artifacts()
    payload, config = loaded if loaded is not None else (None, None)
    st.title("Q-MedAI")
    st.caption("Hybrid Quantum-Classical Disease Detection")
    st.warning("This software is a research prototype and is not a medical diagnostic device.")
    st.info("Quantum models are simulated using PennyLane default.qubit. This MVP does not run on a physical quantum computer.")

    pages = ["Overview", "Dataset", "Run New Experiment"]
    if payload is not None:
        pages = ["Overview", "Dataset", "Final Results", "Visualizations", "Explainability", "Experiment Summary", "Run New Experiment"]
    page = st.sidebar.radio("Navigate", pages, index=0)

    if page == "Overview":
        _overview(payload)
    elif page == "Dataset":
        _dataset_view(config)
    elif page == "Final Results" and payload is not None:
        _final_results(payload, config)
    elif page == "Visualizations" and payload is not None:
        _visualizations(payload)
    elif page == "Explainability" and payload is not None:
        _explainability(payload)
    elif page == "Experiment Summary" and payload is not None:
        _experiment_summary(payload, config)
    else:
        _run_new_experiment()


if __name__ == "__main__":
    main()
