"""Streamlit composition for the combined Q-MedAI faculty experience."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MAX_KERNEL_SUBSET_SIZE, SUPPORTED_FEATURE_COUNTS, ExperimentConfig
from src.demo.artifacts import inspect_framingham_artifacts, load_framingham_artifacts
from src.demo.contracts import ArtifactError, ArtifactStatus, PatientValidationError, TaskEvidence, UtilityVerdict
from src.demo.evidence import load_breast_cancer_evidence, load_framingham_evidence
from src.demo.framingham import PATIENT_FIELDS, predict_patient
from src.demo.styles import THEME_CSS
from src.demo.utility import evaluate_utility, evidence_table
from src.runner import MODEL_NAMES, run_experiment


PAGES = (
    "Command Center",
    "Patient Risk",
    "Quantum Lab",
    "Model Arena",
    "Breast Cancer Evidence",
    "Quantum Utility",
    "Provenance & Safety",
    "Research Runner",
)
QUANTUM_FEATURES = ("age", "sysBP", "prevalentHyp", "diaBP")
FEATURE_MAP_SEQUENCE = "H → RY(x) → RZ(0.5x) → ring-CZ → RY(x²/π)"
FEATURED_BREAST_PLOT = Path("results/plots/matched_classical_quantum_judge_summary.png")


@dataclass(frozen=True)
class DemoContext:
    project_root: Path
    framingham: TaskEvidence
    breast_cancer: TaskEvidence
    framingham_verdict: UtilityVerdict
    breast_cancer_verdict: UtilityVerdict
    artifact_status: ArtifactStatus
    artifact_directory: Path | None = None


def _metric_frame(evidence: TaskEvidence) -> pd.DataFrame:
    frame = evidence_table(evidence).copy()
    frame = frame.rename(
        columns={
            "model": "Model",
            "family": "Family",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
            "auprc": "AUPRC",
            "false_negatives": "False negatives",
            "training_seconds": "Training time (s)",
        }
    )
    return frame.dropna(axis=1, how="all")


def _verdict_card(label: str, verdict: UtilityVerdict) -> None:
    st.markdown(
        f"<div class='qm-verdict'><strong>{label}</strong><br>"
        f"<code>{verdict.status}</code> — {verdict.headline}</div>",
        unsafe_allow_html=True,
    )


def _download(path: Path, label: str, mime: str) -> None:
    if path.is_file():
        st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime)


def _breast_plot_paths(project_root: Path) -> tuple[Path, ...]:
    """Return unique authoritative plot paths with the judge summary first."""
    evidence_path = project_root / "results/final_results.json"
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        listed = payload["plots"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        st.warning(f"Authoritative Breast Cancer plot inventory is unavailable: {exc}")
        return ()
    if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
        st.warning("Authoritative Breast Cancer plot inventory must be a list of paths.")
        return ()

    relative_paths = [Path(item) for item in listed]
    ordered = [FEATURED_BREAST_PLOT, *(path for path in relative_paths if path != FEATURED_BREAST_PLOT)]
    unique: list[Path] = []
    for path in ordered:
        if path in relative_paths and path not in unique:
            unique.append(path)
    return tuple(unique)


def _render_breast_plot(project_root: Path, relative_path: Path) -> None:
    path = project_root / relative_path
    if not path.is_file():
        st.warning(f"Authoritative Breast Cancer plot is listed but unavailable: {relative_path.as_posix()}")
        return
    st.image(str(path), caption=path.name, width="stretch")
    st.download_button(
        f"Download {path.name}",
        path.read_bytes(),
        file_name=path.name,
        mime="image/png",
        key=f"breast_plot_{relative_path.as_posix()}",
    )


def _signature(config: ExperimentConfig, selected_models: set[str]) -> str:
    payload = {"config": asdict(config), "models": sorted(selected_models)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def render_command_center(context: DemoContext) -> None:
    st.subheader("Faculty command center")
    st.markdown(
        "<div class='qm-card'><strong>One platform, two evidence roles.</strong><br>"
        "Framingham is the primary prospective 10-year CHD research workflow. "
        "Breast Cancer is a separate, matched quantum evidence benchmark.</div>",
        unsafe_allow_html=True,
    )
    first, second, third = st.columns(3)
    first.metric("Framingham cohort", f"{context.framingham.cohort['rows']:,}")
    second.metric("Framingham held-out rows", f"{context.framingham.cohort['test_rows']:,}")
    third.metric("Artifact gate", "READY" if context.artifact_status.ready else "EVIDENCE ONLY")
    st.markdown("#### Hybrid decision path")
    st.markdown(
        "**Baseline variables** → verified preprocessing → **Classical / Quantum / Hybrid research scores** "
        "→ task-specific utility verdict. The utility verdict does not assert universal quantum advantage."
    )
    left, right = st.columns(2)
    with left:
        _verdict_card("Framingham · primary workflow", context.framingham_verdict)
    with right:
        _verdict_card("Breast Cancer · evidence benchmark", context.breast_cancer_verdict)
    if context.artifact_status.ready:
        st.success(context.artifact_status.message)
    else:
        st.warning(f"{context.artifact_status.message} Patient inference is safely disabled.")
    st.info("Quantum circuits run on PennyLane default.qubit, a classical simulator—not physical quantum hardware.")


def _patient_form(ready: bool) -> tuple[dict[str, float], bool]:
    values: dict[str, float] = {}
    with st.form("framingham_patient_form", clear_on_submit=False):
        columns = st.columns(3)
        for index, field in enumerate(PATIENT_FIELDS):
            column = columns[index % len(columns)]
            if field.binary:
                values[field.name] = float(
                    column.selectbox(
                        field.label,
                        options=(0, 1),
                        index=int(field.default),
                        format_func=lambda value: "Yes" if value else "No",
                    )
                )
            else:
                values[field.name] = float(
                    column.number_input(
                        field.label,
                        min_value=float(field.minimum),
                        max_value=float(field.maximum),
                        value=float(field.default),
                        step=float(field.step),
                    )
                )
        submitted = st.form_submit_button("Run verified inference", type="primary") if ready else False
    return values, submitted


def render_patient_risk(context: DemoContext) -> None:
    st.subheader("Framingham patient risk research workflow")
    st.caption("Prospective 10-year CHD research · exact 15-field notebook input contract")
    st.info(
        "Patient values remain in this Streamlit session only. They are not written to files, analytics, "
        "or authoritative evidence artifacts."
    )
    st.caption(
        "Outputs are uncalibrated research-model scores, not validated 10-year probabilities, diagnoses, "
        "risk bands, or treatment recommendations."
    )
    values, submitted = _patient_form(context.artifact_status.ready)
    if not context.artifact_status.ready:
        st.warning("Verified Framingham artifacts are not installed, so Q-MedAI will produce no patient score.")
        st.button("Run verified inference", type="primary", disabled=True)
        if context.artifact_status.missing_files:
            st.write("**Missing trusted files:**", ", ".join(context.artifact_status.missing_files))
        st.caption(
            "Export the five frozen files with kaggle/framingham_artifact_export.py and place them in "
            "artifacts/framingham. The application never substitutes example or synthetic scores."
        )
        return
    if submitted:
        try:
            artifact_directory = context.artifact_directory or context.project_root / "artifacts/framingham"
            artifacts = load_framingham_artifacts(artifact_directory)
            result = predict_patient(artifacts, values)
        except (ArtifactError, PatientValidationError) as exc:
            st.error(f"Verified inference stopped safely: {exc}")
            return
        first, second, third = st.columns(3)
        first.metric("Classical research-model score", f"{result.classical_score:.3f}")
        second.metric("Quantum research score", f"{result.quantum_score:.3f}")
        third.metric("Hybrid research score", f"{result.hybrid_score:.3f}")
        st.success(f"Evidence-supported pathway: {result.recommended_pathway}")
        if result.contributions:
            st.markdown("#### Logistic Regression model associations")
            st.dataframe(
                pd.DataFrame(result.contributions, columns=["Feature", "Model contribution"]),
                hide_index=True,
                width="stretch",
            )


def render_quantum_lab(context: DemoContext) -> None:
    st.subheader("Quantum Lab · Framingham feature map")
    st.write("**Mutual-information-selected variables:** " + " · ".join(QUANTUM_FEATURES))
    st.code(
        "selected = [age, sysBP, prevalentHyp, diaBP]\n"
        "encoded = clip(standardized[selected], -3, 3) * (pi / 3)\n"
        f"feature_map = {FEATURE_MAP_SEQUENCE}\n"
        "kernel(x, z) = |<state(x)|state(z)>|^2",
        language="text",
    )
    st.markdown(
        "The four-qubit circuit creates a simulated quantum state, then compares it with frozen training "
        "reference states through a fidelity kernel. `ring-CZ` entangles each qubit with its neighbour."
    )
    st.warning(
        "Backend disclosure: PennyLane default.qubit is an analytic classical simulator with no shot noise. "
        "This demonstrates quantum-circuit semantics, not physical-hardware execution or computational speedup."
    )
    st.caption(f"Feature-map contract: framingham_v1_h_rz_cz_reupload · {context.framingham.backend}")


def render_model_arena(context: DemoContext) -> None:
    st.subheader("Model Arena · Framingham")
    _verdict_card("Task-specific utility", context.framingham_verdict)
    st.dataframe(_metric_frame(context.framingham), hide_index=True, width="stretch")
    st.warning(
        "Unmatched resource comparison: quantum and hybrid models use 500 training rows and four features; "
        "full-data classical models use 3,306 rows and 15 features. These results cannot establish quantum advantage."
    )
    st.markdown("**How to read the table**")
    st.write(
        "ROC-AUC and AUPRC summarize ranking across thresholds. Accuracy, sensitivity, specificity, precision, "
        "and F1 depend on the selected decision threshold. Runtime on default.qubit is simulator context only."
    )
    st.caption(
        "One executed held-out split; per-model bootstrap only; no paired uncertainty, external validation, "
        "or demonstrated probability calibration."
    )


def render_breast_cancer_evidence(context: DemoContext) -> None:
    st.subheader("Breast Cancer · quantum evidence benchmark")
    st.caption("Separate cross-sectional diagnostic-classification benchmark; not a Framingham patient calculator.")
    _verdict_card("Matched 150-row verdict", context.breast_cancer_verdict)
    plot_paths = _breast_plot_paths(context.project_root)
    if plot_paths:
        st.markdown("#### Featured matched judge figure")
        _render_breast_plot(context.project_root, plot_paths[0])
    st.dataframe(_metric_frame(context.breast_cancer), hide_index=True, width="stretch")
    deltas = context.breast_cancer_verdict.deltas
    first, second, third = st.columns(3)
    first.metric("Sensitivity Δ vs Random Forest", f"{deltas['sensitivity']:+.4f}")
    second.metric("False-negative Δ vs Random Forest", f"{deltas['false_negatives']:+.0f}")
    third.metric("ROC-AUC Δ vs Random Forest", f"{deltas['roc_auc']:+.4f}")
    st.info(
        "The Quantum Kernel shows a selective sensitivity benefit and one fewer false negative versus Random "
        "Forest on this matched split. RBF SVM remains the strongest overall model by the declared ranking rule."
    )
    st.markdown("#### Authoritative evidence downloads")
    download_columns = st.columns(4)
    downloads = (
        (context.project_root / "results/final_results.csv", "Download matched CSV", "text/csv"),
        (context.project_root / "results/final_results.json", "Download evidence JSON", "application/json"),
        (context.project_root / "results/final_config.json", "Download configuration", "application/json"),
        (context.project_root / "SIH_FINAL_SUMMARY.md", "Download SIH final summary", "text/markdown"),
    )
    for column, (path, label, mime) in zip(download_columns, downloads):
        with column:
            _download(path, label, mime)
    if len(plot_paths) > 1:
        st.markdown("#### Additional authoritative result figures")
        for plot_path in plot_paths[1:]:
            _render_breast_plot(context.project_root, plot_path)
    for limitation in context.breast_cancer.limitations:
        st.caption(limitation)


def render_quantum_utility(context: DemoContext) -> None:
    st.subheader("Quantum Utility Engine")
    st.write(
        "Verdicts are deterministic and task-specific. They combine model deltas, comparison fairness, "
        "uncertainty status, and backend—not promotional wording."
    )
    for evidence, verdict in (
        (context.framingham, context.framingham_verdict),
        (context.breast_cancer, context.breast_cancer_verdict),
    ):
        st.markdown(f"#### {evidence.title}")
        _verdict_card(evidence.role.replace("_", " ").title(), verdict)
        st.write(
            f"**Reference:** {verdict.reference_model}  ·  **Candidate:** {verdict.candidate_model}  ·  "
            f"**Best overall:** {verdict.best_overall_model}"
        )
        st.dataframe(
            pd.DataFrame(
                ((metric.replace("_", " ").title(), value) for metric, value in verdict.deltas.items()),
                columns=["Candidate minus reference", "Delta"],
            ),
            hide_index=True,
            width="stretch",
        )
        for reason in verdict.rationale:
            st.caption(reason)
    st.warning("No result here is a universal quantum-advantage claim.")


def render_provenance(context: DemoContext) -> None:
    st.subheader("Provenance & Safety")
    st.markdown("#### Evidence sources")
    st.write(f"**Framingham:** {context.framingham.source}")
    st.write(f"**Breast Cancer:** {context.breast_cancer.source}")
    st.write("**Framingham cohort:**", dict(context.framingham.cohort))
    st.write("**Breast Cancer matched cohort:**", dict(context.breast_cancer.cohort))
    st.write(f"**Artifact status:** {context.artifact_status.code} — {context.artifact_status.message}")
    if context.artifact_status.ready and context.artifact_status.manifest is not None:
        st.markdown("#### Verified frozen-artifact manifest")
        st.json(dict(context.artifact_status.manifest))
    else:
        st.warning("No manifest data is displayed because a complete verified artifact set is not installed.")
        if context.artifact_status.missing_files:
            st.write("**Missing files:**", ", ".join(context.artifact_status.missing_files))
    st.markdown("#### Full Framingham limitations")
    for limitation in context.framingham.limitations:
        st.write(f"- {limitation}")
    st.markdown("#### Safety boundary")
    st.write(
        "Research and faculty demonstration only. Not a medical device; not clinically validated; no diagnosis, "
        "treatment recommendation, calibrated probability, or patient-data persistence. Arbitrary uploaded "
        "pickle/joblib files are unsupported because deserialization can execute code."
    )
    files = (
        (context.project_root / "results/framingham/notebook_metrics.json", "Download Framingham metrics"),
        (context.project_root / "results/final_results.json", "Download Breast Cancer evidence"),
    )
    for path, label in files:
        _download(path, label, "application/json")


def render_research_runner() -> None:
    st.subheader("Research Runner")
    st.caption("This is exploratory and session-only. Session-only runs never overwrite authoritative artifacts.")
    method = st.radio("Feature reduction", ("SelectKBest", "PCA"), horizontal=True)
    feature_count = st.selectbox("Output features / quantum qubits", SUPPORTED_FEATURE_COUNTS, index=0)
    if method == "PCA":
        st.info("PCA components are transformed combinations of original biomarkers, not individual biomarkers.")
    selected = {name for name in MODEL_NAMES if st.checkbox(name, value=True, key=f"explore_{name}")}
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
            st.warning("Configuration changed after this result. Run again to replace stale session results.")
        else:
            st.dataframe(result.comparison_table, width="stretch", hide_index=True)


def _run_new_experiment() -> None:
    """Compatibility wrapper for the formerly top-level exploratory runner."""
    render_research_runner()


def render_app(project_root: Path, artifact_directory: Path | None = None) -> None:
    """Render the faculty app using a local trusted artifact directory.

    ``artifact_directory`` supports controlled deployments and isolated tests;
    no UI accepts arbitrary artifact uploads.
    """
    framingham = load_framingham_evidence(project_root / "results/framingham/notebook_metrics.json")
    breast = load_breast_cancer_evidence(project_root / "results/final_results.json")
    resolved_artifact_directory = artifact_directory or project_root / "artifacts/framingham"
    status = inspect_framingham_artifacts(resolved_artifact_directory)
    context = DemoContext(
        project_root=project_root,
        framingham=framingham,
        breast_cancer=breast,
        framingham_verdict=evaluate_utility(framingham),
        breast_cancer_verdict=evaluate_utility(breast),
        artifact_status=status,
        artifact_directory=resolved_artifact_directory,
    )

    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.title("Q-MedAI Clinical Intelligence")
    st.markdown(
        "<div class='qm-hero'><div class='qm-eyebrow'>SIH 26139 · FACULTY EVIDENCE CONSOLE</div>"
        "<h3>Framingham early-risk workflow + Breast Cancer quantum evidence</h3>"
        "Task-specific classical, quantum, and hybrid evidence with verification-gated inference.</div>",
        unsafe_allow_html=True,
    )
    st.caption("Research prototype · no patient data persistence · not a medical device")
    page = st.sidebar.radio("Navigate", PAGES, index=0)
    st.sidebar.caption("Framingham is the primary workflow. Breast Cancer is a separate benchmark.")

    renderers = {
        "Command Center": render_command_center,
        "Patient Risk": render_patient_risk,
        "Quantum Lab": render_quantum_lab,
        "Model Arena": render_model_arena,
        "Breast Cancer Evidence": render_breast_cancer_evidence,
        "Quantum Utility": render_quantum_utility,
        "Provenance & Safety": render_provenance,
    }
    if page == "Research Runner":
        render_research_runner()
    else:
        renderers[page](context)


__all__ = [
    "DemoContext",
    "render_app",
    "render_breast_cancer_evidence",
    "render_command_center",
    "render_model_arena",
    "render_patient_risk",
    "render_provenance",
    "render_quantum_lab",
    "render_quantum_utility",
    "render_research_runner",
]
