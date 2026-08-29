"""Streamlit composition for the combined Q-MedAI faculty experience."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MAX_KERNEL_SUBSET_SIZE, SUPPORTED_FEATURE_COUNTS, ExperimentConfig
from src.demo.artifacts import inspect_framingham_artifacts, load_framingham_artifacts
from src.demo.contracts import ArtifactError, ArtifactStatus, PatientValidationError, TaskEvidence, UtilityVerdict
from src.demo.evidence import load_breast_cancer_evidence, load_framingham_evidence
from src.demo.framingham import PATIENT_FIELDS, predict_patient
from src.demo.quantum_viz import build_quantum_trace
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

PAGE_COPY = {
    "Command Center": (
        "Faculty Command Center",
        "One view of the prospective Framingham workflow, the separate breast-cancer evidence benchmark, "
        "and the evidence-derived architecture recommendation.",
    ),
    "Patient Risk": (
        "Patient Risk Analysis",
        "Real frozen-model inference from the exact 15-field Framingham input contract. "
        "The classical model remains the primary evidence-supported pathway.",
    ),
    "Quantum Lab": (
        "Quantum Lab",
        "Inspect the qubit-efficient feature representation and simulated fidelity-kernel pathway used by the research engine.",
    ),
    "Model Arena": (
        "Model Arena",
        "Compare classical, quantum, and hybrid evidence without hiding resource-budget differences or uncertainty limitations.",
    ),
    "Breast Cancer Evidence": (
        "Breast Cancer Evidence",
        "A separate matched diagnostic benchmark used to study selective quantum competitiveness—not a Framingham calculator.",
    ),
    "Quantum Utility": (
        "Quantum Utility Engine",
        "Task-specific architecture recommendations generated from measured evidence rather than assuming quantum is always better.",
    ),
    "Provenance & Safety": (
        "Provenance & Safety",
        "Trace the evidence sources, frozen-artifact gate, limitations, and simulator disclosures behind every displayed result.",
    ),
    "Research Runner": (
        "Advanced Research Runner",
        "Optional exploratory training. This is intentionally separated from the fast, frozen-artifact faculty demonstration.",
    ),
}


@dataclass(frozen=True)
class DemoContext:
    project_root: Path
    framingham: TaskEvidence
    breast_cancer: TaskEvidence
    framingham_verdict: UtilityVerdict
    breast_cancer_verdict: UtilityVerdict
    artifact_status: ArtifactStatus
    artifact_directory: Path | None = None


def _safe(value: object) -> str:
    return html.escape(str(value), quote=True)


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


def _section(title: str, copy: str | None = None) -> None:
    body = f"<div class='qm-section'><div class='qm-section-title'>{_safe(title)}</div>"
    if copy:
        body += f"<div class='qm-section-copy'>{_safe(copy)}</div>"
    body += "</div>"
    st.markdown(body, unsafe_allow_html=True)


def _pill(text: str, purple: bool = False, dark: bool = False) -> str:
    class_name = "qm-pill"
    if purple:
        class_name += " qm-pill-purple"
    if dark:
        class_name += " qm-pill-dark"
    return f"<span class='{class_name}'>{_safe(text)}</span>"


def _verdict_card(label: str, verdict: UtilityVerdict) -> None:
    st.markdown(
        "<div class='qm-verdict'>"
        f"<div class='qm-verdict-label'>{_safe(label)}</div>"
        f"<div class='qm-verdict-headline'>{_pill(verdict.status)} {_safe(verdict.headline)}</div>"
        "</div>",
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


def _artifact_badge(context: DemoContext) -> str:
    if context.artifact_status.ready:
        return _pill("ARTIFACTS VERIFIED")
    return _pill("EVIDENCE ONLY", dark=True)


def _stat_grid(items: tuple[tuple[str, str, str], ...]) -> None:
    cards = "".join(
        "<div class='qm-stat'>"
        f"<div class='qm-stat-label'>{_safe(label)}</div>"
        f"<div class='qm-stat-value'>{_safe(value)}</div>"
        f"<div class='qm-stat-note'>{_safe(note)}</div>"
        "</div>"
        for label, value, note in items
    )
    st.markdown(f"<div class='qm-stat-grid'>{cards}</div>", unsafe_allow_html=True)


def render_command_center(context: DemoContext) -> None:
    _section(
        "Faculty command center",
        "A concise product view of what Q-MedAI does, what evidence supports each pathway, "
        "and where quantum processing is or is not currently justified.",
    )

    st.markdown(
        "<div class='qm-card'>"
        "<div class='qm-card-title'>One platform, two evidence roles</div>"
        "<div class='qm-card-copy'>"
        "Framingham is the primary prospective 10-year CHD research workflow. "
        "Breast Cancer remains a separate matched diagnostic benchmark used to evaluate quantum competitiveness."
        "</div></div>",
        unsafe_allow_html=True,
    )

    _stat_grid(
        (
            ("Framingham cohort", f"{context.framingham.cohort['rows']:,}", "baseline patient records"),
            ("Held-out evaluation", f"{context.framingham.cohort['test_rows']:,}", "Framingham test rows"),
            ("Inference gate", "READY" if context.artifact_status.ready else "LOCKED", "verified frozen artifacts"),
        )
    )

    _section(
        "Hybrid decision path",
        "Q-MedAI does not force quantum inference. It evaluates classical, quantum, and hybrid evidence, "
        "then recommends the architecture supported by the current task.",
    )

    st.markdown(
        "<div class='qm-flow'>"
        "<div class='qm-flow-step'><div class='qm-flow-num'>01 · INPUT</div>"
        "<div class='qm-flow-title'>Baseline variables</div><div class='qm-flow-copy'>15-field Framingham contract</div></div>"
        "<div class='qm-flow-step'><div class='qm-flow-num'>02 · PROCESS</div>"
        "<div class='qm-flow-title'>Verified preprocessing</div><div class='qm-flow-copy'>Frozen feature order and transforms</div></div>"
        "<div class='qm-flow-step'><div class='qm-flow-num'>03 · INFER</div>"
        "<div class='qm-flow-title'>CML · QML · Hybrid</div><div class='qm-flow-copy'>Parallel research pathways</div></div>"
        "<div class='qm-flow-step'><div class='qm-flow-num'>04 · AUDIT</div>"
        "<div class='qm-flow-title'>Utility decision</div><div class='qm-flow-copy'>Evidence-derived recommendation</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="medium")
    with left:
        _verdict_card("Framingham · primary workflow", context.framingham_verdict)
    with right:
        _verdict_card("Breast Cancer · evidence benchmark", context.breast_cancer_verdict)

    if context.artifact_status.ready:
        st.success("Verified frozen Framingham artifacts are ready for real patient inference.")
    else:
        st.warning(f"{context.artifact_status.message} Patient inference is safely disabled.")

    st.info(
        "Quantum circuits run on PennyLane default.qubit, a classical simulator. "
        "The prototype does not claim physical-quantum speedup."
    )


# Field groups preserve the exact notebook contract while making the UI scannable.
_PATIENT_GROUPS = (
    (
        "Demographics & behavior",
        ("sex_male", "age", "education", "currentSmoker", "cigsPerDay"),
    ),
    (
        "Vitals & laboratory",
        ("totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"),
    ),
    (
        "Clinical history",
        ("BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"),
    ),
)


def _field_lookup() -> dict[str, object]:
    return {field.name: field for field in PATIENT_FIELDS}


def _render_field(column, field) -> float:
    if field.binary:
        return float(
            column.selectbox(
                field.label,
                options=(0, 1),
                index=int(field.default),
                format_func=lambda value: "Yes" if value else "No",
                key=f"patient_{field.name}",
            )
        )
    return float(
        column.number_input(
            field.label,
            min_value=float(field.minimum),
            max_value=float(field.maximum),
            value=float(field.default),
            step=float(field.step),
            key=f"patient_{field.name}",
        )
    )


def _patient_form(ready: bool) -> tuple[dict[str, float], bool]:
    values: dict[str, float] = {}
    lookup = _field_lookup()

    with st.form("framingham_patient_form", clear_on_submit=False):
        group_columns = st.columns(3, gap="large")

        for group_column, (group_name, field_names) in zip(group_columns, _PATIENT_GROUPS):
            with group_column:
                st.markdown(f"<div class='qm-field-group'>{_safe(group_name)}</div>", unsafe_allow_html=True)
                for field_name in field_names:
                    values[field_name] = _render_field(group_column, lookup[field_name])

        submitted = (
            st.form_submit_button(
                "Run verified inference",
                type="primary",
                width="stretch",
            )
            if ready
            else False
        )

    return values, submitted


def _score_cards(classical_score: float, quantum_score: float, hybrid_score: float) -> None:
    st.markdown(
        "<div class='qm-score-grid'>"
        "<div class='qm-score-card'>"
        "<div class='qm-score-label'>Classical · primary</div>"
        f"<div class='qm-score-value'>{classical_score:.3f}</div>"
        "<div class='qm-score-note'>Evidence-supported Framingham pathway</div></div>"
        "<div class='qm-score-card'>"
        "<div class='qm-score-label'>Quantum · research</div>"
        f"<div class='qm-score-value'>{quantum_score:.3f}</div>"
        "<div class='qm-score-note'>4-qubit fidelity-kernel branch</div></div>"
        "<div class='qm-score-card'>"
        "<div class='qm-score-label'>Hybrid · research</div>"
        f"<div class='qm-score-value'>{hybrid_score:.3f}</div>"
        "<div class='qm-score-note'>Classical + quantum fusion branch</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _contribution_table(contributions) -> None:
    if not contributions:
        return

    frame = pd.DataFrame(contributions, columns=["Feature", "Model contribution"]).copy()
    frame["Absolute contribution"] = frame["Model contribution"].abs()
    frame = frame.sort_values("Absolute contribution", ascending=False).drop(columns=["Absolute contribution"])

    _section(
        "Top model associations",
        "Largest Logistic Regression contributions for this patient. These are model associations, not causal medical explanations.",
    )
    st.dataframe(frame.head(8), hide_index=True, width="stretch")


def render_patient_risk(context: DemoContext) -> None:
    _section(
        "Framingham patient risk research workflow",
        "Enter baseline measurements once. Q-MedAI applies the frozen preprocessing and displays the "
        "classical primary score beside the quantum and hybrid research branches.",
    )

    st.markdown(
        "<div class='qm-card'>"
        "<div class='qm-card-title'>Privacy & interpretation boundary</div>"
        "<div class='qm-card-copy'>Patient values remain in this Streamlit session only and are not persisted. "
        "Outputs are uncalibrated research-model scores—not diagnoses, treatment recommendations, "
        "or clinically validated 10-year probabilities.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    values, submitted = _patient_form(context.artifact_status.ready)

    if not context.artifact_status.ready:
        st.warning("Verified Framingham artifacts are not installed, so Q-MedAI will produce no patient score.")
        st.button("Run verified inference", type="primary", disabled=True, width="stretch")
        if context.artifact_status.missing_files:
            st.write("**Missing trusted files:**", ", ".join(context.artifact_status.missing_files))
        st.caption(
            "Export the frozen files with kaggle/framingham_artifact_export.py and place them in artifacts/framingham. "
            "The application never substitutes synthetic scores."
        )
        return

    if not submitted:
        st.markdown(
            "<div class='qm-empty'><strong>Ready for verified inference.</strong><br>"
            "Complete or review the baseline fields above, then select <strong>Run verified inference</strong>. "
            "The result uses the installed frozen artifacts; no retraining occurs.</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        artifact_directory = context.artifact_directory or context.project_root / "artifacts/framingham"
        artifacts = load_framingham_artifacts(artifact_directory)
        result = predict_patient(artifacts, values)

        # Session-only bridge to Quantum Lab.
        # Nothing is written to disk or authoritative evidence files.
        st.session_state["qmedai_latest_verified_patient"] = dict(values)
        st.session_state["qmedai_latest_quantum_score"] = float(result.quantum_score)

    except (ArtifactError, PatientValidationError) as exc:
        st.error(f"Verified inference stopped safely: {exc}")
        return

    _section(
        "Patient result",
        "The classical pathway is shown as primary because the locked Framingham evidence currently favors it.",
    )

    st.markdown(
        "<div class='qm-result-primary'>"
        "<div class='qm-result-kicker'>Primary evidence-supported output</div>"
        f"<div class='qm-result-score'>{result.classical_score:.3f}</div>"
        "<div class='qm-result-copy'>Classical Logistic Regression research-model score · uncalibrated</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    first, second, third = st.columns(3, gap="medium")
    first.metric(
        "Classical research-model score",
        f"{result.classical_score:.3f}",
    )
    second.metric(
        "Quantum research score",
        f"{result.quantum_score:.3f}",
    )
    third.metric(
        "Hybrid research score",
        f"{result.hybrid_score:.3f}",
    )

    st.success(f"Evidence-supported pathway: {result.recommended_pathway}")
    _verdict_card("Recommended architecture", context.framingham_verdict)
    _contribution_table(result.contributions)

    st.caption(
        "Research prototype only. A change in model score is not itself a clinical risk category, diagnosis, "
        "or treatment recommendation."
    )


def render_quantum_lab(context: DemoContext) -> None:
    _section(
        "Quantum Lab · Framingham feature map",
        "The quantum branch compresses the clinical representation into four selected dimensions and "
        "evaluates similarity through a simulated fidelity kernel.",
    )

    st.markdown(
        "<div class='qm-card'><div class='qm-card-title'>Selected quantum representation</div>"
        "<div class='qm-feature-row'>"
        + "".join(_pill(feature, purple=True) for feature in QUANTUM_FEATURES)
        + "</div>"
        "<div class='qm-card-copy'>Mutual-information-selected compact representation used by the frozen quantum branch.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    _stat_grid(
        (
            ("Qubits", "4", "compact patient representation"),
            ("Backend", "default.qubit", "classical quantum-circuit simulator"),
            ("Kernel", "Fidelity", "|⟨φ(x)|φ(z)⟩|²"),
        )
    )

    st.markdown(
        "<div class='qm-flow'>"
        "<div class='qm-flow-step'><div class='qm-flow-num'>01</div>"
        "<div class='qm-flow-title'>Select</div><div class='qm-flow-copy'>age · sysBP · prevalentHyp · diaBP</div></div>"
        "<div class='qm-flow-step'><div class='qm-flow-num'>02</div>"
        "<div class='qm-flow-title'>Encode</div><div class='qm-flow-copy'>clip standardized values and map to angles</div></div>"
        "<div class='qm-flow-step'><div class='qm-flow-num'>03</div>"
        "<div class='qm-flow-title'>Entangle</div><div class='qm-flow-copy'>H · RY · RZ · ring-CZ · re-upload</div></div>"
        "<div class='qm-flow-step'><div class='qm-flow-num'>04</div>"
        "<div class='qm-flow-title'>Compare</div><div class='qm-flow-copy'>fidelity to frozen reference states</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    latest_patient = st.session_state.get("qmedai_latest_verified_patient")

    if latest_patient is None:
        st.info(
            "Run verified inference on Patient Risk to inspect the latest patient's "
            "real quantum encoding, circuit, and fidelity-kernel trace."
        )

        with st.expander("View exact feature-map contract"):
            st.code(
                "selected = [age, sysBP, prevalentHyp, diaBP]\n"
                "encoded = clip(standardized[selected], -3, 3) * (pi / 3)\n"
                f"feature_map = {FEATURE_MAP_SEQUENCE}\n"
                "kernel(x, z) = |<state(x)|state(z)>|^2",
                language="text",
            )

        st.warning(
            "Backend disclosure: PennyLane default.qubit is an analytic classical simulator with no shot noise. "
            "This demonstrates quantum-circuit semantics—not physical-hardware execution or computational speedup."
        )
        st.caption(
            f"Feature-map contract: framingham_v1_h_rz_cz_reupload · {context.framingham.backend}"
        )
        return

    if not context.artifact_status.ready:
        st.warning(
            "The latest patient trace cannot be reconstructed because the verified "
            "Framingham artifact gate is not ready."
        )
        return

    try:
        artifact_directory = (
            context.artifact_directory
            or context.project_root / "artifacts/framingham"
        )

        artifacts = load_framingham_artifacts(artifact_directory)

        trace = build_quantum_trace(
            artifacts,
            latest_patient,
        )

    except (ArtifactError, PatientValidationError) as exc:
        st.error(f"Verified quantum trace stopped safely: {exc}")
        return

    _section(
        "Latest verified patient quantum trace",
        "These values come from the same frozen preprocessing, selected feature indices, "
        "feature map, and training reference states used by verified quantum inference.",
    )

    st.markdown(
        "**Raw value → transformed value → Encoded angle**"
    )

    trace_frame = pd.DataFrame(
        {
            "Quantum feature": trace.feature_names,
            "Raw value": trace.raw_values,
            "Transformed value": trace.transformed_values,
            "Encoded angle": trace.encoded_angles,
        }
    )

    st.dataframe(
        trace_frame,
        hide_index=True,
        width="stretch",
    )

    _section(
        "Patient-specific 4-qubit circuit",
        "PennyLane renders the exact feature-map operations evaluated for this patient.",
    )

    st.code(
        trace.circuit_text,
        language="text",
    )

    st.caption(
        "Circuit semantics: H → RY(θ) → RZ(0.5θ) → ring-CZ → RY(θ²/π). "
        "Controlled-Z gates may appear as connected ● / Z symbols in the PennyLane drawer."
    )

    _section(
        "Fidelity-kernel response",
        "The patient state is compared with the frozen quantum training reference states using "
        "|⟨φ(x)|φ(z)⟩|².",
    )

    first, second, third, fourth = st.columns(4)

    first.metric(
        "Kernel max fidelity",
        f"{trace.kernel_max:.3f}",
    )

    second.metric(
        "Kernel mean fidelity",
        f"{trace.kernel_mean:.3f}",
    )

    third.metric(
        "Kernel median fidelity",
        f"{trace.kernel_median:.3f}",
    )

    fourth.metric(
        "References compared",
        str(len(trace.kernel_similarities)),
    )

    quantum_score = st.session_state.get("qmedai_latest_quantum_score")

    if quantum_score is not None:
        st.metric(
            "Latest quantum research score",
            f"{float(quantum_score):.3f}",
        )

    similarities = pd.DataFrame(
        {
            "Reference": [
                f"Reference {index + 1}"
                for index in range(len(trace.kernel_similarities))
            ],
            "Fidelity": trace.kernel_similarities,
        }
    )

    strongest = (
        similarities
        .sort_values("Fidelity", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )

    with st.expander(
        "Strongest frozen-reference similarities",
        expanded=False,
    ):
        st.dataframe(
            strongest,
            hide_index=True,
            width="stretch",
        )

        st.bar_chart(
            strongest.set_index("Reference")["Fidelity"],
            height=300,
        )

    with st.expander("View exact feature-map contract"):
        st.code(
            "selected = [age, sysBP, prevalentHyp, diaBP]\n"
            "encoded = clip(standardized[selected], -3, 3) * (pi / 3)\n"
            f"feature_map = {FEATURE_MAP_SEQUENCE}\n"
            "kernel(x, z) = |<state(x)|state(z)>|^2",
            language="text",
        )

    st.warning(
        "Backend disclosure: PennyLane default.qubit is an analytic classical simulator with no shot noise. "
        "This demonstrates quantum-circuit semantics—not physical-hardware execution or computational speedup."
    )

    st.caption(
        f"Feature-map contract: framingham_v1_h_rz_cz_reupload · {context.framingham.backend}"
    )

def render_model_arena(context: DemoContext) -> None:
    _section(
        "Model Arena · Framingham",
        "The table preserves the real locked evidence. Use ranking metrics such as ROC-AUC and AUPRC "
        "instead of accuracy alone on this imbalanced CHD task.",
    )

    _verdict_card("Task-specific utility", context.framingham_verdict)
    st.dataframe(_metric_frame(context.framingham), hide_index=True, width="stretch")

    st.warning(
        "Resource-budget warning: quantum and hybrid models use 500 training rows and four features; "
        "full-data classical models use 3,306 rows and 15 features. This table alone cannot establish quantum advantage."
    )

    with st.expander("How to read these metrics"):
        st.write(
            "ROC-AUC and AUPRC summarize ranking across thresholds. Accuracy, sensitivity, specificity, precision, "
            "and F1 depend on the decision threshold. Runtime on default.qubit is simulator context only."
        )
        st.caption(
            "One executed held-out split; per-model bootstrap only; no paired uncertainty, external validation, "
            "or demonstrated probability calibration."
        )


def render_breast_cancer_evidence(context: DemoContext) -> None:
    _section(
        "Breast Cancer · quantum evidence benchmark",
        "A separate cross-sectional diagnostic-classification benchmark used to test whether the quantum kernel "
        "shows a selective benefit under a matched training budget.",
    )

    _verdict_card("Matched 150-row verdict", context.breast_cancer_verdict)
    plot_paths = _breast_plot_paths(context.project_root)

    if plot_paths:
        _section("Featured matched judge figure")
        _render_breast_plot(context.project_root, plot_paths[0])

    st.dataframe(_metric_frame(context.breast_cancer), hide_index=True, width="stretch")

    deltas = context.breast_cancer_verdict.deltas
    _stat_grid(
        (
            ("Sensitivity Δ", f"{deltas['sensitivity']:+.4f}", "Quantum Kernel vs Random Forest"),
            ("False-negative Δ", f"{deltas['false_negatives']:+.0f}", "matched split"),
            ("ROC-AUC Δ", f"{deltas['roc_auc']:+.4f}", "Quantum Kernel vs Random Forest"),
        )
    )

    st.info(
        "The Quantum Kernel shows a selective sensitivity benefit and one fewer false negative versus Random Forest "
        "on this matched split. RBF SVM remains strongest overall by the declared ranking rule."
    )

    with st.expander("Authoritative evidence downloads"):
        download_columns = st.columns(4)
        downloads = (
            (context.project_root / "results/final_results.csv", "Matched CSV", "text/csv"),
            (context.project_root / "results/final_results.json", "Evidence JSON", "application/json"),
            (context.project_root / "results/final_config.json", "Configuration", "application/json"),
            (context.project_root / "SIH_FINAL_SUMMARY.md", "Download SIH final summary", "text/markdown"),
        )
        for column, (path, label, mime) in zip(download_columns, downloads):
            with column:
                _download(path, label, mime)

    if len(plot_paths) > 1:
        with st.expander("Additional authoritative result figures"):
            for plot_path in plot_paths[1:]:
                _render_breast_plot(context.project_root, plot_path)

    with st.expander("Benchmark limitations"):
        for limitation in context.breast_cancer.limitations:
            st.caption(limitation)


def _utility_block(evidence: TaskEvidence, verdict: UtilityVerdict) -> None:
    st.markdown(f"### {_safe(evidence.title)}")
    _verdict_card(evidence.role.replace("_", " ").title(), verdict)

    _stat_grid(
        (
            ("Reference", verdict.reference_model, "comparison baseline"),
            ("Candidate", verdict.candidate_model, "architecture under evaluation"),
            ("Best overall", verdict.best_overall_model, "declared ranking rule"),
        )
    )

    with st.expander("Measured candidate-minus-reference deltas"):
        st.dataframe(
            pd.DataFrame(
                ((metric.replace("_", " ").title(), value) for metric, value in verdict.deltas.items()),
                columns=["Metric", "Delta"],
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander("Why Q-MedAI reached this verdict"):
        for reason in verdict.rationale:
            st.write(f"• {reason}")


def render_quantum_utility(context: DemoContext) -> None:
    _section(
        "Quantum Utility Engine",
        "The platform makes a task-specific architecture recommendation from measured deltas, comparison fairness, "
        "uncertainty status, and backend constraints.",
    )

    _utility_block(context.framingham, context.framingham_verdict)
    st.markdown("<div class='qm-divider'></div>", unsafe_allow_html=True)
    _utility_block(context.breast_cancer, context.breast_cancer_verdict)

    st.warning("No result on this page should be interpreted as a universal quantum-advantage claim.")


def render_provenance(context: DemoContext) -> None:
    _section(
        "Provenance & Safety",
        "Every displayed result is tied to a local evidence source or a checksum-verified frozen artifact set.",
    )

    _stat_grid(
        (
            ("Framingham rows", f"{context.framingham.cohort['rows']:,}", "prospective CHD cohort"),
            ("Breast-cancer train", f"{context.breast_cancer.cohort['train_rows']:,}", "matched benchmark rows"),
            ("Artifact gate", context.artifact_status.code, context.artifact_status.message),
        )
    )

    st.markdown(
        "<div class='qm-card'><div class='qm-card-title'>Evidence sources</div>"
        f"<div class='qm-card-copy'><strong>Framingham:</strong> {_safe(context.framingham.source)}<br>"
        f"<strong>Breast Cancer:</strong> {_safe(context.breast_cancer.source)}</div></div>",
        unsafe_allow_html=True,
    )

    with st.expander("Cohort details"):
        st.write("**Framingham cohort:**", dict(context.framingham.cohort))
        st.write("**Breast Cancer matched cohort:**", dict(context.breast_cancer.cohort))

    if context.artifact_status.ready and context.artifact_status.manifest is not None:
        with st.expander("Verified frozen-artifact manifest"):
            st.json(dict(context.artifact_status.manifest))
    else:
        st.warning("No manifest data is displayed because a complete verified artifact set is not installed.")
        if context.artifact_status.missing_files:
            st.write("**Missing files:**", ", ".join(context.artifact_status.missing_files))

    with st.expander("Full Framingham limitations"):
        for limitation in context.framingham.limitations:
            st.write(f"• {limitation}")

    _section("Safety boundary")
    st.markdown(
        "<div class='qm-card'><div class='qm-card-copy'>"
        "<strong>Research and faculty demonstration only.</strong> Not a medical device; not clinically validated; "
        "no diagnosis, treatment recommendation, calibrated probability, or patient-data persistence. "
        "Arbitrary uploaded pickle/joblib files are unsupported because deserialization can execute code."
        "</div></div>",
        unsafe_allow_html=True,
    )

    files = (
        (context.project_root / "results/framingham/notebook_metrics.json", "Download Framingham metrics"),
        (context.project_root / "results/final_results.json", "Download Breast Cancer evidence"),
    )
    for path, label in files:
        _download(path, label, "application/json")


def render_research_runner() -> None:
    _section(
        "Advanced Research Runner",
        "Optional exploratory training for development use. Faculty patient inference does not require this page.",
    )

    st.warning(
        "This page can take many minutes when VQC or the quantum kernel is enabled. "
        "Use Patient Risk for the fast frozen-artifact demonstration."
    )

    with st.expander("Configure exploratory experiment", expanded=False):
        method = st.radio("Feature reduction", ("SelectKBest", "PCA"), horizontal=True)
        feature_count = st.selectbox("Output features / quantum qubits", SUPPORTED_FEATURE_COUNTS, index=0)

        if method == "PCA":
            st.info("PCA components are transformed combinations of original biomarkers, not individual biomarkers.")

        selected = {name for name in MODEL_NAMES if st.checkbox(name, value=True, key=f"explore_{name}")}

        first, second, third = st.columns(3)
        layers = first.number_input("VQC layers", min_value=1, max_value=3, value=3, step=1)
        iterations = second.number_input("VQC iterations", min_value=1, max_value=100, value=100, step=1)
        subset = third.number_input(
            "Quantum-kernel training subset",
            min_value=2,
            max_value=MAX_KERNEL_SUBSET_SIZE,
            value=150,
            step=1,
        )

        config = ExperimentConfig(
            reduction_method=method,
            n_features=int(feature_count),
            vqc_layers=int(layers),
            vqc_iterations=int(iterations),
            kernel_subset_size=int(subset),
        )

        signature = _signature(config, selected)

        if st.button("Run exploratory experiment", type="primary"):
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


def _render_page_hero(page: str, context: DemoContext) -> None:
    title, copy = PAGE_COPY[page]
    st.markdown(
        "<div class='qm-shell'>"
        "<div class='qm-hero'>"
        "<div class='qm-eyebrow'>Q-MedAI · SIH 26139 · faculty prototype</div>"
        f"<div class='qm-hero-title'>{_safe(title)}</div>"
        f"<div class='qm-hero-copy'>{_safe(copy)}</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


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

    page = st.sidebar.radio("Navigate", PAGES, index=0)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Q-MedAI**")
    st.sidebar.caption("Framingham is the primary prospective workflow.")
    st.sidebar.caption("Breast Cancer is a separate quantum evidence benchmark.")
    st.sidebar.caption("Research prototype · not a medical device")

    st.title("Q-MedAI Clinical Intelligence")
    _render_page_hero(page, context)

    if page == "Research Runner":
        render_research_runner()
        return

    renderers = {
        "Command Center": render_command_center,
        "Patient Risk": render_patient_risk,
        "Quantum Lab": render_quantum_lab,
        "Model Arena": render_model_arena,
        "Breast Cancer Evidence": render_breast_cancer_evidence,
        "Quantum Utility": render_quantum_utility,
        "Provenance & Safety": render_provenance,
    }
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
