"""Patient-input validation and frozen, notebook-parity Framingham inference.

The returned values are research-model scores.  Feature contributions describe
model associations only; they do not establish clinical causality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pennylane as qml

from src.demo.contracts import FraminghamArtifacts, InferenceResult, PatientValidationError


FEATURE_MAP_VERSION = "framingham_v1_h_rz_cz_reupload"


@dataclass(frozen=True)
class PatientField:
    name: str
    label: str
    minimum: float
    maximum: float
    default: float
    step: float
    binary: bool = False


PATIENT_FIELDS = (
    PatientField("sex_male", "Sex: male", 0, 1, 0, 1, True),
    PatientField("age", "Age (years)", 32, 75, 49, 1),
    PatientField("education", "Education indicator", 0, 1, 1, 1, True),
    PatientField("currentSmoker", "Current smoker", 0, 1, 0, 1, True),
    PatientField("cigsPerDay", "Cigarettes per day", 0, 70, 0, 1),
    PatientField("BPMeds", "Blood-pressure medication", 0, 1, 0, 1, True),
    PatientField("prevalentStroke", "Previous stroke", 0, 1, 0, 1, True),
    PatientField("prevalentHyp", "Prevalent hypertension", 0, 1, 0, 1, True),
    PatientField("diabetes", "Diabetes", 0, 1, 0, 1, True),
    PatientField("totChol", "Total cholesterol (mg/dL)", 100, 600, 220, 1),
    PatientField("sysBP", "Systolic BP (mmHg)", 80, 250, 125, 1),
    PatientField("diaBP", "Diastolic BP (mmHg)", 40, 150, 80, 1),
    PatientField("BMI", "BMI (kg/m²)", 12, 60, 25, 0.1),
    PatientField("heartRate", "Heart rate (bpm)", 35, 220, 75, 1),
    PatientField("glucose", "Glucose (mg/dL)", 40, 400, 80, 1),
)

_FIELD_NAMES = tuple(field.name for field in PATIENT_FIELDS)


def validate_patient(values: Mapping[str, float], feature_order: list[str]) -> np.ndarray:
    """Return one strictly contract-ordered patient row without persisting it."""
    if list(feature_order) != list(_FIELD_NAMES):
        raise PatientValidationError("Artifact feature order does not match the exact 15-field Framingham contract.")
    fields = {field.name: field for field in PATIENT_FIELDS}
    row: list[float] = []
    for name in feature_order:
        field = fields[name]
        try:
            value = float(values[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise PatientValidationError(f"{name} must be a numeric value.") from exc
        if not np.isfinite(value) or not field.minimum <= value <= field.maximum:
            raise PatientValidationError(f"{name} must be between {field.minimum:g} and {field.maximum:g}.")
        if field.binary and value not in (0.0, 1.0):
            raise PatientValidationError(f"{name} must be 0 or 1.")
        row.append(value)
    return np.asarray([row], dtype=float)


def encode_quantum_features(transformed: np.ndarray, indices: list[int]) -> np.ndarray:
    """Clip selected transformed values to [-3, 3] and scale them by pi/3."""
    values = np.asarray(transformed, dtype=float)
    selected = np.asarray(indices, dtype=int)
    if values.ndim != 2:
        raise PatientValidationError("Quantum preprocessing must return a two-dimensional feature matrix.")
    if not selected.size or np.any(selected < 0) or np.any(selected >= values.shape[1]):
        raise PatientValidationError("Quantum selected feature indices do not match transformed feature dimensions.")
    return np.clip(values[:, selected], -3.0, 3.0) * (np.pi / 3.0)


def feature_state(encoded: np.ndarray) -> np.ndarray:
    """Recreate H -> RY(x) -> RZ(.5x) -> ring-CZ -> RY(x²/pi)."""
    values = np.asarray(encoded, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise PatientValidationError("Quantum feature-map input must be a finite one-dimensional vector with at least two qubits.")
    try:
        device = qml.device("default.qubit", wires=len(values), shots=None)

        @qml.qnode(device)
        def circuit():
            for wire, value in enumerate(values):
                qml.Hadamard(wires=wire)
                qml.RY(value, wires=wire)
                qml.RZ(0.5 * value, wires=wire)
            for wire in range(len(values)):
                qml.CZ(wires=[wire, (wire + 1) % len(values)])
            for wire, value in enumerate(values):
                qml.RY(value**2 / np.pi, wires=wire)
            return qml.state()

        return np.asarray(circuit(), dtype=complex)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PatientValidationError(f"Unable to execute the notebook-parity quantum feature map: {exc}") from exc


def fidelity_kernel_row(patient_state: np.ndarray, train_states: np.ndarray) -> np.ndarray:
    """Return |<patient_state|training_state>|² for each frozen reference state."""
    states = np.asarray(train_states, dtype=complex)
    state = np.asarray(patient_state, dtype=complex)
    if state.ndim != 1 or states.ndim != 2 or states.shape[1] != state.shape[0]:
        raise PatientValidationError("Patient and training quantum-state dimensions do not match.")
    if not len(states) or not np.all(np.isfinite(states)) or not np.all(np.isfinite(state)):
        raise PatientValidationError("Quantum states must be finite and include at least one training reference state.")
    return (np.abs(state[None, :] @ states.conj().T) ** 2).astype(float)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PatientValidationError(f"Verified {label} bundle is unavailable.")
    return value


def _manifest_contract(artifacts: FraminghamArtifacts) -> tuple[list[str], list[int], int]:
    manifest = _mapping(artifacts.manifest, "manifest")
    feature_order = manifest.get("feature_order")
    quantum = _mapping(manifest.get("quantum"), "manifest quantum")
    feature_map = _mapping(quantum.get("feature_map"), "manifest feature-map")
    indices = quantum.get("selected_indices")
    n_qubits = quantum.get("n_qubits")
    if feature_map.get("version") != FEATURE_MAP_VERSION:
        raise PatientValidationError("Artifact feature-map version is not the verified Framingham notebook feature map.")
    if not isinstance(feature_order, list) or list(feature_order) != list(_FIELD_NAMES):
        raise PatientValidationError("Artifact feature order does not match the exact 15-field Framingham contract.")
    if (
        not isinstance(indices, list)
        or not indices
        or len(set(indices)) != len(indices)
        or not all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(_FIELD_NAMES) for index in indices)
    ):
        raise PatientValidationError("Artifact quantum selected feature indices are invalid for the Framingham contract.")
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits != len(indices):
        raise PatientValidationError("Artifact n_qubits must exactly match the selected quantum feature count.")
    return list(feature_order), list(indices), n_qubits


def _verified_bundle(artifacts: FraminghamArtifacts, role: str, required: tuple[str, ...]) -> Mapping[str, Any]:
    bundle = _mapping(getattr(artifacts, role), role)
    missing = [name for name in required if name not in bundle]
    if missing:
        raise PatientValidationError(f"Verified {role} bundle is missing required interface(s): {', '.join(missing)}.")
    manifest_version = artifacts.manifest.get("model_version_id")
    if not isinstance(manifest_version, str) or bundle.get("model_version_id") != manifest_version:
        raise PatientValidationError(f"Verified {role} bundle model_version_id does not match the manifest.")
    return bundle


def _transform(bundle: Mapping[str, Any], raw: np.ndarray, role: str) -> np.ndarray:
    try:
        transformed = np.asarray(bundle["preprocess"].transform(raw), dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PatientValidationError(f"Verified {role} preprocessing could not transform the patient input: {exc}") from exc
    if transformed.shape != (1, len(_FIELD_NAMES)) or not np.all(np.isfinite(transformed)):
        raise PatientValidationError(f"Verified {role} preprocessing returned incompatible feature dimensions.")
    return transformed


def _positive(model: Any, values: np.ndarray) -> float:
    try:
        probabilities = np.asarray(model.predict_proba(values), dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PatientValidationError(f"Verified model could not produce a research-model score: {exc}") from exc
    if probabilities.shape != (1, 2) or not np.all(np.isfinite(probabilities)):
        raise PatientValidationError(f"Expected a finite (1, 2) probability matrix, got {probabilities.shape}.")
    score = float(probabilities[0, 1])
    if not 0.0 <= score <= 1.0:
        raise PatientValidationError("Research-model score must be between 0 and 1.")
    return score


def _contributions(bundle: Mapping[str, Any], transformed: np.ndarray, feature_order: list[str]) -> tuple[tuple[str, float], ...]:
    """Return coefficient associations, not causal explanations."""
    model = bundle["model"]
    if not hasattr(model, "coef_"):
        return ()
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    if len(coefficients) != transformed.shape[1]:
        return ()
    rows = [(name, float(value * weight)) for name, value, weight in zip(feature_order, transformed[0], coefficients)]
    return tuple(sorted(rows, key=lambda item: abs(item[1]), reverse=True))


def _kernel(bundle: Mapping[str, Any], transformed: np.ndarray, indices: list[int], n_qubits: int, role: str) -> np.ndarray:
    encoded = encode_quantum_features(transformed, indices)
    if encoded.shape != (1, n_qubits):
        raise PatientValidationError(f"Verified {role} quantum encoding dimensions do not match n_qubits.")
    state = feature_state(encoded[0])
    if state.shape != (2**n_qubits,):
        raise PatientValidationError(f"Verified {role} quantum-state dimensions do not match n_qubits.")
    return fidelity_kernel_row(state, bundle["train_states"])


def predict_patient(artifacts: FraminghamArtifacts, values: Mapping[str, float]) -> InferenceResult:
    """Score one patient with verified frozen bundles; nothing is stored."""
    feature_order, indices, n_qubits = _manifest_contract(artifacts)
    classical = _verified_bundle(artifacts, "classical", ("preprocess", "model"))
    quantum = _verified_bundle(artifacts, "quantum", ("preprocess", "train_states", "qsvc"))
    hybrid = _verified_bundle(artifacts, "hybrid", ("preprocess", "classical_base_model", "train_states", "qsvc", "meta_model"))
    raw = validate_patient(values, feature_order)

    classical_x = _transform(classical, raw, "classical")
    classical_score = _positive(classical["model"], classical_x)

    quantum_x = _transform(quantum, raw, "quantum")
    quantum_score = _positive(quantum["qsvc"], _kernel(quantum, quantum_x, indices, n_qubits, "quantum"))

    hybrid_x = _transform(hybrid, raw, "hybrid")
    classical_base = _positive(hybrid["classical_base_model"], hybrid_x)
    quantum_base = _positive(hybrid["qsvc"], _kernel(hybrid, hybrid_x, indices, n_qubits, "hybrid"))
    hybrid_score = _positive(hybrid["meta_model"], np.asarray([[classical_base, quantum_base]], dtype=float))

    return InferenceResult(
        classical_score=classical_score,
        quantum_score=quantum_score,
        hybrid_score=hybrid_score,
        recommended_pathway="CLASSICAL-PREFERRED",
        contributions=_contributions(classical, classical_x, feature_order),
    )
