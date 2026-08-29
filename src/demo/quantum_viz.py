"""Verified, session-only quantum visualization helpers for Q-MedAI.

This module does not define an alternative inference path. It derives
visualization data from the same verified Framingham artifact contract and
checks circuit parity against the feature-map semantics used by inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pennylane as qml

from src.demo.contracts import FraminghamArtifacts, PatientValidationError
from src.demo.framingham import (
    _manifest_contract,
    _transform,
    _verified_bundle,
    encode_quantum_features,
    feature_state,
    fidelity_kernel_row,
    validate_patient,
)


@dataclass(frozen=True)
class QuantumPatientTrace:
    feature_names: tuple[str, ...]
    raw_values: tuple[float, ...]
    transformed_values: tuple[float, ...]
    encoded_angles: tuple[float, ...]
    kernel_similarities: tuple[float, ...]
    kernel_max: float
    kernel_mean: float
    kernel_median: float
    circuit_text: str


def _visualization_circuit(encoded: np.ndarray):
    values = np.asarray(encoded, dtype=float)

    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise PatientValidationError(
            "Quantum visualization requires a finite one-dimensional encoded feature vector."
        )

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

    return circuit


def visualization_feature_state(encoded: np.ndarray) -> np.ndarray:
    """Return the state produced by the visualization circuit."""
    circuit = _visualization_circuit(encoded)
    return np.asarray(circuit(), dtype=complex)


def _circuit_text(encoded: np.ndarray) -> str:
    circuit = _visualization_circuit(encoded)

    try:
        return str(qml.draw(circuit)())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PatientValidationError(
            f"Unable to render the verified quantum feature-map circuit: {exc}"
        ) from exc


def build_quantum_trace(
    artifacts: FraminghamArtifacts,
    values: Mapping[str, float],
) -> QuantumPatientTrace:
    """Build session-only visualization data from verified frozen artifacts."""

    feature_order, indices, n_qubits, feature_map_version = _manifest_contract(
        artifacts
    )

    quantum = _verified_bundle(
        artifacts,
        "quantum",
        ("preprocess", "train_states", "qsvc"),
        feature_order,
        indices,
        n_qubits,
        feature_map_version,
    )

    raw = validate_patient(values, feature_order)

    transformed = _transform(
        quantum,
        raw,
        feature_order,
        "quantum",
    )

    encoded = encode_quantum_features(transformed, indices)

    if encoded.shape != (1, n_qubits):
        raise PatientValidationError(
            "Quantum visualization encoding does not match the verified qubit count."
        )

    # Kernel computation uses the exact feature-state implementation
    # already used by verified inference.
    patient_state = feature_state(encoded[0])

    similarities = np.asarray(
        fidelity_kernel_row(patient_state, quantum["train_states"]),
        dtype=float,
    ).reshape(-1)

    if (
        len(similarities) == 0
        or not np.all(np.isfinite(similarities))
        or np.any(similarities < -1e-12)
        or np.any(similarities > 1.0 + 1e-12)
    ):
        raise PatientValidationError(
            "Verified fidelity similarities are unavailable or outside the expected [0, 1] range."
        )

    # Make sure the visualization circuit is state-equivalent
    # to the inference circuit before displaying it.
    visualization_state = visualization_feature_state(encoded[0])

    if not np.allclose(
        visualization_state,
        patient_state,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise PatientValidationError(
            "Quantum visualization circuit does not match the verified inference feature map."
        )

    feature_names = tuple(feature_order[index] for index in indices)
    raw_values = tuple(float(raw[0, index]) for index in indices)
    transformed_values = tuple(
        float(transformed[0, index]) for index in indices
    )
    encoded_angles = tuple(float(value) for value in encoded[0])

    return QuantumPatientTrace(
        feature_names=feature_names,
        raw_values=raw_values,
        transformed_values=transformed_values,
        encoded_angles=encoded_angles,
        kernel_similarities=tuple(float(value) for value in similarities),
        kernel_max=float(np.max(similarities)),
        kernel_mean=float(np.mean(similarities)),
        kernel_median=float(np.median(similarities)),
        circuit_text=_circuit_text(encoded[0]),
    )


__all__ = [
    "QuantumPatientTrace",
    "build_quantum_trace",
    "visualization_feature_state",
]
