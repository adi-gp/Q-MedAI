from pathlib import Path

import numpy as np

from src.demo.artifacts import load_framingham_artifacts
from src.demo.framingham import PATIENT_FIELDS, feature_state
from src.demo.quantum_viz import build_quantum_trace, visualization_feature_state


ROOT = Path(__file__).resolve().parents[1]


def _default_patient() -> dict[str, float]:
    return {field.name: float(field.default) for field in PATIENT_FIELDS}


def test_visualization_circuit_matches_verified_inference_feature_map():
    encoded = np.asarray([0.20, -0.35, 0.0, 0.42], dtype=float)

    expected = feature_state(encoded)
    actual = visualization_feature_state(encoded)

    assert actual.shape == expected.shape == (16,)
    assert np.allclose(actual, expected, atol=1e-10)


def test_real_quantum_trace_uses_verified_four_feature_contract():
    artifacts = load_framingham_artifacts(ROOT / "artifacts/framingham")

    trace = build_quantum_trace(artifacts, _default_patient())

    assert trace.feature_names == ("age", "sysBP", "prevalentHyp", "diaBP")
    assert len(trace.raw_values) == 4
    assert len(trace.transformed_values) == 4
    assert len(trace.encoded_angles) == 4
    assert np.all(np.isfinite(trace.encoded_angles))


def test_real_quantum_trace_kernel_is_finite_and_bounded():
    artifacts = load_framingham_artifacts(ROOT / "artifacts/framingham")

    trace = build_quantum_trace(artifacts, _default_patient())

    similarities = np.asarray(trace.kernel_similarities, dtype=float)

    assert similarities.ndim == 1
    assert len(similarities) > 0
    assert np.all(np.isfinite(similarities))
    assert np.all(similarities >= 0.0)
    assert np.all(similarities <= 1.0 + 1e-12)
    assert 0.0 <= trace.kernel_max <= 1.0 + 1e-12
    assert 0.0 <= trace.kernel_mean <= 1.0 + 1e-12
    assert 0.0 <= trace.kernel_median <= 1.0 + 1e-12


def test_trace_contains_a_real_pennylane_circuit_rendering():
    artifacts = load_framingham_artifacts(ROOT / "artifacts/framingham")

    trace = build_quantum_trace(artifacts, _default_patient())

    assert isinstance(trace.circuit_text, str)
    assert "RY" in trace.circuit_text
    assert "RZ" in trace.circuit_text

    # PennyLane renders CZ graphically as a control dot connected
    # to a Z target rather than printing the literal string "CZ".
    assert "●" in trace.circuit_text
    assert "Z" in trace.circuit_text
    assert "State" in trace.circuit_text
