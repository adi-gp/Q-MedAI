import numpy as np
import pytest

from src.demo.contracts import FraminghamArtifacts, PatientValidationError
from src.demo.framingham import (
    PATIENT_FIELDS,
    encode_quantum_features,
    feature_state,
    fidelity_kernel_row,
    predict_patient,
    validate_patient,
)


class IdentityPreprocess:
    def transform(self, values):
        return np.asarray(values, dtype=float)


class FixedModel:
    def __init__(self, positive):
        self.positive = positive

    def predict_proba(self, values):
        return np.tile([1.0 - self.positive, self.positive], (len(values), 1))


class FixedMeta:
    def predict_proba(self, values):
        score = np.clip(np.asarray(values).mean(axis=1), 0, 1)
        return np.c_[1.0 - score, score]


class NeverScore:
    def predict_proba(self, values):
        pytest.fail("semantic bundle validation must occur before scoring")


class CoefficientModel(FixedModel):
    def __init__(self, positive, coefficients):
        super().__init__(positive)
        self.coef_ = coefficients


def _values():
    return {field.name: field.default for field in PATIENT_FIELDS}


def _artifacts(*, manifest=None, quantum_states=None, hybrid_states=None):
    names = [field.name for field in PATIENT_FIELDS]
    encoded = encode_quantum_features(validate_patient(_values(), names), [0, 1, 2, 3])[0]
    state = feature_state(encoded)
    version = "test-v1"
    manifest = manifest or {
        "model_version_id": version,
        "feature_order": names,
        "quantum": {
            "selected_indices": [0, 1, 2, 3],
            "n_qubits": 4,
            "feature_map": {"version": "framingham_v1_h_rz_cz_reupload"},
        },
        "calibration": {"status": "NOT_DEMONSTRATED"},
    }
    return FraminghamArtifacts(
        manifest=manifest,
        classical={"model_version_id": version, "feature_order": names, "preprocess": IdentityPreprocess(), "model": FixedModel(0.2)},
        quantum={"model_version_id": version, "feature_order": names, "selected_indices": [0, 1, 2, 3], "n_qubits": 4, "feature_map_version": "framingham_v1_h_rz_cz_reupload", "preprocess": IdentityPreprocess(), "train_states": np.asarray([state]) if quantum_states is None else quantum_states, "qsvc": FixedModel(0.4)},
        hybrid={"model_version_id": version, "feature_order": names, "selected_indices": [0, 1, 2, 3], "n_qubits": 4, "feature_map_version": "framingham_v1_h_rz_cz_reupload", "preprocess": IdentityPreprocess(), "classical_base_model": FixedModel(0.3), "train_states": np.asarray([state]) if hybrid_states is None else hybrid_states, "qsvc": FixedModel(0.5), "meta_model": FixedMeta()},
    )


def test_patient_order_and_range_validation():
    names = [field.name for field in PATIENT_FIELDS]
    row = validate_patient(_values(), names)
    assert row.shape == (1, 15)
    bad = _values()
    bad["sysBP"] = 999
    with pytest.raises(PatientValidationError, match="sysBP"):
        validate_patient(bad, names)


def test_quantum_angle_encoding_matches_notebook():
    transformed = np.array([[-4.0, -1.0, 1.0, 4.0]])
    encoded = encode_quantum_features(transformed, [0, 1, 2, 3])
    np.testing.assert_allclose(encoded, [[-np.pi, -np.pi / 3, np.pi / 3, np.pi]])


def test_state_and_fidelity_kernel_are_normalized():
    encoded = np.array([0.1, -0.2, 0.3, -0.4])
    state = feature_state(encoded)
    assert np.vdot(state, state).real == pytest.approx(1.0)
    kernel = fidelity_kernel_row(state, np.asarray([state]))
    assert kernel.shape == (1, 1)
    assert kernel[0, 0] == pytest.approx(1.0)


def test_prediction_composes_real_bundle_interfaces():
    result = predict_patient(_artifacts(), _values())
    assert result.classical_score == pytest.approx(0.2)
    assert result.quantum_score == pytest.approx(0.4)
    assert result.hybrid_score == pytest.approx(0.4)
    assert result.recommended_pathway == "CLASSICAL-PREFERRED"


def test_prediction_rejects_unknown_feature_map_version_before_scoring():
    artifacts = _artifacts()
    manifest = {**artifacts.manifest, "quantum": {**artifacts.manifest["quantum"], "feature_map": {"version": "unverified"}}}
    with pytest.raises(PatientValidationError, match="feature-map"):
        predict_patient(_artifacts(manifest=manifest), _values())


def test_prediction_rejects_manifest_qubit_count_mismatch_before_scoring():
    artifacts = _artifacts()
    manifest = {**artifacts.manifest, "quantum": {**artifacts.manifest["quantum"], "n_qubits": 3}}
    with pytest.raises(PatientValidationError, match="n_qubits"):
        predict_patient(_artifacts(manifest=manifest), _values())


def test_prediction_translates_quantum_state_dimension_mismatch_to_domain_error():
    with pytest.raises(PatientValidationError, match="dimensions"):
        predict_patient(_artifacts(quantum_states=np.ones((1, 3))), _values())


@pytest.mark.parametrize(
    ("role", "field", "value"),
    [
        ("classical", "feature_order", ["age"]),
        ("quantum", "feature_order", ["age"]),
        ("hybrid", "feature_order", ["age"]),
        ("quantum", "selected_indices", [3, 2, 1, 0]),
        ("hybrid", "selected_indices", [3, 2, 1, 0]),
        ("quantum", "n_qubits", 3),
        ("hybrid", "n_qubits", 3),
        ("quantum", "feature_map_version", "other-map"),
        ("hybrid", "feature_map_version", "other-map"),
    ],
)
def test_prediction_rejects_embedded_bundle_semantic_mismatch_before_any_score(role, field, value):
    artifacts = _artifacts()
    getattr(artifacts, role)[field] = value
    artifacts.classical["model"] = NeverScore()
    artifacts.quantum["qsvc"] = NeverScore()
    artifacts.hybrid["classical_base_model"] = NeverScore()
    artifacts.hybrid["qsvc"] = NeverScore()
    artifacts.hybrid["meta_model"] = NeverScore()

    with pytest.raises(PatientValidationError, match=field):
        predict_patient(artifacts, _values())


@pytest.mark.parametrize("coefficients", [["not-a-number"], np.ones(14), np.ones((1, 1, 15))])
def test_prediction_translates_corrupt_contribution_coefficients_to_domain_error(coefficients):
    artifacts = _artifacts()
    artifacts.classical["model"] = CoefficientModel(0.2, coefficients)

    with pytest.raises(PatientValidationError, match="coefficients"):
        predict_patient(artifacts, _values())
