import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kaggle.framingham_artifact_export import export_from_namespace
from src.demo.artifacts import load_framingham_artifacts, sha256_file
from src.demo.framingham import (
    FEATURE_MAP_VERSION,
    PATIENT_FIELDS,
    encode_quantum_features,
    feature_state,
    predict_patient,
)


def _notebook_namespace():
    feature_order = [field.name for field in PATIENT_FIELDS]
    X = pd.DataFrame(np.array([
        [0, 35, 1, 0, 0, 0, 0, 0, 0, 150, 110, 70, 20, 60, 70],
        [1, 45, 1, 1, 5, 0, 0, 1, 0, 180, 135, 85, 25, 75, 90],
        [0, 55, 0, 0, 0, 1, 0, 1, 1, 220, 150, 95, 30, 85, 110],
        [1, 65, 1, 1, 20, 1, 1, 1, 1, 260, 165, 100, 35, 95, 130],
    ], dtype=float), columns=feature_order)
    y = np.array([0, 1, 1, 0])
    preprocess = Pipeline([("scaler", StandardScaler())]).fit(X)
    transformed = preprocess.transform(X)
    classical = LogisticRegression().fit(transformed, y)
    c_final = SVC(probability=True).fit(transformed, y)
    indices = [1, 10, 7, 11]
    states = np.asarray([feature_state(values) for values in encode_quantum_features(transformed, indices)])
    kernel = np.abs(states @ states.conj().T) ** 2
    qsvc = SVC(kernel="precomputed", probability=True).fit(kernel, y)
    meta = LogisticRegression().fit(np.c_[classical.predict_proba(transformed)[:, 1], qsvc.predict_proba(kernel)[:, 1]], y)
    return {
        "preprocess": preprocess, "classical_models": {"Logistic Regression": classical},
        "feature_names": np.array(feature_order),
        "selected": ["age", "sysBP", "prevalentHyp", "diaBP"], "idx": [1, 10, 7, 11],
        "S_train": states, "qsvc": qsvc, "c_final": c_final, "q_final": qsvc,
        "Sh": states, "meta": meta, "summary": pd.DataFrame([{"Model": "Logistic Regression", "AUROC": 0.7}]),
        "SEED": 42, "X_train": X, "X_test": X.iloc[:1], "TARGET": "TenYearCHD",
    }


@pytest.mark.filterwarnings(
    "ignore:The `probability` parameter was deprecated in 1\\.9 and will be removed in version 1\\.11.*:FutureWarning:sklearn\\.svm\\._base",
)
@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated in NumPy 2\\.5\\..*:DeprecationWarning:joblib\\.numpy_pickle",
)
def test_exporter_creates_loader_compatible_hashed_artifacts(tmp_path):
    artifact_dir = export_from_namespace(_notebook_namespace(), tmp_path)

    loaded = load_framingham_artifacts(artifact_dir)
    result = predict_patient(loaded, {field.name: field.default for field in PATIENT_FIELDS})

    assert loaded.manifest["schema_version"] == 1
    assert loaded.manifest["calibration"]["status"] == "NOT_DEMONSTRATED"
    assert loaded.manifest["artifacts"] == {
        "classical": "classical_pipeline.joblib",
        "quantum": "quantum_bundle.joblib",
        "hybrid": "hybrid_bundle.joblib",
        "metrics": "frozen_notebook_metrics.json",
    }
    assert (tmp_path / "qmedai_framingham_artifacts.zip").is_file()
    assert 0.0 <= result.classical_score <= 1.0
    assert 0.0 <= result.quantum_score <= 1.0
    assert 0.0 <= result.hybrid_score <= 1.0
    for name, digest in loaded.manifest["checksums_sha256"].items():
        assert sha256_file(artifact_dir / name) == digest


@pytest.mark.filterwarnings(
    "ignore:The `probability` parameter was deprecated in 1\\.9 and will be removed in version 1\\.11.*:FutureWarning:sklearn\\.svm\\._base",
)
@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated in NumPy 2\\.5\\..*:DeprecationWarning:joblib\\.numpy_pickle",
)
def test_exporter_embeds_strict_inference_metadata_and_content_version(tmp_path):
    artifact_dir = export_from_namespace(_notebook_namespace(), tmp_path)
    manifest = json.loads((artifact_dir / "model_manifest.json").read_text(encoding="utf-8"))
    feature_order = [field.name for field in PATIENT_FIELDS]

    assert manifest["feature_order"] == feature_order
    assert manifest["quantum"]["feature_map"]["version"] == FEATURE_MAP_VERSION
    assert manifest["backend"] == "PennyLane default.qubit (classical quantum-circuit simulator)"
    assert manifest["model_version_id"].startswith("qmedai-framingham-")
    for name in ("classical_pipeline.joblib", "quantum_bundle.joblib", "hybrid_bundle.joblib"):
        bundle = joblib.load(artifact_dir / name)
        assert bundle["model_version_id"] == manifest["model_version_id"]
        assert bundle["feature_order"] == feature_order
    for name in ("quantum_bundle.joblib", "hybrid_bundle.joblib"):
        bundle = joblib.load(artifact_dir / name)
        assert bundle["selected_indices"] == [1, 10, 7, 11]
        assert bundle["n_qubits"] == 4
        assert bundle["feature_map_version"] == FEATURE_MAP_VERSION


@pytest.mark.filterwarnings(
    "ignore:The `probability` parameter was deprecated in 1\\.9 and will be removed in version 1\\.11.*:FutureWarning:sklearn\\.svm\\._base",
)
@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated in NumPy 2\\.5\\..*:DeprecationWarning:joblib\\.numpy_pickle",
)
def test_exporter_identity_changes_for_changed_fitted_bundle_content(tmp_path):
    unchanged = _notebook_namespace()
    changed = _notebook_namespace()
    changed["classical_models"]["Logistic Regression"].coef_[0, 0] += 0.25

    first = export_from_namespace(unchanged, tmp_path / "first")
    repeated = export_from_namespace(unchanged, tmp_path / "repeated")
    second = export_from_namespace(changed, tmp_path / "second")

    first_manifest = json.loads((first / "model_manifest.json").read_text(encoding="utf-8"))
    repeated_manifest = json.loads((repeated / "model_manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "model_manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["model_version_id"] == repeated_manifest["model_version_id"]
    assert first_manifest["model_version_id"] != second_manifest["model_version_id"]


@pytest.mark.filterwarnings(
    "ignore:The `probability` parameter was deprecated in 1\\.9 and will be removed in version 1\\.11.*:FutureWarning:sklearn\\.svm\\._base",
)
@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated in NumPy 2\\.5\\..*:DeprecationWarning:joblib\\.numpy_pickle",
)
def test_exported_named_dataframe_preprocessors_round_trip_without_feature_name_warning(tmp_path):
    artifacts = load_framingham_artifacts(export_from_namespace(_notebook_namespace(), tmp_path))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = predict_patient(artifacts, {field.name: field.default for field in PATIENT_FIELDS})

    assert 0.0 <= result.classical_score <= 1.0
    assert 0.0 <= result.quantum_score <= 1.0
    assert 0.0 <= result.hybrid_score <= 1.0
