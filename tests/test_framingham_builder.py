import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.demo.artifacts import inspect_framingham_artifacts, load_framingham_artifacts
from src.demo.framingham import PATIENT_FIELDS, predict_patient


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = ROOT / "artifacts/framingham"

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated in NumPy 2\\.5\\..*:DeprecationWarning:joblib\\.numpy_pickle"
)


def test_builder_module_import_has_no_filesystem_side_effect(tmp_path):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)

    completed = subprocess.run(
        [sys.executable, "-c", "import scripts.build_framingham_artifacts"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_committed_real_artifacts_are_ready_and_score_distinct_patients():
    status = inspect_framingham_artifacts(ARTIFACT_DIRECTORY)
    assert status.ready, status.message
    assert status.code == "READY"

    manifest = status.manifest
    assert manifest is not None
    assert manifest["quantum"]["selected_features"] == ["age", "sysBP", "prevalentHyp", "diaBP"]
    assert manifest["quantum"]["training_reference_rows"] == 500
    assert manifest["calibration"]["status"] == "NOT_DEMONSTRATED"
    assert manifest["backend"] == "PennyLane default.qubit (classical quantum-circuit simulator)"
    provenance = manifest["dataset_provenance"]
    assert provenance["kaggle_slug"] == "captainozlem/framingham-chd-preprocessed-data"
    assert provenance["file"] == "CHD_preprocessed.csv"
    assert len(provenance["dataset_sha256"]) == 64
    assert "data_path" not in provenance

    artifacts = load_framingham_artifacts(ARTIFACT_DIRECTORY)
    typical = {field.name: field.default for field in PATIENT_FIELDS}
    high_risk = dict(typical)
    high_risk.update(
        {
            "sex_male": 1,
            "age": 72,
            "currentSmoker": 1,
            "cigsPerDay": 30,
            "BPMeds": 1,
            "prevalentStroke": 1,
            "prevalentHyp": 1,
            "diabetes": 1,
            "totChol": 340,
            "sysBP": 205,
            "diaBP": 115,
            "BMI": 38,
            "heartRate": 105,
            "glucose": 220,
        }
    )

    typical_result = predict_patient(artifacts, typical)
    high_result = predict_patient(artifacts, high_risk)
    typical_scores = np.asarray(
        [typical_result.classical_score, typical_result.quantum_score, typical_result.hybrid_score]
    )
    high_scores = np.asarray([high_result.classical_score, high_result.quantum_score, high_result.hybrid_score])

    assert np.all(np.isfinite(typical_scores))
    assert np.all(np.isfinite(high_scores))
    assert np.all((0.0 <= typical_scores) & (typical_scores <= 1.0))
    assert np.all((0.0 <= high_scores) & (high_scores <= 1.0))
    assert not np.allclose(typical_scores, high_scores)
    assert len(typical_result.contributions) == len(PATIENT_FIELDS)
    assert len(high_result.contributions) == len(PATIENT_FIELDS)


def test_patient_risk_page_submits_real_frozen_inference():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Patient Risk").run()

    submit = next(button for button in app.button if button.label == "Run verified inference")
    submit.click().run()

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert set(metrics) >= {
        "Classical research-model score",
        "Quantum research score",
        "Hybrid research score",
    }
    assert all(0.0 <= float(metrics[label]) <= 1.0 for label in metrics if "research" in label.lower())
    assert any("CLASSICAL-PREFERRED" in item.value for item in app.success)
    assert app.dataframe


def test_committed_frozen_metrics_match_model_identity():
    manifest = json.loads((ARTIFACT_DIRECTORY / "model_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((ARTIFACT_DIRECTORY / "frozen_notebook_metrics.json").read_text(encoding="utf-8"))

    assert metrics["exported_model_version_id"] == manifest["model_version_id"]
    assert metrics["notebook_summary"]
