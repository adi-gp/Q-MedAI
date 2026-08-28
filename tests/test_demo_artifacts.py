import hashlib
import json
from pathlib import Path

import joblib
import pytest

from src.demo.artifacts import inspect_framingham_artifacts, load_framingham_artifacts
from src.demo.contracts import ArtifactError


FILES = ("classical_pipeline.joblib", "quantum_bundle.joblib", "hybrid_bundle.joblib")


def _artifact_dir(tmp_path: Path) -> Path:
    for name in FILES:
        joblib.dump({"model_version_id": "demo-v1"}, tmp_path / name)
    (tmp_path / "frozen_notebook_metrics.json").write_text("{}", encoding="utf-8")
    checksums = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in (*FILES, "frozen_notebook_metrics.json")
    }
    manifest = {
        "schema_version": 1,
        "model_version_id": "demo-v1",
        "feature_order": ["age", "sysBP", "prevalentHyp", "diaBP"],
        "input_schema": {},
        "quantum": {
            "selected_indices": [0, 1, 2, 3], "n_qubits": 4,
            "feature_map": {"version": "framingham_v1_h_rz_cz_reupload", "clip": [-3.0, 3.0], "angle_scale": "pi/3"},
        },
        "versions": {}, "calibration": {"status": "NOT_DEMONSTRATED"},
        "backend": "PennyLane default.qubit (classical simulator)",
        "artifacts": {"classical": FILES[0], "quantum": FILES[1], "hybrid": FILES[2], "metrics": "frozen_notebook_metrics.json"},
        "checksums_sha256": checksums,
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _manifest(directory: Path) -> dict:
    return json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(directory: Path, manifest: dict) -> None:
    (directory / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_missing_artifacts_return_safe_degraded_status(tmp_path):
    status = inspect_framingham_artifacts(tmp_path)
    assert status.ready is False
    assert status.code == "ARTIFACTS_MISSING"
    assert "model_manifest.json" in status.missing_files


def test_checksum_mismatch_is_rejected_before_joblib_load(tmp_path, monkeypatch):
    directory = _artifact_dir(tmp_path)
    (directory / "classical_pipeline.joblib").write_bytes(b"tampered")
    monkeypatch.setattr(joblib, "load", lambda _: pytest.fail("joblib.load must not run"))
    with pytest.raises(ArtifactError, match="SHA-256"):
        load_framingham_artifacts(directory)


def test_valid_artifacts_load_after_integrity_check(tmp_path):
    directory = _artifact_dir(tmp_path)
    loaded = load_framingham_artifacts(directory)
    assert loaded.manifest["model_version_id"] == "demo-v1"
    assert loaded.classical["model_version_id"] == "demo-v1"


def test_manifest_path_traversal_is_rejected(tmp_path):
    directory = _artifact_dir(tmp_path)
    manifest = _manifest(directory)
    manifest["artifacts"]["classical"] = "../outside.joblib"
    _write_manifest(directory, manifest)
    with pytest.raises(ArtifactError, match="inside the trusted artifact directory"):
        load_framingham_artifacts(directory)


def test_declared_incompatible_sklearn_version_is_rejected(tmp_path):
    directory = _artifact_dir(tmp_path)
    manifest = _manifest(directory)
    manifest["versions"]["scikit-learn"] = "0.0.0"
    _write_manifest(directory, manifest)
    with pytest.raises(ArtifactError, match="incompatible scikit-learn"):
        load_framingham_artifacts(directory)


def test_non_utf8_manifest_returns_invalid_status_and_load_error(tmp_path):
    directory = _artifact_dir(tmp_path)
    (directory / "model_manifest.json").write_bytes(b"\xff\xfe")

    status = inspect_framingham_artifacts(directory)

    assert status.ready is False
    assert status.code == "ARTIFACTS_INVALID"
    with pytest.raises(ArtifactError, match="Cannot read model manifest"):
        load_framingham_artifacts(directory)


def test_swapped_artifact_roles_are_rejected_before_joblib_load(tmp_path, monkeypatch):
    directory = _artifact_dir(tmp_path)
    manifest = _manifest(directory)
    manifest["artifacts"]["classical"] = "hybrid_bundle.joblib"
    manifest["artifacts"]["hybrid"] = "classical_pipeline.joblib"
    _write_manifest(directory, manifest)
    monkeypatch.setattr(joblib, "load", lambda _: pytest.fail("joblib.load must not run"))

    status = inspect_framingham_artifacts(directory)

    assert status.ready is False
    assert status.code == "ARTIFACTS_INVALID"
    with pytest.raises(ArtifactError, match="canonical role-to-filename mapping"):
        load_framingham_artifacts(directory)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("artifacts",), []),
        (("artifacts", "classical"), None),
        (("checksums_sha256",), []),
        (("checksums_sha256", "classical_pipeline.joblib"), 42),
        (("versions",), []),
        (("quantum", "selected_indices"), "0,1,2,3"),
    ],
)
def test_malformed_nested_manifest_returns_invalid_status(tmp_path, path, value):
    directory = _artifact_dir(tmp_path)
    manifest = _manifest(directory)
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _write_manifest(directory, manifest)

    status = inspect_framingham_artifacts(directory)

    assert status.ready is False
    assert status.code == "ARTIFACTS_INVALID"
    with pytest.raises(ArtifactError):
        load_framingham_artifacts(directory)
