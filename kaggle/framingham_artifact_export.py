"""Export already-fitted Framingham notebook objects as verified demo artifacts.

The module is deliberately import-safe: importing it only exposes
``export_from_namespace``.  It never trains models and it never exports until
the caller explicitly supplies the fitted notebook namespace.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.demo.framingham import PATIENT_FIELDS


SCHEMA_VERSION = 1
FEATURE_MAP = {
    "version": "framingham_v1_h_rz_cz_reupload",
    "clip": [-3.0, 3.0],
    "angle_scale": "pi/3",
    "gates": ["H", "RY(x)", "RZ(0.5*x)", "ring-CZ", "RY(x^2/pi)"],
    "kernel": "abs(inner_product(patient_state, training_state))**2",
}
INPUT_SCHEMA = {
    field.name: {"minimum": field.minimum, "maximum": field.maximum, "binary": field.binary}
    for field in PATIENT_FIELDS
}
_ARTIFACTS = {
    "classical": "classical_pipeline.joblib",
    "quantum": "quantum_bundle.joblib",
    "hybrid": "hybrid_bundle.joblib",
    "metrics": "frozen_notebook_metrics.json",
}
_REQUIRED_NAMESPACE_VALUES = (
    "preprocess", "classical_models", "feature_names", "selected", "idx", "S_train", "qsvc",
    "c_final", "q_final", "Sh", "meta", "summary", "SEED", "X_train", "X_test", "TARGET",
)


def _require(ns: Mapping[str, Any]) -> None:
    missing = [name for name in _REQUIRED_NAMESPACE_VALUES if name not in ns]
    if missing:
        raise KeyError("Missing required fitted notebook variable(s): " + ", ".join(missing))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize frozen notebook metric value {type(value).__name__}.")


def _jsonable_metrics(summary: Any) -> list[dict[str, Any]]:
    if isinstance(summary, pd.DataFrame):
        return json.loads(summary.to_json(orient="records"))
    if isinstance(summary, list):
        encoded = json.dumps(summary, default=_json_default, sort_keys=True, separators=(",", ":"))
        value = json.loads(encoded)
        if all(isinstance(row, dict) for row in value):
            return value
    return []


def _package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import pennylane as qml

        versions["pennylane"] = qml.__version__
    except ImportError:
        versions["pennylane"] = "not-imported-during-export"
    return versions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _content_hashes(bundles: Mapping[str, Mapping[str, Any]], frozen_metrics: Mapping[str, Any]) -> dict[str, str]:
    """Hash the version-free fitted content used to derive a stable model ID."""
    with tempfile.TemporaryDirectory(prefix="qmedai-framingham-identity-") as temporary_directory:
        root = Path(temporary_directory)
        hashes: dict[str, str] = {}
        for role, bundle in bundles.items():
            path = root / f"{role}.joblib"
            joblib.dump(dict(bundle), path)
            hashes[role] = _sha256_file(path)
    hashes["metrics"] = hashlib.sha256(_canonical_json_bytes(frozen_metrics)).hexdigest()
    return hashes


def _dataset_provenance(ns: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    raw_path = ns.get("DATA_PATH")
    if raw_path is None:
        return None, {"data_path": None, "dataset_sha256": None}
    path = Path(raw_path)
    dataset_hash = _sha256_file(path) if path.is_file() else None
    return dataset_hash, {"data_path": str(path), "dataset_sha256": dataset_hash}


def _notebook_contract(ns: Mapping[str, Any]) -> tuple[list[str], list[str], list[int], int]:
    feature_order = [str(value) for value in list(ns["feature_names"])]
    expected_order = [field.name for field in PATIENT_FIELDS]
    if feature_order != expected_order:
        raise ValueError("feature_names must match the exact 15-field Framingham patient contract.")
    selected = [str(value) for value in list(ns["selected"])]
    selected_indices = [int(value) for value in list(ns["idx"])]
    if not selected_indices or len(selected) != len(selected_indices):
        raise ValueError("Quantum selected feature/index counts must be equal and non-empty.")
    if len(set(selected_indices)) != len(selected_indices) or any(index < 0 or index >= len(feature_order) for index in selected_indices):
        raise ValueError("Quantum selected indices must be unique Framingham feature positions.")
    if selected != [feature_order[index] for index in selected_indices]:
        raise ValueError("Quantum selected features must match selected indices in feature order.")
    return feature_order, selected, selected_indices, len(selected_indices)


def export_from_namespace(ns: Mapping[str, Any], output_dir: Path) -> Path:
    """Serialize fitted notebook objects, write a verified manifest, and zip them.

    No estimator ``fit`` method is called.  The exporter only preserves the
    notebook's already-fitted models and preprocessing state.
    """
    _require(ns)
    feature_order, selected, selected_indices, n_qubits = _notebook_contract(ns)
    classical_models = ns["classical_models"]
    primary_name = "Logistic Regression"
    if not isinstance(classical_models, Mapping) or primary_name not in classical_models:
        raise KeyError("classical_models must contain fitted Logistic Regression")

    output_dir = Path(output_dir)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics = _jsonable_metrics(ns["summary"])
    dataset_sha256, dataset_provenance = _dataset_provenance(ns)
    classical_bundle = {
        "feature_order": list(feature_order),
        "model_name": primary_name,
        "preprocess": ns["preprocess"],
        "model": classical_models[primary_name],
    }
    quantum_bundle = {
        "feature_order": list(feature_order),
        "preprocess": ns["preprocess"],
        "selected_features": list(selected),
        "selected_indices": list(selected_indices),
        "n_qubits": n_qubits,
        "feature_map_version": FEATURE_MAP["version"],
        "train_states": np.asarray(ns["S_train"], dtype=complex),
        "qsvc": ns["qsvc"],
    }
    hybrid_bundle = {
        "feature_order": list(feature_order),
        "preprocess": ns["preprocess"],
        "classical_base_model": ns["c_final"],
        "selected_features": list(selected),
        "selected_indices": list(selected_indices),
        "n_qubits": n_qubits,
        "feature_map_version": FEATURE_MAP["version"],
        "train_states": np.asarray(ns["Sh"], dtype=complex),
        "qsvc": ns["q_final"],
        "meta_model": ns["meta"],
    }
    identity_metrics = {
        "notebook_summary": metrics,
        "dataset_provenance": dataset_provenance,
    }
    identity_hashes = _content_hashes(
        {"classical": classical_bundle, "quantum": quantum_bundle, "hybrid": hybrid_bundle}, identity_metrics,
    )
    identity_inputs = {
        "feature_order": feature_order,
        "selected_indices": selected_indices,
        "seed": int(ns["SEED"]),
        "metrics": metrics,
        "dataset_sha256": dataset_sha256,
        "content_sha256": identity_hashes,
    }
    version_id = "qmedai-framingham-" + hashlib.sha256(_canonical_json_bytes(identity_inputs)).hexdigest()[:12]
    classical_bundle["model_version_id"] = version_id
    quantum_bundle["model_version_id"] = version_id
    hybrid_bundle["model_version_id"] = version_id
    joblib.dump(classical_bundle, artifact_dir / _ARTIFACTS["classical"])
    joblib.dump(quantum_bundle, artifact_dir / _ARTIFACTS["quantum"])
    joblib.dump(hybrid_bundle, artifact_dir / _ARTIFACTS["hybrid"])
    frozen_metrics = {
        "notebook_summary": metrics,
        "exported_model_version_id": version_id,
        "dataset_provenance": dataset_provenance,
        "identity": {"algorithm": "sha256", "inputs": identity_inputs},
    }
    (artifact_dir / _ARTIFACTS["metrics"]).write_text(
        json.dumps(frozen_metrics, indent=2, sort_keys=True, default=_json_default), encoding="utf-8",
    )
    checksums = {name: _sha256_file(artifact_dir / name) for name in _ARTIFACTS.values()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_version_id": version_id,
        "project": "Q-MedAI",
        "dataset": "Framingham Heart Study-compatible cohort loaded by the notebook",
        "dataset_provenance": dataset_provenance,
        "notebook_provenance": {"source": "already-fitted Kaggle notebook namespace", "exporter": __name__},
        "task": "Prospective 10-year coronary heart disease risk prediction from baseline clinical measurements",
        "target": str(ns["TARGET"]),
        "split": {"seed": int(ns["SEED"]), "test_size": float(len(ns["X_test"]) / (len(ns["X_train"]) + len(ns["X_test"])))},
        "feature_order": list(feature_order),
        "input_schema": INPUT_SCHEMA,
        "quantum": {
            "selected_features": list(selected),
            "selected_indices": list(selected_indices),
            "n_qubits": n_qubits,
            "feature_map": FEATURE_MAP,
            "training_reference_rows": int(np.asarray(ns["S_train"]).shape[0]),
        },
        "versions": _package_versions(),
        "evaluation_metrics": metrics,
        "calibration": {"status": "NOT_DEMONSTRATED"},
        "backend": "PennyLane default.qubit (classical quantum-circuit simulator)",
        "artifacts": _ARTIFACTS,
        "checksums_sha256": checksums,
        "provenance_notes": [
            "Artifacts are serialized from already-fitted notebook objects; the exporter does not retrain models.",
            "Primary patient-facing prediction is the frozen Logistic Regression probability.",
            "Quantum and hybrid outputs are research comparisons and are not promoted as validated clinical estimates.",
        ],
    }
    (artifact_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    shutil.make_archive(str(output_dir / "qmedai_framingham_artifacts"), "zip", root_dir=artifact_dir)
    return artifact_dir


if __name__ == "__main__":
    artifact_dir = export_from_namespace(globals(), Path("/kaggle/working/qmedai_framingham_export"))
    print("Q-MedAI FROZEN ARTIFACT EXPORT COMPLETE")
    print("Artifact directory:", artifact_dir)
    print("Download this ZIP after creating it from the artifact directory.")
