"""Trusted loading for the frozen Framingham demonstration artifacts.

``joblib`` deserializes Python pickle data, so it is intentionally reached only
after the manifest has passed structural, containment, and checksum checks.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.demo.contracts import ArtifactError, ArtifactStatus, FraminghamArtifacts


REQUIRED_FILES = (
    "model_manifest.json",
    "classical_pipeline.joblib",
    "quantum_bundle.joblib",
    "hybrid_bundle.joblib",
    "frozen_notebook_metrics.json",
)
REQUIRED_MANIFEST_KEYS = {
    "schema_version", "model_version_id", "feature_order", "input_schema", "quantum",
    "versions", "calibration", "backend", "artifacts", "checksums_sha256",
}
ARTIFACT_FILENAMES = {
    "classical": "classical_pipeline.joblib",
    "quantum": "quantum_bundle.joblib",
    "hybrid": "hybrid_bundle.joblib",
    "metrics": "frozen_notebook_metrics.json",
}
SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it as code."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"Manifest field {label!r} must be an object.")
    if not all(isinstance(key, str) for key in value):
        raise ArtifactError(f"Manifest field {label!r} must use string keys.")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"Manifest field {label!r} must be a non-empty string.")
    return value


def _validate_manifest(data: Any) -> dict[str, Any]:
    manifest = _require_mapping(data, "manifest")
    missing = sorted(REQUIRED_MANIFEST_KEYS - manifest.keys())
    if missing:
        raise ArtifactError("Manifest is missing: " + ", ".join(missing))

    schema_version = manifest["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ArtifactError("Manifest field 'schema_version' must be an integer.")
    if schema_version != 1:
        raise ArtifactError(f"Unsupported manifest schema_version: {schema_version}")
    _require_string(manifest["model_version_id"], "model_version_id")

    feature_order = manifest["feature_order"]
    if not isinstance(feature_order, list) or not feature_order or not all(isinstance(item, str) and item for item in feature_order):
        raise ArtifactError("Manifest field 'feature_order' must be a non-empty list of strings.")
    _require_mapping(manifest["input_schema"], "input_schema")

    quantum = _require_mapping(manifest["quantum"], "quantum")
    selected_indices = quantum.get("selected_indices")
    if not isinstance(selected_indices, list) or not selected_indices or not all(isinstance(index, int) and not isinstance(index, bool) and index >= 0 for index in selected_indices):
        raise ArtifactError("Manifest field 'quantum.selected_indices' must be a non-empty list of non-negative integers.")
    n_qubits = quantum.get("n_qubits")
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits <= 0:
        raise ArtifactError("Manifest field 'quantum.n_qubits' must be a positive integer.")
    feature_map = _require_mapping(quantum.get("feature_map"), "quantum.feature_map")
    _require_string(feature_map.get("version"), "quantum.feature_map.version")

    versions = _require_mapping(manifest["versions"], "versions")
    if not all(isinstance(version, str) and version for version in versions.values()):
        raise ArtifactError("Manifest field 'versions' must map package names to non-empty version strings.")
    calibration = _require_mapping(manifest["calibration"], "calibration")
    _require_string(calibration.get("status"), "calibration.status")
    _require_string(manifest["backend"], "backend")

    artifacts = _require_mapping(manifest["artifacts"], "artifacts")
    if dict(artifacts) != ARTIFACT_FILENAMES:
        raise ArtifactError(
            "Manifest artifact filenames must stay inside the trusted artifact directory "
            "and equal the canonical role-to-filename mapping."
        )
    checksums = _require_mapping(manifest["checksums_sha256"], "checksums_sha256")
    if set(checksums) != set(ARTIFACT_FILENAMES.values()) or not all(
        isinstance(digest, str) and SHA256_HEX.fullmatch(digest) for digest in checksums.values()
    ):
        raise ArtifactError("Manifest checksums_sha256 must contain a SHA-256 digest for every canonical artifact file.")
    return dict(manifest)


def _manifest(directory: Path) -> dict[str, Any]:
    try:
        data = json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read model manifest: {exc}") from exc
    return _validate_manifest(data)


def _trusted_path(directory: Path, name: str) -> Path:
    """Return a canonical direct-child artifact path, rejecting traversal/symlinks."""
    if not isinstance(name, str) or name not in ARTIFACT_FILENAMES.values():
        raise ArtifactError(f"Artifact {name!r} must stay inside the trusted artifact directory.")
    root = directory.resolve()
    candidate = root / name
    if candidate.parent != root or candidate.is_symlink() or candidate.resolve().parent != root:
        raise ArtifactError(f"Artifact {name!r} must stay inside the trusted artifact directory.")
    if not candidate.is_file():
        raise ArtifactError(f"Required artifact {name!r} is missing from the trusted artifact directory.")
    return candidate


def _check_runtime(manifest: Mapping[str, Any]) -> None:
    current = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for package, expected in manifest["versions"].items():
        installed = current.get(package)
        if installed is not None and installed.split(".")[:2] != expected.split(".")[:2]:
            raise ArtifactError(
                f"Artifact requires incompatible {package} {expected}; runtime has {installed}. "
                "Matching major.minor versions are required before a faculty demonstration can score a patient."
            )


def _verify_artifacts(directory: Path, manifest: Mapping[str, Any]) -> None:
    _check_runtime(manifest)
    for role, name in manifest["artifacts"].items():
        path = _trusted_path(directory, name)
        expected = manifest["checksums_sha256"][name]
        try:
            actual = sha256_file(path)
        except OSError as exc:
            raise ArtifactError(f"Cannot calculate SHA-256 for {name}: {exc}") from exc
        if actual != expected:
            raise ArtifactError(f"SHA-256 verification failed for {name} ({role} artifact).")


def inspect_framingham_artifacts(directory: Path) -> ArtifactStatus:
    """Check whether all frozen artifacts are present and trustworthy, without unpickling."""
    missing = tuple(name for name in REQUIRED_FILES if not (directory / name).is_file())
    if missing:
        return ArtifactStatus(False, "ARTIFACTS_MISSING", "Real Framingham artifacts are not installed.", missing_files=missing)
    try:
        manifest = _manifest(directory)
        _verify_artifacts(directory, manifest)
    except ArtifactError as exc:
        return ArtifactStatus(False, "ARTIFACTS_INVALID", str(exc))
    return ArtifactStatus(True, "READY", "Verified frozen Framingham artifacts are ready.", manifest=manifest)


def load_framingham_artifacts(directory: Path) -> FraminghamArtifacts:
    """Load verified artifact bundles, never deserializing an unchecked file."""
    status = inspect_framingham_artifacts(directory)
    if not status.ready or status.manifest is None:
        raise ArtifactError(status.message)
    manifest = status.manifest
    # Repeat verification immediately before the first pickle load to narrow the
    # gap between inspection and deserialization.
    _verify_artifacts(directory, manifest)
    bundles = {
        role: joblib.load(_trusted_path(directory, manifest["artifacts"][role]))
        for role in ("classical", "quantum", "hybrid")
    }
    if not all(isinstance(bundle, Mapping) for bundle in bundles.values()):
        raise ArtifactError("Each verified model bundle must be a mapping.")
    versions = {bundle.get("model_version_id") for bundle in bundles.values()}
    if versions != {manifest["model_version_id"]}:
        raise ArtifactError("Bundle model_version_id does not match the manifest.")
    return FraminghamArtifacts(manifest, bundles["classical"], bundles["quantum"], bundles["hybrid"])
