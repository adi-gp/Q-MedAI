# Q-MedAI Combined Faculty Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one faculty-facing Streamlit application in which Framingham is the primary prospective CHD workflow and the existing breast-cancer study is the quantum-evidence benchmark.

**Architecture:** Keep `app.py` as a thin Streamlit entry point and put demo contracts, evidence adapters, artifact security, Framingham inference, utility decisions, and UI composition in focused `src/demo/` modules. The application always runs in evidence mode, enables patient inference only after trusted frozen artifacts pass schema and SHA-256 checks, and never invents a probability.

**Tech Stack:** Python 3.12, Streamlit 1.62, NumPy 2.5, Pandas 3.0, scikit-learn 1.9, PennyLane 0.45, Matplotlib 3.11, joblib, pytest 9.1.

**Spec:** `docs/superpowers/specs/2026-08-28-combined-faculty-demo-design.md`

## Global Constraints

- Framingham is the default patient workflow; breast cancer remains a separate evidence benchmark.
- Frozen artifacts are the only source of patient inference. Streamlit must not retrain models.
- Missing, incompatible, or integrity-failing artifacts must disable inference without producing substitute scores.
- Framingham outputs are labelled **research-model score** unless the manifest records demonstrated calibration.
- PennyLane `default.qubit` must always be disclosed as classical quantum-circuit simulation.
- Load `joblib` only from the repository's trusted `artifacts/framingham/` directory and only after SHA-256 verification.
- Do not package test doubles or synthetic trained models as demo artifacts.
- Do not persist patient inputs.
- Preserve the existing breast-cancer result JSON, plots, downloads, caveats, and exploratory runner.
- Do not show Experiment H/I metrics unless a validated originating evidence file is present.

## Planned File Structure

```text
app.py                                      Thin Streamlit entry point
src/demo/__init__.py                       Public demo interfaces
src/demo/contracts.py                      Typed evidence/artifact/result contracts
src/demo/evidence.py                       Framingham JSON and breast-cancer adapters
src/demo/utility.py                        Evidence-derived utility verdicts
src/demo/artifacts.py                      Trusted artifact inspection and loading
src/demo/framingham.py                     Input validation and frozen inference
src/demo/styles.py                         Contained Streamlit CSS
src/demo/pages.py                          Page rendering and navigation
results/framingham/notebook_metrics.json   Supplied notebook's executed evidence
artifacts/framingham/README.md              Artifact installation/security contract
kaggle/framingham_artifact_export.py       Corrected Kaggle exporter
tests/test_demo_evidence.py                 Evidence adapters
tests/test_demo_utility.py                  Utility decisions
tests/test_demo_artifacts.py                Artifact safety and loading
tests/test_demo_framingham.py               Input/kernel/inference parity
tests/test_framingham_exporter.py           Exporter manifest and hashes
tests/test_demo_app.py                      Streamlit degraded-mode smoke test
README.md                                   Combined-demo instructions and story
```

---

### Task 1: Typed Evidence Contracts and Evidence Loaders

**Files:**
- Create: `src/demo/__init__.py`
- Create: `src/demo/contracts.py`
- Create: `src/demo/evidence.py`
- Create: `results/framingham/notebook_metrics.json`
- Create: `tests/test_demo_evidence.py`

**Interfaces:**
- Produces: `MetricRecord`, `TaskEvidence`, `EvidenceError`, `load_framingham_evidence(path: Path) -> TaskEvidence`, and `load_breast_cancer_evidence(path: Path) -> TaskEvidence`.
- Consumes: existing `results/final_results.json` breast-cancer schema.

- [ ] **Step 1: Write failing evidence-loader tests**

```python
# tests/test_demo_evidence.py
from pathlib import Path

import pytest

from src.demo.evidence import load_breast_cancer_evidence, load_framingham_evidence
from src.demo.contracts import EvidenceError


ROOT = Path(__file__).resolve().parents[1]


def test_framingham_evidence_matches_executed_notebook():
    evidence = load_framingham_evidence(ROOT / "results/framingham/notebook_metrics.json")
    by_name = {row.model: row for row in evidence.metrics}
    assert evidence.task_id == "framingham_chd_10y"
    assert evidence.role == "PRIMARY_CLINICAL_WORKFLOW"
    assert evidence.comparison_fairness == "UNMATCHED_RESOURCE_BUDGET"
    assert by_name["Logistic Regression"].roc_auc == pytest.approx(0.7345)
    assert by_name["Hybrid CML + QML"].roc_auc == pytest.approx(0.7062)
    assert by_name["Quantum Kernel SVM"].roc_auc == pytest.approx(0.6581)
    assert evidence.calibration_status == "NOT_DEMONSTRATED"


def test_breast_cancer_adapter_preserves_matched_quantum_result():
    evidence = load_breast_cancer_evidence(ROOT / "results/final_results.json")
    by_name = {row.model: row for row in evidence.metrics}
    assert evidence.task_id == "breast_cancer_wdbc"
    assert evidence.comparison_fairness == "MATCHED_150_ROWS"
    assert by_name["Quantum Kernel SVM"].sensitivity == pytest.approx(0.9047619048)
    assert by_name["Random Forest"].false_negatives == 5
    assert by_name["Quantum Kernel SVM"].false_negatives == 4


def test_invalid_evidence_has_a_domain_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"task_id": "broken"}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="metrics"):
        load_framingham_evidence(path)
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `.venv/bin/python -m pytest tests/test_demo_evidence.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.demo'`.

- [ ] **Step 3: Create the typed contracts**

```python
# src/demo/contracts.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class DemoError(RuntimeError):
    """Base error for the combined faculty demo."""


class EvidenceError(DemoError):
    """Evidence is missing or violates the expected schema."""


class ArtifactError(DemoError):
    """Frozen model artifacts cannot be trusted or used."""


class PatientValidationError(DemoError):
    """Patient input does not satisfy the frozen model contract."""


@dataclass(frozen=True)
class MetricRecord:
    model: str
    family: str
    accuracy: float
    sensitivity: float
    specificity: float
    f1: float
    roc_auc: float
    auprc: float | None = None
    precision: float | None = None
    false_negatives: int | None = None
    training_seconds: float | None = None


@dataclass(frozen=True)
class TaskEvidence:
    task_id: str
    title: str
    role: str
    task_type: str
    source: str
    cohort: Mapping[str, Any]
    comparison_fairness: str
    backend: str
    metrics: tuple[MetricRecord, ...]
    focus_reference_model: str
    focus_candidate_model: str
    prioritized_metric: str
    uncertainty_status: str
    calibration_status: str
    limitations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class UtilityVerdict:
    task_id: str
    status: str
    headline: str
    reference_model: str
    candidate_model: str
    deltas: Mapping[str, float]
    best_overall_model: str
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactStatus:
    ready: bool
    code: str
    message: str
    manifest: Mapping[str, Any] | None = None
    missing_files: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FraminghamArtifacts:
    manifest: Mapping[str, Any]
    classical: Mapping[str, Any]
    quantum: Mapping[str, Any]
    hybrid: Mapping[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    classical_score: float
    quantum_score: float
    hybrid_score: float
    recommended_pathway: str
    contributions: tuple[tuple[str, float], ...]
```

- [ ] **Step 4: Add the executed Framingham evidence record**

Create `results/framingham/notebook_metrics.json` with this exact top-level structure and the six executed model rows from notebook cell 34:

```json
{
  "schema_version": 1,
  "task_id": "framingham_chd_10y",
  "title": "Prospective 10-year CHD risk research",
  "role": "PRIMARY_CLINICAL_WORKFLOW",
  "task_type": "PROSPECTIVE_BINARY_RISK",
  "source": "framingham_with_artifact_export.ipynb; executed Kaggle input CHD_preprocessed.csv",
  "cohort": {"rows": 4133, "train_rows": 3306, "test_rows": 827, "event_rate": 0.151948},
  "comparison_fairness": "UNMATCHED_RESOURCE_BUDGET",
  "backend": "PennyLane default.qubit (classical simulator)",
  "focus_reference_model": "Logistic Regression",
  "focus_candidate_model": "Hybrid CML + QML",
  "prioritized_metric": "roc_auc",
  "uncertainty_status": "PER_MODEL_BOOTSTRAP_ONLY",
  "calibration_status": "NOT_DEMONSTRATED",
  "metrics": [
    {"model": "Logistic Regression", "family": "Classical", "accuracy": 0.6989, "precision": 0.2811, "sensitivity": 0.6270, "specificity": 0.7118, "f1": 0.3882, "roc_auc": 0.7345, "auprc": 0.3603, "training_seconds": 0.0278},
    {"model": "Hybrid CML + QML", "family": "Hybrid", "accuracy": 0.7207, "precision": 0.2667, "sensitivity": 0.4762, "specificity": 0.7646, "f1": 0.3419, "roc_auc": 0.7062, "auprc": 0.2727, "training_seconds": 12.3486},
    {"model": "Random Forest", "family": "Classical", "accuracy": 0.8380, "precision": 0.4167, "sensitivity": 0.1587, "specificity": 0.9601, "f1": 0.2299, "roc_auc": 0.7016, "auprc": 0.3362, "training_seconds": 1.4572},
    {"model": "RBF-SVM", "family": "Classical", "accuracy": 0.8476, "precision": 0.0, "sensitivity": 0.0, "specificity": 1.0, "f1": 0.0, "roc_auc": 0.6886, "auprc": 0.2581, "training_seconds": 2.6535},
    {"model": "HistGradientBoosting", "family": "Classical", "accuracy": 0.8452, "precision": 0.4667, "sensitivity": 0.1111, "specificity": 0.9772, "f1": 0.1795, "roc_auc": 0.6775, "auprc": 0.2903, "training_seconds": 0.4299},
    {"model": "Quantum Kernel SVM", "family": "Quantum", "accuracy": 0.8452, "precision": 0.25, "sensitivity": 0.0079, "specificity": 0.9957, "f1": 0.0154, "roc_auc": 0.6581, "auprc": 0.2535, "training_seconds": 4.4046}
  ],
  "limitations": [
    "Quantum and hybrid models use 500 training rows and four features while full-data classical models use 3306 rows and 15 features.",
    "Mutual-information feature selection was learned before the stacking folds.",
    "The source is a public preprocessed CSV whose original cohort provenance was not independently verified.",
    "Quantum execution used a classical simulator.",
    "No external clinical validation or demonstrated probability calibration is available."
  ]
}
```

- [ ] **Step 5: Implement strict evidence adapters**

```python
# src/demo/evidence.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.demo.contracts import EvidenceError, MetricRecord, TaskEvidence


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Cannot read evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"Evidence {path} must be a JSON object.")
    return payload


def _metric(row: dict[str, Any]) -> MetricRecord:
    required = {"model", "family", "accuracy", "sensitivity", "specificity", "f1", "roc_auc"}
    missing = sorted(required - row.keys())
    if missing:
        raise EvidenceError("Metric row is missing: " + ", ".join(missing))
    return MetricRecord(**{name: row.get(name) for name in MetricRecord.__dataclass_fields__})


def load_framingham_evidence(path: Path) -> TaskEvidence:
    data = _read_json(path)
    if not isinstance(data.get("metrics"), list) or not data["metrics"]:
        raise EvidenceError("Framingham evidence requires a non-empty metrics list.")
    return TaskEvidence(
        task_id=data["task_id"], title=data["title"], role=data["role"],
        task_type=data["task_type"], source=data["source"], cohort=data["cohort"],
        comparison_fairness=data["comparison_fairness"], backend=data["backend"],
        metrics=tuple(_metric(row) for row in data["metrics"]),
        focus_reference_model=data["focus_reference_model"],
        focus_candidate_model=data["focus_candidate_model"],
        prioritized_metric=data["prioritized_metric"],
        uncertainty_status=data["uncertainty_status"],
        calibration_status=data["calibration_status"],
        limitations=tuple(data.get("limitations", [])),
    )


def load_breast_cancer_evidence(path: Path) -> TaskEvidence:
    data = _read_json(path)
    rows = data["matched_data_comparison"]["models"]
    metrics = []
    for row in rows:
        matrix = row.get("confusion_matrix") or [[0, 0], [0, 0]]
        metrics.append(MetricRecord(
            model=row["model"], family=row["type"], accuracy=row["accuracy"],
            precision=row["precision"], sensitivity=row["recall"],
            specificity=row["specificity"], f1=row["f1"], roc_auc=row["roc_auc"],
            false_negatives=int(matrix[1][0]), training_seconds=row["training_time_seconds"],
        ))
    return TaskEvidence(
        task_id="breast_cancer_wdbc", title="Breast cancer diagnostic classification",
        role="QUANTUM_EVIDENCE_BENCHMARK", task_type="CROSS_SECTIONAL_CLASSIFICATION",
        source=str(path), cohort={"train_rows": 150, "test_rows": data["matched_data_comparison"]["test_rows"]},
        comparison_fairness="MATCHED_150_ROWS", backend="PennyLane default.qubit (classical simulator)",
        metrics=tuple(metrics), focus_reference_model="Random Forest",
        focus_candidate_model="Quantum Kernel SVM", prioritized_metric="sensitivity",
        uncertainty_status="NOT_COMPUTED", calibration_status="NOT_APPLICABLE",
        limitations=(data["analysis"]["statistical_confidence"], data["analysis"]["interpretation"]),
    )
```

- [ ] **Step 6: Export the public demo interfaces**

```python
# src/demo/__init__.py
from src.demo.evidence import load_breast_cancer_evidence, load_framingham_evidence

__all__ = ["load_breast_cancer_evidence", "load_framingham_evidence"]
```

- [ ] **Step 7: Run the evidence tests**

Run: `.venv/bin/python -m pytest tests/test_demo_evidence.py -q`

Expected: `3 passed`.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/demo results/framingham/notebook_metrics.json tests/test_demo_evidence.py
git commit -m "feat: add traceable disease evidence contracts"
```

---

### Task 2: Quantum Utility Engine

**Files:**
- Create: `src/demo/utility.py`
- Create: `tests/test_demo_utility.py`
- Modify: `src/demo/__init__.py`

**Interfaces:**
- Consumes: `TaskEvidence` and `MetricRecord` from Task 1.
- Produces: `evaluate_utility(evidence: TaskEvidence) -> UtilityVerdict` and `evidence_table(evidence: TaskEvidence) -> pandas.DataFrame`.

- [ ] **Step 1: Write failing verdict tests**

```python
# tests/test_demo_utility.py
from pathlib import Path

import pytest

from src.demo.evidence import load_breast_cancer_evidence, load_framingham_evidence
from src.demo.utility import evaluate_utility


ROOT = Path(__file__).resolve().parents[1]


def test_framingham_is_classical_preferred():
    evidence = load_framingham_evidence(ROOT / "results/framingham/notebook_metrics.json")
    verdict = evaluate_utility(evidence)
    assert verdict.status == "CLASSICAL-PREFERRED"
    assert verdict.best_overall_model == "Logistic Regression"
    assert verdict.deltas["roc_auc"] == pytest.approx(-0.0283)
    assert "unmatched" in " ".join(verdict.rationale).lower()


def test_breast_cancer_reports_selective_quantum_benefit():
    evidence = load_breast_cancer_evidence(ROOT / "results/final_results.json")
    verdict = evaluate_utility(evidence)
    assert verdict.status == "QUANTUM-COMPETITIVE"
    assert verdict.best_overall_model == "RBF SVM"
    assert verdict.deltas["sensitivity"] == pytest.approx(0.0238095238)
    assert verdict.deltas["false_negatives"] == pytest.approx(-1.0)
```

- [ ] **Step 2: Run tests and verify the missing-function failure**

Run: `.venv/bin/python -m pytest tests/test_demo_utility.py -q`

Expected: import or assertion failure because `evaluate_utility` is absent or incomplete.

- [ ] **Step 3: Implement deterministic utility decisions**

```python
# src/demo/utility.py
from __future__ import annotations

import pandas as pd

from src.demo.contracts import EvidenceError, MetricRecord, TaskEvidence, UtilityVerdict


def _models(evidence: TaskEvidence) -> dict[str, MetricRecord]:
    return {row.model: row for row in evidence.metrics}


def _delta(candidate: MetricRecord, reference: MetricRecord, field: str) -> float:
    left = getattr(candidate, field)
    right = getattr(reference, field)
    if left is None or right is None:
        raise EvidenceError(f"Cannot compare missing metric {field}.")
    return float(left - right)


def evaluate_utility(evidence: TaskEvidence) -> UtilityVerdict:
    rows = _models(evidence)
    try:
        reference = rows[evidence.focus_reference_model]
        candidate = rows[evidence.focus_candidate_model]
    except KeyError as exc:
        raise EvidenceError(f"Utility comparison model is missing: {exc.args[0]}") from exc
    best = max(evidence.metrics, key=lambda row: row.roc_auc)
    fields = ("accuracy", "sensitivity", "specificity", "f1", "roc_auc")
    deltas = {field: _delta(candidate, reference, field) for field in fields}
    if candidate.false_negatives is not None and reference.false_negatives is not None:
        deltas["false_negatives"] = float(candidate.false_negatives - reference.false_negatives)

    if deltas["roc_auc"] < -0.01:
        status = "CLASSICAL-PREFERRED"
        headline = f"{reference.model} remains the evidence-supported pathway."
    elif (
        evidence.comparison_fairness.startswith("MATCHED")
        and deltas.get(evidence.prioritized_metric, 0.0) > 0
        and deltas["roc_auc"] >= -0.01
    ):
        status = "QUANTUM-COMPETITIVE" if candidate.family == "Quantum" else "HYBRID-CANDIDATE"
        headline = f"{candidate.model} shows a selective {evidence.prioritized_metric} benefit."
    else:
        status = "INSUFFICIENT-EVIDENCE"
        headline = "No task-specific quantum utility conclusion is established."

    rationale = [
        f"Comparison fairness: {evidence.comparison_fairness.replace('_', ' ').lower()}.",
        f"Best overall AUROC model: {best.model} ({best.roc_auc:.4f}).",
        f"Backend: {evidence.backend}.",
        f"Uncertainty: {evidence.uncertainty_status.replace('_', ' ').lower()}.",
    ]
    return UtilityVerdict(
        task_id=evidence.task_id, status=status, headline=headline,
        reference_model=reference.model, candidate_model=candidate.model,
        deltas=deltas, best_overall_model=best.model, rationale=tuple(rationale),
    )


def evidence_table(evidence: TaskEvidence) -> pd.DataFrame:
    return pd.DataFrame([row.__dict__ for row in evidence.metrics])
```

- [ ] **Step 4: Run utility and evidence tests**

Run: `.venv/bin/python -m pytest tests/test_demo_evidence.py tests/test_demo_utility.py -q`

Expected: `5 passed`.

Update `src/demo/__init__.py` in this task to import and export `evaluate_utility` alongside the two evidence loaders.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/demo/utility.py src/demo/__init__.py tests/test_demo_utility.py
git commit -m "feat: add evidence-derived quantum utility engine"
```

---

### Task 3: Trusted Frozen-Artifact Loader

**Files:**
- Create: `src/demo/artifacts.py`
- Create: `artifacts/framingham/README.md`
- Create: `tests/test_demo_artifacts.py`

**Interfaces:**
- Consumes: `ArtifactError`, `ArtifactStatus`, and `FraminghamArtifacts` from Task 1.
- Produces: `inspect_framingham_artifacts(directory: Path) -> ArtifactStatus`, `load_framingham_artifacts(directory: Path) -> FraminghamArtifacts`, and `sha256_file(path: Path) -> str`.

- [ ] **Step 1: Write failing artifact-safety tests**

```python
# tests/test_demo_artifacts.py
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
            "feature_map": {"version": "framingham_v1_h_rz_cz_reupload", "clip": [-3.0, 3.0], "angle_scale": "pi/3"}
        },
        "versions": {}, "calibration": {"status": "NOT_DEMONSTRATED"},
        "backend": "PennyLane default.qubit (classical simulator)",
        "artifacts": {"classical": FILES[0], "quantum": FILES[1], "hybrid": FILES[2], "metrics": "frozen_notebook_metrics.json"},
        "checksums_sha256": checksums
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_missing_artifacts_return_safe_degraded_status(tmp_path):
    status = inspect_framingham_artifacts(tmp_path)
    assert status.ready is False
    assert status.code == "ARTIFACTS_MISSING"
    assert "model_manifest.json" in status.missing_files


def test_checksum_mismatch_is_rejected_before_joblib_load(tmp_path):
    directory = _artifact_dir(tmp_path)
    (directory / "classical_pipeline.joblib").write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="SHA-256"):
        load_framingham_artifacts(directory)


def test_valid_artifacts_load_after_integrity_check(tmp_path):
    directory = _artifact_dir(tmp_path)
    loaded = load_framingham_artifacts(directory)
    assert loaded.manifest["model_version_id"] == "demo-v1"
    assert loaded.classical["model_version_id"] == "demo-v1"


def test_manifest_path_traversal_is_rejected(tmp_path):
    directory = _artifact_dir(tmp_path)
    manifest_path = directory / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["classical"] = "../outside.joblib"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="inside the trusted artifact directory"):
        load_framingham_artifacts(directory)


def test_declared_incompatible_sklearn_version_is_rejected(tmp_path):
    directory = _artifact_dir(tmp_path)
    manifest_path = directory / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["versions"]["scikit-learn"] = "0.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="incompatible scikit-learn"):
        load_framingham_artifacts(directory)
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `.venv/bin/python -m pytest tests/test_demo_artifacts.py -q`

Expected: collection fails because `src.demo.artifacts` does not exist.

- [ ] **Step 3: Implement inspection, schema validation, path containment, and checksums**

```python
# src/demo/artifacts.py
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.demo.contracts import ArtifactError, ArtifactStatus, FraminghamArtifacts


REQUIRED_FILES = (
    "model_manifest.json", "classical_pipeline.joblib", "quantum_bundle.joblib",
    "hybrid_bundle.joblib", "frozen_notebook_metrics.json",
)
REQUIRED_MANIFEST_KEYS = {
    "schema_version", "model_version_id", "feature_order", "input_schema", "quantum",
    "versions", "calibration", "backend", "artifacts", "checksums_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(directory: Path) -> dict[str, Any]:
    try:
        data = json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read model manifest: {exc}") from exc
    missing = sorted(REQUIRED_MANIFEST_KEYS - data.keys())
    if missing:
        raise ArtifactError("Manifest is missing: " + ", ".join(missing))
    if data["schema_version"] != 1:
        raise ArtifactError(f"Unsupported manifest schema_version: {data['schema_version']}")
    return data


def _trusted_path(directory: Path, name: str) -> Path:
    root = directory.resolve()
    path = (root / name).resolve()
    if path.parent != root:
        raise ArtifactError(f"Artifact {name!r} must stay inside the trusted artifact directory.")
    return path


def _check_runtime(manifest: dict[str, Any]) -> None:
    current = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
    }
    for package, expected in manifest["versions"].items():
        if package in current and current[package].split(".")[:2] != str(expected).split(".")[:2]:
            raise ArtifactError(
                f"Artifact requires incompatible {package} {expected}; runtime has {current[package]}."
            )


def inspect_framingham_artifacts(directory: Path) -> ArtifactStatus:
    missing = tuple(name for name in REQUIRED_FILES if not (directory / name).is_file())
    if missing:
        return ArtifactStatus(False, "ARTIFACTS_MISSING", "Real Framingham artifacts are not installed.", missing_files=missing)
    try:
        manifest = _manifest(directory)
        _check_runtime(manifest)
        for role, name in manifest["artifacts"].items():
            path = _trusted_path(directory, name)
            expected = manifest["checksums_sha256"].get(name)
            if expected is None or sha256_file(path) != expected:
                raise ArtifactError(f"SHA-256 verification failed for {name}.")
    except ArtifactError as exc:
        return ArtifactStatus(False, "ARTIFACTS_INVALID", str(exc))
    return ArtifactStatus(True, "READY", "Verified frozen Framingham artifacts are ready.", manifest=manifest)


def load_framingham_artifacts(directory: Path) -> FraminghamArtifacts:
    status = inspect_framingham_artifacts(directory)
    if not status.ready or status.manifest is None:
        raise ArtifactError(status.message)
    manifest = status.manifest
    bundles = {
        role: joblib.load(_trusted_path(directory, manifest["artifacts"][role]))
        for role in ("classical", "quantum", "hybrid")
    }
    versions = {bundle.get("model_version_id") for bundle in bundles.values()}
    if versions != {manifest["model_version_id"]}:
        raise ArtifactError("Bundle model_version_id does not match the manifest.")
    return FraminghamArtifacts(manifest, bundles["classical"], bundles["quantum"], bundles["hybrid"])
```

- [ ] **Step 4: Write the trusted-install README**

Document these exact rules in `artifacts/framingham/README.md`:

```markdown
# Framingham frozen artifacts

Place only artifacts produced by `kaggle/framingham_artifact_export.py` here. Q-MedAI loads local joblib files only after manifest and SHA-256 verification. Joblib uses Python pickle internally; never place downloaded or untrusted joblib files in this directory.

Required files: `model_manifest.json`, `classical_pipeline.joblib`, `quantum_bundle.joblib`, `hybrid_bundle.joblib`, and `frozen_notebook_metrics.json`.

Without all five verified files, the faculty demo remains in safe evidence mode and produces no patient score.
```

- [ ] **Step 5: Run artifact tests**

Run: `.venv/bin/python -m pytest tests/test_demo_artifacts.py -q`

Expected: `5 passed`.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/demo/artifacts.py artifacts/framingham/README.md tests/test_demo_artifacts.py
git commit -m "feat: enforce trusted frozen artifact loading"
```

---

### Task 4: Framingham Patient Contract and Notebook-Parity Inference

**Files:**
- Create: `src/demo/framingham.py`
- Create: `tests/test_demo_framingham.py`

**Interfaces:**
- Consumes: `FraminghamArtifacts`, `InferenceResult`, and `PatientValidationError` from Task 1.
- Produces: `PATIENT_FIELDS`, `validate_patient(values, feature_order) -> numpy.ndarray`, `encode_quantum_features(transformed, indices) -> numpy.ndarray`, `feature_state(encoded) -> numpy.ndarray`, `fidelity_kernel_row(patient_state, train_states) -> numpy.ndarray`, and `predict_patient(artifacts, values) -> InferenceResult`.

- [ ] **Step 1: Write failing validation, kernel, and inference tests**

```python
# tests/test_demo_framingham.py
import numpy as np
import pytest

from src.demo.contracts import FraminghamArtifacts, PatientValidationError
from src.demo.framingham import (
    PATIENT_FIELDS, encode_quantum_features, fidelity_kernel_row,
    feature_state, predict_patient, validate_patient,
)


class IdentityPreprocess:
    def transform(self, values):
        return np.asarray(values, dtype=float)


class FixedModel:
    def __init__(self, positive): self.positive = positive
    def predict_proba(self, values):
        return np.tile([1.0 - self.positive, self.positive], (len(values), 1))


class FixedMeta:
    def predict_proba(self, values):
        score = np.clip(np.asarray(values).mean(axis=1), 0, 1)
        return np.c_[1.0 - score, score]


def _values():
    return {field.name: field.default for field in PATIENT_FIELDS}


def test_patient_order_and_range_validation():
    names = [field.name for field in PATIENT_FIELDS]
    row = validate_patient(_values(), names)
    assert row.shape == (1, 15)
    bad = _values(); bad["sysBP"] = 999
    with pytest.raises(PatientValidationError, match="sysBP"):
        validate_patient(bad, names)


def test_quantum_angle_encoding_matches_notebook():
    transformed = np.array([[-4.0, -1.0, 1.0, 4.0]])
    encoded = encode_quantum_features(transformed, [0, 1, 2, 3])
    np.testing.assert_allclose(encoded, [[-np.pi, -np.pi/3, np.pi/3, np.pi]])


def test_state_and_fidelity_kernel_are_normalized():
    encoded = np.array([0.1, -0.2, 0.3, -0.4])
    state = feature_state(encoded)
    assert np.vdot(state, state).real == pytest.approx(1.0)
    kernel = fidelity_kernel_row(state, np.asarray([state]))
    assert kernel.shape == (1, 1)
    assert kernel[0, 0] == pytest.approx(1.0)


def test_prediction_composes_real_bundle_interfaces():
    names = [field.name for field in PATIENT_FIELDS]
    encoded = encode_quantum_features(validate_patient(_values(), names), [0, 1, 2, 3])[0]
    state = feature_state(encoded)
    version = "test-v1"
    manifest = {
        "model_version_id": version, "feature_order": names,
        "quantum": {"selected_indices": [0, 1, 2, 3], "n_qubits": 4,
                    "feature_map": {"version": "framingham_v1_h_rz_cz_reupload"}},
        "calibration": {"status": "NOT_DEMONSTRATED"},
    }
    artifacts = FraminghamArtifacts(
        manifest=manifest,
        classical={"model_version_id": version, "preprocess": IdentityPreprocess(), "model": FixedModel(0.2)},
        quantum={"model_version_id": version, "preprocess": IdentityPreprocess(), "train_states": np.asarray([state]), "qsvc": FixedModel(0.4)},
        hybrid={"model_version_id": version, "preprocess": IdentityPreprocess(), "classical_base_model": FixedModel(0.3), "train_states": np.asarray([state]), "qsvc": FixedModel(0.5), "meta_model": FixedMeta()},
    )
    result = predict_patient(artifacts, _values())
    assert result.classical_score == pytest.approx(0.2)
    assert result.quantum_score == pytest.approx(0.4)
    assert result.hybrid_score == pytest.approx(0.4)
    assert result.recommended_pathway == "CLASSICAL-PREFERRED"
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `.venv/bin/python -m pytest tests/test_demo_framingham.py -q`

Expected: collection fails because `src.demo.framingham` does not exist.

- [ ] **Step 3: Implement the patient schema and strict validation**

```python
# beginning of src/demo/framingham.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pennylane as qml

from src.demo.contracts import FraminghamArtifacts, InferenceResult, PatientValidationError


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


def validate_patient(values: Mapping[str, float], feature_order: list[str]) -> np.ndarray:
    fields = {field.name: field for field in PATIENT_FIELDS}
    if set(feature_order) != set(fields):
        raise PatientValidationError("Artifact feature order does not match the 15-field Framingham contract.")
    row = []
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
```

- [ ] **Step 4: Implement exact angle encoding, feature state, and fidelity row**

```python
def encode_quantum_features(transformed: np.ndarray, indices: list[int]) -> np.ndarray:
    values = np.asarray(transformed, dtype=float)[:, np.asarray(indices, dtype=int)]
    return np.clip(values, -3.0, 3.0) * (np.pi / 3.0)


def feature_state(encoded: np.ndarray) -> np.ndarray:
    values = np.asarray(encoded, dtype=float)
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


def fidelity_kernel_row(patient_state: np.ndarray, train_states: np.ndarray) -> np.ndarray:
    states = np.asarray(train_states, dtype=complex)
    state = np.asarray(patient_state, dtype=complex)
    if states.ndim != 2 or states.shape[1] != state.shape[0]:
        raise PatientValidationError("Patient and training quantum-state dimensions do not match.")
    return (np.abs(state[None, :] @ states.conj().T) ** 2).astype(float)
```

- [ ] **Step 5: Implement score composition and contributions**

```python
def _positive(model, values: np.ndarray) -> float:
    probabilities = np.asarray(model.predict_proba(values), dtype=float)
    if probabilities.shape != (1, 2):
        raise PatientValidationError(f"Expected a (1, 2) probability matrix, got {probabilities.shape}.")
    return float(probabilities[0, 1])


def _contributions(bundle, transformed: np.ndarray, feature_order: list[str]) -> tuple[tuple[str, float], ...]:
    model = bundle["model"]
    if not hasattr(model, "coef_"):
        return ()
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    if len(coefficients) != transformed.shape[1]:
        return ()
    rows = [(name, float(value * weight)) for name, value, weight in zip(feature_order, transformed[0], coefficients)]
    return tuple(sorted(rows, key=lambda item: abs(item[1]), reverse=True))


def predict_patient(artifacts: FraminghamArtifacts, values: Mapping[str, float]) -> InferenceResult:
    feature_order = list(artifacts.manifest["feature_order"])
    raw = validate_patient(values, feature_order)
    classical_x = np.asarray(artifacts.classical["preprocess"].transform(raw), dtype=float)
    classical_score = _positive(artifacts.classical["model"], classical_x)

    indices = list(artifacts.manifest["quantum"]["selected_indices"])
    quantum_x = np.asarray(artifacts.quantum["preprocess"].transform(raw), dtype=float)
    encoded = encode_quantum_features(quantum_x, indices)[0]
    q_kernel = fidelity_kernel_row(feature_state(encoded), artifacts.quantum["train_states"])
    quantum_score = _positive(artifacts.quantum["qsvc"], q_kernel)

    hybrid_x = np.asarray(artifacts.hybrid["preprocess"].transform(raw), dtype=float)
    classical_base = _positive(artifacts.hybrid["classical_base_model"], hybrid_x)
    hybrid_encoded = encode_quantum_features(hybrid_x, indices)[0]
    hybrid_kernel = fidelity_kernel_row(feature_state(hybrid_encoded), artifacts.hybrid["train_states"])
    quantum_base = _positive(artifacts.hybrid["qsvc"], hybrid_kernel)
    hybrid_score = _positive(artifacts.hybrid["meta_model"], np.asarray([[classical_base, quantum_base]]))

    return InferenceResult(
        classical_score=classical_score, quantum_score=quantum_score, hybrid_score=hybrid_score,
        recommended_pathway="CLASSICAL-PREFERRED",
        contributions=_contributions(artifacts.classical, classical_x, feature_order),
    )
```

- [ ] **Step 6: Run Framingham tests**

Run: `.venv/bin/python -m pytest tests/test_demo_framingham.py -q`

Expected: `4 passed`.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/demo/framingham.py tests/test_demo_framingham.py
git commit -m "feat: add frozen Framingham inference service"
```

---

### Task 5: Corrected Kaggle Artifact Exporter

**Files:**
- Create: `kaggle/framingham_artifact_export.py`
- Create: `tests/test_framingham_exporter.py`

**Interfaces:**
- Produces: `export_from_namespace(ns: Mapping[str, Any], output_dir: Path) -> Path`.
- Produces bundles and a manifest accepted by `load_framingham_artifacts()` from Task 3.
- Consumes the notebook variables `preprocess`, `classical_models`, `feature_names`, `selected`, `idx`, `S_train`, `qsvc`, `c_final`, `q_final`, `Sh`, `meta`, `summary`, `SEED`, `X_train`, `X_test`, `TARGET`, and optional `DATA_PATH`.

- [ ] **Step 1: Write a failing exporter compatibility test**

```python
# tests/test_framingham_exporter.py
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kaggle.framingham_artifact_export import export_from_namespace
from src.demo.artifacts import load_framingham_artifacts, sha256_file


def test_exporter_creates_loader_compatible_hashed_artifacts(tmp_path):
    X = np.array([[0., 0., 0., 0.], [1., 1., 0., 0.], [0., 1., 1., 0.], [1., 0., 0., 1.]])
    y = np.array([0, 1, 1, 0])
    preprocess = Pipeline([("scaler", StandardScaler())]).fit(X)
    Xt = preprocess.transform(X)
    classical = LogisticRegression().fit(Xt, y)
    c_final = SVC(probability=True).fit(Xt, y)
    kernel = np.eye(4)
    qsvc = SVC(kernel="precomputed", probability=True).fit(kernel, y)
    meta = LogisticRegression().fit(np.c_[classical.predict_proba(Xt)[:, 1], qsvc.predict_proba(kernel)[:, 1]], y)
    states = np.eye(4, dtype=complex)
    ns = {
        "preprocess": preprocess, "classical_models": {"Logistic Regression": classical},
        "feature_names": np.array(["age", "sysBP", "prevalentHyp", "diaBP"]),
        "selected": ["age", "sysBP", "prevalentHyp", "diaBP"], "idx": [0, 1, 2, 3],
        "S_train": states, "qsvc": qsvc, "c_final": c_final, "q_final": qsvc,
        "Sh": states, "meta": meta, "summary": pd.DataFrame([{"Model": "Logistic Regression", "AUROC": 0.7}]),
        "SEED": 42, "X_train": pd.DataFrame(X), "X_test": pd.DataFrame(X[:1]), "TARGET": "TenYearCHD",
    }
    artifact_dir = export_from_namespace(ns, tmp_path)
    loaded = load_framingham_artifacts(artifact_dir)
    assert loaded.manifest["schema_version"] == 1
    assert loaded.manifest["calibration"]["status"] == "NOT_DEMONSTRATED"
    for name, digest in loaded.manifest["checksums_sha256"].items():
        assert sha256_file(artifact_dir / name) == digest
```

- [ ] **Step 2: Run the exporter test and verify the missing-module failure**

Run: `.venv/bin/python -m pytest tests/test_framingham_exporter.py -q`

Expected: collection fails because `kaggle.framingham_artifact_export` does not exist.

- [ ] **Step 3: Implement an import-safe exporter**

Create `kaggle/framingham_artifact_export.py` by extracting the supplied notebook's cell 36 into functions, then make these exact contract changes:

```python
SCHEMA_VERSION = 1
FEATURE_MAP = {
    "version": "framingham_v1_h_rz_cz_reupload",
    "clip": [-3.0, 3.0],
    "angle_scale": "pi/3",
    "gates": ["H", "RY(x)", "RZ(0.5*x)", "ring-CZ", "RY(x^2/pi)"],
    "kernel": "abs(inner_product(patient_state, training_state))**2",
}
```

Write bundles first, write `frozen_notebook_metrics.json`, calculate SHA-256 for those four files, and then write `model_manifest.json`. Include this input contract:

```python
INPUT_SCHEMA = {
    field.name: {"minimum": field.minimum, "maximum": field.maximum, "binary": field.binary}
    for field in PATIENT_FIELDS
}
```

Use a content-derived model identifier based on feature order, selected indices, seed, metrics JSON, dataset hash when available, and artifact hashes:

```python
identity_payload = json.dumps({
    "feature_order": feature_order,
    "selected_indices": selected_indices,
    "seed": int(ns["SEED"]),
    "metrics": metrics,
    "dataset_sha256": dataset_sha256,
}, sort_keys=True, separators=(",", ":"))
version_id = "qmedai-framingham-" + hashlib.sha256(identity_payload.encode()).hexdigest()[:12]
```

Set `model_version_id` in each bundle before dumping. The manifest's `artifacts` mapping must use roles `classical`, `quantum`, `hybrid`, and `metrics`. Do not execute export at import time. Put the Kaggle-cell entry point under:

```python
if __name__ == "__main__":
    artifact_dir = export_from_namespace(globals(), Path("/kaggle/working/qmedai_framingham_export"))
    print("Q-MedAI FROZEN ARTIFACT EXPORT COMPLETE")
    print("Artifact directory:", artifact_dir)
    print("Download this ZIP after creating it from the artifact directory.")
```

- [ ] **Step 4: Ensure the script creates the download ZIP**

At the end of `export_from_namespace`, create `qmedai_framingham_artifacts.zip` beside the artifact directory using `shutil.make_archive`, while still returning the `artifact_dir` path used by the loader test.

- [ ] **Step 5: Run exporter and artifact tests together**

Run: `.venv/bin/python -m pytest tests/test_framingham_exporter.py tests/test_demo_artifacts.py -q`

Expected: every exporter and artifact test passes.

- [ ] **Step 6: Commit Task 5**

```bash
git add kaggle/framingham_artifact_export.py tests/test_framingham_exporter.py
git commit -m "feat: add reproducible Framingham artifact exporter"
```

---

### Task 6: Combined Streamlit Faculty Experience

**Files:**
- Create: `src/demo/styles.py`
- Create: `src/demo/pages.py`
- Modify: `app.py`
- Create: `tests/test_demo_app.py`

**Interfaces:**
- Consumes: loaders, utility verdicts, artifact status, `PATIENT_FIELDS`, and `predict_patient` from Tasks 1–4.
- Produces: `render_app(project_root: Path) -> None` and seven faculty-facing pages plus the preserved exploratory runner.

- [ ] **Step 1: Write the degraded-mode Streamlit smoke test**

```python
# tests/test_demo_app.py
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_app_starts_without_framingham_artifacts():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Q-MedAI Clinical Intelligence"
    text = " ".join(item.value for item in app.markdown)
    assert "Framingham" in text
    assert "Breast Cancer" in text


def test_patient_page_never_infers_without_artifacts():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Patient Risk").run()
    assert not app.exception
    assert any(button.label == "Run verified inference" and button.disabled for button in app.button)
    assert any("no patient score" in item.value.lower() for item in app.warning)
```

- [ ] **Step 2: Run the smoke test and verify it fails against the old app**

Run: `.venv/bin/python -m pytest tests/test_demo_app.py -q`

Expected: assertions fail because the old title/navigation does not implement the combined demo.

- [ ] **Step 3: Add contained visual styling**

```python
# src/demo/styles.py
THEME_CSS = """
<style>
:root { --qm-navy:#071A2F; --qm-cyan:#24D6D0; --qm-violet:#8B7CFF; --qm-panel:#102A43; }
.stApp { background: linear-gradient(145deg, #F7FBFF 0%, #EEF6FF 55%, #F7F4FF 100%); }
.qm-hero { padding: 1.4rem 1.6rem; border-radius: 22px; color: white;
  background: linear-gradient(120deg, var(--qm-navy), #123E67 62%, #3D2D78); box-shadow: 0 18px 44px #17324d25; }
.qm-eyebrow { color: #8FF8F1; font-size: .78rem; letter-spacing: .14em; font-weight: 700; }
.qm-card { background: #ffffffd9; border: 1px solid #D8E8F5; border-radius: 18px;
  padding: 1rem 1.1rem; min-height: 132px; box-shadow: 0 8px 24px #17324d12; }
.qm-verdict { border-left: 5px solid var(--qm-cyan); background: white; padding: .85rem 1rem; border-radius: 12px; }
.qm-footnote { color:#52677B; font-size:.86rem; }
</style>
"""
```

- [ ] **Step 4: Implement page composition in `src/demo/pages.py`**

Implement concrete bodies for these exact page-function interfaces: `render_command_center(context: DemoContext) -> None`, `render_patient_risk(context: DemoContext) -> None`, `render_quantum_lab(context: DemoContext) -> None`, `render_model_arena(context: DemoContext) -> None`, `render_breast_cancer_evidence(context: DemoContext) -> None`, `render_quantum_utility(context: DemoContext) -> None`, `render_provenance(context: DemoContext) -> None`, `render_research_runner() -> None`, and `render_app(project_root: Path) -> None`.

Create a page context:

```python
@dataclass(frozen=True)
class DemoContext:
    project_root: Path
    framingham: TaskEvidence
    breast_cancer: TaskEvidence
    framingham_verdict: UtilityVerdict
    breast_cancer_verdict: UtilityVerdict
    artifact_status: ArtifactStatus
```

`render_app` must load:

```python
framingham = load_framingham_evidence(project_root / "results/framingham/notebook_metrics.json")
breast = load_breast_cancer_evidence(project_root / "results/final_results.json")
status = inspect_framingham_artifacts(project_root / "artifacts/framingham")
```

Use this navigation order:

```python
pages = (
    "Command Center", "Patient Risk", "Quantum Lab", "Model Arena",
    "Breast Cancer Evidence", "Quantum Utility", "Provenance & Safety", "Research Runner",
)
```

Patient page behavior must follow this exact branch:

```python
if not context.artifact_status.ready:
    st.warning("Verified Framingham artifacts are not installed, so Q-MedAI will produce no patient score.")
    st.button("Run verified inference", type="primary", disabled=True)
else:
    if st.button("Run verified inference", type="primary"):
        artifacts = load_framingham_artifacts(context.project_root / "artifacts/framingham")
        result = predict_patient(artifacts, values)
        st.metric("Classical research-model score", f"{result.classical_score:.3f}")
        st.metric("Quantum research score", f"{result.quantum_score:.3f}")
        st.metric("Hybrid research score", f"{result.hybrid_score:.3f}")
        if result.contributions:
            st.dataframe(pd.DataFrame(result.contributions, columns=["Feature", "Model contribution"]), hide_index=True)
```

Use `st.number_input` for continuous fields and `st.selectbox` for binary fields. Put all fields inside `st.form`, but place the disabled button outside the form in degraded mode so Streamlit testing can inspect its disabled state.

`render_model_arena` must show the Framingham table and its unmatched-resource warning. `render_breast_cancer_evidence` must show `results/plots/matched_classical_quantum_judge_summary.png`, matched metrics, false-negative deltas, and downloads. `render_quantum_lab` must display the four selected features and the exact feature-map sequence as text/code. `render_provenance` must show manifest data only when the artifact status is ready.

- [ ] **Step 5: Preserve the exploratory runner**

Move the existing `app.py` helpers `_signature` and `_run_new_experiment` into `src/demo/pages.py`, retaining `ExperimentConfig`, `MODEL_NAMES`, `MAX_KERNEL_SUBSET_SIZE`, `SUPPORTED_FEATURE_COUNTS`, and `run_experiment`. Render it only on the `Research Runner` page and retain the statement that session-only runs never overwrite authoritative artifacts.

- [ ] **Step 6: Replace `app.py` with the thin entry point**

```python
"""Q-MedAI combined faculty demonstration."""
from pathlib import Path

import streamlit as st

from src.demo.pages import render_app


PROJECT_ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Q-MedAI", page_icon="🧬", layout="wide")


if __name__ == "__main__":
    render_app(PROJECT_ROOT)
```

- [ ] **Step 7: Run the Streamlit smoke tests**

Run: `.venv/bin/python -m pytest tests/test_demo_app.py -q`

Expected: `2 passed` and no Streamlit exceptions.

- [ ] **Step 8: Run all demo tests**

Run: `.venv/bin/python -m pytest tests/test_demo_evidence.py tests/test_demo_utility.py tests/test_demo_artifacts.py tests/test_demo_framingham.py tests/test_framingham_exporter.py tests/test_demo_app.py -q`

Expected: every listed demo test passes.

- [ ] **Step 9: Commit Task 6**

```bash
git add app.py src/demo/pages.py src/demo/styles.py tests/test_demo_app.py
git commit -m "feat: build combined Q-MedAI faculty demo"
```

---

### Task 7: Documentation, Full Verification, and Faculty Package

**Files:**
- Modify: `README.md`
- Modify: `SIH_FINAL_SUMMARY.md`
- Create: `FACULTY_DEMO_GUIDE.md`
- Verify: all source, tests, evidence, and generated app state

**Interfaces:**
- Consumes: the completed combined application.
- Produces: launch instructions, a six-step faculty walkthrough, artifact activation instructions, and a distributable source ZIP outside git.

- [ ] **Step 1: Add a failing documentation contract test**

Append to `tests/test_demo_app.py`:

```python
def test_faculty_guide_contains_required_claim_boundaries():
    guide = (ROOT / "FACULTY_DEMO_GUIDE.md").read_text(encoding="utf-8")
    assert "Framingham = Primary Clinical Workflow" in guide
    assert "Breast Cancer = Quantum Evidence Benchmark" in guide
    assert "CLASSICAL-PREFERRED" in guide
    assert "QUANTUM-COMPETITIVE" in guide
    assert "research-model score" in guide
    assert "default.qubit" in guide
    assert "not a medical diagnostic device" in guide
```

- [ ] **Step 2: Run the documentation test and verify the missing-file failure**

Run: `.venv/bin/python -m pytest tests/test_demo_app.py::test_faculty_guide_contains_required_claim_boundaries -q`

Expected: fails with `FileNotFoundError: FACULTY_DEMO_GUIDE.md`.

- [ ] **Step 3: Write the faculty guide**

Create `FACULTY_DEMO_GUIDE.md` with these sections and exact claim boundaries:

```markdown
# Q-MedAI Faculty Demonstration

Framingham = Primary Clinical Workflow  
Breast Cancer = Quantum Evidence Benchmark

Q-MedAI is a research prototype and not a medical diagnostic device.

## Six-minute walkthrough
1. Command Center: explain task-specific model selection.
2. Patient Risk: show the artifact-gated Framingham workflow.
3. Quantum Lab: show the real four-qubit feature-map simulation.
4. Model Arena: explain why Framingham is CLASSICAL-PREFERRED.
5. Breast Cancer Evidence: show the matched QUANTUM-COMPETITIVE result.
6. Quantum Utility: conclude that Q-MedAI selects evidence-supported strategies rather than forcing a quantum winner.

## Required wording
Framingham outputs are research-model scores, not calibrated clinical 10-year probabilities. PennyLane default.qubit is a classical simulator. Breast-cancer results show a selective benefit against Random Forest, while RBF SVM remains strongest overall.
```

Include artifact activation steps using `kaggle/framingham_artifact_export.py` and `artifacts/framingham/`.

- [ ] **Step 4: Update repository documentation**

Update the first screen of `README.md` to introduce the combined story and add navigation, safe degraded mode, Framingham artifact activation, and faculty launch instructions. Preserve the existing final breast-cancer methodology and result details.

Update `SIH_FINAL_SUMMARY.md` with one new section titled `Combined faculty demo` that links to `FACULTY_DEMO_GUIDE.md` and states both task-specific verdicts without changing the saved breast-cancer metrics.

- [ ] **Step 5: Run the documentation and full test suite**

Run: `.venv/bin/python -m pytest -q`

Expected: all tests pass; the prior suite remains green, with only previously documented warnings/skips.

- [ ] **Step 6: Validate Python and JSON artifacts**

Run:

```bash
.venv/bin/python -m compileall -q app.py src kaggle/framingham_artifact_export.py
.venv/bin/python -m json.tool results/framingham/notebook_metrics.json >/dev/null
.venv/bin/python -m json.tool results/final_results.json >/dev/null
git diff --check
```

Expected: every command exits `0` with no output from `git diff --check`.

- [ ] **Step 7: Perform a headless Streamlit boot smoke test**

Run:

```bash
.venv/bin/streamlit run app.py --server.headless true --server.port 8511 --browser.gatherUsageStats false
```

Expected: Streamlit prints a local URL and stays running without a Python exception. Stop it with `Ctrl-C` after the startup message.

- [ ] **Step 8: Inspect the app visually**

Open the local Streamlit URL and verify:

- Command Center is the default page;
- Framingham and Breast Cancer are both visible above the fold;
- Patient Risk shows the form and disabled inference action when artifacts are absent;
- Quantum Lab shows the four-feature/four-qubit flow and simulator disclosure;
- Model Arena labels the Framingham comparison as unmatched;
- Breast Cancer Evidence renders the existing matched judge figure;
- Utility page shows `CLASSICAL-PREFERRED` and `QUANTUM-COMPETITIVE`; and
- no component labels an uncalibrated output as a clinical probability.

- [ ] **Step 9: Commit documentation**

```bash
git add README.md SIH_FINAL_SUMMARY.md FACULTY_DEMO_GUIDE.md tests/test_demo_app.py
git commit -m "docs: add combined faculty demo walkthrough"
```

- [ ] **Step 10: Create the faculty source ZIP outside the repository**

Run from the repository root after the documentation commit:

```bash
git archive --format=zip --output=/Users/mymac/Downloads/Q-MedAI_Combined_Faculty_Demo.zip HEAD
unzip -t /Users/mymac/Downloads/Q-MedAI_Combined_Faculty_Demo.zip
```

Expected: `/Users/mymac/Downloads/Q-MedAI_Combined_Faculty_Demo.zip` exists and `unzip -t` reports no errors. The ZIP contains source and committed evidence but no synthetic Framingham model artifacts.

- [ ] **Step 11: Run final clean-state verification**

Run:

```bash
.venv/bin/python -m pytest -q
git status --short --branch
git log -8 --oneline
```

Expected: the complete suite passes and the worktree is clean. The branch may be ahead of `origin/main` until the user authorizes or requests a push.

---

## Completion Evidence to Report

The final handoff must include:

- total passing/skipped/failed test counts from the final run;
- headless Streamlit startup result;
- confirmation that artifact-absent mode produced no patient score;
- confirmation that no production mock model was packaged;
- exact commit SHA(s);
- clickable links to `app.py`, `FACULTY_DEMO_GUIDE.md`, the design, the plan, and the faculty ZIP; and
- the verified frozen Framingham artifact ID and a successful real patient-inference smoke result.

---

## Scope Update — Real Ready-Mode Demonstration (2026-08-29)

The user explicitly expanded the final acceptance criteria after Task 6: the delivered faculty demo must include real frozen Framingham inference and an end-to-end browser walkthrough. This supersedes the earlier package-only/degraded-mode completion assumption.

Before documentation and final packaging, implement and verify the following:

1. Reproduce the executed Framingham notebook's seed-42 preprocessing, four-feature quantum map, 500-row quantum subset, quantum-kernel SVM, and leakage-safe hybrid stack from the exact public `CHD_preprocessed.csv` cohort.
2. Use `kaggle/framingham_artifact_export.py` to export the already-fitted namespace into the canonical trusted directory `artifacts/framingham/`; do not handcraft, mock, or synthesize model outputs.
3. Add a reproducible local build entry point and focused tests for ready-mode artifact loading and patient scoring.
4. Verify SHA-256 checksums, matching runtime versions, exact feature/bundle metadata, and finite `[0, 1]` classical, quantum, and hybrid research scores for at least two valid patient inputs.
5. Run the Streamlit app on port 8511, navigate it in a real browser, submit the Patient Risk form, verify all three score cards and the contribution table render, inspect every main page, and preserve screenshots as verification evidence.
6. Commit the real frozen artifacts and browser-ready documentation so `git archive` creates a self-contained faculty demo. Keep all outputs labelled as uncalibrated research-model scores and keep the `default.qubit` classical-simulator disclosure.
