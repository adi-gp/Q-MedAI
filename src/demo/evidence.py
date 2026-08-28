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
    if not isinstance(row, dict):
        raise EvidenceError("Metric row must be a JSON object.")
    required = {"model", "family", "accuracy", "sensitivity", "specificity", "f1", "roc_auc"}
    missing = sorted(required - row.keys())
    if missing:
        raise EvidenceError("Metric row is missing: " + ", ".join(missing))
    return MetricRecord(**{name: row.get(name) for name in MetricRecord.__dataclass_fields__})


def _task(data: dict[str, Any], metrics: tuple[MetricRecord, ...]) -> TaskEvidence:
    required = {
        "task_id", "title", "role", "task_type", "source", "cohort",
        "comparison_fairness", "backend", "focus_reference_model",
        "focus_candidate_model", "prioritized_metric", "uncertainty_status",
        "calibration_status",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise EvidenceError("Evidence is missing: " + ", ".join(missing))
    return TaskEvidence(
        task_id=data["task_id"], title=data["title"], role=data["role"],
        task_type=data["task_type"], source=data["source"], cohort=data["cohort"],
        comparison_fairness=data["comparison_fairness"], backend=data["backend"],
        metrics=metrics, focus_reference_model=data["focus_reference_model"],
        focus_candidate_model=data["focus_candidate_model"],
        prioritized_metric=data["prioritized_metric"],
        uncertainty_status=data["uncertainty_status"],
        calibration_status=data["calibration_status"],
        limitations=tuple(data.get("limitations", [])),
    )


def load_framingham_evidence(path: Path) -> TaskEvidence:
    data = _read_json(path)
    rows = data.get("metrics")
    if not isinstance(rows, list) or not rows:
        raise EvidenceError("Framingham evidence requires a non-empty metrics list.")
    return _task(data, tuple(_metric(row) for row in rows))


def load_breast_cancer_evidence(path: Path) -> TaskEvidence:
    data = _read_json(path)
    try:
        comparison = data["matched_data_comparison"]
        rows = comparison["models"]
    except (KeyError, TypeError) as exc:
        raise EvidenceError("Breast-cancer evidence requires matched_data_comparison.models.") from exc
    if not isinstance(rows, list) or not rows:
        raise EvidenceError("Breast-cancer evidence requires a non-empty models list.")
    metrics = []
    for row in rows:
        if not isinstance(row, dict):
            raise EvidenceError("Breast-cancer metric row must be a JSON object.")
        try:
            matrix = row["confusion_matrix"]
            if (
                not isinstance(matrix, list) or len(matrix) != 2
                or any(not isinstance(line, list) or len(line) != 2 for line in matrix)
            ):
                raise EvidenceError("Invalid confusion_matrix; expected a 2x2 matrix.")
            metrics.append(MetricRecord(
                model=row["model"], family=row["type"], accuracy=row["accuracy"],
                precision=row["precision"], sensitivity=row["recall"],
                specificity=row["specificity"], f1=row["f1"], roc_auc=row["roc_auc"],
                false_negatives=int(matrix[1][0]),
                training_seconds=row["training_time_seconds"],
            ))
        except KeyError as exc:
            raise EvidenceError(f"Breast-cancer metric is missing: {exc.args[0]}") from exc
        except (IndexError, TypeError, ValueError) as exc:
            raise EvidenceError("Invalid breast-cancer metric row.") from exc
    try:
        test_rows = comparison["test_rows"]
        analysis = data["analysis"]
        confidence = analysis["statistical_confidence"]
        interpretation = analysis["interpretation"]
    except KeyError as exc:
        raise EvidenceError(f"Breast-cancer evidence is missing: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise EvidenceError("Breast-cancer evidence has invalid nested fields.") from exc
    if not isinstance(test_rows, int):
        raise EvidenceError("Breast-cancer evidence requires matched_data_comparison.test_rows.")
    if not isinstance(analysis, dict):
        raise EvidenceError("Breast-cancer evidence requires analysis.")
    return TaskEvidence(
        task_id="breast_cancer_wdbc", title="Breast cancer diagnostic classification",
        role="QUANTUM_EVIDENCE_BENCHMARK", task_type="CROSS_SECTIONAL_CLASSIFICATION",
        source=str(path), cohort={"train_rows": 150, "test_rows": test_rows},
        comparison_fairness="MATCHED_150_ROWS",
        backend="PennyLane default.qubit (classical simulator)", metrics=tuple(metrics),
        focus_reference_model="Random Forest", focus_candidate_model="Quantum Kernel SVM",
        prioritized_metric="sensitivity", uncertainty_status="NOT_COMPUTED",
        calibration_status="NOT_APPLICABLE",
        limitations=(confidence, interpretation),
    )
