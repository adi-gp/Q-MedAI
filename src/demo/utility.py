"""Deterministic, evidence-derived utility decisions for the faculty demo."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from src.demo.contracts import EvidenceError, MetricRecord, TaskEvidence, UtilityVerdict


def _models(evidence: TaskEvidence) -> dict[str, MetricRecord]:
    """Index metric records by model name for the configured comparison."""
    return {row.model: row for row in evidence.metrics}


def _delta(candidate: MetricRecord, reference: MetricRecord, field: str) -> float:
    left = getattr(candidate, field)
    right = getattr(reference, field)
    if left is None or right is None:
        raise EvidenceError(f"Cannot compare missing metric {field}.")
    return float(left - right)


def _utility_key(row: MetricRecord) -> tuple[float, float, float, float]:
    """Rank models by the deterministic overall utility tuple.

    ROC-AUC is primary; accuracy, F1, and specificity resolve ties in that
    order.  This explicitly makes the breast-cancer RBF SVM win its AUROC tie.
    """
    fields = ("roc_auc", "accuracy", "f1", "specificity")
    values = tuple(getattr(row, field) for field in fields)
    if any(value is None for value in values):
        missing = fields[next(i for i, value in enumerate(values) if value is None)]
        raise EvidenceError(f"Cannot rank model with missing metric {missing}.")
    return tuple(float(value) for value in values)


def evaluate_utility(evidence: TaskEvidence) -> UtilityVerdict:
    """Evaluate task utility without claiming a universal quantum advantage."""
    rows = _models(evidence)
    try:
        reference = rows[evidence.focus_reference_model]
        candidate = rows[evidence.focus_candidate_model]
    except KeyError as exc:
        raise EvidenceError(f"Utility comparison model is missing: {exc.args[0]}") from exc

    best = max(evidence.metrics, key=_utility_key)
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

    rationale = (
        f"Comparison fairness: {evidence.comparison_fairness.replace('_', ' ').lower()}.",
        f"Best overall model by (ROC-AUC, accuracy, F1, specificity): {best.model} ({best.roc_auc:.4f}).",
        f"Backend: {evidence.backend}.",
        f"Uncertainty: {evidence.uncertainty_status.replace('_', ' ').lower()}.",
    )
    return UtilityVerdict(
        task_id=evidence.task_id,
        status=status,
        headline=headline,
        reference_model=reference.model,
        candidate_model=candidate.model,
        deltas=deltas,
        best_overall_model=best.model,
        rationale=rationale,
    )


def evidence_table(evidence: TaskEvidence) -> pd.DataFrame:
    """Return metric records as a tabular view suitable for the demo UI."""
    return pd.DataFrame([asdict(row) for row in evidence.metrics])
