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
