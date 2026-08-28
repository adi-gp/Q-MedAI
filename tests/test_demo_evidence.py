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
