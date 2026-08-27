import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import evaluate_predictions


def test_metrics_specificity_and_continuous_roc_auc_are_correct():
    y_true = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 0, 1])
    continuous_scores = np.array([0.1, 0.8, 0.2, 0.9])
    metrics, matrix, fpr, tpr = evaluate_predictions(y_true, predictions, continuous_scores)

    assert matrix.tolist() == [[1, 1], [1, 1]]
    assert metrics["Specificity"] == 0.5
    assert metrics["ROC-AUC"] == roc_auc_score(y_true, continuous_scores)
    assert metrics["ROC-AUC"] != roc_auc_score(y_true, predictions)
    assert len(fpr) == len(tpr)
