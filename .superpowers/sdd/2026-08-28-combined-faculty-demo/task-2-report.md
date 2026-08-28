# Task 2 Report: Quantum Utility Engine

## RED

Command:

```text
/Users/mymac/Desktop/Q-MedAI/.venv/bin/python -m pytest tests/test_demo_utility.py -q
```

Result: collection failed as expected with `ModuleNotFoundError: No module named 'src.demo.utility'`.

## GREEN

Focused command:

```text
/Users/mymac/Desktop/Q-MedAI/.venv/bin/python -m pytest tests/test_demo_evidence.py tests/test_demo_utility.py -q
```

Result: `8 passed`.

Full-suite command:

```text
/Users/mymac/Desktop/Q-MedAI/.venv/bin/python -m pytest -q
```

Result: `24 passed, 1 warning`.

## Files

- `src/demo/utility.py`: deterministic utility evaluation and evidence table.
- `src/demo/__init__.py`: exports utility functions with evidence loaders.
- `tests/test_demo_utility.py`: Framingham and breast-cancer verdict tests.

## Summary

The engine compares configured reference and candidate models, computes metric deltas, and ranks every model by `(roc_auc, accuracy, f1, specificity)` so the RBF SVM deterministically wins the breast-cancer ROC-AUC tie. It preserves `CLASSICAL-PREFERRED` for Framingham and reports matched breast-cancer evidence as `QUANTUM-COMPETITIVE` with selective sensitivity and false-negative benefit, without claiming universal quantum advantage.

## Concerns

- The full suite retains one pre-existing scikit-learn `FutureWarning` about the SVM `probability` parameter.
- `evidence_table` exposes the metric-record schema directly; downstream UI should tolerate optional null columns.
