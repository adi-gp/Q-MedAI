# Task 1 report: typed evidence contracts and loaders

## Implementation summary

Added immutable typed contracts for demo errors, metric records, task evidence, and the downstream artifact/inference value objects. Added JSON evidence loaders for the executed Framingham metrics and the existing matched breast-cancer result schema. The breast-cancer adapter preserves the matched 150-row quantum result and derives false negatives from each confusion matrix without changing the frozen result artifact.

## Files

- `src/demo/__init__.py`
- `src/demo/contracts.py`
- `src/demo/evidence.py`
- `results/framingham/notebook_metrics.json`
- `tests/test_demo_evidence.py`

## Test commands and results

- `python -m pytest tests/test_demo_evidence.py -q` — **3 passed**.
- `python -m pytest -q` — **19 passed**.

## RED evidence

The prescribed `.venv/bin/python -m pytest tests/test_demo_evidence.py -q` command could not start because this checkout has no `.venv`. After installing pytest into the available system Python, the initial test collection failed with `ModuleNotFoundError: No module named 'src'` when invoking the standalone `pytest` shim. Running the module form from the repository root then exercised the tests successfully after implementation.

## GREEN evidence

`python -m pytest tests/test_demo_evidence.py -q` completed with `3 passed in 0.10s`; the full suite completed with `19 passed in 18.88s`.

## Self-review

Reviewed the new files for required field coverage, immutable dataclasses, error translation for malformed JSON and malformed metric rows, preservation of source path and matched comparison metadata, and accidental modifications to `results/final_results.json`. `git diff --check` is clean.

## Concerns

The repository did not include the brief’s `.venv`, so the exact prescribed interpreter path was unavailable and pytest had to be installed in the available Python environment. The loader enforces required schema presence and domain errors, while numeric range/type validation remains outside this task’s requested contract.
