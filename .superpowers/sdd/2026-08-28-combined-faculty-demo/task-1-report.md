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

## Fix round 1 review response

### Changes

- Copied the supplied executed notebook to `kaggle/Framingham_CHD_Executed_Evidence.ipynb` without modifying its outputs.
- Changed the Framingham evidence source to the repository-relative notebook path; the test now asserts that the resolved source exists. The later corrected exporter script supersedes the unexecuted exporter cell in the notebook.
- Added strict 2x2 confusion-matrix validation. Missing or malformed matrices now raise `EvidenceError`; false negatives are never invented.
- Added domain-error coverage for missing nested breast-cancer `test_rows` and `analysis` fields and translated nested schema failures to `EvidenceError`.

### TDD RED evidence

With the new tests present, the production package was temporarily moved out of `src/demo`, then this exact command was run:

```text
/Users/mymac/Desktop/Q-MedAI/.venv/bin/python -m pytest tests/test_demo_evidence.py -q
```

Result: collection failed with `ModuleNotFoundError: No module named 'src.demo'`.

Before the loader fixes, the new behavior tests also failed as expected: 3 failed, 3 passed. The failures were the missing confusion matrix not raising an error, and raw `KeyError`/non-matching domain errors for missing `test_rows` and `analysis`.

### TDD GREEN evidence

```text
/Users/mymac/Desktop/Q-MedAI/.venv/bin/python -m pytest tests/test_demo_evidence.py -q
......                                                                   [100%]
6 passed in 0.03s

/Users/mymac/Desktop/Q-MedAI/.venv/bin/python -m pytest -q
......................                                                   [100%]
22 passed, 1 warning in 3.95s
```

### Fix-round self-review

Verified the copied notebook is present in the delivered tree, the Framingham source assertion resolves successfully, malformed and missing nested schemas produce `EvidenceError`, and no existing breast-cancer result artifact was changed. `git diff --check` is clean. The full suite retains one pre-existing sklearn deprecation warning.
