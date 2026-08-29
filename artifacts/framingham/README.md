# Framingham frozen artifacts

This directory contains the real, bundled READY-mode Framingham artifact set for the combined faculty demo.

## Bundled identity

- Model ID: `qmedai-framingham-b98c306b5b44`
- Task: prospective 10-year coronary heart disease research from baseline measurements
- Current utility verdict: `CLASSICAL-PREFERRED`
- Calibration: `NOT_DEMONSTRATED`
- Backend: PennyLane `default.qubit` (classical quantum-circuit simulator)
- Selected quantum features: `age`, `sysBP`, `prevalentHyp`, `diaBP`
- Feature map: `H → RY(x) → RZ(0.5x) → ring-CZ → RY(x²/π)` with fidelity kernel

All displayed outputs are **uncalibrated research-model scores**, not validated clinical probabilities, diagnoses, or treatment recommendations. These artifacts have no external clinical validation and do not establish universal quantum advantage.

## Canonical files

Q-MedAI requires exactly these five files:

1. `model_manifest.json`
2. `classical_pipeline.joblib`
3. `quantum_bundle.joblib`
4. `hybrid_bundle.joblib`
5. `frozen_notebook_metrics.json`

On startup, the trusted loader validates the manifest, canonical filenames, SHA-256 hashes, runtime compatibility, and bundle/model identity before any joblib file is deserialized. The committed hashes are recorded in `model_manifest.json`.

Joblib uses Python pickle internally and can execute code during deserialization. Do not add arbitrary uploads, downloaded bundles, or files from an untrusted source. The application intentionally provides no upload control.

## Verify the bundled READY state

From the repository root:

```bash
.venv/bin/python -m pytest tests/test_framingham_builder.py::test_committed_real_artifacts_are_ready_and_score_distinct_patients -q
```

The test verifies that the canonical artifact set is READY, loads through the trusted loader, and produces distinct finite `[0, 1]` research scores for two valid inputs.

## Rebuild from the public cohort

Rebuilding is optional because the real artifact set is already bundled. If a rebuild is required, obtain the exact public file `CHD_preprocessed.csv` from Kaggle dataset `captainozlem/framingham-chd-preprocessed-data`, then run:

```bash
.venv/bin/python -m scripts.build_framingham_artifacts \
  --data /absolute/path/to/CHD_preprocessed.csv \
  --output-dir artifacts/framingham
```

The deterministic builder reproduces the executed notebook procedure and calls `kaggle/framingham_artifact_export.py` with already-fitted objects. It does not handcraft or mock patient outputs. Before presenting a rebuild:

1. Confirm the builder exits successfully and selects exactly `age`, `sysBP`, `prevalentHyp`, `diaBP`.
2. Review the new `model_manifest.json`, model ID, provenance, runtime versions, and checksums.
3. Run the focused READY-state test above.
4. Restart Streamlit so the app re-inspects the artifact set.

The public raw CSV is not bundled. Its source license metadata was `Unknown`, so only derived fitted artifacts and provenance hashes are committed.

## Safe failure mode

If a canonical file is missing, altered, incompatible, or invalid, the faculty demo remains in safe evidence mode. Patient Risk still shows the complete form and evidence, but inference is disabled and Q-MedAI produces **no patient score**.
