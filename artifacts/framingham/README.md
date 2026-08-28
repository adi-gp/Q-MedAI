# Framingham frozen artifacts

Place only artifacts produced by `kaggle/framingham_artifact_export.py` here. Q-MedAI loads local joblib files only after manifest and SHA-256 verification. Joblib uses Python pickle internally; never place downloaded or untrusted joblib files in this directory.

Required files: `model_manifest.json`, `classical_pipeline.joblib`, `quantum_bundle.joblib`, `hybrid_bundle.joblib`, and `frozen_notebook_metrics.json`.

Without all five verified files, the faculty demo remains in safe evidence mode and produces no patient score.
