# Q-MedAI Combined Faculty Demo Design

**Date:** 2026-08-28  
**Status:** Approved in chat for specification  
**Problem statement:** SIH 2026 PS 26139 — Hybrid Quantum Machine Learning Platform for Early Disease Detection

## 1. Purpose

The faculty-facing Q-MedAI application will demonstrate that the appropriate classical, quantum, or hybrid strategy depends on the biomedical task:

- **Framingham is the primary clinical-workflow demo.** It models a future 10-year CHD outcome from baseline variables and provides a patient-input workflow when real frozen artifacts are installed.
- **Breast cancer is the quantum-evidence benchmark.** It preserves the existing matched classical-versus-quantum experiment and its selective Quantum Kernel benefit against Random Forest.

The product will not force a universal quantum-advantage claim. It will expose an evidence-based Quantum Utility Engine whose current Framingham verdict is `CLASSICAL-PREFERRED` and whose current breast-cancer verdict is `QUANTUM-COMPETITIVE / SELECTIVE BENEFIT`.

This is a research and faculty-demonstration prototype, not a medical device.

## 2. Success Criteria

The combined demo is successful when:

1. `streamlit run app.py` opens one coherent Q-MedAI application.
2. Framingham is the default landing workflow and breast cancer remains accessible as a separate evidence module.
3. The application starts safely even when Framingham model artifacts are absent.
4. Missing, incompatible, or integrity-failing artifacts never produce a patient score.
5. Compatible frozen artifacts produce classical, quantum, and hybrid research outputs through the same transformations used by the notebook.
6. The Framingham output is called a **research-model score** unless the artifact manifest explicitly records validated calibration.
7. Existing breast-cancer results, figures, downloads, and scientific caveats remain intact.
8. Every metric shown in the application is traceable to a local evidence file or validated artifact manifest.
9. The application clearly discloses PennyLane `default.qubit` as a classical simulator.
10. Automated tests cover the artifact contract, inference composition, utility decisions, patient-input validation, and artifact-missing application state.

## 3. Considered Approaches

### 3.1 Unified application — selected

One Streamlit application contains the Framingham workflow and breast-cancer evidence. Shared navigation, provenance, safety language, and utility decisions create a single platform story.

### 3.2 Separate applications — rejected

Separate apps would reduce routing changes but weaken the platform narrative and duplicate safety, styling, and provenance logic.

### 3.3 Retrain models during application startup — rejected

Startup training is slow, environment-dependent, and unsuitable for a faculty demonstration. The application will use frozen artifacts only. Training remains in Kaggle/notebook workflows.

## 4. User Experience

### 4.1 Command Center

The landing page will show:

- PS 26139 alignment;
- the two-task product story;
- a compact hybrid architecture diagram rendered with Streamlit-native components;
- current task-specific utility verdicts;
- artifact readiness and simulator status; and
- research/medical disclaimers.

### 4.2 Patient Risk

The Framingham form will collect the exact ordered baseline variables used by the notebook. Inputs will use explicit units, types, plausible prototype ranges, and binary/category controls.

When artifacts are ready, the page will show:

- primary classical research-model score;
- quantum research score;
- hybrid research score;
- the evidence-supported recommended pathway;
- patient-specific Logistic Regression contributions when coefficients are available; and
- plain-language limitations.

When artifacts are not ready, the form remains visible for demonstration, but the prediction action is disabled. The page will state which artifacts are missing and give the exact Kaggle export/install steps. It will not substitute sample probabilities.

Patient values will stay in Streamlit session memory only. The application will not write them to files or analytics logs.

### 4.3 Quantum Lab

This page will explain and display:

- the mutual-information-selected Framingham variables;
- feature scaling and angle encoding;
- four-qubit circuit structure;
- fidelity-kernel calculation;
- reference-state/kernel flow; and
- simulator and hardware limitations.

The page will distinguish genuine quantum-circuit simulation from physical quantum-hardware execution and computational quantum advantage.

### 4.4 Model Arena

This page will separate:

- practical full-data comparisons;
- matched-row and matched-feature controls when available;
- threshold-dependent metrics from threshold-independent metrics;
- uncertainty evidence; and
- runtime context.

The currently supplied Framingham notebook evidence will be labelled as one executed split. It will not be presented as a matched quantum-advantage experiment unless a matching evidence artifact is installed.

### 4.5 Breast Cancer Evidence

This page will load the existing authoritative repository artifacts and show:

- the matched-data judge summary figure;
- the complete classical/quantum metric table;
- Quantum Kernel versus Random Forest metric deltas;
- false-negative comparison; and
- the boundary that RBF SVM remains strongest overall.

No breast-cancer patient-risk calculator is required for this version.

### 4.6 Quantum Utility Engine

The engine will compute deterministic, explainable verdicts from evidence records rather than hard-coded promotional wording.

For each task it will report:

- best classical reference;
- best quantum or hybrid candidate;
- relevant metric deltas;
- comparison fairness level;
- uncertainty availability;
- backend type; and
- one of `CLASSICAL-PREFERRED`, `HYBRID-CANDIDATE`, `QUANTUM-COMPETITIVE`, or `INSUFFICIENT-EVIDENCE`.

Current expected decisions are:

- Framingham: `CLASSICAL-PREFERRED` because Logistic Regression has AUROC 0.7345, versus 0.7062 for the hybrid and 0.6581 for the Quantum Kernel in the supplied notebook.
- Breast cancer: `QUANTUM-COMPETITIVE / SELECTIVE BENEFIT` because the matched Quantum Kernel improves sensitivity, F1, and ROC-AUC against Random Forest while RBF SVM remains strongest overall.

### 4.7 Provenance and Safety

The page will show dataset/task identity, split details, feature order, selected quantum features, model and feature-map versions, library versions, backend, timestamps, integrity status, metrics source, limitations, and downloadable evidence.

## 5. Architecture

`app.py` remains the Streamlit entry point but delegates domain logic to a new `src/demo/` package:

- `src/demo/contracts.py` — typed artifact/evidence contracts and domain errors.
- `src/demo/artifacts.py` — trusted local artifact discovery, checksum verification, manifest/schema validation, and compatibility checks.
- `src/demo/framingham.py` — patient schema, validation, notebook-parity preprocessing, feature-map/state generation, kernel calculation, classical/quantum/hybrid inference, and Logistic Regression contributions.
- `src/demo/evidence.py` — Framingham evidence and existing breast-cancer artifact loading.
- `src/demo/utility.py` — deterministic utility verdicts and metric deltas.
- `src/demo/pages.py` — Streamlit page composition.
- `src/demo/styles.py` — contained visual theme/CSS.

Supporting files will include:

- `results/framingham/notebook_metrics.json` — traceable metrics extracted from the supplied executed notebook, with source and limitation metadata.
- `artifacts/framingham/README.md` — exact artifact installation and trust requirements.
- `kaggle/framingham_artifact_export.py` — corrected export logic for Kaggle.

The existing breast-cancer experiment code and authoritative `results/final_results.json` remain unchanged except for adapters needed by the combined UI.

## 6. Frozen Artifact Contract

The Framingham artifact directory will require:

- `model_manifest.json`;
- `classical_pipeline.joblib`;
- `quantum_bundle.joblib`;
- `hybrid_bundle.joblib`; and
- `frozen_notebook_metrics.json`.

The manifest will include:

- schema version and model version;
- exact feature order and input schema;
- preprocessing and selected-feature specification;
- complete feature-map parameters;
- qubit and reference-state counts;
- dataset filename, shape, column schema, and SHA-256 when available;
- notebook/run identity;
- Python and package versions;
- model-evaluation metrics;
- calibration status;
- backend disclosure; and
- SHA-256 for every loadable artifact.

Only files beneath the configured local artifact directory will be loaded. `joblib` files are trusted-code artifacts and will be loaded only after filename, path, size, and SHA-256 checks pass. The UI will explicitly state that arbitrary uploaded pickle/joblib files are unsafe and unsupported.

## 7. Framingham Inference Flow

The inference service will:

1. Validate all patient inputs and construct the exact manifest-declared feature order.
2. Apply the frozen training preprocessing.
3. Produce the primary classical score from the frozen Logistic Regression.
4. Select the manifest-declared quantum indices.
5. Apply notebook-parity clipping to `[-3, 3]` and multiply by `pi / 3`.
6. Recreate the manifest-declared `H -> RY(x) -> RZ(0.5x) -> ring-CZ -> RY(x^2/pi)` state map.
7. Calculate `|<patient_state|training_state>|^2` against the frozen reference states.
8. Produce the Quantum Kernel SVM score.
9. Combine the frozen classical and quantum base scores through the frozen meta-model for the hybrid score.
10. Compute classical model contributions as transformed value times coefficient when the artifact supports them.

Any mismatch in feature order, feature-map version, dimensions, model version, or package compatibility will stop inference with a faculty-readable error.

## 8. Probability and Medical-Safety Policy

The current primary Logistic Regression was trained with balanced class weights and has no demonstrated calibration. Its `predict_proba` output will therefore be labelled **research-model score**, not a validated 10-year event probability.

Low/elevated/high percentage risk bands will appear only if the manifest records a calibration method, validation evidence, and an approved band policy. Otherwise the interface will show continuous model scores and model-to-model comparison without clinical categories.

The interface will distinguish:

- model score;
- display category;
- screening threshold; and
- diagnosis.

No page will recommend treatment or claim diagnosis.

## 9. Evidence Integrity

Framingham metrics in the supplied notebook and metrics described in the earlier prototype report are different experiments. The application will initially use only the metrics present in the supplied executed notebook. Experiment H/I values will not be shown until their originating evidence file is provided and validated.

The application will identify limitations including:

- public preprocessed dataset provenance;
- one held-out split for quantum/hybrid results;
- unmatched training size and feature count in the basic comparison;
- preprocessing/feature selection outside stacking folds;
- simulator-only quantum execution;
- no external validation; and
- no validated probability calibration or clinical threshold.

## 10. Error Handling

Errors are categorized as:

- artifacts absent;
- integrity failure;
- schema/version incompatibility;
- invalid patient input;
- inference dimension mismatch;
- simulator/model execution failure; and
- evidence unavailable.

The UI will display concise recovery guidance while preserving diagnostic detail for local logs. It will never catch an inference failure and replace it with a fabricated result.

## 11. Testing

Tests will be added for:

- manifest parsing and required-field validation;
- path traversal and checksum rejection;
- missing-artifact degraded mode;
- input order, type, binary value, and numeric range checks;
- quantum feature-map and fidelity-kernel parity against fixed vectors;
- classical, quantum, and hybrid inference composition using test-only fixtures;
- contribution calculations;
- utility-engine verdicts and metric deltas;
- Framingham evidence provenance;
- preservation of breast-cancer artifact loading; and
- import/startup smoke behavior without Framingham artifacts.

Test fixtures may use small test doubles, but no synthetic trained model will be packaged as a production/demo artifact.

## 12. Out of Scope

This version will not:

- claim clinical validation or regulatory readiness;
- claim universal or computational quantum advantage;
- run on physical quantum hardware;
- upload arbitrary user-provided joblib files;
- persist patient inputs;
- retrain models inside Streamlit; or
- invent Framingham prediction outputs when artifacts are absent.

## 13. Delivery and Demonstration Flow

The repository will run immediately in evidence/degraded mode using its committed results. After the corrected Kaggle exporter is run, the user installs the resulting verified artifact files under `artifacts/framingham/`, restarts Streamlit, and the Patient Risk page changes to real-inference mode.

The recommended faculty walkthrough is:

1. Command Center and problem-statement alignment.
2. Framingham patient workflow and artifact/provenance status.
3. Quantum Lab circuit and kernel evidence.
4. Framingham `CLASSICAL-PREFERRED` Utility Engine verdict.
5. Breast-cancer matched comparison and selective quantum benefit.
6. Final explanation: Q-MedAI chooses the evidence-supported strategy per disease task.
