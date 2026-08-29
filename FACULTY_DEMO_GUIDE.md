# Q-MedAI Faculty Demonstration

**Framingham = Primary Prospective Clinical Workflow**  
**Breast Cancer = Separate Quantum Evidence Benchmark**

Q-MedAI is a research prototype, not a medical diagnostic device. It compares classical, quantum, and hybrid research pathways per task instead of forcing a quantum winner.

## Demonstration claim boundaries

- The bundled Framingham artifacts are READY and identify themselves as `qmedai-framingham-b98c306b5b44`.
- Framingham outputs are **uncalibrated research-model scores**, not validated 10-year event probabilities, clinical risk categories, diagnoses, or treatment recommendations.
- The current Framingham verdict is **`CLASSICAL-PREFERRED`**: Logistic Regression has the strongest supplied ROC-AUC evidence (0.7345), ahead of the hybrid (0.7062) and Quantum Kernel SVM (0.6581). The Framingham comparison also uses unmatched resource budgets, so it cannot establish quantum advantage.
- The matched Breast Cancer verdict is **`QUANTUM-COMPETITIVE` with a selective sensitivity benefit** against Random Forest. The Quantum Kernel SVM has one fewer false negative and sensitivity higher by 0.0238 on the saved split; **RBF SVM remains strongest overall**.
- PennyLane `default.qubit` is a **classical quantum-circuit simulator**, not physical quantum hardware and not evidence of computational speedup.
- Neither task is externally clinically validated. Results from one dataset and one split do not establish universal quantum advantage.

## Launch the bundled READY-mode demo

From this checked-out worktree:

```bash
cd /Users/mymac/Desktop/Q-MedAI/.worktrees/combined-faculty-demo
/Users/mymac/Desktop/Q-MedAI/.venv/bin/streamlit run app.py --server.port 8511 --browser.gatherUsageStats false
```

Open <http://localhost:8511> if the browser does not open automatically. Keep that browser tab open throughout the walkthrough so Patient Risk and Quantum Lab share the same Streamlit session.

For a fresh clone or extracted faculty package:

```bash
cd /path/to/Q-MedAI
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/test_framingham_builder.py::test_committed_real_artifacts_are_ready_and_score_distinct_patients -q
.venv/bin/streamlit run app.py --server.port 8511 --browser.gatherUsageStats false
```

The focused test verifies checksum-protected artifact loading and distinct, finite research scores before the presentation.

## Six-minute walkthrough

### 0:00–0:45 — Command Center

1. Start on **Command Center**.
2. Introduce the two evidence roles: Framingham is the primary prospective 10-year CHD workflow; Breast Cancer is the separate matched diagnostic benchmark.
3. Point to the READY artifact gate and the two task-specific verdict cards.
4. Say: “Q-MedAI chooses the architecture supported by the evidence for each task; it does not assume quantum must win.”

### 0:45–2:15 — Patient Risk

1. Select **Patient Risk** in the sidebar.
2. Note the 15-field frozen Framingham contract and the privacy notice: entered values stay in the current Streamlit session and are not persisted.
3. Use the pre-filled valid demonstration values or adjust values within the displayed ranges.
4. Select **Run verified inference**.
5. Show the primary Classical score, the Quantum and Hybrid research branches, `CLASSICAL-PREFERRED`, and the Logistic Regression contribution table.
6. Use the exact wording: “These are uncalibrated research-model scores, not clinical probabilities or diagnoses.”

### 2:15–3:45 — Quantum Lab, same-session patient trace

1. Without refreshing or opening a new browser session, select **Quantum Lab** immediately after Patient Risk.
2. Point out the same patient’s four selected quantum features: `age`, `sysBP`, `prevalentHyp`, and `diaBP`.
3. Follow the patient-specific table from raw value → frozen transformed value → encoded angle.
4. Show the exact four-qubit circuit sequence: `H → RY(θ) → RZ(0.5θ) → ring-CZ → RY(θ²/π)`.
5. Show the fidelity-kernel summary, frozen-reference similarities, and the same session’s Quantum research score.
6. State that `default.qubit` classically simulates genuine quantum-circuit semantics; it is not quantum hardware or a speedup claim.

### 3:45–4:30 — Model Arena

1. Select **Model Arena**.
2. Show the full Framingham evidence table and `CLASSICAL-PREFERRED` verdict.
3. Explain the visible fairness warning: quantum and hybrid models used 500 rows/four features, whereas full-data classical models used 3,306 rows/15 features.
4. Emphasize that the result is an honest task-level architecture choice, not a negative or positive universal verdict on quantum ML.

### 4:30–5:30 — Breast Cancer Evidence

1. Select **Breast Cancer Evidence**.
2. Show the matched 150-row judge summary and metric table; all matched models use the same 114 held-out rows.
3. Compare Quantum Kernel SVM with Random Forest: equal accuracy, sensitivity +0.0238, one fewer false negative, F1 +0.0018, and ROC-AUC +0.0033.
4. State the complete verdict: “Quantum is competitive with a selective sensitivity benefit against Random Forest on this split, while RBF SVM remains strongest overall.”
5. Point out that every authoritative plot and the saved CSV/JSON/configuration/summary are downloadable.

### 5:30–6:00 — Quantum Utility and safety close

1. Select **Quantum Utility**.
2. Contrast `CLASSICAL-PREFERRED` for Framingham with `QUANTUM-COMPETITIVE` for the matched Breast Cancer benchmark.
3. Close with: “The platform reports task-specific utility. It makes no claim of clinical validation, medical-device status, physical quantum execution, or universal quantum advantage.”

## Page map

| Page | Faculty purpose |
|---|---|
| Command Center | Two-task product story, readiness, and current verdicts |
| Patient Risk | Verified Framingham inference using the bundled frozen artifacts |
| Quantum Lab | Same-session patient encoding, circuit, fidelity trace, and simulator disclosure |
| Model Arena | Full Framingham evidence, fairness warning, thresholds, and limitations |
| Breast Cancer Evidence | Matched benchmark, selective quantum benefit, complete plots, and downloads |
| Quantum Utility | Deterministic task-specific verdict comparison |
| Provenance & Safety | Artifact manifest, evidence sources, limitations, and trust boundary |
| Research Runner | Optional session-only exploratory experiments; never overwrites authoritative artifacts |

## Bundled artifact readiness and trust

The repository already contains the five canonical Framingham files under `artifacts/framingham/`:

- `model_manifest.json`
- `classical_pipeline.joblib`
- `quantum_bundle.joblib`
- `hybrid_bundle.joblib`
- `frozen_notebook_metrics.json`

At startup, Q-MedAI checks canonical filenames, SHA-256 values, manifest structure, model identity, and compatible runtime versions before deserializing the joblib bundles. Do not replace these files with downloaded or untrusted pickle/joblib data.

To verify READY mode:

```bash
.venv/bin/python -m pytest tests/test_framingham_builder.py::test_committed_real_artifacts_are_ready_and_score_distinct_patients -q
```

To deterministically rebuild from the exact public `CHD_preprocessed.csv` rather than use the bundled files:

```bash
.venv/bin/python -m scripts.build_framingham_artifacts \
  --data /absolute/path/to/CHD_preprocessed.csv \
  --output-dir artifacts/framingham
```

The builder reproduces the executed notebook procedure and passes already-fitted objects to `kaggle/framingham_artifact_export.py`. Review the new manifest/model ID and run the focused test again before presenting. The raw CSV is not included in the repository.

## If READY mode is not available

If any canonical file is absent, modified, incompatible, or fails validation, Q-MedAI stays in safe evidence mode. The full form and evidence remain visible, but **Run verified inference** is disabled and the app produces **no patient score**. Never bypass this gate or substitute a sample probability.

## Final faculty Q&A

- **Is the Framingham number a 10-year probability?** No. Calibration is `NOT_DEMONSTRATED`; it is an uncalibrated research-model score.
- **Why is the classical output primary?** It is the evidence-supported Framingham pathway under the supplied results (`CLASSICAL-PREFERRED`).
- **Where is quantum useful here?** The Breast Cancer Quantum Kernel is competitive and selectively improves sensitivity versus Random Forest on the matched split; RBF SVM is still strongest overall.
- **Is this running on quantum hardware?** No. PennyLane `default.qubit` is a classical simulator.
- **Does the project prove quantum advantage?** No. It reports bounded, task-specific observations without a universal or computational advantage claim.
- **Can these outputs guide care?** No. The platform has no external clinical validation and is not a medical diagnostic device.
