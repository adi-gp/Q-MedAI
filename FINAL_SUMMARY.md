# Q-MedAI — SIH FINAL SUMMARY

## Problem and solution

Early disease-classification workflows can benefit from research ML assistance. Q-MedAI is a hybrid platform that compares strong classical baselines with quantum ML under one controlled, leakage-safe methodology. It is research software, not a medical diagnostic device.

## Combined faculty demo

The self-contained faculty application presents two evidence roles without merging their claims:

- **Framingham is the primary prospective clinical workflow.** The real bundled frozen artifact set is READY with model ID `qmedai-framingham-b98c306b5b44`. Patient Risk produces classical, quantum, and hybrid **uncalibrated research-model scores**; these are not validated 10-year probabilities, diagnoses, risk categories, or treatment recommendations. The current utility verdict is **`CLASSICAL-PREFERRED`** because the supplied Framingham evidence favors Logistic Regression.
- **Breast Cancer is a separate quantum evidence benchmark.** On the matched 150-row experiment, the utility verdict is **`QUANTUM-COMPETITIVE` with a selective sensitivity benefit against Random Forest**: the Quantum Kernel has sensitivity higher by 0.0238 and one fewer false negative. **RBF SVM remains strongest overall.**

PennyLane `default.qubit` is a classical quantum-circuit simulator, not physical quantum hardware or evidence of computational speedup. Neither task is externally clinically validated, and neither result establishes universal quantum advantage.

Launch from the current worktree:

```bash
cd /Users/mymac/Desktop/Q-MedAI/.worktrees/combined-faculty-demo
/Users/mymac/Desktop/Q-MedAI/.venv/bin/streamlit run app.py --server.port 8511 --browser.gatherUsageStats false
```

Use the timed [Faculty Demo Guide](FACULTY_DEMO_GUIDE.md): run verified inference on **Patient Risk**, then open **Quantum Lab in the same browser session** to show that patient’s frozen transformation, encoded angles, four-qubit circuit, fidelity-kernel response, and Quantum research score.

## Architecture and methodology

Local CSV → validation (B=0, M=1) → stratified split (seed 42) → train-only imputation/scaling/SelectKBest(`f_classif`) → models → held-out metrics and plots. Classical models are Logistic Regression, RBF SVM, and Random Forest. Quantum models are a 4-qubit VQC and a fidelity quantum-kernel SVM, both simulated with PennyLane `default.qubit`.

## Why the matched-data comparison matters

The earlier **LOCAL BASELINE** used only 50 kernel-training rows while classical models used the complete training set, so it was not a fair head-to-head comparison. The **FINAL EXPERIMENT** uses 150 kernel rows, then tests all matched models on the exact same 150 rows with a separate training-only preprocessing pipeline and the same held-out test set.

## FINAL EXPERIMENT — full-data results

| Model | Type | Status | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Training time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Classical | COMPLETED | 0.9561 | 0.9512 | 0.9286 | 0.9722 | 0.9398 | 0.9954 | 0.003 |
| RBF SVM | Classical | COMPLETED | 0.9386 | 0.9730 | 0.8571 | 0.9861 | 0.9114 | 0.9954 | 0.004 |
| Random Forest | Classical | COMPLETED | 0.9386 | 0.9487 | 0.8810 | 0.9722 | 0.9136 | 0.9922 | 0.089 |
| VQC | Quantum | TIMED OUT | — | — | — | — | — | — | 605.106 |
| Quantum Kernel SVM | Quantum | COMPLETED | 0.9474 | 0.9500 | 0.9048 | 0.9722 | 0.9268 | 0.9940 | 4.845 |

## MATCHED-DATA COMPARISON — 150 rows for all models

| Model | Type | Status | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Training time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Classical | COMPLETED | 0.9561 | 0.9512 | 0.9286 | 0.9722 | 0.9398 | 0.9960 | 0.012 |
| RBF SVM | Classical | COMPLETED | 0.9649 | 0.9750 | 0.9286 | 0.9861 | 0.9512 | 0.9960 | 0.002 |
| Random Forest | Classical | COMPLETED | 0.9474 | 0.9737 | 0.8810 | 0.9861 | 0.9250 | 0.9907 | 0.085 |
| Quantum Kernel SVM | Quantum | COMPLETED | 0.9474 | 0.9500 | 0.9048 | 0.9722 | 0.9268 | 0.9940 | 4.799 |

![Fair matched classical-vs-quantum comparison](results/plots/matched_classical_quantum_judge_summary.png)

## What quantum contributed

Full-data context: Quantum Kernel SVM minus Logistic Regression: ROC-AUC -0.0013; accuracy -0.0088; F1 -0.0129. The kernel used fewer training rows than full-data classical models, so this is contextual only.

Fair matched-data result: Quantum Kernel SVM minus Logistic Regression: ROC-AUC -0.0020; accuracy -0.0088; F1 -0.0129. This is an observed performance difference, not quantum advantage.

Compared specifically with Random Forest on the fair matched data, the Quantum Kernel tied accuracy and improved sensitivity by 0.0238, F1 by 0.0018, and ROC-AUC by 0.0033. RBF SVM remained strongest overall.

## Limitations and future work

One supplied dataset and one train/test split are not clinical validation or general evidence. The quantum models run on a classical simulator and may be computationally slower. Future work: more datasets and splits, confidence intervals, larger carefully controlled circuits/kernels, appropriate GPU-accelerated simulation where supported, and eventual hardware studies.

The most important next step is repeated-seed or nested cross-validation with paired confidence intervals, followed by external biomedical validation. Another graph cannot replace that evidence. Because the current dataset is cross-sectional, it supports malignant-class detection research but does not prove earlier-in-time diagnosis.

## 30-second answer: Why quantum?

“We are not assuming quantum is better. Q-MedAI gives us a controlled way to compare quantum and classical approaches on the same held-out data. The project’s value is the fair methodology, including a matched 150-sample comparison for the quantum kernel, and a reusable platform for future experiments.”

## 20-second answer: Classical model has better accuracy—why use quantum?

“That is a valid outcome. This experiment tests the question rather than assuming a winner. On this split, we report the actual classical and quantum scores openly; the prototype demonstrates a controlled evaluation framework, not a claim of quantum superiority.”

## 20-second answer: Was the old 90% quantum-kernel result just less data?

“The old 50-sample kernel result had a training-size mismatch with full-data classical models, so it was not a fair head-to-head claim. The final study explicitly fixes that with the same 150 training rows, separate subset-only preprocessing, and the same held-out test set for all compared models.”

## Judge Q&A

- **Why only 4 qubits?** Four selected features map directly to four qubits and keep local simulation feasible.
- **Why this dataset?** It is the supplied binary breast-cancer CSV; its external provenance was not independently verified from project files.
- **Why only 150 kernel rows?** Fidelity-kernel matrices scale quadratically in training rows; 150 is the configured controlled limit.
- **Is that unfair?** It would be if compared directly to full-data models, which is why the matched-data comparison uses exactly those 150 rows for every model.
- **Is quantum slower?** Training times are reported from the run; `default.qubit` is classical simulation and not a speed claim.
- **Are you using a quantum computer?** No. PennyLane `default.qubit` simulates circuits classically.
- **Where is the quantum advantage?** This project makes no quantum-advantage claim. A higher score on one split would only be an observed performance difference.
- **How is leakage prevented?** The split happens first; preprocessing is fitted only on training rows, including a separate pipeline for the matched subset.
- **Is this clinically validated?** No. It is a research prototype and not a medical diagnostic device.
