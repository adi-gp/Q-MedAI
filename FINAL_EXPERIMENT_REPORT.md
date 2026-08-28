# Q-MedAI FINAL EXPERIMENT REPORT

## Objective

Q-MedAI experimentally compares classical and quantum machine-learning approaches under a controlled methodology. This FINAL EXPERIMENT is distinct from the prior **LOCAL BASELINE**, which used a 2-layer/35-iteration VQC and a 50-row quantum-kernel subset and was not rerun.

## Dataset, split, target, and leakage prevention

Dataset source: `local CSV (data/breast_cancer.csv)`. The supplied project files do not independently establish external provenance beyond the CSV schema, so provenance is reported as: **Source provenance not independently verified from supplied project files**. The dataset has 569 rows and 30 numeric features after documented removals. Target encoding is B = 0 (benign, negative) and M = 1 (malignant, positive). A single stratified 20% train/test split used `random_state=42`.

Imputation, scaling, and SelectKBest(`f_classif`) were fit only on training rows. The held-out test set was never used to fit preprocessing. The final configuration used SelectKBest with 4 selected features/qubits.

## Quantum configuration and backend

VQC: 4 qubits, 3 layers, 100 optimization iterations. Quantum kernel: 150 stratified training rows. Each quantum component had a cooperative 600 second timeout. Backend: PennyLane default.qubit — classical quantum-circuit simulation.

## FINAL EXPERIMENT — full-data results

Classical models and the VQC used all 455 full-training rows. The quantum-kernel model was constrained to 150 rows, so its full-data row is not an apples-to-apples training-size comparison with full-data classical models.

| Model | Type | Status | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Training time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Classical | COMPLETED | 0.9561 | 0.9512 | 0.9286 | 0.9722 | 0.9398 | 0.9954 | 0.003 |
| RBF SVM | Classical | COMPLETED | 0.9386 | 0.9730 | 0.8571 | 0.9861 | 0.9114 | 0.9954 | 0.004 |
| Random Forest | Classical | COMPLETED | 0.9386 | 0.9487 | 0.8810 | 0.9722 | 0.9136 | 0.9922 | 0.089 |
| VQC | Quantum | TIMED OUT | — | — | — | — | — | — | 605.106 |
| Quantum Kernel SVM | Quantum | COMPLETED | 0.9474 | 0.9500 | 0.9048 | 0.9722 | 0.9268 | 0.9940 | 4.845 |

## MATCHED-DATA COMPARISON — 150 training samples for all models

The exact same stratified 150 original training rows and unchanged held-out test set were used for all four models. A separate preprocessing pipeline was fit on those 150 rows only before transforming the test set.

| Model | Type | Status | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Training time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Classical | COMPLETED | 0.9561 | 0.9512 | 0.9286 | 0.9722 | 0.9398 | 0.9960 | 0.012 |
| RBF SVM | Classical | COMPLETED | 0.9649 | 0.9750 | 0.9286 | 0.9861 | 0.9512 | 0.9960 | 0.002 |
| Random Forest | Classical | COMPLETED | 0.9474 | 0.9737 | 0.8810 | 0.9861 | 0.9250 | 0.9907 | 0.085 |
| Quantum Kernel SVM | Quantum | COMPLETED | 0.9474 | 0.9500 | 0.9048 | 0.9722 | 0.9268 | 0.9940 | 4.799 |

![Fair matched classical-vs-quantum comparison](results/plots/matched_classical_quantum_judge_summary.png)

## Observed performance difference

Full-data reference: Quantum Kernel SVM minus Logistic Regression: ROC-AUC -0.0013; accuracy -0.0088; F1 -0.0129.

Matched-data fair comparison: Quantum Kernel SVM minus Logistic Regression: ROC-AUC -0.0020; accuracy -0.0088; F1 -0.0129.

These are observed differences from one split, not quantum advantage. Statistical confidence intervals were not computed for this run; results reflect a single train/test split.

Compared specifically with Random Forest in the fair matched condition, the Quantum Kernel tied accuracy and improved sensitivity by 0.0238, F1 by 0.0018, and ROC-AUC by 0.0033. RBF SVM remained the strongest overall model.

## Computational cost, interpretation, and limitations

Quantum-kernel timing details are saved in `results/final_results.json`; VQC and kernel training times are shown above. This feasibility/prototype study on one supplied dataset is not evidence that quantum ML is generally superior or inferior for medical diagnosis. PennyLane `default.qubit` is a classical simulator, not physical quantum hardware. The study has one dataset, one split, limited qubit count, and no clinical validation.

The most important next evidence is repeated-seed or nested cross-validation with paired confidence intervals and an external biomedical dataset. The current cross-sectional dataset evaluates malignant-class detection and cannot establish earlier-in-time diagnosis.

## Conclusion

Q-MedAI provides a leakage-safe, controlled classical-plus-quantum comparison with a matched-data fairness condition. It does not establish quantum advantage merely by predictive accuracy.
