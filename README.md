# Q-MedAI

Q-MedAI is an SIH 2026 (Problem Statement 26139) hybrid quantum-classical research platform for binary disease-classification experiments. It compares established classical ML baselines with quantum ML under one leakage-safe methodology.

This software is a research prototype and is not a medical diagnostic device.

This project does not establish quantum advantage merely by achieving high predictive accuracy. Quantum advantage would require a rigorous comparison under appropriate computational and statistical conditions.

## FINAL EXPERIMENT

The authoritative executed artifacts are in [`results/`](results/): [JSON](results/final_results.json), [CSV](results/final_results.csv), [exact configuration](results/final_config.json), and presentation plots. [FINAL_EXPERIMENT_REPORT.md](FINAL_EXPERIMENT_REPORT.md) and [SIH_FINAL_SUMMARY.md](SIH_FINAL_SUMMARY.md) are generated from the saved final artifact.

The final methodology uses:

- supplied `data/breast_cancer.csv`; its external source provenance was not independently verified from the supplied project files;
- B = 0 (benign / negative) and M = 1 (malignant / positive);
- stratified 80/20 train/test split, `random_state=42`;
- train-only median imputation, scaling, and `SelectKBest(f_classif)` with four features;
- full-data Logistic Regression, RBF SVM, Random Forest, and a 4-qubit VQC (3 layers, 100 iterations);
- 150-row stratified fidelity quantum-kernel SVM; and
- a separate matched-data comparison where all four compared models use the exact same 150 raw training rows and separately fitted subset-only preprocessing.

Quantum circuits use PennyLane `default.qubit`, a classical simulator—not a physical quantum computer. The final VQC has a cooperative 600-second limit; statuses and missing metrics are preserved exactly in the artifacts.

### LOCAL BASELINE

The earlier verified local baseline (2-layer/35-iteration VQC and 50-row quantum kernel) is preserved as historical context only. It was not rerun and is not the FINAL EXPERIMENT. Its 50-row kernel result had a training-size mismatch with full-data classical models, which the final matched-data study explicitly addresses.

## Architecture

```text
CSV → validation and B/M encoding → stratified split
    → train-only preprocessing → classical / VQC / quantum-kernel models
    → held-out metrics, saved plots, final artifacts, Streamlit demo
```

The code in `src/` owns data validation, preprocessing, models, metrics, explainability, the generic exploratory runner, and the one-shot final artifact runner. `app.py` presents saved final artifacts in Demo Mode and only trains under **Run New Experiment**.

## QUICK START

```bash
cd /Users/mymac/Desktop/Q-MedAI
source .venv/bin/activate
streamlit run app.py
```

This command starts the Q-MedAI web application in your browser. If the browser does not open, use the local URL printed in the terminal, normally <http://localhost:8501>.

The default landing view is **DEMO / FINAL RESULTS** and does not retrain models. It shows saved full-data and matched-data results, actual plots, the final configuration, and downloadable artifacts.

## Installation and verified environment

Tested with Python **3.12.12**.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Verified direct dependency versions: `numpy==2.5.2`, `pandas==3.0.5`, `scikit-learn==1.9.0`, `pennylane==0.45.1`, `streamlit==1.62.0`, `matplotlib==3.11.1`, and `pytest==9.1.1`.

## Kaggle beginner instructions

1. Open Kaggle and create or open a Notebook.
2. Upload/add the complete Q-MedAI project directory and `breast_cancer.csv` as Kaggle inputs.
3. Open `kaggle/Q_MedAI_Final_Experiment.ipynb` and run the dataset-discovery cell; it prints the exact path used and fails clearly if the CSV is missing.
4. Enable an accelerator/GPU only if supported by the notebook/backend. PennyLane `default.qubit` does not automatically use a GPU.
5. Run all cells once. The notebook prints full-data and matched-data tables and saves actual results/plots under Kaggle working storage.
6. Review the final tables and plots. Kaggle execution was **not** performed during this local Codex run.

## Limitations and future work

This is a feasibility study on one supplied dataset and one split, without clinical validation, confidence intervals, quantum hardware, hyperparameter sweeps, or a quantum-advantage claim. Future work can add more datasets/splits, statistical analysis, larger carefully controlled circuits/kernels, appropriately supported GPU-accelerated simulation, and eventual hardware studies.

## Project structure

```text
app.py                         Streamlit final-results demo and exploratory view
src/final_experiment.py        One-shot final study and artifact generation
results/                       Authoritative final JSON/CSV/config and PNG plots
kaggle/Q_MedAI_Final_Experiment.ipynb
FINAL_EXPERIMENT_REPORT.md     Artifact-derived technical report
SIH_FINAL_SUMMARY.md           Artifact-derived SIH talking points and Q&A
tests/                         Lightweight validation, leakage, model, and metric tests
```
