"""Reproduce the executed Framingham training procedure and freeze artifacts.

The module is import-safe. Training starts only through :func:`build_artifacts`
or the command-line entry point. The raw public CSV is read in place and is
never copied into the output artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kaggle.framingham_artifact_export import export_from_namespace
from src.demo.framingham import PATIENT_FIELDS, encode_quantum_features, feature_state


SEED = 42
TARGET = "TenYearCHD"
EXPECTED_SELECTED_FEATURES = ["age", "sysBP", "prevalentHyp", "diaBP"]
KAGGLE_SLUG = "captainozlem/framingham-chd-preprocessed-data"
KAGGLE_FILE = "CHD_preprocessed.csv"
CANONICAL_FILES = (
    "model_manifest.json",
    "classical_pipeline.joblib",
    "quantum_bundle.joblib",
    "hybrid_bundle.joblib",
    "frozen_notebook_metrics.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_matrix(encoded: np.ndarray) -> np.ndarray:
    return np.asarray([feature_state(row) for row in encoded], dtype=complex)


def _kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left @ right.conj().T) ** 2


def _metrics_row(name: str, y_true: pd.Series, probabilities: np.ndarray, seconds: float) -> dict[str, Any]:
    predicted = (np.asarray(probabilities) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predicted),
        "Balanced Acc": balanced_accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Sensitivity": recall_score(y_true, predicted, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "F1": f1_score(y_true, predicted, zero_division=0),
        "AUROC": roc_auc_score(y_true, probabilities),
        "AUPRC": average_precision_score(y_true, probabilities),
        "Time_s": seconds,
    }


def _authoritative_summary(project_root: Path) -> pd.DataFrame:
    payload = json.loads((project_root / "results/framingham/notebook_metrics.json").read_text(encoding="utf-8"))
    field_names = {
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "f1": "F1",
        "roc_auc": "AUROC",
        "auprc": "AUPRC",
        "training_seconds": "Time_s",
    }
    return pd.DataFrame(
        [{field_names[key]: value for key, value in row.items() if key in field_names} for row in payload["metrics"]]
    )


def _load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not data_path.is_file() or data_path.name != KAGGLE_FILE:
        raise FileNotFoundError(f"Expected the exact public Kaggle file named {KAGGLE_FILE}: {data_path}")
    frame = pd.read_csv(data_path).rename(columns={"male": "sex_male", "Male": "sex_male"})
    frame = frame.drop(columns=[name for name in frame if name.lower() in {"id", "patientid", "patient_id"}])
    if TARGET not in frame:
        raise ValueError(f"Dataset must contain target column {TARGET}.")
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    valid = target.notna()
    features = frame.loc[valid].drop(columns=[TARGET]).apply(pd.to_numeric, errors="coerce")
    features = features.dropna(axis=1, how="all")
    expected_order = [field.name for field in PATIENT_FIELDS]
    if list(features.columns) != expected_order:
        raise ValueError(
            "Dataset feature order must match the exact Framingham patient contract: " + ", ".join(expected_order)
        )
    return features, target.loc[valid].astype(int)


def _fit_namespace(data_path: Path, project_root: Path) -> dict[str, Any]:
    np.random.seed(SEED)
    random.seed(SEED)
    features, target = _load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, stratify=target, random_state=SEED
    )
    preprocess = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    Xtr = preprocess.fit_transform(X_train)
    Xte = preprocess.transform(X_test)
    feature_names = X_train.columns.to_numpy()
    mutual_information = mutual_info_classif(Xtr, y_train, random_state=SEED)
    ranking = pd.DataFrame(
        {"feature": feature_names, "mutual_information": mutual_information}
    ).sort_values("mutual_information", ascending=False)
    selected = ranking.head(4)["feature"].tolist()
    if selected != EXPECTED_SELECTED_FEATURES:
        raise RuntimeError(
            f"Mutual-information selection drifted: expected {EXPECTED_SELECTED_FEATURES}, got {selected}. Export stopped."
        )
    selected_indices = [list(feature_names).index(name) for name in selected]
    Xq_train = encode_quantum_features(Xtr, selected_indices)
    Xq_test = encode_quantum_features(Xte, selected_indices)

    classical_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=SEED
        ),
        "RBF-SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            min_samples_leaf=3,
            random_state=SEED,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15, random_state=SEED
        ),
    }
    locally_measured_rows: list[dict[str, Any]] = []
    for name, model in classical_models.items():
        started = time.perf_counter()
        model.fit(Xtr, y_train)
        probabilities = model.predict_proba(Xte)[:, 1]
        locally_measured_rows.append(_metrics_row(name, y_test, probabilities, time.perf_counter() - started))

    quantum_indices, _ = train_test_split(
        np.arange(len(Xq_train)), train_size=500, stratify=y_train, random_state=SEED
    )
    Xq_subset = Xq_train[quantum_indices]
    yq_subset = y_train.iloc[quantum_indices].reset_index(drop=True)
    S_train = _state_matrix(Xq_subset)
    K_train = _kernel(S_train, S_train)
    qsvc = SVC(kernel="precomputed", probability=True, class_weight="balanced", random_state=SEED)
    qsvc.fit(K_train, yq_subset)

    Xc_hybrid = Xtr[quantum_indices]
    Xq_hybrid = Xq_train[quantum_indices]
    y_hybrid = y_train.iloc[quantum_indices].reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    classical_oof = np.zeros(len(y_hybrid))
    quantum_oof = np.zeros(len(y_hybrid))
    for fold, (training_indices, validation_indices) in enumerate(folds.split(Xc_hybrid, y_hybrid), 1):
        classical_fold = SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=SEED + fold
        )
        classical_fold.fit(Xc_hybrid[training_indices], y_hybrid.iloc[training_indices])
        classical_oof[validation_indices] = classical_fold.predict_proba(Xc_hybrid[validation_indices])[:, 1]

        train_states = _state_matrix(Xq_hybrid[training_indices])
        validation_states = _state_matrix(Xq_hybrid[validation_indices])
        quantum_fold = SVC(
            kernel="precomputed", probability=True, class_weight="balanced", random_state=SEED + fold
        )
        quantum_fold.fit(_kernel(train_states, train_states), y_hybrid.iloc[training_indices])
        quantum_oof[validation_indices] = quantum_fold.predict_proba(
            _kernel(validation_states, train_states)
        )[:, 1]

    meta = LogisticRegression(class_weight="balanced", random_state=SEED)
    meta.fit(np.c_[classical_oof, quantum_oof], y_hybrid)
    c_final = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)
    c_final.fit(Xc_hybrid, y_hybrid)
    Sh = _state_matrix(Xq_hybrid)
    q_final = SVC(kernel="precomputed", probability=True, class_weight="balanced", random_state=SEED)
    q_final.fit(_kernel(Sh, Sh), y_hybrid)

    return {
        "preprocess": preprocess,
        "classical_models": classical_models,
        "feature_names": feature_names,
        "selected": selected,
        "idx": selected_indices,
        "S_train": S_train,
        "qsvc": qsvc,
        "c_final": c_final,
        "q_final": q_final,
        "Sh": Sh,
        "meta": meta,
        "summary": _authoritative_summary(project_root),
        "SEED": SEED,
        "X_train": X_train,
        "X_test": X_test,
        "TARGET": TARGET,
        "DATASET_PROVENANCE": {
            "kaggle_slug": KAGGLE_SLUG,
            "file": KAGGLE_FILE,
            "dataset_sha256": _sha256_file(data_path),
        },
        "NOTEBOOK_PROVENANCE": {
            "source": "local deterministic reproduction of the executed notebook procedure",
            "evidence_notebook": "kaggle/Framingham_CHD_Executed_Evidence.ipynb",
            "builder": "scripts.build_framingham_artifacts",
            "exporter": "kaggle.framingham_artifact_export",
        },
        "LOCAL_EVALUATION": locally_measured_rows,
    }


def build_artifacts(data_path: Path, output_directory: Path) -> Path:
    """Fit the frozen notebook-parity models and install five canonical files."""
    data_path = Path(data_path)
    output_directory = Path(output_directory)
    project_root = Path(__file__).resolve().parents[1]
    namespace = _fit_namespace(data_path, project_root)
    with tempfile.TemporaryDirectory(prefix="qmedai-framingham-build-") as temporary_directory:
        exported = export_from_namespace(namespace, Path(temporary_directory))
        output_directory.mkdir(parents=True, exist_ok=True)
        for name in CANONICAL_FILES:
            shutil.copy2(exported / name, output_directory / name)
    return output_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help=f"Path to exact {KAGGLE_FILE}")
    parser.add_argument("--output-dir", required=True, type=Path, help="Trusted artifact installation directory")
    arguments = parser.parse_args(argv)
    artifact_directory = build_artifacts(arguments.data, arguments.output_dir)
    manifest = json.loads((artifact_directory / "model_manifest.json").read_text(encoding="utf-8"))
    print(f"READY {manifest['model_version_id']} -> {artifact_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
