import base64
import json
import sys
import types
from pathlib import Path

import pandas as pd

from src import final_experiment


MATCHED_RECORDS = [
    {
        "model": "Logistic Regression",
        "type": "Classical",
        "status": "COMPLETED",
        "accuracy": 0.9561,
        "precision": 0.9512,
        "recall": 0.9286,
        "specificity": 0.9722,
        "f1": 0.9398,
        "roc_auc": 0.9960,
        "training_time_seconds": 0.01,
        "confusion_matrix": [[70, 2], [3, 39]],
    },
    {
        "model": "RBF SVM",
        "type": "Classical",
        "status": "COMPLETED",
        "accuracy": 0.9649,
        "precision": 0.9750,
        "recall": 0.9286,
        "specificity": 0.9861,
        "f1": 0.9512,
        "roc_auc": 0.9960,
        "training_time_seconds": 0.01,
        "confusion_matrix": [[71, 1], [3, 39]],
    },
    {
        "model": "Random Forest",
        "type": "Classical",
        "status": "COMPLETED",
        "accuracy": 0.9474,
        "precision": 0.9737,
        "recall": 0.8810,
        "specificity": 0.9861,
        "f1": 0.9250,
        "roc_auc": 0.9907,
        "training_time_seconds": 0.09,
        "confusion_matrix": [[71, 1], [5, 37]],
    },
    {
        "model": "Quantum Kernel SVM",
        "type": "Quantum",
        "status": "COMPLETED",
        "accuracy": 0.9474,
        "precision": 0.9500,
        "recall": 0.9048,
        "specificity": 0.9722,
        "f1": 0.9268,
        "roc_auc": 0.9940,
        "training_time_seconds": 4.80,
        "confusion_matrix": [[70, 2], [4, 38]],
    },
]


def test_judge_summary_plot_writes_a_real_png_from_matched_results(tmp_path, monkeypatch):
    monkeypatch.setattr(final_experiment, "PLOTS_DIR", tmp_path)

    output = final_experiment._save_judge_summary_plot(
        MATCHED_RECORDS,
        "judge-summary.png",
    )

    output_path = Path(output)
    assert output_path == tmp_path / "judge-summary.png"
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output_path.stat().st_size > 10_000


def test_quantum_kernel_head_to_head_reports_only_supported_random_forest_gains():
    comparison = final_experiment._quantum_kernel_vs_random_forest(MATCHED_RECORDS)

    assert comparison == {
        "accuracy_difference": 0.0,
        "sensitivity_difference": 0.0238,
        "f1_difference": 0.0018,
        "roc_auc_difference": 0.0033,
        "quantum_false_negatives": 4,
        "random_forest_false_negatives": 5,
        "fewer_malignant_cases_missed": 1,
    }


def test_kaggle_results_cell_displays_every_saved_plot_even_with_agg_backend(
    tmp_path,
    monkeypatch,
):
    notebook_path = Path("kaggle/Q_MedAI_Final_Experiment.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "FULL-DATA FINAL RESULTS" in "".join(cell["source"])
    )

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    first_plot = tmp_path / "first.png"
    second_plot = tmp_path / "second.png"
    first_plot.write_bytes(png_bytes)
    second_plot.write_bytes(png_bytes)

    displayed = []

    class FakeImage:
        def __init__(self, filename):
            self.filename = filename

    fake_display_module = types.ModuleType("IPython.display")
    fake_display_module.Image = FakeImage
    fake_display_module.display = lambda value: displayed.append(value)
    fake_ipython_module = types.ModuleType("IPython")
    fake_ipython_module.display = fake_display_module
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython_module)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display_module)

    payload = {
        "full_data_final_results": {"models": MATCHED_RECORDS},
        "matched_data_comparison": {"models": MATCHED_RECORDS},
        "analysis": {
            "matched_quantum_kernel_vs_best_classical": {"status": "AVAILABLE"},
            "statistical_confidence": "Single split; no confidence interval.",
        },
        "plots": [str(first_plot), str(second_plot)],
    }

    exec(
        compile(source, str(notebook_path), "exec"),
        {
            "final_payload": payload,
            "pd": pd,
            "Path": Path,
            "PROJECT_ROOT": tmp_path,
            "OUTPUT_DIR": tmp_path,
            "display": fake_display_module.display,
        },
    )

    displayed_images = [
        value
        for value in displayed
        if isinstance(value, FakeImage)
    ]
    assert [Path(image.filename) for image in displayed_images] == [
        first_plot,
        second_plot,
    ]
