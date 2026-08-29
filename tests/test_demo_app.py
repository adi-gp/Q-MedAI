import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.demo.styles import THEME_CSS


ROOT = Path(__file__).resolve().parents[1]


def _app_with_empty_artifact_directory(tmp_path: Path) -> AppTest:
    artifact_directory = tmp_path / "empty-framingham-artifacts"
    artifact_directory.mkdir()
    wrapper = tmp_path / "app_for_test.py"
    wrapper.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                "import sys",
                f"sys.path.insert(0, {str(ROOT)!r})",
                "import streamlit as st",
                "from src.demo.pages import render_app",
                "st.set_page_config(page_title='Q-MedAI', page_icon='🧬', layout='wide')",
                f"render_app(Path({str(ROOT)!r}), artifact_directory=Path({str(artifact_directory)!r}))",
            )
        ),
        encoding="utf-8",
    )
    return AppTest.from_file(str(wrapper), default_timeout=30).run()


def test_app_starts_without_framingham_artifacts(tmp_path):
    app = _app_with_empty_artifact_directory(tmp_path)
    assert not app.exception
    assert app.title[0].value == "Q-MedAI Clinical Intelligence"
    text = " ".join(item.value for item in app.markdown)
    assert "Framingham" in text
    assert "Breast Cancer" in text


def test_patient_page_never_infers_without_artifacts(tmp_path):
    app = _app_with_empty_artifact_directory(tmp_path)
    app.sidebar.radio[0].set_value("Patient Risk").run()
    assert not app.exception
    assert any(button.label == "Run verified inference" and button.disabled for button in app.button)
    assert any("no patient score" in item.value.lower() for item in app.warning)


def test_breast_cancer_page_reaches_every_authoritative_plot_and_summary(tmp_path):
    app = _app_with_empty_artifact_directory(tmp_path)
    app.sidebar.radio[0].set_value("Breast Cancer Evidence").run()
    assert not app.exception

    payload = json.loads((ROOT / "results/final_results.json").read_text(encoding="utf-8"))
    listed = [Path(relative) for relative in payload["plots"]]
    featured = Path("results/plots/matched_classical_quantum_judge_summary.png")
    ordered = [featured, *(path for path in listed if path != featured)]
    expected = [path for path in ordered if (ROOT / path).is_file()]

    captions = [image.proto.imgs[0].caption for image in app.get("image")]
    labels = [button.label for button in app.get("download_button")]
    assert captions == [path.name for path in expected]
    for path in expected:
        assert labels.count(f"Download {path.name}") == 1
    assert labels.count("Download SIH final summary") == 1


def test_breast_cancer_page_warns_when_a_listed_plot_is_missing(tmp_path):
    view_root = tmp_path / "missing-plot-view"
    (view_root / "results").mkdir(parents=True)
    (view_root / "results/final_results.json").write_text(
        json.dumps({"plots": ["results/plots/not-installed.png"]}),
        encoding="utf-8",
    )
    wrapper = tmp_path / "missing_plot_page.py"
    wrapper.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                "import sys",
                f"sys.path.insert(0, {str(ROOT)!r})",
                "from src.demo.artifacts import inspect_framingham_artifacts",
                "from src.demo.evidence import load_breast_cancer_evidence, load_framingham_evidence",
                "from src.demo.pages import DemoContext, render_breast_cancer_evidence",
                "from src.demo.utility import evaluate_utility",
                f"source = Path({str(ROOT)!r})",
                f"view = Path({str(view_root)!r})",
                "framingham = load_framingham_evidence(source / 'results/framingham/notebook_metrics.json')",
                "breast = load_breast_cancer_evidence(source / 'results/final_results.json')",
                "context = DemoContext(view, framingham, breast, evaluate_utility(framingham), "
                "evaluate_utility(breast), inspect_framingham_artifacts(view / 'artifacts/framingham'))",
                "render_breast_cancer_evidence(context)",
            )
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(wrapper), default_timeout=30).run()
    assert not app.exception
    assert any("not-installed.png" in warning.value for warning in app.warning)


def test_light_theme_surfaces_declare_dark_foreground_colors():
    assert ".stApp { color: var(--qm-navy);" in THEME_CSS
    assert ".qm-card { color: var(--qm-navy);" in THEME_CSS
    assert ".qm-verdict { color: var(--qm-navy);" in THEME_CSS
