from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_app_starts_without_framingham_artifacts():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Q-MedAI Clinical Intelligence"
    text = " ".join(item.value for item in app.markdown)
    assert "Framingham" in text
    assert "Breast Cancer" in text


def test_patient_page_never_infers_without_artifacts():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Patient Risk").run()
    assert not app.exception
    assert any(button.label == "Run verified inference" and button.disabled for button in app.button)
    assert any("no patient score" in item.value.lower() for item in app.warning)
