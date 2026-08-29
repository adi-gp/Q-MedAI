from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def _app():
    return AppTest.from_file(
        str(ROOT / "app.py"),
        default_timeout=45,
    ).run()


def test_quantum_lab_waits_for_a_verified_patient_trace():
    app = _app()

    app.sidebar.radio[0].set_value("Quantum Lab").run()

    assert not app.exception

    messages = [item.value for item in app.info]

    assert any(
        "Run verified inference on Patient Risk" in message
        for message in messages
    )


def test_quantum_lab_renders_latest_verified_patient_trace():
    app = _app()

    # Produce a real frozen-artifact inference first.
    app.sidebar.radio[0].set_value("Patient Risk").run()

    submit = next(
        button
        for button in app.button
        if button.label == "Run verified inference"
    )

    submit.click().run()

    assert not app.exception

    # Same Streamlit session -> Quantum Lab should now receive
    # the latest verified patient trace.
    app.sidebar.radio[0].set_value("Quantum Lab").run()

    assert not app.exception

    metric_values = {
        metric.label: metric.value
        for metric in app.metric
    }

    assert "Kernel max fidelity" in metric_values
    assert "Kernel mean fidelity" in metric_values
    assert "Kernel median fidelity" in metric_values
    assert "References compared" in metric_values

    assert 0.0 <= float(metric_values["Kernel max fidelity"]) <= 1.0
    assert 0.0 <= float(metric_values["Kernel mean fidelity"]) <= 1.0
    assert 0.0 <= float(metric_values["Kernel median fidelity"]) <= 1.0
    assert int(metric_values["References compared"]) > 0

    markdown = "\n".join(item.value for item in app.markdown)

    assert "Latest verified patient quantum trace" in markdown
    assert "Raw value" in markdown
    assert "Encoded angle" in markdown
