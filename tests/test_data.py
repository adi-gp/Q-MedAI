import pandas as pd
import pytest

from src.data.loader import CSV_SOURCE, SKLEARN_SOURCE, load_dataset
from src.data.validation import DataValidationError, validate_csv_dataset


def test_local_csv_loads_with_expected_encoding_and_id_removal():
    bundle = load_dataset()
    assert bundle.report.source == CSV_SOURCE
    assert bundle.report.target_encoding == {"B": 0, "M": 1}
    assert "id" not in bundle.X.columns
    assert bundle.X.shape[1] == 30
    assert set(bundle.y.unique()) == {0, 1}
    assert any(item["column"] == "id" for item in bundle.report.removed_columns)


def test_missing_csv_uses_sklearn_fallback(tmp_path):
    bundle = load_dataset(tmp_path / "not-present.csv")
    assert bundle.report.source == SKLEARN_SOURCE
    assert bundle.X.shape[0] > 100
    assert bundle.X.shape[1] == 30
    assert set(bundle.y.unique()) == {0, 1}


def test_target_encoding_and_unnamed_column_removal():
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "diagnosis": ["B", "M", "B", "M"],
            "radius": [1.0, 2.0, 3.0, 4.0],
            "Unnamed: 32": [None, None, None, None],
        }
    )
    X, y, report = validate_csv_dataset(frame, source="test")
    assert list(X.columns) == ["radius"]
    assert y.tolist() == [0, 1, 0, 1]
    assert {item["column"] for item in report.removed_columns} == {"id", "Unnamed: 32"}


def test_unexpected_non_numeric_column_is_not_silently_discarded():
    frame = pd.DataFrame(
        {
            "diagnosis": ["B", "M", "B", "M"],
            "radius": [1.0, 2.0, 3.0, 4.0],
            "unexpected_note": ["a", "b", "c", "d"],
        }
    )
    with pytest.raises(DataValidationError, match="Unexpected non-numeric"):
        validate_csv_dataset(frame, source="test")
