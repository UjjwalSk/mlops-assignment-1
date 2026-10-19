"""Unit tests for data loading, missing value handling, and schema validation."""
from pathlib import Path
import pandas as pd
import pytest

from src.heart_risk.data import validate_frame
from src.heart_risk.schema import FEATURES

ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = ROOT / "data" / "processed" / "heart_disease_cleveland.csv"


def test_validation_accepts_valid_schema():
    frame = pd.DataFrame([{**{feature: 1.0 for feature in FEATURES}, "target": 0}])
    validate_frame(frame)


def test_validation_rejects_bad_target():
    frame = pd.DataFrame([{**{feature: 1.0 for feature in FEATURES}, "target": 2}])
    with pytest.raises(ValueError, match="Target must be binary"):
        validate_frame(frame)


def test_validation_rejects_missing_column():
    frame = pd.DataFrame([{"age": 50.0, "target": 1}])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_frame(frame)


def test_validation_rejects_empty_frame():
    frame = pd.DataFrame(columns=FEATURES + ["target"])
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_frame(frame)


def test_cleaned_dataset_integrity():
    if not CLEANED_DATA_PATH.exists():
        pytest.skip("Processed dataset not yet generated.")
    data = pd.read_csv(CLEANED_DATA_PATH)
    validate_frame(data)
    assert len(data) == 303
    assert set(data["target"].unique()) == {0, 1}
    assert (data["target"] == 0).sum() == 164
    assert (data["target"] == 1).sum() == 139
