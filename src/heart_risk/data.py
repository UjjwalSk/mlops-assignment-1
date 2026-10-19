from pathlib import Path

import pandas as pd

from src.heart_risk.schema import COLUMN_NAMES, FEATURES


def load_processed_data(path: str | Path) -> pd.DataFrame:
    """Load UCI processed Cleveland data and create a binary disease target."""
    frame = pd.read_csv(path, names=COLUMN_NAMES, na_values="?", skipinitialspace=True)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame["target"] = (frame["num"] > 0).astype(int)
    return frame.drop(columns=["num"])


def validate_frame(frame: pd.DataFrame) -> None:
    missing = set(FEATURES + ["target"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not set(frame["target"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target must be binary (0 or 1).")
    if frame.empty:
        raise ValueError("Dataset is empty.")
