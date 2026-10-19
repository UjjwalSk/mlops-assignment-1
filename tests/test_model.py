"""Unit tests for ML model pipeline, preprocessing, training, and artifact loading."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.heart_risk.schema import FEATURES
from src.heart_risk.train import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "heart_risk_pipeline.joblib"


@pytest.fixture
def sample_dataset():
    """Create a minimal synthetic dataset conforming to heart risk schema."""
    np.random.seed(42)
    n = 40
    data = {
        "age": np.random.uniform(30, 75, n),
        "sex": np.random.choice([0, 1], n),
        "cp": np.random.choice([1, 2, 3, 4], n),
        "trestbps": np.random.uniform(90, 180, n),
        "chol": np.random.uniform(150, 350, n),
        "fbs": np.random.choice([0, 1], n),
        "restecg": np.random.choice([0, 1, 2], n),
        "thalach": np.random.uniform(90, 200, n),
        "exang": np.random.choice([0, 1], n),
        "oldpeak": np.random.uniform(0.0, 4.0, n),
        "slope": np.random.choice([1, 2, 3], n),
        "ca": np.random.choice([0, 1, 2, 3], n),
        "thal": np.random.choice([3, 6, 7], n),
        "target": np.random.choice([0, 1], n),
    }
    return pd.DataFrame(data)


def test_build_preprocessor():
    """Verify that the ColumnTransformer preprocesses numeric, categorical, and binary features."""
    preprocessor = build_preprocessor()
    assert preprocessor is not None
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "numeric" in transformer_names
    assert "categorical" in transformer_names
    assert "binary" in transformer_names


def test_pipeline_fit_and_predict(sample_dataset):
    """Test full pipeline end-to-end fit and inference probability output."""
    X = sample_dataset[FEATURES]
    y = sample_dataset["target"]

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=500)),
    ])
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (len(X), 2)
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_serialized_model_artifact():
    """Verify that the saved model artifact exists, loads cleanly, and predicts correctly."""
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not yet generated. Run training first.")

    model = joblib.load(MODEL_PATH)
    sample_input = pd.DataFrame([{
        "age": 63.0, "sex": 1, "cp": 1, "trestbps": 145.0, "chol": 233.0,
        "fbs": 1, "restecg": 2, "thalach": 150.0, "exang": 0, "oldpeak": 2.3,
        "slope": 3, "ca": 0.0, "thal": 6.0
    }])[FEATURES]

    proba = model.predict_proba(sample_input)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0
