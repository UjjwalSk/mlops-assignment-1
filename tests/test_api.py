"""Unit tests for FastAPI endpoints: /health, /predict, and /metrics."""
from fastapi.testclient import TestClient
from src.heart_risk.api import app

client = TestClient(app)

SAMPLE_VALID_PATIENT = {
    "age": 63.0,
    "sex": 1,
    "cp": 1,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150.0,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0.0,
    "thal": 6.0,
}


def test_health_endpoint():
    """Verify /health returns status 200 and health check payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_success():
    """Verify /predict returns status 200 and correct prediction structure with valid patient."""
    response = client.post("/predict", json=SAMPLE_VALID_PATIENT)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["risk_label"] in ["disease_risk", "no_disease_risk"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["disease_probability"] <= 1.0
    assert "model_version" in data


def test_predict_rejects_negative_age():
    """Verify Pydantic validation rejects impossible negative age with HTTP 422."""
    bad_payload = {**SAMPLE_VALID_PATIENT, "age": -5}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_features():
    """Verify out-of-range categorical feature (e.g. cp=99) is rejected with 422."""
    bad_payload = {**SAMPLE_VALID_PATIENT, "cp": 99}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_metrics_endpoint():
    """Verify /metrics returns Prometheus metric format with 200."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "heart_risk_predictions_total" in response.text
