from functools import lru_cache
from pathlib import Path
import json
import logging
import time

from fastapi import FastAPI, HTTPException, Request
import joblib
import pandas as pd
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from src.heart_risk.schema import FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "heart_risk_pipeline.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("heart_risk.api")

REQUESTS = Counter("heart_risk_predictions_total", "Prediction requests", ["outcome"])
LATENCY = Histogram("heart_risk_prediction_latency_seconds", "Prediction latency in seconds")

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production-ready MLOps API for clinical risk classification using the UCI Cleveland dataset.",
    version="1.0.0",
)
app.mount("/metrics", make_asgi_app())


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years", json_schema_extra={"example": 63.0})
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male; 0 = female)", json_schema_extra={"example": 1})
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1: typical, 2: atypical, 3: non-anginal, 4: asymptomatic)", json_schema_extra={"example": 1})
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure in mm Hg", json_schema_extra={"example": 145.0})
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol in mg/dl", json_schema_extra={"example": 233.0})
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", json_schema_extra={"example": 1})
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)", json_schema_extra={"example": 2})
    thalach: float = Field(..., ge=40, le=250, description="Maximum heart rate achieved", json_schema_extra={"example": 150.0})
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes; 0 = no)", json_schema_extra={"example": 0})
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise relative to rest", json_schema_extra={"example": 2.3})
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1: up, 2: flat, 3: down)", json_schema_extra={"example": 3})
    ca: float = Field(..., ge=0, le=3, description="Number of major vessels (0-3) colored by fluoroscopy", json_schema_extra={"example": 0.0})
    thal: float = Field(..., ge=3, le=7, description="Thallium stress test (3 = normal; 6 = fixed defect; 7 = reversible defect)", json_schema_extra={"example": 6.0})


@lru_cache(maxsize=1)
def get_cached_model():
    """Load model into memory once and cache across requests."""
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable. Run model training first (python -m src.heart_risk.train).",
        )
    return joblib.load(MODEL_PATH)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Structured request logging middleware."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    if request.url.path not in ["/metrics", "/health"]:
        LOGGER.info(
            json.dumps({
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "client_host": request.client.host if request.client else "unknown",
            })
        )
    return response


@app.get("/health", tags=["Health"])
def health():
    """Liveness and readiness health probe."""
    model_loaded = False
    try:
        get_cached_model()
        model_loaded = True
    except HTTPException:
        pass
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "service": "heart-risk-api",
        "version": "1.0.0",
    }


@app.post("/predict", tags=["Inference"])
def predict(patient: Patient):
    """Predict heart disease presence/absence and confidence score."""
    started = time.perf_counter()
    try:
        model = get_cached_model()
        input_data = pd.DataFrame([patient.model_dump()])[FEATURES]
        probabilities = model.predict_proba(input_data)[0]
        prob_disease = float(probabilities[1])
        prediction = int(prob_disease >= 0.5)

        REQUESTS.labels(outcome=str(prediction)).inc()
        risk_label = "disease_risk" if prediction == 1 else "no_disease_risk"

        return {
            "prediction": prediction,
            "risk_label": risk_label,
            "confidence": round(prob_disease if prediction == 1 else (1 - prob_disease), 4),
            "disease_probability": round(prob_disease, 4),
            "model_version": "production-v1",
        }
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error("Prediction inference failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        LATENCY.observe(time.perf_counter() - started)
