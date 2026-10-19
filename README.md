# Heart Disease Risk Prediction — End-to-End MLOps Solution
**BITS Pilani — Work Integrated Learning Programmes (WILP)**
**Machine Learning Operations (MLOps) AIMLCZG523 — Assignment 01**

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)

**Student ID:** 2025AE05260
**BITS Email:** 2025ae05260@wilp.bits-pilani.ac.in

A reproducible MLOps assignment implementation for binary coronary heart-disease risk prediction using the UCI Cleveland dataset. It includes data ingestion, EDA, feature engineering, two tuned classification models, MLflow tracking, automated tests, Docker packaging, cloud-ready Kubernetes manifests, and Prometheus/Grafana monitoring configuration.

---

## Architecture Overview

The decoupled architecture consists of five core layers:
1. **Data Ingestion & Preprocessing:** Automated data acquisition, missing value (`?`) handling, and binary target formulation.
2. **Feature Engineering & Modeling:** `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder` for multi-categorical features. Stratified 5-Fold Cross-Validation and hyperparameter optimization for Logistic Regression and Random Forest.
3. **Experiment Tracking & Packaging:** Full MLflow logging (hyperparameters, metrics, confusion matrices, ROC/PR curves, feature importances, and model binaries). Serialized production pipeline in `artifacts/heart_risk_pipeline.joblib`.
4. **Automated Quality Controls (CI/CD):** Multi-stage GitHub Actions workflow executing `ruff` linting, data preparation, model training, and `pytest` verification (13 passing unit tests).
5. **Containerization, Orchestration & Monitoring:** Unprivileged Docker container, Kubernetes Deployment/Service/Ingress/HPA manifests, a Helm chart, and Prometheus/Grafana configuration.

---

## Quick Start (Local Setup)

### 1. Environment Setup & Dependency Installation
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Ingest Data, Generate EDA & Train Models
```bash
# Extract and clean dataset
python scripts/prepare_data.py

# Generate comprehensive EDA figures in reports/figures/
python scripts/run_eda.py

# Run hyperparameter tuning, model training, and MLflow tracking
python -m src.heart_risk.train --experiment-name heart-risk-production
```

### 3. Launch MLflow Tracking Server
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```
Open [http://localhost:5000](http://localhost:5000) to view logged parameters, metrics, confusion matrices, and ROC/PR curve artifacts.

### 4. Run Unit Tests & Linting
```bash
# Run the complete test suite (13 tests)
python -m pytest -v

# Run code style and lint check
python -m ruff check src tests scripts
```

### 5. Launch Serving API
```bash
uvicorn src.heart_risk.api:app --host 0.0.0.0 --port 8000
```
* **Swagger Interactive Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
* **Prometheus Metrics:** [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## Full Observability Stack (Docker Compose)

Spin up the FastAPI service, Prometheus, and Grafana simultaneously with one command:
```bash
docker-compose up --build
```
* **FastAPI Service:** `http://localhost:8000`
* **Prometheus Server:** `http://localhost:9090`
* **Grafana Dashboard:** `http://localhost:3000` (Credentials: `admin` / `admin`)

---

## Kubernetes & Helm Deployment

### Option A: Standard Kubernetes Manifests (`kubectl`)
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

kubectl rollout status deployment/heart-risk-api
kubectl get pods,svc,hpa
```

### Option B: Helm Chart (`helm`)
```bash
# Lint the chart
helm lint k8s/helm/heart-risk-api

# Install release
helm install heart-risk-api k8s/helm/heart-risk-api
```

---

## Project Structure

```
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions multi-stage CI/CD pipeline
├── artifacts/                     # Versioned model artifacts, plots & metrics
│   ├── confusion_matrix_*.png     # Confusion matrices
│   ├── features_*.png             # Feature importance / coefficient charts
│   ├── heart_risk_pipeline.joblib # Production packaged pipeline
│   ├── metrics.json               # Cross-validation & holdout evaluation metrics
│   └── model_comparison.png      # ROC and Precision-Recall comparison plots
├── data/
│   ├── raw/                       # Extracted raw data files
│   └── processed/                 # Cleaned Cleveland dataset (CSV)
├── k8s/                           # Production Kubernetes manifests
│   ├── deployment.yaml            # Deployment with resource limits & probes
│   ├── service.yaml               # LoadBalancer Service
│   ├── ingress.yaml               # Ingress routing manifest
│   ├── hpa.yaml                   # Horizontal Pod Autoscaler manifest
│   └── helm/heart-risk-api/       # Parameterized Helm Chart
├── monitoring/                    # Observability configuration
│   ├── prometheus.yml             # Prometheus scrape configuration
│   └── grafana/                   # Grafana datasource and dashboard provisioning
├── scripts/                       # Automation scripts
│   ├── prepare_data.py            # Data extraction and validation with fallback
│   ├── run_eda.py                 # Comprehensive EDA plot generator
│   └── sample_request.py          # Sample inference request script
├── src/heart_risk/                # Core production package
│   ├── api.py                     # FastAPI serving engine with cached model
│   ├── data.py                    # Data loading and schema validator
│   ├── schema.py                  # Clinical feature definitions
│   └── train.py                   # GridSearchCV tuning & MLflow tracking
├── tests/                         # Unit & integration tests
│   ├── test_api.py                # Tests for /health, /predict, and /metrics
│   ├── test_data.py               # Tests for schema and data integrity
│   └── test_model.py              # Tests for model pipeline & serialization
├── Dockerfile                     # Hardened non-root production Dockerfile
├── docker-compose.yml             # Full stack (FastAPI + Prometheus + Grafana)
├── requirements.txt               # Pinned pip requirements
└── Makefile                       # One-stop build and pipeline automation
```

---

## API Request & Response Sample

### Request: `POST /predict`
```json
{
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
  "thal": 6.0
}
```

### Response: `HTTP 200 OK`
```json
{
  "prediction": 1,
  "risk_label": "disease_risk",
  "confidence": 0.9234,
  "disease_probability": 0.9234,
  "model_version": "production-v1"
}
```

---

## Deliverables & Submission Checklist

- [x] Code, Dockerfile, and requirements.txt
- [x] Cleaned dataset and automated download/extraction script
- [x] Pytest suite across data, model, and API (13 tests passing)
- [x] GitHub Actions CI/CD workflow YAML (`.github/workflows/ci.yml`)
- [x] Kubernetes manifests (Deployment, Service, Ingress, HPA) & Helm Chart
- [x] Real-time monitoring stack (Prometheus + Grafana dashboard)
- [x] GitHub repository and CI/CD workflow configuration
