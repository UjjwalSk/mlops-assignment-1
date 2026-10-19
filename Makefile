.PHONY: setup data eda train test lint api mlflow compose helm-lint clean

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

data:
	python scripts/prepare_data.py

eda: data
	python scripts/run_eda.py

train: data
	python -m src.heart_risk.train --experiment-name heart-risk-production

test:
	python -m pytest -q

lint:
	python -m ruff check src tests scripts

api:
	python -m uvicorn src.heart_risk.api:app --host 0.0.0.0 --port 8000

mlflow:
	python -m mlflow ui --backend-store-uri ./mlruns --port 5000

compose:
	docker compose up --build

helm-lint:
	helm lint k8s/helm/heart-risk-api

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('data/raw'), Path('.pytest_cache'), Path('.ruff_cache'), Path('tmp')]]"
