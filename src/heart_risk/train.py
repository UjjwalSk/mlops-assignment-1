"""Train, compare, tune, track with MLflow, and package heart-risk classifiers."""
from argparse import ArgumentParser
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.heart_risk.data import validate_frame
from src.heart_risk.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
)

ROOT = Path(__file__).resolve().parents[2]


def build_preprocessor() -> ColumnTransformer:
    """Build production preprocessor with scaling for numerics and one-hot encoding for categoricals."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ("binary", binary_pipe, BINARY_FEATURES),
    ])


def main(experiment_name: str) -> None:
    data_path = ROOT / "data" / "processed" / "heart_disease_cleveland.csv"
    if not data_path.exists():
        from scripts.prepare_data import main as prep_main
        prep_main()

    data = pd.read_csv(data_path)
    validate_frame(data)

    X = data[FEATURES]
    y = data["target"]

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Models and hyperparameter grids for systematic tuning (Task 2)
    models = {
        "logistic_regression": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
        },
        "random_forest": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42)),
            ]),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [4, 6, 8, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        },
    }

    # Setup MLflow local tracking
    tracking_uri = f"file:///{(ROOT / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    results = {}
    fitted_models = {}

    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for name, config in models.items():
        print(f"\n--- Tuning and Training {name} ---")
        grid_search = GridSearchCV(
            config["pipeline"],
            config["params"],
            cv=cv,
            scoring="roc_auc",
            # Keep execution predictable on student laptops and in CI. The
            # search space is small, so parallel worker processes are not needed.
            n_jobs=1,
            refit=True,
        )

        with mlflow.start_run(run_name=name):
            grid_search.fit(X_train, y_train)
            best_pipe = grid_search.best_estimator_
            fitted_models[name] = best_pipe

            # Predict probabilities and classes on holdout test set
            y_pred_proba = best_pipe.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Test metrics
            test_acc = accuracy_score(y_test, y_pred)
            test_prec = precision_score(y_test, y_pred)
            test_rec = recall_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            test_auc = roc_auc_score(y_test, y_pred_proba)

            metrics = {
                "cv_best_score_roc_auc": float(grid_search.best_score_),
                "test_accuracy": float(test_acc),
                "test_precision": float(test_prec),
                "test_recall": float(test_rec),
                "test_f1": float(test_f1),
                "test_roc_auc": float(test_auc),
            }
            results[name] = metrics

            # Log parameters, metrics to MLflow
            clean_params = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
            mlflow.log_params({"model_type": name, "cv_folds": 5, "random_state": 42, **clean_params})
            mlflow.log_metrics(metrics)

            # 1. Confusion Matrix Plot
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_estimator(
                best_pipe, X_test, y_test, display_labels=["No Disease", "Disease"],
                cmap="Blues", ax=ax_cm, values_format="d"
            )
            ax_cm.set_title(f"Confusion Matrix: {name.replace('_', ' ').title()}")
            fig_cm.tight_layout()
            mlflow.log_figure(fig_cm, f"confusion_matrix_{name}.png")
            fig_cm.savefig(artifacts_dir / f"confusion_matrix_{name}.png", dpi=180)
            plt.close(fig_cm)

            # 2. ROC and Precision-Recall Curves
            fig_roc, ax_roc = plt.subplots(1, 2, figsize=(12, 5))
            RocCurveDisplay.from_estimator(best_pipe, X_test, y_test, name=name, ax=ax_roc[0])
            ax_roc[0].plot([0, 1], [0, 1], "k--", label="Chance")
            ax_roc[0].set_title("ROC Curve")
            PrecisionRecallDisplay.from_estimator(best_pipe, X_test, y_test, name=name, ax=ax_roc[1])
            ax_roc[1].set_title("Precision-Recall Curve")
            fig_roc.tight_layout()
            mlflow.log_figure(fig_roc, f"curves_{name}.png")
            plt.close(fig_roc)

            # 3. Feature Importances / Coefficients Plot
            fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
            try:
                feature_names = best_pipe.named_steps["preprocessor"].get_feature_names_out()
                if hasattr(best_pipe.named_steps["classifier"], "feature_importances_"):
                    importances = best_pipe.named_steps["classifier"].feature_importances_
                    idx = np.argsort(importances)[::-1][:15]
                    ax_feat.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1], color="#2b5c8f")
                    ax_feat.set_title(f"Top 15 Feature Importances: {name}")
                elif hasattr(best_pipe.named_steps["classifier"], "coef_"):
                    coefs = best_pipe.named_steps["classifier"].coef_[0]
                    idx = np.argsort(np.abs(coefs))[::-1][:15]
                    ax_feat.barh(np.array(feature_names)[idx][::-1], coefs[idx][::-1], color="#2b5c8f")
                    ax_feat.set_title(f"Top 15 Coefficients: {name}")
                fig_feat.tight_layout()
                mlflow.log_figure(fig_feat, f"features_{name}.png")
                fig_feat.savefig(artifacts_dir / f"features_{name}.png", dpi=180)
            except Exception as e:
                print(f"Feature plot notice for {name}: {e}")
            finally:
                plt.close(fig_feat)

            # 4. Classification Report Text Artifact
            cls_report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])
            mlflow.log_text(cls_report, f"classification_report_{name}.txt")

            # 5. Log Model to MLflow
            mlflow.sklearn.log_model(
                sk_model=best_pipe,
                artifact_path="model",
                input_example=X_train.iloc[:2],
            )
            print(f"Logged {name} to MLflow. Test ROC-AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}")

    # Select Best Model based on Holdout Test ROC-AUC
    best_name = max(results, key=lambda n: results[n]["test_roc_auc"])
    best_pipeline = fitted_models[best_name]

    # Package Final Model (Joblib)
    joblib.dump(best_pipeline, artifacts_dir / "heart_risk_pipeline.joblib")

    # Save Metrics JSON
    metrics_summary = {
        "selected_model": best_name,
        "selection_metric": "test_roc_auc",
        "results": results,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2))

    # Combined Comparison Curves Plot
    fig_comp, axes_comp = plt.subplots(1, 2, figsize=(12, 5))
    for name, pipe in fitted_models.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name.replace("_", " ").title(), ax=axes_comp[0])
        PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, name=name.replace("_", " ").title(), ax=axes_comp[1])
    axes_comp[0].plot([0, 1], [0, 1], "k--", label="Chance")
    axes_comp[0].set_title("ROC Curves Comparison")
    axes_comp[1].set_title("Precision-Recall Curves Comparison")
    fig_comp.tight_layout()
    fig_comp.savefig(artifacts_dir / "model_comparison.png", dpi=200)
    plt.close(fig_comp)

    print("\n================ FINAL MODEL SUMMARY ================")
    print(json.dumps(metrics_summary, indent=2))
    print(f"Best pipeline serialized to: {artifacts_dir / 'heart_risk_pipeline.joblib'}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train and track heart disease classifiers.")
    parser.add_argument("--experiment-name", default="heart-risk-production", help="MLflow experiment name")
    args = parser.parse_args()
    main(args.experiment_name)
