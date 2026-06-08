"""
Model training with MLflow experiment tracking.
Trains XGBoost and LightGBM; picks the best by ROC-AUC on the val set.
"""

import json, mlflow, mlflow.sklearn, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_pipeline import download_data, clean_data, split_data
from src.feature_engineering import add_interaction_features, build_preprocessor

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
mlflow.set_experiment("churn_prediction")


def evaluate_model(model, X_val, y_val) -> dict:
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    return {
        "roc_auc": round(roc_auc_score(y_val, y_prob), 4),
        "avg_precision": round(average_precision_score(y_val, y_prob), 4),
    }


def run_training():
    df = clean_data(download_data())
    df = add_interaction_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_val_t   = preprocessor.transform(X_val)
    X_test_t  = preprocessor.transform(X_test)
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")

    models = {
        "xgboost": XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="auc", early_stopping_rounds=30,
            random_state=42, verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, verbosity=-1,
        ),
    }

    best_auc, best_name, best_model = -1, None, None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            if name == "xgboost":
                model.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
            else:
                model.fit(X_train_t, y_train)
            metrics = evaluate_model(model, X_val_t, y_val)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("val_roc_auc", metrics["roc_auc"])
            mlflow.log_metric("val_avg_precision", metrics["avg_precision"])
            mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"[{name}] val_roc_auc={metrics['roc_auc']} avg_precision={metrics['avg_precision']}")
            if metrics["roc_auc"] > best_auc:
                best_auc, best_name, best_model = metrics["roc_auc"], name, model

    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_t)[:, 1])
    print(f"\nBest: {best_name} | Test ROC-AUC: {test_auc:.4f}")
    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")
    with open(ARTIFACTS_DIR / "model_meta.json", "w") as f:
        json.dump({"model": best_name, "test_roc_auc": test_auc}, f, indent=2)
    print(f"Artifacts saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    run_training()
