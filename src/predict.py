"""
Prediction module — wraps model + preprocessor into a clean interface.
"""

import joblib
import pandas as pd
from pathlib import Path
from src.feature_engineering import add_interaction_features


# FIX 1: absolute-safe path (always points to project root/artifacts)
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


class ChurnPredictor:
    def __init__(self):
        model_path = ARTIFACTS_DIR / "best_model.joblib"
        preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"

        # FIX 2: clear error messages instead of cryptic crash
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run training script first (train.py)."
            )

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {preprocessor_path}. "
                f"Run training script first (train.py)."
            )

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, customer: dict) -> dict:
        df = pd.DataFrame([customer])

        # safe feature engineering
        df = add_interaction_features(df)

        X = self.preprocessor.transform(df)
        prob = float(self.model.predict_proba(X)[0, 1])

        label = int(prob >= 0.5)

        if prob >= 0.75:
            risk = "HIGH"
        elif prob >= 0.45:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "churn_probability": round(prob, 4),
            "churn_label": label,
            "risk_tier": risk
        }

    def predict_batch(self, customers: list) -> list:
        return [self.predict(c) for c in customers]