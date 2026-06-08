"""
SHAP-based explainability.
"""

import shap
import joblib
import pandas as pd
import numpy as np
import io, base64, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.feature_engineering import add_interaction_features

ARTIFACTS_DIR = Path("artifacts")


class ChurnExplainer:
    def __init__(self):
        self.model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
        self.preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, customer: dict, top_n: int = 6) -> dict:
        df = pd.DataFrame([customer])
        df = add_interaction_features(df)
        X = self.preprocessor.transform(df)
        feature_names = self.preprocessor.get_feature_names_out()
        shap_values = self.explainer.shap_values(X)
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        reasons = [
            {
                "feature": name.split("__")[-1],
                "shap_value": round(float(val), 4),
                "direction": "increases churn risk" if val > 0 else "reduces churn risk",
            }
            for name, val in pairs
        ]
        chart_b64 = self._waterfall_chart(feature_names, sv, top_n)
        return {"top_reasons": reasons, "waterfall_chart_b64": chart_b64}

    def _waterfall_chart(self, names, values, top_n) -> str:
        idx = np.argsort(np.abs(values))[-top_n:]
        top_names = [names[i].split("__")[-1] for i in idx]
        top_vals  = [values[i] for i in idx]
        colors = ["#E85D30" if v > 0 else "#1D9E75" for v in top_vals]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top_names, top_vals, color=colors)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on churn probability)")
        ax.set_title("Top feature contributions")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
