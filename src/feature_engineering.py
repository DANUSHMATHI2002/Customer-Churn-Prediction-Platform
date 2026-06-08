"""
Feature engineering: encoding, scaling, interaction terms, SMOTE.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
]
MULTI_CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["charge_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["is_new_customer"] = (df["tenure"] <= 3).astype(int)
    df["is_long_term"] = (df["tenure"] >= 24).astype(int)
    df["high_value"] = (df["MonthlyCharges"] > 70).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_features = NUMERIC_COLS + [
        "charge_per_month", "is_new_customer", "is_long_term", "high_value"
    ]
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_features = [c for c in BINARY_COLS if c != "Churn"]
    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("bin", binary_transformer, binary_features),
        ("cat", cat_transformer, MULTI_CAT_COLS),
    ], remainder="drop")


def get_feature_pipeline(use_smote: bool = True) -> ImbPipeline:
    steps = [("preprocessor", build_preprocessor())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    return ImbPipeline(steps)


def save_preprocessor(preprocessor, path=None):
    path = path or str(ARTIFACTS_DIR / "preprocessor.joblib")
    joblib.dump(preprocessor, path)


def load_preprocessor(path=None):
    path = path or str(ARTIFACTS_DIR / "preprocessor.joblib")
    return joblib.load(path)
