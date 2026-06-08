"""
Data ingestion and validation pipeline.
Downloads the Telco Customer Churn dataset and runs quality checks.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_data() -> pd.DataFrame:
    cache_path = DATA_DIR / "raw_churn.csv"
    if cache_path.exists():
        logger.info("Loading cached data from %s", cache_path)
        return pd.read_csv(cache_path)
    logger.info("Downloading dataset...")
    df = pd.read_csv(RAW_DATA_URL)
    df.to_csv(cache_path, index=False)
    logger.info("Saved %d rows to %s", len(df), cache_path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    required = {"customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn", "Contract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    logger.info("Schema validation passed.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df.drop(columns=["customerID"], inplace=True)
    logger.info("Cleaned data: %d rows, %d columns.", *df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=relative_val, stratify=y_tmp, random_state=random_state
    )
    logger.info("Split — train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df_raw = download_data()
    validate_schema(df_raw)
    df_clean = clean_data(df_raw)
    df_clean.to_csv(DATA_DIR / "clean_churn.csv", index=False)
    print(df_clean.head())
    print(df_clean["Churn"].value_counts(normalize=True))
