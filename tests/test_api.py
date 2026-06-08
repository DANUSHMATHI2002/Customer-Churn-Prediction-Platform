"""Integration tests for the prediction API."""

import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from main import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 24, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.50, "TotalCharges": 1908.0,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_schema():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_label"] in (0, 1)
    assert data["risk_tier"] in ("LOW", "MEDIUM", "HIGH")


def test_batch_predict():
    response = client.post("/predict/batch", json=[SAMPLE_CUSTOMER] * 3)
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_invalid_input_rejected():
    bad = {**SAMPLE_CUSTOMER, "MonthlyCharges": -5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
