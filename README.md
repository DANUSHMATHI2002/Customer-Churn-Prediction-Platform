# 🔴 Customer Churn Prediction Platform
### End-to-End Production ML System | XGBoost · LightGBM · FastAPI · SHAP · MLflow · Docker

---
## Customer Churn Analysis
![Customer Churn](images/customer%20churn.png)

## SHAP Feature Importance
![SHAP Values](images/shap%20values.png)
---

## 📌 Project Overview

This is a **production-grade machine learning platform** that predicts which telecom customers are likely to cancel their subscription (churn). It goes far beyond a basic notebook — it is a fully deployable system with a REST API, a live monitoring dashboard, explainable AI, experiment tracking, caching, database logging, and a CI/CD pipeline.

The system is built on the **IBM Telco Customer Churn dataset** (~7,000 customers, 20 features) and achieves a **ROC-AUC of ~0.85+** on the held-out test set.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│   Raw CSV (IBM Telco) → Validation → Cleaning → Feature Eng.   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      TRAINING LAYER                             │
│   XGBoost vs LightGBM → MLflow Tracking → Best Model Saved     │
│   SMOTE Oversampling → Stratified Split → SHAP Explainability   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       API LAYER (FastAPI)                       │
│   /predict → Redis Cache → PostgreSQL Audit Log → Response      │
│   /predict/batch  /stats  /health                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    DASHBOARD LAYER (Streamlit)                  │
│   Live Stats · Risk Distribution · SHAP Waterfall Chart         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option A — Run Locally

```bash
# 1. Clone and install
git clone <your-repo-url>
cd churn-prediction
pip install -r requirements.txt

# 2. Train the model (auto-downloads dataset)
cd src
python train.py
cd ..

# 3. Start the REST API
uvicorn api.main:app --reload --port 8000

# 4. Open API docs
# Visit → http://localhost:8000/docs

# 5. Start the monitoring dashboard (new terminal)
streamlit run dashboard/app.py
# Visit → http://localhost:8501

# 6. Run the test suite
pytest tests/ -v
```

### Option B — Full Docker Stack (Recommended)

```bash
# Spin up API + Dashboard + PostgreSQL + Redis in one command
docker-compose up --build

# Services:
# API          → http://localhost:8000
# Dashboard    → http://localhost:8501
# API Docs     → http://localhost:8000/docs
# MLflow UI    → mlflow ui  →  http://localhost:5000
```

---

## 📁 Project Structure

```
churn-prediction/
├── data/                        # Raw & processed data (auto-downloaded)
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis
├── src/
│   ├── data_pipeline.py         # Download, validate, clean, split
│   ├── feature_engineering.py   # Encoding, scaling, SMOTE, interactions
│   ├── train.py                 # MLflow training loop (XGBoost + LightGBM)
│   ├── predict.py               # Inference wrapper (ChurnPredictor class)
│   └── explainer.py             # SHAP TreeExplainer + waterfall charts
├── api/
│   ├── main.py                  # FastAPI app, endpoints, Redis cache
│   ├── schemas.py               # Pydantic request/response models
│   └── database.py              # SQLAlchemy audit log (PostgreSQL/SQLite)
├── dashboard/
│   └── app.py                   # Streamlit monitoring dashboard
├── tests/
│   ├── test_pipeline.py         # Unit tests for data pipeline
│   └── test_api.py              # Integration tests for API endpoints
├── artifacts/                   # Saved model + preprocessor (after training)
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml     # GitHub Actions CI/CD pipeline
├── requirements.txt
└── README.md
```

---

## 🔌 API Reference

### `POST /predict`
Predict churn probability for a single customer.

**Request body:**
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 24,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 79.50,
  "TotalCharges": 1908.0
}
```

**Response:**
```json
{
  "churn_probability": 0.7843,
  "churn_label": 1,
  "risk_tier": "HIGH"
}
```

### `POST /predict?explain=true`
Returns prediction + top SHAP feature contributions.

### `POST /predict/batch`
Send a JSON array of customers for bulk scoring.

### `GET /stats`
Returns aggregated prediction statistics from the audit log.

### `GET /health`
Returns service health and model load status.

---

## 🧠 ML Pipeline Details

### Data
- **Source:** IBM Telco Customer Churn (7,043 rows, 20 features)
- **Target:** Binary — `Churn` (Yes/No → 1/0)
- **Class imbalance:** ~73% No-churn, ~27% Churn → handled with SMOTE

### Feature Engineering
- **Interaction features:** `charge_per_month`, `is_new_customer`, `is_long_term`, `high_value`
- **Encoding:** OrdinalEncoder for categorical features
- **Scaling:** StandardScaler for numeric features
- **Imputation:** Median (numeric), Most-frequent (categorical)

### Models Compared
| Model | Val ROC-AUC | Notes |
|---|---|---|
| XGBoost | ~0.84 | With early stopping, class weight |
| LightGBM | ~0.85 | With balanced class weight |

### Explainability
- **SHAP TreeExplainer** computes feature-level contribution scores
- Waterfall chart visualises the top 6 drivers for each prediction
- Surfaced in both the API response and the Streamlit dashboard

---

## ⚙️ Infrastructure

| Component | Technology |
|---|---|
| ML Models | XGBoost, LightGBM, scikit-learn |
| Experiment Tracking | MLflow |
| REST API | FastAPI + Uvicorn |
| Input Validation | Pydantic v2 |
| Caching | Redis (TTL: 24h) |
| Database | PostgreSQL (prod) / SQLite (dev) |
| Dashboard | Streamlit |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest + FastAPI TestClient |

---

## 📊 MLflow Experiment Tracking

```bash
# After training, view all experiment runs
mlflow ui
# Open → http://localhost:5000
```

Each run logs:
- Model hyperparameters
- Val ROC-AUC and Average Precision
- Serialised model artifact
- Training metadata

---

## 🧪 Testing

```bash
pytest tests/ -v --tb=short
```

Tests cover:
- Schema validation (valid and invalid inputs)
- Prediction response structure
- Batch prediction endpoint
- Health check endpoint

---

## 🔐 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./churn_audit.db` | PostgreSQL or SQLite URL |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |

---

## 📈 Performance

- **Test ROC-AUC:** ~0.85
- **API response time:** <50ms (cached), <200ms (first request)
- **Throughput:** Handles batch predictions of 1,000+ customers

---

## 🗺️ Roadmap / Possible Extensions

- [ ] Real-time data drift monitoring with Evidently AI
- [ ] Hyperparameter tuning with Optuna
- [ ] Model serving with BentoML or TorchServe
- [ ] Kubernetes deployment manifests
- [ ] A/B testing framework for model versions

---

## 📄 License

MIT License. Free to use for educational and commercial purposes.
