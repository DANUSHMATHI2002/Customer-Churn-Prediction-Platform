# Customer Churn Prediction Platform

End-to-end ML platform: data ingestion → feature engineering → model training → FastAPI → monitoring dashboard.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train model (downloads data automatically)
cd src && python train.py

# Start API
uvicorn api.main:app --reload --port 8000

# Start dashboard
streamlit run dashboard/app.py

# Run tests
pytest tests/ -v

# Full Docker stack
docker-compose up --build
```

## API Docs
Visit http://localhost:8000/docs

## MLflow UI
```bash
mlflow ui  # http://localhost:5000
```
