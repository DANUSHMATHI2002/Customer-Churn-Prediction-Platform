import os

import hashlib, json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import redis

from api.schemas import CustomerInput, PredictionResponse
from api.database import init_db, get_db, PredictionLog
from src.predict import ChurnPredictor
from src.explainer import ChurnExplainer

app = FastAPI(
    title="Churn Prediction API",
    description="Real-time customer churn prediction with SHAP explanations.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = explainer = redis_client = None


@app.on_event("startup")
async def startup():
    global predictor, explainer, redis_client
    init_db()
    predictor = ChurnPredictor()
    explainer = ChurnExplainer()
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
        )
        redis_client.ping()
    except Exception:
        redis_client = None


def _cache_key(data: dict) -> str:
    return "churn:" + hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput, explain: bool = False, db: Session = Depends(get_db)):
    payload = customer.model_dump()
    key = _cache_key(payload)
    if redis_client:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)

    result = predictor.predict(payload)
    if explain:
        exp = explainer.explain(payload)
        result["top_reasons"] = exp["top_reasons"]

    db.add(PredictionLog(
        tenure=customer.tenure,
        monthly_charges=customer.MonthlyCharges,
        contract=customer.Contract,
        churn_prob=result["churn_probability"],
        risk_tier=result["risk_tier"],
    ))
    db.commit()

    if redis_client:
        redis_client.setex(key, 86400, json.dumps(result))

    return result


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerInput]):
    return [predictor.predict(c.model_dump()) for c in customers]


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).all()
    if not logs:
        return {"total_predictions": 0}
    probs = [l.churn_prob for l in logs]
    tiers = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for l in logs:
        tiers[l.risk_tier] += 1
    return {
        "total_predictions": len(logs),
        "avg_churn_probability": round(sum(probs) / len(probs), 4),
        "risk_distribution": tiers,
    }
