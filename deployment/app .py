"""FastAPI service for the customer transaction prediction model.

Run from the project root:
    uvicorn deployment.app:app --reload
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "customer_transaction_model.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES: list[str] = bundle["features"]
THRESHOLD: float = bundle["threshold"]

app = FastAPI(
    title="Customer Transaction Prediction API",
    description="Predicts whether a bank customer will make a transaction in the future.",
    version="1.0.0",
)


class Customer(BaseModel):
    features: dict[str, float] = Field(
        ..., description="All 200 anonymized features, keyed var_0 ... var_199")


class CustomerBatch(BaseModel):
    customers: list[Customer] = Field(..., min_length=1)


class Prediction(BaseModel):
    transaction_probability: float
    prediction: int
    prediction_label: str


def _to_frame(customers: list[Customer]) -> pd.DataFrame:
    for i, customer in enumerate(customers):
        missing = [f for f in FEATURES if f not in customer.features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Customer {i}: missing {len(missing)} feature(s), e.g. {missing[:5]}")
    return pd.DataFrame([c.features for c in customers])[FEATURES]


def _predict(frame: pd.DataFrame) -> list[Prediction]:
    proba = model.predict_proba(frame)[:, 1]
    return [
        Prediction(
            transaction_probability=round(float(p), 4),
            prediction=int(p >= THRESHOLD),
            prediction_label="Transaction" if p >= THRESHOLD else "No transaction",
        )
        for p in proba
    ]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": bundle["model_name"],
        "threshold": THRESHOLD,
        "n_features": len(FEATURES),
        "top_features": bundle.get("top_features", [])[:10],
        "test_metrics": bundle["metrics"],
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return _predict(_to_frame([customer]))[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: CustomerBatch):
    return _predict(_to_frame(batch.customers))
