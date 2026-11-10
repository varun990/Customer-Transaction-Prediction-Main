"""End-to-end tests for the saved model and the prediction API.

Model loads -> Input accepted -> Prediction generated -> Probability returned

Run from the project root:
    python -m pytest tests -v
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deployment.app import app  # noqa: E402

MODEL_PATH = ROOT / "models" / "customer_transaction_model.pkl"
EXAMPLE_PATH = ROOT / "deployment" / "example_request.json"


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def example():
    return json.loads(EXAMPLE_PATH.read_text())


def test_model_loads(bundle):
    assert hasattr(bundle["model"], "predict_proba")
    assert len(bundle["features"]) == 200
    assert 0 < bundle["threshold"] < 1


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_input_accepted(client, example):
    response = client.post("/predict", json=example)
    assert response.status_code == 200


def test_prediction_generated(client, example):
    result = client.post("/predict", json=example).json()
    assert result["prediction"] in (0, 1)
    expected_label = "Transaction" if result["prediction"] == 1 else "No transaction"
    assert result["prediction_label"] == expected_label


def test_probability_returned(client, example, bundle):
    result = client.post("/predict", json=example).json()
    probability = result["transaction_probability"]
    assert 0.0 <= probability <= 1.0
    # The API must return the same probability as the saved model itself
    frame = pd.DataFrame([example["features"]])[bundle["features"]]
    assert probability == pytest.approx(bundle["model"].predict_proba(frame)[0, 1], abs=1e-4)
    assert result["prediction"] == int(probability >= bundle["threshold"])


def test_batch_prediction(client, example):
    response = client.post("/predict/batch", json={"customers": [example, example]})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_missing_features_rejected(client):
    response = client.post("/predict", json={"features": {"var_0": 1.0}})
    assert response.status_code == 422
    assert "missing" in response.json()["detail"]
