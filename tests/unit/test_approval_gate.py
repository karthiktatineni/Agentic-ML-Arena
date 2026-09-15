"""Unit tests for Approval Gate."""

import os
import joblib
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from app.api.endpoints import experiments, predict
import app.api.endpoints.experiments

fastapi_app = FastAPI()
fastapi_app.include_router(experiments.router, prefix="/api/approval")
fastapi_app.include_router(predict.router, prefix="/api")

client = TestClient(fastapi_app)

def test_approval_gate_default_deny(tmp_path, monkeypatch):
    # Mock the registry location so we don't mess up real files
    registry_path = tmp_path / "approvals.json"
    pending_path = tmp_path / "pending.json"
    
    import app.api.endpoints.experiments
    monkeypatch.setattr(app.api.endpoints.experiments, "REGISTRY_FILE", registry_path)
    monkeypatch.setattr(app.api.endpoints.experiments, "PENDING_APPROVALS_FILE", pending_path)
    
    # Force import predict module so we can patch its REGISTRY_FILE
    import app.api.endpoints.predict
    monkeypatch.setattr(app.api.endpoints.predict, "REGISTRY_FILE", registry_path)
    monkeypatch.setattr(app.api.endpoints.predict, "PENDING_APPROVALS_FILE", pending_path)
    
    # Create a fake bundle in models/run_unapproved/champion/
    run_id = "run_unapproved"
    models_dir = tmp_path / "models"
    
    # Needs a real pipeline object for prediction
    pipeline = LogisticRegression()
    import numpy as np
    pipeline.fit(np.array([[0], [1]]), np.array([0, 1]))
    
    # Create a dict bundle to match predict.py expectation
    bundle = {
        "model": pipeline,
        "feature_names": ["age"],
        "model_name": "logistic",
        "run_id": run_id,
        "best_threshold": 0.5
    }
    joblib_path = tmp_path / "champion.joblib"
    joblib.dump(bundle, joblib_path)
    joblib_path = str(joblib_path)
    
    # Mock the pending approvals file so the approve endpoint works
    with open(pending_path, "w") as f:
        json.dump({
            run_id: {
                "experiment_hash": "testhash",
                "run_id": run_id,
                "model_family": "logistic",
                "joblib_path": joblib_path
            }
        }, f)
    
    # 1. Attempt prediction without approval -> 403 Forbidden
    response = client.post("/api/predict", json={
        "model_id": run_id,
        "features": {"age": 30}
    })
    
    assert response.status_code == 403
    assert "pending human approval" in response.json()["detail"].lower()
    
    # 2. Approve the model
    app_response = client.post("/api/approval/approve", data={
        "run_id": run_id
    })
    assert app_response.status_code == 200
    
    # 3. Attempt prediction again -> 200 OK
    response2 = client.post("/api/predict", json={
        "model_id": run_id,
        "features": {"age": 30}
    })
    
    assert response2.status_code == 200
    
    assert "prediction" in response2.json()
