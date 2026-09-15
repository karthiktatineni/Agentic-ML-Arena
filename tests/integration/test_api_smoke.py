"""API smoke tests against the full FastAPI application."""

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.api.main import app

client = TestClient(app)


def test_health_routes_respond():
    r = client.get("/api/v1/system/status")
    assert r.status_code == 200
    body = r.json()
    assert "cpu_percent" in body or "status" in body or isinstance(body, dict)


def test_predict_features_without_models():
    r = client.get("/api/predict/features")
    assert r.status_code in (200, 404)


def test_experiments_pending_endpoint():
    r = client.get("/api/experiments/pending")
    assert r.status_code == 200
    assert "pending" in r.json()


def test_ws_recent_events():
    r = client.get("/api/ws/events/recent")
    assert r.status_code == 200


def test_predict_with_approved_model(tmp_path, monkeypatch):
    import app.api.endpoints.experiments as exp_mod
    import app.api.endpoints.predict as pred_mod

    registry_path = tmp_path / "registry.json"
    pending_path = tmp_path / "pending.json"
    monkeypatch.setattr(exp_mod, "REGISTRY_FILE", registry_path)
    monkeypatch.setattr(exp_mod, "PENDING_APPROVALS_FILE", pending_path)
    monkeypatch.setattr(pred_mod, "REGISTRY_FILE", registry_path)
    monkeypatch.setattr(pred_mod, "PENDING_APPROVALS_FILE", pending_path)

    run_id = "smoke_run"
    model = LogisticRegression()
    model.fit(np.array([[0], [1], [0], [1]]), np.array([0, 1, 0, 1]))
    joblib_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": ["age"],
            "model_name": "logistic",
            "run_id": run_id,
            "task_type": "classification",
            "best_threshold": 0.5,
        },
        joblib_path,
    )

    import json
    from datetime import datetime

    registry_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": datetime.now().isoformat(),
                    "run_id": run_id,
                    "champion": {"experiment_hash": "hash1", "model_family": "logistic"},
                    "joblib_path": str(joblib_path),
                }
            ]
        ),
        encoding="utf-8",
    )

    r = client.post("/api/predict", json={"model_id": run_id, "features": {"age": 25}})
    assert r.status_code == 200
    assert "prediction" in r.json()
