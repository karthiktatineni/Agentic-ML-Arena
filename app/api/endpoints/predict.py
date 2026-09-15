"""Prediction API — Load registered joblib models and serve predictions."""

import json
import logging
import joblib
import numpy as np
import pandas as pd
import io
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

PENDING_APPROVALS_FILE = Path("data/pending_approvals.json")
REGISTRY_FILE = Path("data/registry.json")


class PredictionRequest(BaseModel):
    """Input features for prediction."""
    features: Dict[str, Any]  # {"col1": val1, "col2": val2, ...}
    model_id: Optional[str] = None  # If None, use latest approved model


class PredictionResponse(BaseModel):
    """Prediction result."""
    prediction: Any
    confidence: Optional[float] = None
    model_id: str
    model_name: str
    task_type: Optional[str] = "regression"
    latency_ms: Optional[float] = 0.0


def _get_latest_model_path() -> Optional[Path]:
    """Find the latest approved model from the registry."""
    registry_file = REGISTRY_FILE
    if not registry_file.exists():
        return None
    try:
        with open(registry_file, "r") as f:
            registry = json.load(f)
        if not registry:
            return None
        # Search backwards for first entry whose file actually exists
        for entry in reversed(registry):
            joblib_path = entry.get("joblib_path")
            if joblib_path and Path(joblib_path).exists():
                return Path(joblib_path)
            # Check data/runs/{run_id}/
            run_id = entry.get("run_id")
            if run_id and Path(f"data/runs/{run_id}").exists():
                matches = list(Path(f"data/runs/{run_id}").glob("*.joblib"))
                if matches:
                    return matches[0]
    except Exception as e:
        logger.error(f"Failed to read registry: {e}")
    
    # Fallback to any joblib in data/runs or data/models
    for folder in [Path("data/runs"), Path("data/models")]:
        if folder.exists():
            files = list(folder.glob("**/*.joblib"))
            if files:
                return files[-1]
    return None


def _get_model_path_by_id(model_id: str) -> Optional[Path]:
    """Find a specific model by run_id, experiment_hash, or file path."""
    registry_file = REGISTRY_FILE
    if registry_file.exists():
        try:
            with open(registry_file, "r") as f:
                registry = json.load(f)
            for entry in registry:
                champ = entry.get("champion", {})
                if (entry.get("run_id") == model_id or 
                    champ.get("experiment_hash") == model_id or 
                    model_id in str(entry.get("joblib_path", ""))):
                    joblib_path = entry.get("joblib_path")
                    if joblib_path and Path(joblib_path).exists():
                        return Path(joblib_path)
                    run_id = entry.get("run_id")
                    if run_id and Path(f"data/runs/{run_id}").exists():
                        matches = list(Path(f"data/runs/{run_id}").glob("*.joblib"))
                        if matches:
                            return matches[0]
        except Exception as e:
            logger.error(f"Failed to read registry: {e}")

    # Search directly in filesystem
    for folder in [Path("data/runs"), Path("data/models")]:
        if folder.exists():
            direct = folder / f"{model_id}.joblib"
            if direct.exists():
                return direct
            matches = list(folder.glob(f"**/*{model_id}*.joblib"))
            if matches:
                return matches[0]
            # Check if model_id is a run folder
            run_dir = folder / model_id
            if run_dir.is_dir():
                sub_matches = list(run_dir.glob("*.joblib"))
                if sub_matches:
                    return sub_matches[0]
    return None


@router.post("", response_model=PredictionResponse)
@router.post("/", response_model=PredictionResponse)
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Load a registered model and return predictions."""
    import time
    start_t = time.perf_counter()

    # Find the model
    if request.model_id:
        if PENDING_APPROVALS_FILE.exists():
            try:
                with open(PENDING_APPROVALS_FILE, "r", encoding="utf-8") as f:
                    pending = json.load(f)
                if request.model_id in pending:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Model '{request.model_id}' deployment is pending human approval. Auto-deploy blocked."
                    )
            except HTTPException:
                raise
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read pending approvals: {e}")

        model_path = _get_model_path_by_id(request.model_id)
        if model_path is None:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found in registry")
    else:
        model_path = _get_latest_model_path()
        if model_path is None:
            raise HTTPException(status_code=404, detail="No approved models found in registry. Run the pipeline and approve a model first.")

    # Load model bundle (LRU cache avoids repeated disk reads)
    try:
        from app.utils.model_cache import model_bundle_cache
        bundle = model_bundle_cache.get(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    model = bundle.get("model")
    feature_names = bundle.get("feature_names", [])
    model_name = bundle.get("model_name", "Unknown Model")
    model_id = bundle.get("run_id", str(model_path.stem))
    task_type = bundle.get("task_type", "regression")

    if model is None:
        raise HTTPException(status_code=500, detail="Model bundle is corrupted — no 'model' key found")

    # Build input DataFrame
    try:
        input_df = pd.DataFrame([request.features])
        if feature_names:
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build input features: {e}")

    # Predict
    try:
        best_threshold = bundle.get("best_threshold", 0.5)
        if hasattr(model, "predict_proba") and task_type == "classification":
            proba = model.predict_proba(input_df)
            if proba.shape[1] == 2:
                prediction = int(proba[0, 1] >= best_threshold)
                confidence = float(max(proba[0]))
            else:
                prediction = int(np.argmax(proba[0]))
                confidence = float(proba[0, prediction])
        else:
            raw_pred = model.predict(input_df)[0]
            prediction = round(float(raw_pred), 2)
            confidence = None

        latency = round((time.perf_counter() - start_t) * 1000, 2)

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_id=model_id,
            model_name=model_name,
            task_type=task_type,
            latency_ms=latency,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router.get("/features")
async def get_model_features(model_id: Optional[str] = None):
    """Return the input schema and feature names for a model."""
    if model_id:
        model_path = _get_model_path_by_id(model_id)
    else:
        model_path = _get_latest_model_path()
        
    if model_path is None or not model_path.exists():
        raise HTTPException(status_code=404, detail="Model bundle not found")

    try:
        bundle = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect model bundle: {e}")

    feature_names = bundle.get("feature_names", [])
    model_name = bundle.get("model_name", "Trained Model")
    run_id = bundle.get("run_id", model_path.stem)
    task_type = bundle.get("task_type", "regression")
    model_hash = model_path.stem

    # Build smart initial sample record
    sample_record = {}
    for feat in feature_names:
        lower = feat.lower()
        if "year" in lower:
            sample_record[feat] = 2018
        elif "km" in lower or "mile" in lower or "odometer" in lower:
            sample_record[feat] = 45000
        elif "price" in lower or "amount" in lower or "cc" in lower:
            sample_record[feat] = 1500
        elif "age" in lower:
            sample_record[feat] = 5
        elif "owner" in lower or "count" in lower:
            sample_record[feat] = 1
        elif "condition" in lower:
            sample_record[feat] = 4
        else:
            sample_record[feat] = 0

    return {
        "model_id": run_id,
        "model_hash": model_hash,
        "model_name": model_name,
        "task_type": task_type,
        "score": bundle.get("score"),
        "feature_names": feature_names,
        "sample_record": sample_record,
        "download_url": f"http://localhost:8000/api/experiments/download/{model_hash}",
        "run_folder": f"data/runs/{run_id}",
    }


@router.get("/models")
async def list_models():
    """List all approved models available for prediction."""
    models = []
    seen_hashes = set()

    # 1. From registry.json
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r") as f:
                registry = json.load(f)
            for entry in reversed(registry):
                champ = entry.get("champion", {})
                hash_id = champ.get("experiment_hash") or Path(entry.get("joblib_path", "")).stem
                if hash_id and hash_id not in seen_hashes:
                    seen_hashes.add(hash_id)
                    models.append({
                        "run_id": entry.get("run_id", "run"),
                        "model_hash": hash_id,
                        "timestamp": entry.get("timestamp"),
                        "model_name": champ.get("model_family", "Model"),
                        "score": champ.get("metrics", {}).get("selection_val_score") or champ.get("selection_val_score"),
                        "joblib_path": entry.get("joblib_path"),
                        "download_url": f"http://localhost:8000/api/experiments/download/{hash_id}",
                    })
        except Exception as e:
            logger.error(f"Failed to read registry: {e}")

    # 2. From data/runs/
    if Path("data/runs").exists():
        for run_dir in Path("data/runs").iterdir():
            if run_dir.is_dir():
                for joblib_file in run_dir.glob("*.joblib"):
                    hash_id = joblib_file.stem
                    if hash_id not in seen_hashes:
                        seen_hashes.add(hash_id)
                        models.append({
                            "run_id": run_dir.name,
                            "model_hash": hash_id,
                            "timestamp": None,
                            "model_name": "Trained Champion",
                            "score": None,
                            "joblib_path": str(joblib_file),
                            "download_url": f"http://localhost:8000/api/experiments/download/{hash_id}",
                        })

    return {"models": models}


@router.post("/batch")
async def batch_predict(file: UploadFile = File(...), model_id: Optional[str] = Form(None)):
    """Run batch predictions on an uploaded CSV file."""
    if model_id:
        model_path = _get_model_path_by_id(model_id)
    else:
        model_path = _get_latest_model_path()
        
    if not model_path or not model_path.exists():
        raise HTTPException(status_code=404, detail="Model bundle not found")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}")

    try:
        bundle = joblib.load(model_path)
        model = bundle["model"]
        feature_names = bundle.get("feature_names", [])
        task_type = bundle.get("task_type", "regression")

        # Align features
        input_df = df.copy()
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        if hasattr(model, "predict_proba") and task_type == "classification":
            proba = model.predict_proba(input_df)
            thresh = bundle.get("best_threshold", 0.5)
            preds = (proba[:, 1] >= thresh).astype(int).tolist()
        else:
            preds = [round(float(p), 2) for p in model.predict(input_df)]

        return {
            "model_id": bundle.get("run_id", model_path.stem),
            "rows_processed": len(preds),
            "predictions_sample": preds[:50],
            "predictions_total": preds,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")
