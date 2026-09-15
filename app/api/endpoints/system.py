"""System Telemetry API — Real host machine & pipeline metrics."""

import os
import sys
import psutil
from pathlib import Path
from fastapi import APIRouter

router = APIRouter()

REGISTRY_FILE = Path("data/registry.json")
RUNS_DIR = Path("data/runs")
MODELS_DIR = Path("data/models")

def _get_dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.glob("**/*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return round(total / (1024 * 1024), 2)

@router.get("/status")
@router.get("/telemetry")
async def get_system_telemetry():
    """Return real system hardware and pipeline execution telemetry."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(".")
    cpu_pct = psutil.cpu_percent(interval=None)
    cpu_count = psutil.cpu_count(logical=True)
    
    # Registered models count
    model_count = 0
    if REGISTRY_FILE.exists():
        try:
            import json
            with open(REGISTRY_FILE, "r") as f:
                data = json.load(f)
                model_count = len(data) if isinstance(data, list) else len(data.get("models", []))
        except Exception:
            pass
            
    # Active runs count
    runs_count = 0
    if RUNS_DIR.exists():
        runs_count = len([d for d in RUNS_DIR.iterdir() if d.is_dir()])
        
    artifacts_mb = _get_dir_size_mb(RUNS_DIR) + _get_dir_size_mb(MODELS_DIR)

    return {
        "status": "ONLINE",
        "platform": f"Host OS ({sys.platform})",
        "python_version": sys.version.split()[0],
        "cpu": {
            "percent": cpu_pct,
            "logical_cores": cpu_count,
        },
        "memory": {
            "total_gb": round(mem.total / (1024 ** 3), 1),
            "used_gb": round(mem.used / (1024 ** 3), 1),
            "available_gb": round(mem.available / (1024 ** 3), 1),
            "percent": mem.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024 ** 3), 1),
            "used_gb": round(disk.used / (1024 ** 3), 1),
            "free_gb": round(disk.free / (1024 ** 3), 1),
            "percent": disk.percent,
        },
        "pipeline": {
            "registered_models": model_count,
            "recorded_runs": runs_count,
            "artifacts_storage_mb": artifacts_mb,
            "worker_engine": "FastAPI Async / Uvicorn",
            "cloud_ready": True,
        }
    }
