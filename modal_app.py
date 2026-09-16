import os
import shutil
from pathlib import Path
import modal

# 1. Initialize Modal App
app = modal.App("automl-arena")

# 2. Container Image Specification
# Uses Python 3.11 on Debian Slim with OpenMP (for LightGBM/XGBoost)
# Code is baked into the image container for fast, reliable startup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("app", remote_path="/root/app", copy=True)
    .add_local_file(".env", remote_path="/root/.env", copy=True)
    .add_local_dir(
        "data",
        remote_path="/root/seed_data",
        copy=True,
        ignore=["*.db-shm", "*.db-wal", "__pycache__", "*.pyc"],
    )
    .add_local_dir(
        "models",
        remote_path="/root/seed_models",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
)

# 3. Persistent Volumes for Persistent State Across Cold Starts
volume_data = modal.Volume.from_name("automl-arena-data", create_if_missing=True)
volume_models = modal.Volume.from_name("automl-arena-models", create_if_missing=True)


def copy_tree_contents(src: Path, dst: Path) -> bool:
    """Recursively copy files from src to dst without overwriting existing files."""
    copied = False
    dst.mkdir(parents=True, exist_ok=True)
    for sub in src.iterdir():
        target_sub = dst / sub.name
        if sub.is_dir():
            if copy_tree_contents(sub, target_sub):
                copied = True
        else:
            if not target_sub.exists() or target_sub.stat().st_size == 0:
                try:
                    shutil.copy2(sub, target_sub)
                    copied = True
                except Exception as e:
                    print(f"Copy warning for {sub.name}: {e}")
    return copied


def seed_storage_if_needed():
    """Seed initial data and models to persistent volumes if they are newly created."""
    data_dir = Path("/root/data")
    models_dir = Path("/root/models")
    seed_data = Path("/root/seed_data")
    seed_models = Path("/root/seed_models")

    seeded = False
    if seed_data.exists():
        if copy_tree_contents(seed_data, data_dir):
            seeded = True

    if seed_models.exists():
        if copy_tree_contents(seed_models, models_dir):
            seeded = True

    if seeded:
        try:
            volume_data.commit()
            volume_models.commit()
        except Exception as e:
            print(f"Volume commit notification: {e}")


# 4. Storage Sync Utility Function
@app.function(
    image=image,
    volumes={"/root/data": volume_data, "/root/models": volume_models},
)
def sync_storage():
    """Explicit one-shot function to sync seed data to persistent volumes and verify."""
    data_dir = Path("/root/data")
    models_dir = Path("/root/models")
    seed_data = Path("/root/seed_data")
    seed_models = Path("/root/seed_models")
    if seed_data.exists():
        copy_tree_contents(seed_data, data_dir)
    if seed_models.exists():
        copy_tree_contents(seed_models, models_dir)
    volume_data.commit()
    volume_models.commit()
    return {
        "data_files": [str(p.relative_to(data_dir)) for p in data_dir.glob("**/*") if p.is_file()],
        "models_files": [str(p.relative_to(models_dir)) for p in models_dir.glob("**/*") if p.is_file()],
    }


# 5. Serverless ASGI FastAPI Endpoint
# - cpu=4.0: 4 vCPUs for fast parallel training & optimization
# - memory=4096: 4 GB RAM (prevents any OOM failures)
# - scaledown_window=60: Scales to 0 instances after 60s idle (minimizes cost!)
# - timeout=1800: Up to 30 min runtime for AutoML pipelines
# - @modal.concurrent(max_inputs=100): Supports simultaneous HTTP and WebSocket traffic
@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/root/data": volume_data, "/root/models": volume_models},
    cpu=4.0,
    memory=4096,
    scaledown_window=60,
    timeout=1800,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def api():
    os.chdir("/root")
    seed_storage_if_needed()
    from app.api.main import app as web_app
    return web_app
