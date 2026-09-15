"""FastAPI App Factory (PRD v2)."""

from fastapi import FastAPI
from app.core.config import GlobalRunConfig

app = FastAPI(title="AutoML Arena API")

@app.get("/health")
def health_check():
    return {"status": "ok"}
