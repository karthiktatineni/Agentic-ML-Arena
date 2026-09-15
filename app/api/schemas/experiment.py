"""Pydantic schemas for Experiments and Runs (PRD v2 Section 31)."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class ExperimentMetrics(BaseModel):
    """Metrics recorded after evaluation."""
    primary_metric: str
    mean_cv_score: float = 0.0
    std_cv_score: float = 0.0
    fold_scores: List[float] = Field(default_factory=list)
    min_score: float = 0.0
    max_score: float = 0.0
    selection_val_score: Optional[float] = None
    selection_val_predictions: List[float] = Field(default_factory=list)
    additional_metrics: Dict[str, Any] = Field(default_factory=dict)

class ExperimentObject(BaseModel):
    """Core experiment schema tracked throughout the pipeline."""
    experiment_hash: str
    run_id: str
    model_family: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_set_version: str = "v1"
    preprocessing_steps: List[str] = Field(default_factory=list)
    dataset_protected_attributes: List[str] = Field(default_factory=list)
    
    state: str = "CREATED"
    created_at: datetime = Field(default_factory=lambda: datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow())
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    metrics: Optional[ExperimentMetrics] = None
    error_message: Optional[str] = None
    worker_id: Optional[str] = None
