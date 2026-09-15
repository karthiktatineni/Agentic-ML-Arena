"""Event Schemas for Real-Time Dashboard (PRD v2 Section 54/55)."""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

class BaseEvent(BaseModel):
    event_type: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    run_id: str

class PipelineStageUpdatedEvent(BaseEvent):
    event_type: str = "PIPELINE_STAGE_UPDATED"
    stage_name: str
    status: str  # "IDLE", "ACTIVE", "COMPLETED", "FAILED"
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentDecisionEvent(BaseEvent):
    event_type: str = "AGENT_DECISION"
    agent_name: str
    decision_action: str
    confidence: float
    reasoning_summary: Optional[str] = None

class ExperimentResultEvent(BaseEvent):
    event_type: str = "EXPERIMENT_RESULT"
    experiment_hash: str
    model_family: str
    cv_score: float
    state: str

class ModelEliminatedEvent(BaseEvent):
    event_type: str = "MODEL_ELIMINATED"
    experiment_hash: str
    reason: str

class ChampionChangedEvent(BaseEvent):
    event_type: str = "CHAMPION_CHANGED"
    experiment_hash: str
    cv_score: float
    is_provisional: bool = True

class ChampionCertifiedEvent(BaseEvent):
    event_type: str = "CHAMPION_CERTIFIED"
    experiment_hash: str
    bundle_path: str
    model_name: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    correction_method: Optional[str] = None
    comparison_count: int = 0
    correction_history: List[Dict[str, Any]] = Field(default_factory=list)
    pipeline_attempt: int = 1
    total_pipeline_attempts: int = 1
    target_metric_achieved: bool = True
    below_target_warning: Optional[str] = None

class CertificationRejectedEvent(BaseEvent):
    event_type: str = "CERTIFICATION_REJECTED"
    experiment_hash: Optional[str] = None
    model_name: Optional[str] = None
    reason: str
    score: Optional[float] = None
    minimum_score_threshold: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    correction_method: Optional[str] = None
    comparison_count: int = 0
    correction_history: List[Dict[str, Any]] = Field(default_factory=list)
    pipeline_attempt: int = 1
    total_pipeline_attempts: int = 1

class LockAcquiredEvent(BaseEvent):
    event_type: str = "LOCK_ACQUIRED"
    lock_key: str
    worker_id: str

class LockExpiredEvent(BaseEvent):
    event_type: str = "LOCK_EXPIRED"
    lock_key: str
    worker_id: str
    recovered_by: str

class DegradedModeEvent(BaseEvent):
    event_type: str = "DEGRADED_MODE_ENTERED"
    subsystem: str
    reason: str

class RunStartedEvent(BaseEvent):
    event_type: str = "RUN_STARTED"
    config: Dict[str, Any]
