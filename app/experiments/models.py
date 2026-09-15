"""SQLAlchemy ORM Models for Experiment Registry (PRD v2 Section 35)."""

from sqlalchemy import Column, String, Float, Integer, JSON, Boolean
from app.core.database import Base

class ExperimentRecord(Base):
    """Database record for an experiment run."""
    __tablename__ = "experiments"
    
    experiment_hash = Column(String, primary_key=True, index=True)
    run_id = Column(String, index=True, nullable=False)
    model_family = Column(String, nullable=False)
    state = Column(String, nullable=False)
    error_message = Column(String, nullable=True)
    
    # JSON Fields
    hyperparameters = Column(JSON, nullable=False, default=dict)
    feature_set_version = Column(String, nullable=False, default="v1")
    dataset_protected_attributes = Column(JSON, nullable=True)
    
    # Metrics extracted for easier querying
    primary_metric = Column(String, nullable=True)
    mean_cv_score = Column(Float, nullable=True)
    selection_val_score = Column(Float, nullable=True)
    
    # The full metrics object as JSON
    metrics_json = Column(JSON, nullable=True)


class ValidationCertificationRecord(Base):
    """Durable certification or rejection decision for a candidate champion."""
    __tablename__ = "validation_certifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_hash = Column(String, index=True, nullable=True)
    run_id = Column(String, index=True, nullable=False)
    model_name = Column(String, nullable=True)
    status = Column(String, nullable=False)
    score = Column(Float, nullable=True)
    minimum_score_threshold = Column(Float, nullable=True)
    rejection_reason = Column(String, nullable=True)
    certified_at = Column(Float, nullable=False)
    bundle_path = Column(String, nullable=True)
    correction_method = Column(String, nullable=True)
    comparison_count = Column(Integer, nullable=False, default=0)
    correction_history_json = Column(JSON, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
