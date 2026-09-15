"""Experiment Registry (PRD v2 Section 35)."""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.api.schemas.experiment import ExperimentObject, ExperimentMetrics
from app.core.database import get_db_session, init_db
from app.experiments.models import ExperimentRecord

# Initialize DB tables
init_db()

class ExperimentRegistry:
    """Manages persistence and retrieval of experiments."""
    
    @staticmethod
    def _to_schema(record: ExperimentRecord) -> ExperimentObject:
        metrics = None
        if record.metrics_json:
            metrics = ExperimentMetrics(**record.metrics_json)
            
        return ExperimentObject(
            experiment_hash=record.experiment_hash,
            run_id=record.run_id,
            model_family=record.model_family,
            state=record.state,
            error_message=record.error_message,
            hyperparameters=record.hyperparameters,
            feature_set_version=record.feature_set_version,
            dataset_protected_attributes=record.dataset_protected_attributes,
            metrics=metrics
        )

    @staticmethod
    def save_experiment(experiment: ExperimentObject) -> None:
        """Save or update an experiment in the registry."""
        with get_db_session() as db:
            # Check if exists
            record = db.query(ExperimentRecord).filter(
                ExperimentRecord.experiment_hash == experiment.experiment_hash
            ).first()
            
            if not record:
                record = ExperimentRecord(experiment_hash=experiment.experiment_hash)
                db.add(record)
                
            record.run_id = experiment.run_id
            record.model_family = experiment.model_family
            record.state = experiment.state
            record.error_message = experiment.error_message
            record.hyperparameters = experiment.hyperparameters
            record.feature_set_version = experiment.feature_set_version
            record.dataset_protected_attributes = experiment.dataset_protected_attributes
            
            if experiment.metrics:
                record.primary_metric = experiment.metrics.primary_metric
                record.mean_cv_score = experiment.metrics.mean_cv_score
                record.selection_val_score = experiment.metrics.selection_val_score
                record.metrics_json = experiment.metrics.model_dump()
                
    @staticmethod
    def get_experiment(experiment_hash: str) -> Optional[ExperimentObject]:
        """Retrieve an experiment by hash."""
        with get_db_session() as db:
            record = db.query(ExperimentRecord).filter(
                ExperimentRecord.experiment_hash == experiment_hash
            ).first()
            if record:
                return ExperimentRegistry._to_schema(record)
        return None
        
    @staticmethod
    def list_experiments(run_id: str, limit: int = 100) -> List[ExperimentObject]:
        """List all experiments for a given run."""
        with get_db_session() as db:
            records = db.query(ExperimentRecord).filter(
                ExperimentRecord.run_id == run_id
            ).order_by(desc(ExperimentRecord.mean_cv_score)).limit(limit).all()
            
            return [ExperimentRegistry._to_schema(r) for r in records]
