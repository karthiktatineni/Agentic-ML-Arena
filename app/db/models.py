"""SQLite ORM models for registry, approvals, and audit trail."""

from sqlalchemy import Column, String, Float, Integer, JSON, Text, DateTime
from sqlalchemy.sql import func

from app.core.database import Base


class RunRecord(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True, index=True)
    status = Column(String, nullable=False, default="pending")
    target_col = Column(String, nullable=True)
    dataset_path = Column(String, nullable=True)
    task_type = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class ModelRegistryRecord(Base):
    __tablename__ = "models"

    experiment_hash = Column(String, primary_key=True, index=True)
    run_id = Column(String, index=True, nullable=False)
    model_name = Column(String, nullable=True)
    score = Column(Float, nullable=True)
    joblib_path = Column(String, nullable=True)
    champion_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    approved_at = Column(DateTime, nullable=True)


class PendingApprovalRecord(Base):
    __tablename__ = "pending_approvals"

    run_id = Column(String, primary_key=True, index=True)
    model_hash = Column(String, nullable=True)
    cv_score = Column(Float, nullable=True)
    metric_name = Column(String, nullable=True)
    attempt = Column(Integer, nullable=True, default=1)
    payload_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, server_default=func.now())


class CertificationAuditRecord(Base):
    __tablename__ = "certifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True, nullable=False)
    experiment_hash = Column(String, nullable=True)
    status = Column(String, nullable=False)
    p_value = Column(Float, nullable=True)
    holm_bonferroni_json = Column(JSON, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class InferenceLogRecord(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_hash = Column(String, index=True, nullable=True)
    run_id = Column(String, index=True, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
