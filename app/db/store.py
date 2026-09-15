"""Thread-safe SQLite-backed registry and pending approval storage."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.database import get_db_session, init_db
from app.db.models import ModelRegistryRecord, PendingApprovalRecord

logger = logging.getLogger(__name__)

REGISTRY_JSON = Path("data/registry.json")
PENDING_JSON = Path("data/pending_approvals.json")


def _migrate_json_if_needed() -> None:
    """One-time import from legacy flat JSON files."""
    init_db()
    with get_db_session() as db:
        if db.query(ModelRegistryRecord).count() == 0 and REGISTRY_JSON.exists():
            try:
                with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                for entry in registry:
                    champ = entry.get("champion", {})
                    record = ModelRegistryRecord(
                        experiment_hash=champ.get("experiment_hash") or entry.get("run_id", "unknown"),
                        run_id=entry.get("run_id", "unknown"),
                        model_name=champ.get("model_family"),
                        score=champ.get("selection_val_score") or champ.get("metrics", {}).get("selection_val_score"),
                        joblib_path=entry.get("joblib_path"),
                        champion_json=entry,
                        approved_at=datetime.fromisoformat(entry["timestamp"]) if entry.get("timestamp") else None,
                    )
                    db.add(record)
                logger.info("Migrated %d registry entries from JSON", len(registry))
            except Exception as exc:
                logger.warning("Registry JSON migration skipped: %s", exc)

        if db.query(PendingApprovalRecord).count() == 0 and PENDING_JSON.exists():
            try:
                with open(PENDING_JSON, "r", encoding="utf-8") as f:
                    pending = json.load(f)
                for run_id, payload in pending.items():
                    record = PendingApprovalRecord(
                        run_id=run_id,
                        model_hash=payload.get("experiment_hash"),
                        cv_score=payload.get("selection_val_score"),
                        metric_name=payload.get("primary_metric"),
                        attempt=payload.get("pipeline_attempt"),
                        payload_json=payload,
                    )
                    db.add(record)
                logger.info("Migrated %d pending approvals from JSON", len(pending))
            except Exception as exc:
                logger.warning("Pending approvals JSON migration skipped: %s", exc)


class RegistryStore:
    @staticmethod
    def list_models() -> List[Dict[str, Any]]:
        _migrate_json_if_needed()
        with get_db_session() as db:
            records = db.query(ModelRegistryRecord).order_by(ModelRegistryRecord.approved_at.desc()).all()
            return [r.champion_json or {
                "run_id": r.run_id,
                "champion": {"experiment_hash": r.experiment_hash, "model_family": r.model_name},
                "joblib_path": r.joblib_path,
                "timestamp": r.approved_at.isoformat() if r.approved_at else None,
            } for r in records]

    @staticmethod
    def append_approved(run_id: str, champion_data: Dict[str, Any], joblib_path: Optional[str]) -> None:
        _migrate_json_if_needed()
        experiment_hash = champion_data.get("experiment_hash", run_id)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "champion": champion_data,
            "joblib_path": joblib_path,
        }
        with get_db_session() as db:
            record = db.query(ModelRegistryRecord).filter(
                ModelRegistryRecord.experiment_hash == experiment_hash
            ).first()
            if not record:
                record = ModelRegistryRecord(experiment_hash=experiment_hash)
                db.add(record)
            record.run_id = run_id
            record.model_name = champion_data.get("model_family")
            record.score = champion_data.get("selection_val_score")
            record.joblib_path = joblib_path
            record.champion_json = entry
            record.approved_at = datetime.now()

        # Keep JSON mirror for backward compatibility
        REGISTRY_JSON.parent.mkdir(parents=True, exist_ok=True)
        registry = RegistryStore.list_models()
        with open(REGISTRY_JSON, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, default=str)


class PendingApprovalStore:
    @staticmethod
    def load_all() -> Dict[str, Any]:
        _migrate_json_if_needed()
        with get_db_session() as db:
            records = db.query(PendingApprovalRecord).all()
            return {r.run_id: r.payload_json for r in records}

    @staticmethod
    def save(run_id: str, payload: Dict[str, Any]) -> None:
        _migrate_json_if_needed()
        with get_db_session() as db:
            record = db.query(PendingApprovalRecord).filter(
                PendingApprovalRecord.run_id == run_id
            ).first()
            if not record:
                record = PendingApprovalRecord(run_id=run_id)
                db.add(record)
            record.model_hash = payload.get("experiment_hash")
            record.cv_score = payload.get("selection_val_score")
            record.metric_name = payload.get("primary_metric")
            record.attempt = payload.get("pipeline_attempt")
            record.payload_json = payload

        PENDING_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(PENDING_JSON, "w", encoding="utf-8") as f:
            json.dump(PendingApprovalStore.load_all(), f, indent=4, default=str)

    @staticmethod
    def pop(run_id: str) -> Optional[Dict[str, Any]]:
        _migrate_json_if_needed()
        payload = None
        with get_db_session() as db:
            record = db.query(PendingApprovalRecord).filter(
                PendingApprovalRecord.run_id == run_id
            ).first()
            if record:
                payload = dict(record.payload_json)
                db.delete(record)

        PENDING_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(PENDING_JSON, "w", encoding="utf-8") as f:
            json.dump(PendingApprovalStore.load_all(), f, indent=4, default=str)
        return payload

    @staticmethod
    def is_pending(run_id: str) -> bool:
        return run_id in PendingApprovalStore.load_all()
