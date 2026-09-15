"""Durable storage for champion certification decisions."""

from typing import Any, Dict, Optional

from app.core.database import get_db_session, init_db
from app.events.schemas import ChampionCertifiedEvent, CertificationRejectedEvent
from app.experiments.models import ValidationCertificationRecord


class ValidationCertificationStore:
    """Persists validation certification decisions to the experiment database."""

    @staticmethod
    def _score_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
        for key in ("selection_val_score", "score", "mean_cv_score"):
            value = metrics.get(key)
            if value is not None:
                return float(value)
        return None

    @staticmethod
    def save_event(event: ChampionCertifiedEvent | CertificationRejectedEvent) -> int:
        """Persist a CHAMPION_CERTIFIED or CERTIFICATION_REJECTED event."""
        init_db()

        metrics = dict(getattr(event, "metrics", {}) or {})
        status = "CERTIFIED" if event.event_type == "CHAMPION_CERTIFIED" else "REJECTED"
        rejection_reason = getattr(event, "reason", None)
        score = getattr(event, "score", None)
        if score is None:
            score = ValidationCertificationStore._score_from_metrics(metrics)

        record = ValidationCertificationRecord(
            experiment_hash=getattr(event, "experiment_hash", None),
            run_id=event.run_id,
            model_name=getattr(event, "model_name", None),
            status=status,
            score=score,
            minimum_score_threshold=getattr(event, "minimum_score_threshold", None),
            rejection_reason=rejection_reason,
            certified_at=event.timestamp,
            bundle_path=getattr(event, "bundle_path", None),
            correction_method=getattr(event, "correction_method", None),
            comparison_count=int(getattr(event, "comparison_count", 0) or 0),
            correction_history_json=getattr(event, "correction_history", []) or [],
            metrics_json=metrics,
            hyperparameters=dict(getattr(event, "hyperparameters", {}) or {}),
        )

        with get_db_session() as db:
            db.add(record)
            db.flush()
            record_id = int(record.id)

        return record_id
