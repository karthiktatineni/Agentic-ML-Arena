"""Deterministic Hashing for Experiments (PRD v2 Section 31)."""

import hashlib
import json
from typing import Dict, Any

class ExperimentHasher:
    """Computes deterministic SHA-256 fingerprint for experiment definitions."""

    @staticmethod
    def compute_hash(
        dataset_hash: str,
        model_family: str,
        hyperparameters: Dict[str, Any],
        preprocessing_steps: list,
        feature_set_version: str,
        seed: int,
    ) -> str:
        """Create a reproducible hash representing exactly what this experiment will compute."""
        # Sort dictionary keys to ensure deterministic serialization
        payload = {
            "dataset_hash": dataset_hash,
            "model_family": model_family,
            "hyperparameters": hyperparameters,
            "preprocessing_steps": preprocessing_steps,
            "feature_set_version": feature_set_version,
            "seed": seed,
        }
        
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
