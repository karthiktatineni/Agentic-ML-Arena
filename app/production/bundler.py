"""Model Artifact Bundler (PRD v2 Section 60)."""

import os
import json
import joblib
import subprocess
from typing import Dict, Any, Optional
from app.api.schemas.experiment import ExperimentObject

class ChampionBundler:
    """Packages a certified champion model for production serving."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        
    def export_champion(
        self, 
        run_id: str, 
        pipeline: Any, 
        experiment: ExperimentObject, 
        optimal_threshold: float = 0.5
    ) -> str:
        """Export the full artifact bundle."""
        bundle_dir = os.path.join(self.models_dir, run_id, "champion")
        os.makedirs(bundle_dir, exist_ok=True)
        
        # 1. pipeline.joblib
        joblib.dump(pipeline, os.path.join(bundle_dir, "pipeline.joblib"))
        
        # 2. metadata.json
        metadata = {
            "run_id": run_id,
            "experiment_hash": experiment.experiment_hash,
            "model_family": experiment.model_family,
            "feature_set_version": experiment.feature_set_version,
            "dataset_protected_attributes": experiment.dataset_protected_attributes
        }
        with open(os.path.join(bundle_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # 3. metrics.json
        metrics = experiment.metrics.model_dump() if experiment.metrics else {}
        with open(os.path.join(bundle_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # 4. threshold.json
        with open(os.path.join(bundle_dir, "threshold.json"), 'w') as f:
            json.dump({"optimal_threshold": optimal_threshold}, f, indent=2)
            
        # 5. environment.lock
        try:
            env_freeze = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
        except Exception:
            env_freeze = "unknown"
        with open(os.path.join(bundle_dir, "environment.lock"), 'w') as f:
            f.write(env_freeze)
            
        return bundle_dir
