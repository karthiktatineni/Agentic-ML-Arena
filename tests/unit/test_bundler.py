"""Unit tests for Champion Bundler."""

import os
from sklearn.linear_model import LogisticRegression
from app.production.bundler import ChampionBundler
from app.api.schemas.experiment import ExperimentObject, ExperimentMetrics

def test_champion_bundler(tmp_path):
    models_dir = tmp_path / "models"
    bundler = ChampionBundler(models_dir=str(models_dir))
    
    experiment = ExperimentObject(
        experiment_hash="hash123",
        run_id="run_test",
        model_family="logistic_regression",
        dataset_protected_attributes=["age"]
    )
    experiment.metrics = ExperimentMetrics(primary_metric="f1", mean_cv_score=0.9)
    
    pipeline = LogisticRegression()
    
    bundle_dir = bundler.export_champion("run_test", pipeline, experiment, optimal_threshold=0.6)
    
    assert os.path.exists(os.path.join(bundle_dir, "pipeline.joblib"))
    assert os.path.exists(os.path.join(bundle_dir, "metadata.json"))
    assert os.path.exists(os.path.join(bundle_dir, "metrics.json"))
    assert os.path.exists(os.path.join(bundle_dir, "threshold.json"))
    assert os.path.exists(os.path.join(bundle_dir, "environment.lock"))
