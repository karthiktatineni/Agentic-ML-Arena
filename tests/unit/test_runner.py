"""Unit tests for ExperimentRunner."""

import pytest
import pandas as pd
from sklearn.datasets import make_classification
from app.core.config import GlobalRunConfig
from app.api.schemas.experiment import ExperimentObject, ExperimentMetrics
from app.experiments.cache import ExperimentCache
from app.reliability.locks import FileLockManager
from app.automl.models.xgboost_model import XGBoostModel
from app.experiments.runner import ExperimentRunner

def test_experiment_runner(tmp_path):
    config = GlobalRunConfig(
        run_id="run_1",
        hpo_trials=1,
        max_optuna_trials_per_candidate=1,
        cv_folds=2,
        primary_metric="accuracy",
        target="target"
    )
    
    lock_manager = FileLockManager(lock_dir=str(tmp_path / ".locks"))
    cache = ExperimentCache(lock_manager=lock_manager, cache_dir=str(tmp_path / ".cache"))
    runner = ExperimentRunner(config=config, cache=cache, worker_id="test_worker")
    
    X_np, y_np = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X_np, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y_np
    
    from app.automl.split_strategy import SplitStrategyEngine
    splitter = SplitStrategyEngine(config)
    splits = splitter.create_splits(df, "target")
    
    experiment = ExperimentObject(
        experiment_hash="testhash123",
        run_id="run_1",
        model_family="xgboost",
    )
    
    model_def = XGBoostModel(task_type="classification")
    
    result = runner.run_experiment(experiment, model_def, splits=splits)
    
    assert result.state == "COMPLETED"
    assert result.metrics is not None
    assert result.metrics.mean_cv_score > 0.0
    assert len(result.metrics.fold_scores) == config.cv_folds
    assert result.metrics.min_score == min(result.metrics.fold_scores)
    assert result.metrics.max_score == max(result.metrics.fold_scores)
    assert result.metrics.std_cv_score >= 0.0
    assert len(result.metrics.additional_metrics["oof_predictions"]) == len(splits.train_pool_df)
    assert result.error_message is None
    
    # Try running again, should hit cache
    experiment2 = ExperimentObject(
        experiment_hash="testhash123",
        run_id="run_2",
        model_family="xgboost",
    )
    
    result2 = runner.run_experiment(experiment2, model_def, df)
    assert result2.state == "COMPLETED"
    assert result2.metrics.mean_cv_score == result.metrics.mean_cv_score
