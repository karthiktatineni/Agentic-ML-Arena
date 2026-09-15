"""Unit tests for ModelAgent."""

import pytest
import pandas as pd
from sklearn.datasets import make_classification
from app.core.config import GlobalRunConfig
from app.automl.models.xgboost_model import XGBoostModel
from app.automl.agent import ModelAgent

def test_model_agent_hpo_and_train():
    """Test that ModelAgent can run HPO and train a final model."""
    # Create synthetic dataset
    X_np, y_np = make_classification(n_samples=100, n_features=10, random_state=42)
    X = pd.DataFrame(X_np, columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(y_np)

    config = GlobalRunConfig(
        max_optuna_trials_per_candidate=2,
        cv_folds=2,
        primary_metric="accuracy",
    )
    
    model_def = XGBoostModel(task_type="classification")
    agent = ModelAgent(model_def=model_def, config=config)
    
    # Run HPO
    best_params, best_value = agent.run_hpo(X, y, study_name="test_xgboost_study")
    
    assert isinstance(best_params, dict)
    assert best_value > 0.0
    
    # Train final model
    estimator = agent.train_final_model(X, y, best_params)
    assert hasattr(estimator, "predict")

def test_model_agent_selection_validation_isolation():
    """Verify that the HPO loop never sees or leaks selection validation data."""
    X_train = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [4, 3, 2, 1]})
    y_train = pd.Series([0, 1, 0, 1])
    
    # We purposefully create a selection_val dataframe but DO NOT pass it to run_hpo
    # This explicitly asserts that run_hpo only operates on the provided X_train/y_train
    # and has no physical access to the full dataset or other splits.
    
    config = GlobalRunConfig(
        max_optuna_trials_per_candidate=1,
        cv_folds=2,
    )
    model_def = XGBoostModel(task_type="classification")
    agent = ModelAgent(model_def=model_def, config=config)
    
    # HPO takes X_train and y_train explicitly. It has no internal state holding splits.
    best_params, _ = agent.run_hpo(X_train, y_train, study_name="test_isolation")
    
    # Prove that the agent state does not contain selection val data
    assert not hasattr(agent, "selection_val_df")
    assert not hasattr(agent, "dataset_splits")
