"""Unit tests for AutoML Arena models."""

import pytest
import optuna
from sklearn.datasets import make_classification
import pandas as pd
from app.automl.models.xgboost_model import XGBoostModel
from app.automl.models.lightgbm_model import LightGBMModel

def test_xgboost_model_initialization():
    """Test XGBoost model initialization and properties."""
    model = XGBoostModel(task_type="classification")
    assert model.model_name == "xgboost"
    assert model.task_type == "classification"

def test_lightgbm_model_initialization():
    """Test LightGBM model initialization and properties."""
    model = LightGBMModel(task_type="regression")
    assert model.model_name == "lightgbm"
    assert model.task_type == "regression"

def test_xgboost_build_estimator():
    """Test XGBoost build_estimator with valid params."""
    model = XGBoostModel(task_type="classification")
    params = {"n_estimators": 100, "max_depth": 5}
    estimator = model.build_estimator(params)
    assert estimator.n_estimators == 100
    assert estimator.max_depth == 5

def test_lightgbm_build_estimator():
    """Test LightGBM build_estimator with valid params."""
    model = LightGBMModel(task_type="classification")
    params = {"n_estimators": 200, "max_depth": 7}
    estimator = model.build_estimator(params)
    assert estimator.n_estimators == 200
    assert estimator.max_depth == 7

def test_xgboost_adds_scale_pos_weight_for_imbalanced_binary_target():
    model = XGBoostModel(task_type="classification")
    y = pd.Series([0] * 40 + [1] * 4)

    params = model.prepare_estimator_params({"n_estimators": 50}, y)

    assert params["scale_pos_weight"] == 10.0
    assert params["class_imbalance_ratio"] == 10.0

def test_lightgbm_adds_balanced_class_weight_for_imbalanced_target():
    model = LightGBMModel(task_type="classification")
    y = pd.Series([0] * 40 + [1] * 4)

    params = model.prepare_estimator_params({"n_estimators": 50}, y)

    assert params["scale_pos_weight"] == 10.0
    assert params["class_imbalance_ratio"] == 10.0

def test_invalid_task_type():
    """Test that an invalid task type raises an error."""
    with pytest.raises(ValueError, match="task_type must be 'classification' or 'regression'"):
        XGBoostModel(task_type="clustering")
