"""Unit tests for Threshold Optimizer."""

import numpy as np
import pytest
from app.automl.threshold import ThresholdOptimizer
from app.api.schemas.experiment import ExperimentObject

def test_threshold_optimizer():
    # True labels: 5 pos, 5 neg
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Probas: model is somewhat confident but shifted.
    # To get perfect F1, threshold should be around 0.3
    y_prob = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.5, 0.6, 0.7])
    
    result = ThresholdOptimizer.optimize_threshold(
        y_true, 
        y_prob, 
        metric="f1", 
        num_thresholds=100
    )
    
    assert result["metric"] == "f1"
    assert result["best_score"] == 1.0
    assert 0.25 < result["best_threshold"] <= 0.35
    
    # Test apply
    binary_preds = ThresholdOptimizer.apply_threshold(y_prob, result["best_threshold"])
    np.testing.assert_array_equal(binary_preds, y_true)

def test_optimize_champion_threshold_enforcement():
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.9])
    
    # Test valid states
    for state in ["COMPLETED", "RANKED", "CERTIFIED"]:
        e_valid = ExperimentObject(experiment_hash="e1", run_id="r1", model_family="xgb", state=state)
        res = ThresholdOptimizer.optimize_champion_threshold(e_valid, y_true, y_prob)
        assert res["best_score"] == 1.0
        
    # Test invalid state
    e_invalid = ExperimentObject(experiment_hash="e2", run_id="r1", model_family="xgb", state="RUNNING")
    with pytest.raises(ValueError, match="Threshold tuning must only occur on certified champions"):
        ThresholdOptimizer.optimize_champion_threshold(e_invalid, y_true, y_prob)
