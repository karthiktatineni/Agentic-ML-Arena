"""Unit tests for Champion Selection."""

from typing import List
from sklearn.metrics import f1_score
import numpy as np

from app.api.schemas.experiment import ExperimentObject, ExperimentMetrics
from app.automl.champion import ChampionSelectionEngine
from app.statistics.correction import MultipleTestingCorrection

def test_champion_selection_bootstrap():
    tracker = MultipleTestingCorrection(alpha=0.05)
    
    y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    # Model 1 is best
    e1 = ExperimentObject(
        experiment_hash="e1", 
        run_id="run_1", 
        model_family="xgboost",
        state="COMPLETED",
        metrics=ExperimentMetrics(
            primary_metric="f1", 
            selection_val_score=1.0, 
            selection_val_predictions=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        )
    )
    
    # Model 2 is slightly worse
    e2 = ExperimentObject(
        experiment_hash="e2", 
        run_id="run_1", 
        model_family="ensemble",
        state="COMPLETED",
        metrics=ExperimentMetrics(
            primary_metric="f1", 
            selection_val_score=0.8, 
            selection_val_predictions=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
        )
    )
    
    # Model 3 is broken
    e3 = ExperimentObject(
        experiment_hash="e3", 
        run_id="run_1", 
        model_family="rf",
        state="FAILED"
    )
    
    def metric_func(y_true_np: np.ndarray, preds_np: np.ndarray) -> float:
        return float(f1_score(y_true_np, preds_np, zero_division=0))
        
    champion = ChampionSelectionEngine.select_champion(
        experiments=[e1, e2, e3],
        y_true_selection_val=y_true,
        metric_func=metric_func,
        correction_tracker=tracker
    )
    
    assert champion is not None
    assert champion.experiment_hash == "e1"
    
    # Validate the ensemble was successfully compared
    assert tracker.total_comparisons_made == 1
    comparison = tracker.history[0]
    assert comparison["candidate_a"] == "e1"
    assert comparison["candidate_b"] == "e2"
    assert comparison["raw_p_value"] <= 1.0

def test_champion_selection_empty():
    def dummy_metric(y, p):
        return 0.0
    assert ChampionSelectionEngine.select_champion([], [], dummy_metric) is None
