"""Fault injection test for Optuna budget cutoff."""

import time
import pytest
import optuna
from app.core.config import GlobalRunConfig
from app.automl.hpo import HPOEngine, HPOBudgetExceeded

def test_optuna_budget_cutoff():
    """Verify that HPO Engine terminates trials when max_time_seconds is reached."""
    config = GlobalRunConfig(
        max_optuna_trials_per_candidate=100,
        max_optuna_time_seconds_per_candidate=1, # extremely short timeout
    )
    
    engine = HPOEngine(config)
    
    def slow_objective(trial: optuna.Trial) -> float:
        # Sleep to simulate slow training
        time.sleep(0.5)
        # Suggest a random param to keep Optuna happy
        x = trial.suggest_float("x", -10, 10)
        return -(x**2)
        
    start_time = time.time()
    best_params, best_value, stats = engine.optimize_candidate(
        study_name="test_timeout_study",
        objective_fn=slow_objective,
        direction="maximize",
    )
    elapsed = time.time() - start_time
    
    # It should not run all 100 trials, but should stop close to 1 second
    assert stats["total_trials"] < 100
    assert elapsed < 3.0  # Allow some overhead margin, but strictly cut off
    assert stats["failed_trials"] == 0 # Timeout is caught via TrialPruned, not fail
    assert stats["elapsed_seconds"] >= 1.0 # Should have waited until timeout
