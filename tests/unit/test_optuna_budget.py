"""Unit test for Optuna trial and wall-clock compute bounds."""

import time
import pytest
import optuna
from app.core.config import GlobalRunConfig
from app.automl.hpo import HPOEngine


def test_optuna_trials_cap():
    """Verify that HPO strictly halts when trial limit is reached."""
    config = GlobalRunConfig(
        max_optuna_trials_per_candidate=10,
        max_optuna_time_seconds_per_candidate=60,
    )
    engine = HPOEngine(config)

    trial_counter = 0

    def dummy_objective(trial: optuna.Trial) -> float:
        nonlocal trial_counter
        trial_counter += 1
        x = trial.suggest_float("x", -10, 10)
        return -(x ** 2)

    best_params, best_val, stats = engine.optimize_candidate(
        study_name="test_trials_cap",
        objective_fn=dummy_objective,
    )

    assert stats["total_trials"] <= 10
    assert trial_counter <= 10
    assert "x" in best_params
    assert stats["completed_trials"] > 0
    assert stats["elapsed_seconds"] < 60


def test_optuna_time_budget_cutoff():
    """Verify that HPO strictly halts when wall-clock limit is exceeded."""
    config = GlobalRunConfig(
        max_optuna_trials_per_candidate=100,
        max_optuna_time_seconds_per_candidate=2,  # 2 second timeout
    )
    engine = HPOEngine(config)

    def slow_objective(trial: optuna.Trial) -> float:
        time.sleep(0.5)  # slow trial
        x = trial.suggest_float("x", 0, 1)
        return x

    start_time = time.time()
    best_params, best_val, stats = engine.optimize_candidate(
        study_name="test_time_cutoff",
        objective_fn=slow_objective,
    )
    duration = time.time() - start_time

    # Should have terminated around 2-3 seconds, well before 100 trials
    assert duration < 5.0
    assert stats["total_trials"] < 15
    assert "completed_trials" in stats
    assert "pruned_trials" in stats
