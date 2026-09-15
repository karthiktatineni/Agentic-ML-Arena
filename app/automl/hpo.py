"""Optuna Hyperparameter Optimization Engine with strict trial and time bounds."""

import time
import logging
from typing import Callable, Dict, Any, Optional, Tuple
import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler

from app.core.config import GlobalRunConfig

logger = logging.getLogger(__name__)


class HPOBudgetExceeded(Exception):
    """Raised when an HPO study reaches its allocated time or trial budget."""
    pass


class HPOEngine:
    """Manages hyperparameter optimization studies with strict trial and wall-clock caps."""

    def __init__(self, config: Optional[GlobalRunConfig] = None):
        self.config = config or GlobalRunConfig()
        self.max_trials = self.config.max_optuna_trials_per_candidate
        self.max_time_seconds = self.config.max_optuna_time_seconds_per_candidate

    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
    ) -> optuna.Study:
        """Create an Optuna study with configured pruner and sampler."""
        if self.config.optuna_pruner == "median":
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
        else:
            pruner = NopPruner()

        sampler = TPESampler(seed=self.config.random_seed)
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=pruner,
            sampler=sampler,
        )
        return study

    def optimize_candidate(
        self,
        study_name: str,
        objective_fn: Callable[[optuna.Trial], float],
        direction: str = "maximize",
        max_trials: Optional[int] = None,
        max_time_seconds: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float, Dict[str, int]]:
        """Run bounded hyperparameter search for a candidate model.

        Returns:
            (best_params, best_value, trial_stats_dict)
        """
        trials_limit = max_trials or self.max_trials
        timeout_limit = max_time_seconds or self.max_time_seconds

        study = self.create_study(study_name=study_name, direction=direction)

        start_time = time.time()

        def bounded_objective(trial: optuna.Trial) -> float:
            elapsed = time.time() - start_time
            if elapsed >= timeout_limit:
                trial.study.stop()
                raise optuna.TrialPruned(f"Timeout limit reached ({elapsed:.1f}s >= {timeout_limit}s)")
            return objective_fn(trial)

        # Optuna suppress verbose logs for clean runs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            bounded_objective,
            n_trials=trials_limit,
            timeout=timeout_limit,
            catch=(Exception,),
        )

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        stats = {
            "total_trials": len(study.trials),
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }

        logger.info(
            f"Candidate [{study_name}] HPO completed: {stats['completed_trials']} completed, "
            f"{stats['pruned_trials']} pruned, {stats['failed_trials']} failed "
            f"in {stats['elapsed_seconds']}s (caps: {trials_limit} trials, {timeout_limit}s)."
        )

        if not completed_trials:
            # Fallback if all pruned/failed
            return {}, 0.0, stats

        return study.best_params, study.best_value, stats
