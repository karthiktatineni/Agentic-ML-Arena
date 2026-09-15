"""Experiment Runner (Isolated Worker Execution)."""

import logging
import traceback
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from app.core.config import GlobalRunConfig
from app.api.schemas.experiment import ExperimentObject, ExperimentMetrics
from app.experiments.cache import ExperimentCache, ClaimStatus
from app.reliability.state_machine import ExperimentStateMachine, ExperimentState
from app.automl.agent import ModelAgent
from app.automl.models.base_model import BaseAutoMLModel
from app.automl.split_strategy import SplitStrategyEngine, DatasetSplits

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Executes a single experiment within the safety bounds of locks and state machines."""
    
    def __init__(
        self,
        config: GlobalRunConfig,
        cache: ExperimentCache,
        worker_id: str,
    ):
        self.config = config
        self.cache = cache
        self.worker_id = worker_id

    def _selection_score(self, y_true, preds, y_prob=None) -> float:
        from sklearn.metrics import f1_score, r2_score, roc_auc_score, accuracy_score

        metric = self.config.primary_metric
        if metric == "r2":
            return float(r2_score(y_true, preds))
        if metric == "f1":
            binary_preds = (np.asarray(preds) >= 0.5).astype(int)
            return float(f1_score(y_true, binary_preds, zero_division=0))
        if metric == "roc_auc" and y_prob is not None:
            prob_pos = y_prob[:, 1] if getattr(y_prob, "ndim", 1) == 2 else y_prob
            return float(roc_auc_score(y_true, prob_pos))
        if metric == "accuracy":
            return float(accuracy_score(y_true, preds))
        if self.config.primary_metric in ("rmse", "mse", "mae"):
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            if metric == "mae":
                return -float(mean_absolute_error(y_true, preds))
            return -float(mean_squared_error(y_true, preds))
        # Task-aware fallback
        if pd.api.types.is_numeric_dtype(pd.Series(y_true)) and pd.Series(y_true).nunique() > 15:
            return float(r2_score(y_true, preds))
        binary_preds = (np.asarray(preds) >= 0.5).astype(int)
        return float(f1_score(y_true, binary_preds, zero_division=0))
        
    def run_experiment(
        self,
        experiment: ExperimentObject,
        model_def: BaseAutoMLModel,
        splits: DatasetSplits,
    ) -> ExperimentObject:
        """Run an experiment safely with locking and state tracking."""
        
        sm = ExperimentStateMachine(experiment.state)
        
        try:
            sm.transition(ExperimentState.VALIDATING)
            experiment.state = sm.current_state
            
            # 1. Attempt lock claim
            sm.transition(ExperimentState.QUEUED)
            experiment.state = sm.current_state
            
            status, cached_data = self.cache.claim_or_get_cached(
                experiment.experiment_hash, 
                self.worker_id,
                ttl=self.config.lock_ttl_seconds,
            )
            
            if status == ClaimStatus.CACHED and cached_data:
                logger.info(f"Experiment {experiment.experiment_hash} was cached. Skipping execution.")
                experiment.metrics = ExperimentMetrics(**cached_data["metrics"])
                experiment.hyperparameters = cached_data["hyperparameters"]
                
                # Jump straight to completed
                sm.transition(ExperimentState.CLAIMED)
                sm.transition(ExperimentState.RUNNING)
                sm.transition(ExperimentState.EVALUATING)
                sm.transition(ExperimentState.COMPLETED)
                experiment.state = sm.current_state
                return experiment
                
            elif status == ClaimStatus.DUPLICATE_IN_PROGRESS:
                logger.info(f"Experiment {experiment.experiment_hash} is running elsewhere.")
                # Could sleep and poll cache, but for now we'll mark as queued to try later
                return experiment
                
            # We acquired the lock
            sm.transition(ExperimentState.CLAIMED)
            experiment.state = sm.current_state
            
            # 2. Execution Phase
            sm.transition(ExperimentState.RUNNING)
            experiment.state = sm.current_state
            
            # HPO & Training
            agent = ModelAgent(model_def=model_def, config=self.config)
            
            X_train = splits.train_pool_df.drop(columns=[self.config.target])
            y_train = splits.train_pool_df[self.config.target]
            
            study_name = f"study_{experiment.experiment_hash}"
            best_params, best_value = agent.run_hpo(X_train, y_train, study_name)
            
            # Evaluate using the ModelAgent cross-validation
            # We already got the mean_cv_score from HPO, but let's do a final CV evaluation to get full metrics
            
            sm.transition(ExperimentState.EVALUATING)
            experiment.state = sm.current_state
            
            final_cv_summary, oof_preds = agent._cross_validate(best_params, X_train, y_train)
            final_cv_score = final_cv_summary["mean_cv_score"]
            
            # Predict on Selection-Validation set (or use OOF for Nested CV)
            if splits.is_nested_cv:
                selection_val_preds = oof_preds.tolist()
                selection_val_score = final_cv_score
            else:
                # Train model on full train_pool_df
                final_estimator = agent.train_final_model(X_train, y_train, best_params)
                
                X_val = splits.selection_val_df.drop(columns=[self.config.target])
                y_val = splits.selection_val_df[self.config.target]
                
                if hasattr(final_estimator, "predict_proba"):
                    y_prob = final_estimator.predict_proba(X_val)
                    preds = y_prob[:, 1] if y_prob.shape[1] == 2 else np.argmax(y_prob, axis=1)
                else:
                    preds = final_estimator.predict(X_val)
                    
                selection_val_preds = preds.tolist()
                selection_val_score = self._selection_score(
                    y_val, preds, y_prob if hasattr(final_estimator, "predict_proba") else None
                )
            
            # Create metrics object
            effective_params = agent.model_def.prepare_estimator_params(best_params, y_train)
            experiment.hyperparameters = effective_params
            metrics = ExperimentMetrics(
                primary_metric=self.config.primary_metric,
                mean_cv_score=final_cv_score,
                std_cv_score=final_cv_summary["std_cv_score"],
                fold_scores=final_cv_summary["fold_scores"],
                min_score=final_cv_summary["min_score"],
                max_score=final_cv_summary["max_score"],
                selection_val_score=selection_val_score,
                selection_val_predictions=selection_val_preds,
                additional_metrics={
                    "oof_predictions": oof_preds.tolist() if oof_preds is not None else [],
                }
            )
            experiment.metrics = metrics
            
            # Store in cache and release lock
            result_data = {
                "metrics": metrics.model_dump(),
                "hyperparameters": best_params,
            }
            self.cache.store_and_release(experiment.experiment_hash, self.worker_id, result_data)
            
            sm.transition(ExperimentState.COMPLETED)
            experiment.state = sm.current_state
            return experiment
            
        except Exception as e:
            logger.error(f"Experiment {experiment.experiment_hash} failed: {e}")
            experiment.error_message = traceback.format_exc()
            print(f"Runner failed: {e}\n{experiment.error_message}", flush=True)
            
            try:
                sm.transition(ExperimentState.FAILED)
                experiment.state = sm.current_state
            except Exception:
                pass
                
            # Abort lock
            self.cache.abort_claim(experiment.experiment_hash, self.worker_id)
            return experiment
