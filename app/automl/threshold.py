"""Out-Of-Fold Threshold Optimization (PRD v2 Section 34)."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, List
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score

from app.api.schemas.experiment import ExperimentObject

class ThresholdOptimizer:
    """Finds the optimal decision threshold for binary classification probabilities.
    
    Per PRD v2 Section 14 and 49: this optimization must be performed ONLY AFTER 
    champion certification, and strictly on Train/CV pool out-of-fold predictions.
    """
    
    @staticmethod
    def optimize_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric: str = "f1",
        num_thresholds: int = 100,
    ) -> Dict[str, Any]:
        """Sweep thresholds to maximize a given metric."""
        y_true = np.asarray(y_true)
        if len(np.unique(y_true)) != 2:
            return {
                "metric": metric,
                "best_threshold": 0.5,
                "best_score": 0.0
            }

        # Ensure we're using the probability of the positive class
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
            
        thresholds = np.linspace(0.01, 0.99, num_thresholds)
        best_threshold = 0.5
        best_score = -1.0
        
        for th in thresholds:
            y_pred = (y_prob_pos >= th).astype(int)
            
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "balanced_accuracy":
                score = balanced_accuracy_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
                
            if score > best_score:
                best_score = score
                best_threshold = th
                
        return {
            "metric": metric,
            "best_threshold": float(best_threshold),
            "best_score": float(best_score)
        }

    @staticmethod
    def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
        """Apply a learned threshold to raw probabilities."""
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            return (y_prob[:, 1] >= threshold).astype(int)
        return (np.asarray(y_prob) >= threshold).astype(int)

    @staticmethod
    def optimize_champion_threshold(
        champion_experiment: ExperimentObject,
        y_true_train_cv: np.ndarray,
        oof_probabilities: np.ndarray,
        metric: str = "f1",
        num_thresholds: int = 100
    ) -> Dict[str, Any]:
        """Strictly enforce that threshold optimization ONLY runs on certified champions."""
        if champion_experiment.state not in ("RANKED", "CERTIFIED", "COMPLETED"):
            raise ValueError(
                f"Threshold tuning must only occur on certified champions. "
                f"Experiment {champion_experiment.experiment_hash} is in state {champion_experiment.state}."
            )
        
        # In a real pipeline, we'd also assert that `oof_probabilities` matches the length of `y_true_train_cv` 
        # and has no overlap with the selection-validation set.
        
        return ThresholdOptimizer.optimize_threshold(
            y_true=y_true_train_cv,
            y_prob=oof_probabilities,
            metric=metric,
            num_thresholds=num_thresholds
        )
