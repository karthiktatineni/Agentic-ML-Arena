"""Ensemble Methods for AutoML Arena (PRD v2 Section 35)."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
from sklearn.linear_model import LogisticRegression, Ridge

class EnsembleEngine:
    """Builds and evaluates ensembles of candidate models."""

    @staticmethod
    def uniform_average(predictions: List[np.ndarray]) -> np.ndarray:
        """Simple unweighted average of multiple model predictions."""
        if not predictions:
            raise ValueError("No predictions provided for ensembling.")
        
        stacked = np.stack(predictions, axis=0)
        return np.mean(stacked, axis=0)
        
    @staticmethod
    def train_stacking_meta_learner(
        oof_predictions: List[np.ndarray], 
        y_true: pd.Series, 
        task_type: str = "classification"
    ) -> Any:
        """Train a meta-learner (LogisticRegression or Ridge) on out-of-fold predictions."""
        if not oof_predictions:
            raise ValueError("No OOF predictions provided for stacking.")
            
        X_meta = np.column_stack(oof_predictions)
        y_array = y_true.values
        
        if task_type == "classification":
            # For classification, use L2 regularized logistic regression
            meta_learner = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
            meta_learner.fit(X_meta, y_array)
        else:
            # For regression, use Ridge regression
            meta_learner = Ridge(alpha=1.0)
            meta_learner.fit(X_meta, y_array)
            
        return meta_learner
        
    @staticmethod
    def predict_stacking(
        meta_learner: Any, 
        test_predictions: List[np.ndarray], 
        task_type: str = "classification"
    ) -> np.ndarray:
        """Predict using the trained meta-learner."""
        X_meta = np.column_stack(test_predictions)
        
        if task_type == "classification":
            # Return probability of positive class
            return meta_learner.predict_proba(X_meta)[:, 1]
        else:
            return meta_learner.predict(X_meta)
