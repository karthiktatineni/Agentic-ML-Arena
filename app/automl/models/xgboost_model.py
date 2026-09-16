"""XGBoost model wrapper for AutoML Arena."""

from typing import Any, Dict, Optional
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor

from app.automl.models.base_model import BaseAutoMLModel

class XGBoostModel(BaseAutoMLModel):
    """XGBoost wrapper supporting hyperparameter search."""

    imbalance_threshold: float = 3.0

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")

    @property
    def model_name(self) -> str:
        return "xgboost"

    def get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define XGBoost search space."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

    def prepare_estimator_params(self, params: Dict[str, Any], y: Optional[Any] = None) -> Dict[str, Any]:
        """Inject fold-local scale_pos_weight for imbalanced binary classification."""
        effective_params = dict(params)
        if self.task_type != "classification" or y is None or "scale_pos_weight" in effective_params:
            return effective_params

        counts = pd.Series(y).value_counts(dropna=False).sort_index()
        if len(counts) != 2 or counts.min() <= 0:
            return effective_params

        negative_count = float(counts.iloc[0])
        positive_count = float(counts.iloc[1])
        imbalance_ratio = float(counts.max() / counts.min())
        if imbalance_ratio > self.imbalance_threshold:
            effective_params["scale_pos_weight"] = negative_count / positive_count
            effective_params["class_imbalance_ratio"] = imbalance_ratio

        return effective_params

    def build_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        """Instantiate XGBClassifier or XGBRegressor."""
        estimator_params = dict(params)
        estimator_params.pop("class_imbalance_ratio", None)
        if self.task_type == "classification":
            return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, **estimator_params)
        else:
            return XGBRegressor(random_state=42, **estimator_params)
