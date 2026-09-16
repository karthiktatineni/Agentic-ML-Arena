"""CatBoost model wrapper for AutoML Arena."""

from typing import Any, Dict, Optional
import optuna
import pandas as pd
from sklearn.base import BaseEstimator

from app.automl.models.base_model import BaseAutoMLModel

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False


class CatBoostModel(BaseAutoMLModel):
    """CatBoost wrapper supporting hyperparameter search."""

    imbalance_threshold: float = 3.0

    def __init__(self, task_type: str = "classification"):
        if not _CATBOOST_AVAILABLE:
            raise ImportError("catboost is required for CatBoostModel")
        self.task_type = task_type
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")

    @property
    def model_name(self) -> str:
        return "catboost"

    def get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 50, 400, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

    def prepare_estimator_params(self, params: Dict[str, Any], y: Optional[Any] = None) -> Dict[str, Any]:
        effective_params = dict(params)
        if self.task_type != "classification" or y is None:
            return effective_params

        counts = pd.Series(y).value_counts(dropna=False).sort_index()
        if len(counts) != 2 or counts.min() <= 0:
            return effective_params

        imbalance_ratio = float(counts.max() / counts.min())
        if imbalance_ratio > self.imbalance_threshold:
            effective_params["scale_pos_weight"] = float(counts.iloc[0] / counts.iloc[1])
            effective_params["class_imbalance_ratio"] = imbalance_ratio
        return effective_params

    def build_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        estimator_params = dict(params)
        estimator_params.pop("class_imbalance_ratio", None)
        common = {"random_seed": 42, "verbose": 0, "allow_writing_files": False}
        if self.task_type == "classification":
            return CatBoostClassifier(**common, **estimator_params)
        return CatBoostRegressor(**common, **estimator_params)
