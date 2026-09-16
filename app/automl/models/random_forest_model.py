"""Random Forest model wrapper for AutoML Arena."""

from typing import Any, Dict
import optuna
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.automl.models.base_model import BaseAutoMLModel

class RandomForestModel(BaseAutoMLModel):
    """Random Forest wrapper supporting hyperparameter search."""

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")

    @property
    def model_name(self) -> str:
        return "random_forest"

    def get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Random Forest search space."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }

    def build_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        """Instantiate RandomForestClassifier or RandomForestRegressor."""
        if self.task_type == "classification":
            return RandomForestClassifier(random_state=42, **params)
        else:
            return RandomForestRegressor(random_state=42, **params)
