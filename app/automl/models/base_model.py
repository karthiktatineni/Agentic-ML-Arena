"""Base model definition for AutoML Arena candidate models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import optuna
from sklearn.base import BaseEstimator

class BaseAutoMLModel(ABC):
    """Abstract class for candidate models in AutoML Arena."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the unique name of the model (e.g., 'xgboost', 'lightgbm')."""
        pass

    @abstractmethod
    def get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define the hyperparameter search space for Optuna."""
        pass

    @abstractmethod
    def build_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        """Instantiate the underlying scikit-learn compatible estimator with the given params."""
        pass

    def prepare_estimator_params(self, params: Dict[str, Any], y: Optional[Any] = None) -> Dict[str, Any]:
        """Return effective estimator params for a specific training target slice.

        Model wrappers can override this to inject fold-local settings such as class
        weights without fitting or mutating data outside the training split.
        """
        return dict(params)
