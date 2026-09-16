"""Out-of-fold stacking ensemble blending XGBoost, LightGBM, and CatBoost."""

from typing import Any, Dict, Optional
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from app.automl.models.base_model import BaseAutoMLModel
from app.automl.models.xgboost_model import XGBoostModel
from app.automl.models.lightgbm_model import LightGBMModel

try:
    from app.automl.models.catboost_model import CatBoostModel
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False


class StackingEnsembleModel(BaseAutoMLModel):
    """Stacking ensemble with Ridge/Logistic meta-learner over tree base models."""

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self._xgb = XGBoostModel(task_type=task_type)
        self._lgb = LightGBMModel(task_type=task_type)
        self._cat = CatBoostModel(task_type=task_type) if _CATBOOST_AVAILABLE else None

    @property
    def model_name(self) -> str:
        return "stacking_ensemble"

    def get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
            "lgb_num_leaves": trial.suggest_int("lgb_num_leaves", 20, 100),
            "meta_alpha": trial.suggest_float("meta_alpha", 0.1, 10.0, log=True),
        }

    def prepare_estimator_params(self, params: Dict[str, Any], y: Optional[Any] = None) -> Dict[str, Any]:
        return dict(params)

    def build_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        xgb_params = {
            "n_estimators": 200,
            "max_depth": params.get("xgb_max_depth", 6),
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        lgb_params = {
            "n_estimators": 200,
            "num_leaves": params.get("lgb_num_leaves", 63),
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        cat_params = {
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.05,
        }

        estimators = [
            ("xgb", self._xgb.build_estimator(xgb_params)),
            ("lgb", self._lgb.build_estimator(lgb_params)),
        ]
        if self._cat is not None:
            estimators.append(("cat", self._cat.build_estimator(cat_params)))

        meta_alpha = params.get("meta_alpha", 1.0)
        if self.task_type == "classification":
            final_estimator = LogisticRegression(penalty="l2", C=1.0 / meta_alpha, max_iter=1000)
            return StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=5,
                stack_method="predict_proba",
                n_jobs=1,
            )

        final_estimator = Ridge(alpha=meta_alpha)
        return StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=1,
        )
