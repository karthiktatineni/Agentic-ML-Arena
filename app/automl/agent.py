"""Model Agent orchestrating HPO and cross-validation for a candidate model."""

import pandas as pd
import numpy as np
import optuna
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import StratifiedKFold, KFold

from app.core.config import GlobalRunConfig
from app.automl.models.base_model import BaseAutoMLModel
from app.automl.hpo import HPOEngine
from app.automl.evaluation import EvaluatorEngine

def _defensive_coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce continuous string features containing units (e.g. '1005 CC', '11.5 kmpl') to numeric."""
    df_out = df.copy()
    for col in df_out.columns:
        if df_out[col].dtype == "object" or df_out[col].dtype.name == "string":
            s = df_out[col].astype(str).str.strip().str.lower()
            extracted = s.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
            num_s = pd.to_numeric(extracted, errors="coerce")
            valid_ratio = num_s.notna().sum() / max(1, s.notna().sum())
            if valid_ratio >= 0.7:
                df_out[col] = num_s
    return df_out


class ModelAgent:
    """Orchestrates hyperparameter optimization and cross-validation for a specific model."""

    def __init__(self, model_def: BaseAutoMLModel, config: Optional[GlobalRunConfig] = None):
        self.model_def = model_def
        self.config = config or GlobalRunConfig()
        self.hpo_engine = HPOEngine(self.config)
        self.evaluator = EvaluatorEngine()

    def _cross_validate(self, params: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """Run K-Fold CV and return the full fold summary plus OOF predictions."""
        X = _defensive_coerce_numeric(X)
        n_splits = self.config.cv_folds
        if self.model_def.task_type == "classification":
            # Check if any class has fewer members than n_splits
            min_class_count = y.value_counts().min()
            if min_class_count < n_splits:
                # Fallback to standard KFold if StratifiedKFold is impossible
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_seed)
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_seed)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_seed)
        
        fold_metrics = []
        oof_preds = np.zeros(len(y)) if self.model_def.task_type == "classification" else np.zeros(len(y))
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            cat_cols = [c for c in X.columns if c not in num_cols]
            from app.automl.preprocessing import build_safe_tabular_pipeline
            preprocessor = build_safe_tabular_pipeline(num_cols, cat_cols)
            
            from sklearn.pipeline import Pipeline
            fold_params = self.model_def.prepare_estimator_params(params, y_train)
            
            # Log the computed weight per fold for auditability
            if "scale_pos_weight" in fold_params:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Fold in-fold weight (scale_pos_weight): {fold_params['scale_pos_weight']:.2f}")
            elif "class_weight" in fold_params:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Fold in-fold weight (class_weight): {fold_params['class_weight']}")

            base_estimator = self.model_def.build_estimator(fold_params)
            estimator = Pipeline([
                ("preprocessor", preprocessor),
                ("model", base_estimator)
            ])

            from app.automl.preprocessing import TargetTransformer
            fit_model, _ = TargetTransformer.wrap(estimator, y_train)

            fit_model.fit(X_train, y_train)
            
            y_pred = fit_model.predict(X_val)
            if hasattr(fit_model, "predict_proba"):
                y_prob = fit_model.predict_proba(X_val)
                oof_preds[val_idx] = y_prob[:, 1] if y_prob.shape[1] == 2 else np.argmax(y_prob, axis=1) # Simplified for binary
            else:
                y_prob = None
                oof_preds[val_idx] = y_pred
                
            if self.model_def.task_type == "classification":
                metrics = self.evaluator.evaluate_classification(y_val, y_pred, y_prob)
            else:
                metrics = self.evaluator.evaluate_regression(y_val, y_pred)
                
            fold_metrics.append(metrics)
            
        summary = self.evaluator.summarize_cv_folds(fold_metrics, self.config.primary_metric)
        return summary, oof_preds

    def run_hpo(self, X: pd.DataFrame, y: pd.Series, study_name: str) -> Tuple[Dict[str, Any], float]:
        """Run HPO using Optuna."""
        def objective(trial: optuna.Trial) -> float:
            params = self.model_def.get_search_space(trial)
            summary, _ = self._cross_validate(params, X, y)
            return summary["mean_cv_score"]
            
        best_params, best_value, stats = self.hpo_engine.optimize_candidate(
            study_name=study_name,
            objective_fn=objective,
            direction="maximize",
        )
        return best_params, best_value

    def train_final_model(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]):
        """Train the model on the full provided dataset with the best parameters."""
        X = _defensive_coerce_numeric(X)
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]
        from app.automl.preprocessing import build_safe_tabular_pipeline
        preprocessor = build_safe_tabular_pipeline(num_cols, cat_cols)
        
        from sklearn.pipeline import Pipeline
        final_params = self.model_def.prepare_estimator_params(params, y)
        base_estimator = self.model_def.build_estimator(final_params)
        estimator = Pipeline([
            ("preprocessor", preprocessor),
            ("model", base_estimator)
        ])

        from app.automl.preprocessing import TargetTransformer
        final_model, _ = TargetTransformer.wrap(estimator, y)

        final_model.fit(X, y)
        return final_model
