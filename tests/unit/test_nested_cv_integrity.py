import pytest
import pandas as pd
import numpy as np
from app.automl.agent import ModelAgent
from app.core.config import GlobalRunConfig

def test_nested_cv_oof_predictions():
    """Verify OOF predictions don't leak inner fold data (PRD v2 Section 14)."""
    
    # We create a synthetic target that exactly matches a feature.
    # A model that leaks data would get a perfect score.
    # An out-of-fold prediction for K-Fold CV should be unable to perfectly memorize
    # if we force the model to be a simple decision tree that easily overfits.
    
    df = pd.DataFrame({
        "feat1": np.random.rand(100),
        "target": np.random.choice([0, 1], size=100)
    })
    
    config = GlobalRunConfig(target="target", cv_folds=5)
    
    # Use Random Forest since it's easy to overfit
    from app.automl.models.random_forest_model import RandomForestModel
    model_def = RandomForestModel(task_type="classification")
    
    agent = ModelAgent(model_def=model_def, config=config)
    
    X = df[["feat1"]]
    y = df["target"]
    
    # Test with standard CV
    params = {"n_estimators": 1, "max_depth": None} # Very high variance, overfits easily
    
    cv_summary, oof_preds = agent._cross_validate(params, X, y)
    
    assert cv_summary["mean_cv_score"] >= 0.0
    assert len(cv_summary["fold_scores"]) == config.cv_folds
    assert oof_preds is not None
    assert len(oof_preds) == 100
    
    # If it leaked, oof_preds would be near perfect (accuracy ~ 1.0)
    # Since it's out-of-fold and the feature is random noise, it should be ~0.5
    accuracy = np.mean(oof_preds == y)
    
    assert accuracy < 0.8, "OOF predictions show severe leakage (memorization)"
