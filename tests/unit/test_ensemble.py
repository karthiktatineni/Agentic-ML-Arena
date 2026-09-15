"""Unit tests for EnsembleEngine."""

import numpy as np
import pandas as pd
from app.automl.ensemble import EnsembleEngine

def test_uniform_average():
    preds1 = np.array([0.1, 0.9, 0.4])
    preds2 = np.array([0.3, 0.7, 0.6])
    
    avg = EnsembleEngine.uniform_average([preds1, preds2])
    np.testing.assert_array_almost_equal(avg, np.array([0.2, 0.8, 0.5]))

def test_stacking_classification():
    preds1 = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9])
    preds2 = np.array([0.2, 0.8, 0.1, 0.7, 0.3, 0.8])
    
    y_true = pd.Series([0, 1, 0, 1, 0, 1])
    
    meta_learner = EnsembleEngine.train_stacking_meta_learner(
        oof_predictions=[preds1, preds2],
        y_true=y_true,
        task_type="classification"
    )
    
    test_preds1 = np.array([0.15, 0.85])
    test_preds2 = np.array([0.25, 0.75])
    
    final_preds = EnsembleEngine.predict_stacking(
        meta_learner,
        test_predictions=[test_preds1, test_preds2],
        task_type="classification"
    )
    
    assert final_preds.shape == (2,)
    assert (final_preds >= 0).all() and (final_preds <= 1).all()
