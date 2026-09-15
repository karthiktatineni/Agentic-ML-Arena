"""Dedicated tests for FitLeakageLinter and Out-Of-Fold Target Encoding isolation."""

import pytest
import pandas as pd
from app.governance.fit_linter import FitLeakageLinter, FitLeakageViolation
from app.automl.preprocessing import OutOfFoldTargetEncoder

def test_linter_enforces_oof_target_encoding():
    """Verify the linter strictly requires out-of-fold computation for target encoding."""
    # A spec that requests target encoding but disables OOF computation
    leaky_spec = {
        "preprocessing": {
            "categorical_encoding": "target_encoding",
            "target_encode_oof": False
        }
    }
    
    violations = FitLeakageLinter.lint_experiment_spec(leaky_spec)
    assert any("out-of-fold" in v.lower() or "oof" in v.lower() for v in violations), "Linter failed to flag leaky target encoding."
    
    with pytest.raises(FitLeakageViolation):
        FitLeakageLinter.assert_no_leakage(leaky_spec)

    # A safe spec
    safe_spec = {
        "preprocessing": {
            "categorical_encoding": "target_encoding",
            "target_encode_oof": True
        }
    }
    assert len(FitLeakageLinter.lint_experiment_spec(safe_spec)) == 0

def test_oof_target_encoder_logic():
    """Verify that OutOfFoldTargetEncoder does not leak fold target information."""
    X = pd.DataFrame({"cat_feature": ["A", "B", "A", "B", "C", "C", "A", "B", "C", "A"]})
    y = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 0])
    
    encoder = OutOfFoldTargetEncoder(categorical_cols=["cat_feature"], n_splits=3, smoothing=1.0, cv_seed=42)
    
    # Fit transform should use OOF
    X_encoded_train = encoder.fit_transform_oof(X, y)
    assert "cat_feature_target_enc" in X_encoded_train.columns
    
    # The global mapping for test-time should also be fit
    assert "cat_feature" in encoder.encodings_
    
    # Transform test data
    X_test = pd.DataFrame({"cat_feature": ["A", "D"]})
    X_encoded_test = encoder.transform(X_test)
    
    # "D" was unseen, should fallback to global mean
    global_mean = y.mean()
    assert X_encoded_test.loc[1, "cat_feature_target_enc"] == global_mean
