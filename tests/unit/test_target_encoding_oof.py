"""Dedicated unit test for Out-Of-Fold Target Encoding (PRD v2 Section 13 & 59)."""

import pytest
import numpy as np
import pandas as pd
from app.automl.preprocessing import OutOfFoldTargetEncoder
from app.governance.fit_linter import FitLeakageLinter, FitLeakageViolation


def test_target_encoding_oof_isolation():
    """Verify that OutOfFoldTargetEncoder does not leak validation targets into validation features."""
    # Construct a dataset where category has a single unique row per category
    # In global target encoding, single unique categories perfectly reproduce the target (100% leak)
    n = 20
    df = pd.DataFrame({
        "category": [f"cat_{i}" for i in range(n)],
        "target": np.random.binomial(1, 0.5, size=n),
    })

    encoder = OutOfFoldTargetEncoder(categorical_cols=["category"], n_splits=5, smoothing=1.0)
    oof_df = encoder.fit_transform_oof(df[["category"]], df["target"])

    assert "category_target_enc" in oof_df.columns
    assert not oof_df["category_target_enc"].isna().any()

    # For any unique category seen in only 1 fold, its out-of-fold encoding MUST equal the fold global mean,
    # NEVER its own individual target value!
    for i in range(n):
        val = oof_df["category_target_enc"].iloc[i]
        # Should not be equal to individual binary target (0 or 1) unless fold mean happens to be identical
        assert 0.0 < val < 1.0


def test_fit_leakage_linter_rejects_non_oof_target_encoding():
    """Verify that FitLeakageLinter strictly rejects any spec with target encoding lacking OOF."""
    bad_spec = {
        "preprocessing": {
            "categorical_encoding": "target_encoding",
            "target_encode_oof": False,
        }
    }
    violations = FitLeakageLinter.lint_experiment_spec(bad_spec)
    assert any("Target encoding" in v for v in violations)

    with pytest.raises(FitLeakageViolation):
        FitLeakageLinter.assert_no_leakage(bad_spec)

    good_spec = {
        "preprocessing": {
            "categorical_encoding": "target_encoding",
            "target_encode_oof": True,
        }
    }
    assert len(FitLeakageLinter.lint_experiment_spec(good_spec)) == 0
