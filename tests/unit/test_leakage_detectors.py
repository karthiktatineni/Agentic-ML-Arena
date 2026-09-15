"""Unit tests for the 4 concrete leakage detectors and fit-linter."""

import pytest
import numpy as np
import pandas as pd
from app.governance.leakage_detectors import LeakageDetectorEngine
from app.governance.fit_linter import FitLeakageLinter, FitLeakageViolation


def test_detector_1_univariate_perfect_predictor():
    """Verify that Detector 1 reliably catches single-feature target leakage."""
    np.random.seed(42)
    n = 200
    y = np.random.binomial(1, 0.5, size=n)

    df = pd.DataFrame({
        "clean_feature_1": np.random.randn(n),
        "clean_feature_2": np.random.randn(n),
        # Leaked feature with 99% accuracy
        "leaked_feature": y * 1.0 + np.random.normal(0, 0.05, size=n),
        "target": y,
    })

    engine = LeakageDetectorEngine(auc_threshold=0.98)
    findings = engine.detector_1_univariate_screen(df, target_col="target", is_classification=True)

    excluded_cols = [f.column for f in findings if f.action == "exclude"]
    assert "leaked_feature" in excluded_cols
    assert "clean_feature_1" not in excluded_cols
    assert "clean_feature_2" not in excluded_cols


def test_detector_2_temporal_availability():
    """Verify that Detector 2 auto-excludes post-outcome / future features."""
    cutoff = pd.Timestamp("2026-01-01 00:00:00")
    metadata = {
        "pre_event_metric": {"generation_timestamp": "2025-12-15 10:00:00"},
        "post_outcome_flag": {"is_post_outcome_event": True},
        "late_arriving_feature": {"generation_timestamp": "2026-01-05 12:00:00"},
    }

    engine = LeakageDetectorEngine()
    findings = engine.detector_2_temporal_availability(metadata, prediction_cutoff_timestamp=cutoff)

    excluded_cols = [f.column for f in findings if f.action == "exclude"]
    assert "post_outcome_flag" in excluded_cols
    assert "late_arriving_feature" in excluded_cols
    assert "pre_event_metric" not in excluded_cols


def test_detector_3_cross_split_duplicates():
    """Verify Detector 3 flags rows appearing across split boundaries."""
    train_data = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [10, 20, 30, 40]})
    val_data = pd.DataFrame({"x1": [5, 6], "x2": [50, 60]})
    # Row (3, 30) accidentally present in both train and test
    test_data = pd.DataFrame({"x1": [3, 7], "x2": [30, 70]})

    engine = LeakageDetectorEngine()
    report = engine.detector_3_cross_split_duplicates(train_data, val_data, test_data)

    assert report["has_leakage"] is True
    assert report["train_test_overlap_count"] == 1
    assert report["train_val_overlap_count"] == 0


def test_proxy_order_check():
    """Verify that serial ID / row index proxy features are flagged."""
    n = 100
    df = pd.DataFrame({
        "serial_id": np.arange(n),
        "random_val": np.random.randn(n),
        "target": np.random.binomial(1, 0.5, size=n),
    })

    engine = LeakageDetectorEngine(proxy_correlation_threshold=0.90)
    findings = engine.proxy_order_check(df, target_col="target")

    flagged = [f.column for f in findings]
    assert "serial_id" in flagged
    assert "random_val" not in flagged


def test_fit_leakage_linter():
    """Verify Detector 4 structural fit-leakage linter blocks unisolated preprocessors."""
    bad_spec_1 = {
        "model": "xgboost",
        "preprocessing": {"fit_on_full_dataset": True}
    }
    violations_1 = FitLeakageLinter.lint_experiment_spec(bad_spec_1)
    assert any("fit_on_full_dataset" in v for v in violations_1)

    bad_spec_2 = {
        "model": "catboost",
        "preprocessing": {"categorical_encoding": "target_encoding", "target_encode_oof": False}
    }
    violations_2 = FitLeakageLinter.lint_experiment_spec(bad_spec_2)
    assert any("Target encoding" in v for v in violations_2)

    good_spec = {
        "model": "lightgbm",
        "preprocessing": {
            "fit_on_full_dataset": False,
            "categorical_encoding": "onehot",
            "scaling": "standard"
        }
    }
    assert len(FitLeakageLinter.lint_experiment_spec(good_spec)) == 0

    with pytest.raises(FitLeakageViolation):
        FitLeakageLinter.assert_no_leakage(bad_spec_1)
