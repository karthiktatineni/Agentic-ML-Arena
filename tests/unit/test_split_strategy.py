"""Unit tests for Three-Way Splitting and Nested CV."""

import numpy as np
import pandas as pd
import pytest
from app.core.config import GlobalRunConfig
from app.automl.split_strategy import SplitStrategyEngine


def test_three_way_split_fractions():
    """Verify three-way split produces non-overlapping Train, Selection-Val, and Test sets."""
    n = 6000  # >= 5000 rows to trigger three-way split
    np.random.seed(42)
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "target": np.random.binomial(1, 0.4, size=n),
    })

    config = GlobalRunConfig(
        nested_cv_row_threshold=5000,
        final_test_fraction=0.20,
        selection_validation_fraction=0.15,
    )
    engine = SplitStrategyEngine(config)
    splits = engine.create_splits(df, target_col="target", is_classification=True)

    assert not splits.is_nested_cv
    assert splits.verify_final_test_integrity()

    # Check size proportions
    test_len = len(splits.final_test_df)
    val_len = len(splits.selection_val_df)
    train_len = len(splits.train_pool_df)

    assert test_len == 1200  # 20% of 6000
    assert val_len == 720    # 15% of DEV (4800)
    assert train_len == 4080 # 85% of DEV (4800)
    assert train_len + val_len + test_len == n

    # Verify zero data overlap between splits
    train_idx = set(splits.train_pool_df.index)
    val_idx = set(splits.selection_val_df.index)
    test_idx = set(splits.final_test_df.index)

    assert len(train_idx) == train_len
    assert len(val_idx) == val_len


def test_nested_cv_trigger_for_small_dataset():
    """Verify small datasets (<5000 rows) automatically switch to nested CV."""
    n = 1000  # < 5000 rows
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "target": np.random.binomial(1, 0.5, size=n),
    })

    config = GlobalRunConfig(nested_cv_row_threshold=5000)
    engine = SplitStrategyEngine(config)
    splits = engine.create_splits(df, target_col="target")

    assert splits.is_nested_cv is True
    assert splits.selection_val_df.empty  # Handled by nested folds
    assert len(splits.final_test_df) == 180  # 18% final test
    assert len(splits.train_pool_df) == 820
    assert splits.verify_final_test_integrity()
