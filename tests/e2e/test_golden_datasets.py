"""Golden Dataset Suite (PRD v2 Section 59)."""

import pytest
import pandas as pd
import numpy as np
import asyncio

from app.core.config import GlobalRunConfig
from app.automl.split_strategy import SplitStrategyEngine

@pytest.fixture
def imbalanced_binary_df():
    """Dataset 1: Imbalanced binary classification."""
    n = 10000
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "target": np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    })
    return df

@pytest.fixture
def multiclass_df():
    """Dataset 2: Multiclass classification."""
    n = 10000
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "target": np.random.choice([0, 1, 2, 3], size=n)
    })
    return df

@pytest.fixture
def time_series_df():
    """Dataset 3: Time-series (temporal split)."""
    n = 1500
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        "feat1": np.random.randn(n),
        "target": np.random.randn(n)
    })
    return df

@pytest.fixture
def small_nested_cv_df():
    """Dataset 4: Small dataset (< threshold)."""
    n = 200
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "target": np.random.choice([0, 1], size=n)
    })
    return df

def test_imbalanced_binary_split(imbalanced_binary_df):
    config = GlobalRunConfig(target="target")
    engine = SplitStrategyEngine(config)
    splits = engine.create_splits(imbalanced_binary_df, "target")
    assert len(splits.train_pool_df) > 0
    assert not splits.is_nested_cv

def test_multiclass_split(multiclass_df):
    config = GlobalRunConfig(target="target")
    engine = SplitStrategyEngine(config)
    splits = engine.create_splits(multiclass_df, "target")
    assert len(splits.train_pool_df) > 0

def test_time_series_temporal_split(time_series_df):
    config = GlobalRunConfig(target="target", time_column="timestamp")
    engine = SplitStrategyEngine(config)
    splits = engine.create_splits(time_series_df, "target")
    
    # Assert temporal ordering is respected (train < val < test)
    train_max = splits.train_pool_df["timestamp"].max()
    val_min = splits.selection_val_df["timestamp"].min() if not splits.selection_val_df.empty else None
    test_min = splits.final_test_df["timestamp"].min()
    
    if val_min:
        assert train_max < val_min
        assert splits.selection_val_df["timestamp"].max() < test_min
    else:
        assert train_max < test_min

def test_nested_cv_routing(small_nested_cv_df):
    config = GlobalRunConfig(target="target")
    engine = SplitStrategyEngine(config)
    splits = engine.create_splits(small_nested_cv_df, "target")
    assert splits.is_nested_cv is True
    assert splits.selection_val_df.empty
