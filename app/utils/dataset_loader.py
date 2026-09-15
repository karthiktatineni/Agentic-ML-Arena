"""Fast dataset loading with optional Polars backend."""

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV using Polars when available, otherwise pandas."""
    path = Path(path)
    try:
        import polars as pl

        return pl.read_csv(path).to_pandas()
    except ImportError:
        return pd.read_csv(path)
