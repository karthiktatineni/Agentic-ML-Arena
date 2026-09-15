"""Three-Way Splitting Strategy and Nested CV Engine (PRD v2 Section 14)."""

import hashlib
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

from app.core.config import GlobalRunConfig
from app.events.bus import event_bus
from app.events.schemas import PipelineStageUpdatedEvent


class DatasetSplits:
    """Encapsulates the immutable three-way split or nested CV data slices."""

    def __init__(
        self,
        train_pool_df: pd.DataFrame,
        selection_val_df: pd.DataFrame,
        final_test_df: pd.DataFrame,
        target_col: str,
        is_nested_cv: bool = False,
        split_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.train_pool_df = train_pool_df
        self.selection_val_df = selection_val_df
        self.final_test_df = final_test_df
        self.target_col = target_col
        self.is_nested_cv = is_nested_cv
        self.split_metadata = split_metadata or {}

        # Compute SHA-256 fingerprint of final test set to verify immutability
        self.final_test_hash = self._hash_df(final_test_df)
        self.selection_val_hash = self._hash_df(selection_val_df)
        self.train_pool_hash = self._hash_df(train_pool_df)

    @staticmethod
    def _hash_df(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    def verify_final_test_integrity(self) -> bool:
        """Confirm final test set has not mutated since initial split."""
        return self._hash_df(self.final_test_df) == self.final_test_hash


class SplitStrategyEngine:
    """Engine responsible for three-way splitting or nested CV fallback."""

    def __init__(self, config: Optional[GlobalRunConfig] = None):
        self.config = config or GlobalRunConfig()

    def create_splits(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_classification: bool = True,
        time_col: Optional[str] = None,
        group_col: Optional[str] = None,
    ) -> DatasetSplits:
        """Create Train+CV Pool, Selection-Validation, and Final Test splits."""
        time_col = time_col or self.config.time_column
        group_col = group_col or self.config.group_column
        
        total_rows = len(df)
        seed = self.config.random_seed

        # Check for small dataset threshold (< 5000 rows)
        if total_rows < self.config.nested_cv_row_threshold:
            # Nested CV scheme:
            # Final Test is still carved out (20%), remaining 80% is processed via nested outer/inner CV
            test_size = self.config.final_test_fraction

            if time_col:
                # Temporal forward-split for nested CV test set
                df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
                test_split_idx = int(total_rows * (1.0 - test_size))
                dev_df = df_sorted.iloc[:test_split_idx].copy()
                final_test_df = df_sorted.iloc[test_split_idx:].copy()
            else:
                stratify = None
                if is_classification and df[target_col].nunique() > 1:
                    if df[target_col].value_counts().min() >= 2:
                        stratify = df[target_col]
                dev_df, final_test_df = train_test_split(
                    df, test_size=test_size, random_state=seed, stratify=stratify
                )

            
            event_bus.publish(PipelineStageUpdatedEvent(
                run_id="unknown_run",  # Will populate later if known
                stage_name="Dataset Splitting",
                status="COMPLETED",
                message=f"Nested CV activated (N={total_rows}). Train/Dev: {len(dev_df)}, Test: {len(final_test_df)}",
                metadata={"strategy": "nested_cv", "total_rows": total_rows}
            ))
            
            # In nested CV mode, selection-val is dynamically evaluated in outer folds
            return DatasetSplits(
                train_pool_df=dev_df,
                selection_val_df=pd.DataFrame(),  # outer folds handle validation
                final_test_df=final_test_df,
                target_col=target_col,
                is_nested_cv=True,
                split_metadata={
                    "total_rows": total_rows,
                    "dev_rows": len(dev_df),
                    "final_test_rows": len(final_test_df),
                    "mode": "nested_cv",
                },
            )

        # Three-way split for standard datasets (>= 5000 rows):
        # 1. Carve out FINAL TEST (e.g. 18%)
        test_size = self.config.final_test_fraction
        stratify = df[target_col] if (is_classification and not time_col) else None

        if time_col:
            # Temporal forward-split: Sort chronologically
            df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
            test_split_idx = int(total_rows * (1.0 - test_size))
            dev_df = df_sorted.iloc[:test_split_idx].copy()
            final_test_df = df_sorted.iloc[test_split_idx:].copy()

            # Selection-validation carved from end of DEV
            dev_rows = len(dev_df)
            val_split_idx = int(dev_rows * (1.0 - self.config.selection_validation_fraction))
            train_pool_df = dev_df.iloc[:val_split_idx].copy()
            selection_val_df = dev_df.iloc[val_split_idx:].copy()

        else:
            # Stratified random split
            dev_df, final_test_df = train_test_split(
                df, test_size=test_size, random_state=seed, stratify=stratify
            )

            # Carve out SELECTION-VALIDATION from DEV
            stratify_dev = None
            if is_classification and dev_df[target_col].nunique() > 1:
                if dev_df[target_col].value_counts().min() >= 2:
                    stratify_dev = dev_df[target_col]
                    
            train_pool_df, selection_val_df = train_test_split(
                dev_df,
                test_size=self.config.selection_validation_fraction,
                random_state=seed,
                stratify=stratify_dev,
            )

        
        event_bus.publish(PipelineStageUpdatedEvent(
            run_id="unknown_run",
            stage_name="Dataset Splitting",
            status="COMPLETED",
            message=f"Three-Way Split (N={total_rows}). Train: {len(train_pool_df)}, Val: {len(selection_val_df)}, Test: {len(final_test_df)}",
            metadata={"strategy": "three_way", "total_rows": total_rows}
        ))
        
        return DatasetSplits(
            train_pool_df=train_pool_df.reset_index(drop=True),
            selection_val_df=selection_val_df.reset_index(drop=True),
            final_test_df=final_test_df.reset_index(drop=True),
            target_col=target_col,
            is_nested_cv=False,
            split_metadata={
                "total_rows": total_rows,
                "train_pool_rows": len(train_pool_df),
                "selection_val_rows": len(selection_val_df),
                "final_test_rows": len(final_test_df),
                "mode": "three_way",
            },
        )
