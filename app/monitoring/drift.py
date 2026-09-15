"""Continuous KS-Test Drift Monitoring (PRD v2 Section 62)."""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Any, List

class DriftMonitor:
    """Monitors incoming data streams for covariate drift using KS tests."""
    
    def __init__(self, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold
        self.baseline_stats: Dict[str, np.ndarray] = {}
        
    def fit_baseline(self, df_baseline: pd.DataFrame, features: List[str]):
        """Compute and store baseline empirical distributions for features."""
        for feat in features:
            if feat in df_baseline.columns and pd.api.types.is_numeric_dtype(df_baseline[feat]):
                # Store the exact values (or a subsample if huge) to run 2-sample KS test later
                self.baseline_stats[feat] = df_baseline[feat].dropna().values
                
    def detect_drift(self, df_new: pd.DataFrame) -> Dict[str, Any]:
        """Compare new data against baseline using KS Test."""
        results = {}
        drift_detected = False
        
        for feat, baseline_vals in self.baseline_stats.items():
            if feat in df_new.columns and pd.api.types.is_numeric_dtype(df_new[feat]):
                new_vals = df_new[feat].dropna().values
                if len(new_vals) == 0:
                    continue
                    
                ks_stat, p_value = ks_2samp(baseline_vals, new_vals)
                is_drift = p_value < self.p_value_threshold
                
                results[feat] = {
                    "ks_stat": float(ks_stat),
                    "p_value": float(p_value),
                    "drift_detected": bool(is_drift)
                }
                
                if is_drift:
                    drift_detected = True
                    
        return {
            "drift_detected": drift_detected,
            "feature_metrics": results
        }
