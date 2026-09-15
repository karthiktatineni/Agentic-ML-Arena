"""Drift Monitoring & Advisory Rollback Signal (PRD v2 Section 62 & 63)."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List

class DriftMonitor:
    """Monitors inference drift and raises an advisory rollback signal."""
    
    @staticmethod
    def calculate_ks_test(reference_scores: List[float], incoming_scores: List[float]) -> Dict[str, Any]:
        """
        Calculate Kolmogorov-Smirnov test to detect distribution drift.
        Returns the KS statistic and p-value. A low p-value (<0.05) indicates drift.
        """
        if len(reference_scores) == 0 or len(incoming_scores) == 0:
            return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}
            
        stat, p_val = stats.ks_2samp(reference_scores, incoming_scores)
        
        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "drift_detected": bool(p_val < 0.05)
        }

    @classmethod
    def evaluate_drift(cls, reference_scores: List[float], incoming_scores: List[float]) -> Dict[str, Any]:
        """
        Evaluate drift and return an advisory signal.
        This does NOT automatically rollback the model. It surfaces the signal
        so a human can review and explicitly approve a swap if necessary.
        """
        ks_results = cls.calculate_ks_test(reference_scores, incoming_scores)
        
        signal = {
            "rollback_signal": ks_results["drift_detected"],
            "reason": f"KS p-value {ks_results['p_value']:.4f} < 0.05" if ks_results["drift_detected"] else "No significant drift",
            "metrics": ks_results,
            "action_required": ks_results["drift_detected"],
            "auto_rollback_executed": False # Explicitly advisory per PRD v2 Section 63
        }
        
        return signal
