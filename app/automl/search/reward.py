"""Normalized Reward Function for Search Controller (PRD v2 Section 56)."""

import numpy as np
from typing import Dict, Any, List

class RewardCalculator:
    """Calculates the composite reward for a given candidate."""

    def __init__(
        self, 
        w_perf: float = 1.0, 
        w_var: float = 0.2, 
        w_overfit: float = 0.3, 
        w_comp: float = 0.1, 
        w_time: float = 0.1
    ):
        """Initialize with weights for each penalty term."""
        self.w_perf = w_perf
        self.w_var = w_var
        self.w_overfit = w_overfit
        self.w_comp = w_comp
        self.w_time = w_time
        
        # Track history for min-max normalization
        self.history_perf = []
        self.history_var = []
        self.history_overfit = []
        self.history_comp = []
        self.history_time = []

    def _update_history(self, perf, var, overfit, comp, time_ratio):
        self.history_perf.append(perf)
        self.history_var.append(var)
        self.history_overfit.append(overfit)
        self.history_comp.append(comp)
        self.history_time.append(time_ratio)

    def _normalize(self, val: float, history: List[float]) -> float:
        """Min-max normalize a value based on historical bounds."""
        if not history:
            return 0.5 # Default middle ground if no history
            
        hist_min = min(history)
        hist_max = max(history)
        
        if hist_max - hist_min < 1e-6:
            return 0.5
            
        return (val - hist_min) / (hist_max - hist_min)

    def calculate_reward(
        self,
        mean_cv: float,
        std_cv: float,
        train_score: float,
        complexity_score: float,
        compute_time: float,
        time_budget: float
    ) -> float:
        """
        Calculate composite reward.
        
        Formula:
        R = w1 * norm(u_cv) 
            + w2 * (1 - norm(std_cv))
            + w3 * (1 - norm(max(0, train - u_cv)))
            - w4 * norm(complexity)
            - w5 * norm(time / budget)
        """
        overfit = max(0.0, train_score - mean_cv)
        time_ratio = compute_time / time_budget if time_budget > 0 else 1.0
        
        self._update_history(mean_cv, std_cv, overfit, complexity_score, time_ratio)
        
        norm_perf = self._normalize(mean_cv, self.history_perf)
        norm_var = self._normalize(std_cv, self.history_var)
        norm_overfit = self._normalize(overfit, self.history_overfit)
        norm_comp = self._normalize(complexity_score, self.history_comp)
        norm_time = self._normalize(time_ratio, self.history_time)
        
        reward = (
            self.w_perf * norm_perf
            + self.w_var * (1.0 - norm_var)
            + self.w_overfit * (1.0 - norm_overfit)
            - self.w_comp * norm_comp
            - self.w_time * norm_time
        )
        
        return reward
