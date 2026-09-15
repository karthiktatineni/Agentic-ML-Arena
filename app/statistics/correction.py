"""Holm-Bonferroni multiple comparisons correction and practical significance."""

from typing import List, Dict, Any, Tuple
import numpy as np


class MultipleTestingCorrection:
    """Tracks running comparisons count and computes Holm-Bonferroni adjusted thresholds."""

    def __init__(self, alpha: float = 0.05, min_practical_effect_size: float = 0.003):
        self.alpha = alpha
        self.min_practical_effect_size = min_practical_effect_size
        self.total_comparisons_made: int = 0
        self.history: List[Dict[str, Any]] = []

    def evaluate_comparison(
        self,
        candidate_a: str,
        candidate_b: str,
        raw_p_value: float,
        mean_diff: float,
    ) -> Dict[str, Any]:
        """Evaluate a single pairwise comparison within the running experiment context."""
        self.total_comparisons_made += 1
        m = self.total_comparisons_made

        # Holm-Bonferroni conservative threshold for comparison m
        bonferroni_threshold = self.alpha / m

        # Check practical significance (mean diff >= 0.003)
        clears_practical = abs(mean_diff) >= self.min_practical_effect_size
        clears_statistical = raw_p_value < bonferroni_threshold

        result = {
            "comparison_index": m,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "raw_p_value": raw_p_value,
            "effective_threshold": bonferroni_threshold,
            "mean_diff": mean_diff,
            "clears_statistical": clears_statistical,
            "clears_practical": clears_practical,
            "is_significant_winner": clears_statistical and clears_practical and (mean_diff > 0),
        }
        self.history.append(result)
        return result

    @staticmethod
    def adjust_p_values_holm(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, float, bool]]:
        """Batch Holm-Bonferroni step-down correction on a list of raw p-values.

        Returns list of (raw_p, adjusted_p, is_significant).
        """
        m = len(p_values)
        if m == 0:
            return []

        # Sort p-values ascending
        indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * m
        significant = [False] * m

        running_max = 0.0
        for rank, (orig_idx, p_val) in enumerate(indexed_p):
            # Holm formula: p * (m - rank)
            adj_p = min(1.0, max(running_max, p_val * (m - rank)))
            running_max = adj_p
            adjusted[orig_idx] = adj_p
            significant[orig_idx] = bool(adj_p < alpha)

        return [(p_values[i], adjusted[i], significant[i]) for i in range(m)]
