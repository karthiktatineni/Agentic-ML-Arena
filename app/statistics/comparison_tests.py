"""Paired fold-level statistical comparison tests (PRD v2 Section 34)."""

from typing import List, Dict, Any
import numpy as np
from scipy import stats


def paired_model_comparison(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Execute paired fold-level test (Wilcoxon signed-rank or paired t-test).

    Args:
        scores_a: Fold-level metric scores for Model A (e.g. F1 across 5 folds).
        scores_b: Fold-level metric scores for Model B on the same folds.
        alpha: Nominal significance level.

    Returns:
        Dict with test_name, p_value, mean_diff, std_diff, is_statistically_significant.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Fold counts must match: {len(scores_a)} vs {len(scores_b)}")
    if len(scores_a) < 2:
        raise ValueError("Need at least 2 folds for paired comparison.")

    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)
    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    # If all fold differences are exactly 0
    if np.all(diffs == 0):
        return {
            "test_name": "identical_scores",
            "p_value": 1.0,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "statistic": 0.0,
            "raw_significant": False,
        }

    # Minimum fold count guard: meaningless p-values on N<3
    if len(arr_a) < 3:
        p_val = 0.0 if mean_diff > 0 else 1.0
        return {
            "test_name": "raw_difference",
            "p_value": p_val,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "statistic": 0.0,
            "raw_significant": mean_diff > 0,
        }

    # For small fold counts (n < 6), Wilcoxon cannot mathematically reach p < 0.05
    # (minimum two-sided p-value for n=5 is 0.0625). Use paired t-test for n < 6.
    if len(arr_a) < 6:
        res = stats.ttest_rel(arr_a, arr_b)
        test_name = "paired_t_test"
        p_val = float(res.pvalue)
        stat = float(res.statistic)
    else:
        try:
            res = stats.wilcoxon(arr_a, arr_b, zero_method="pratt")
            test_name = "wilcoxon_signed_rank"
            p_val = float(res.pvalue)
            stat = float(res.statistic)
        except Exception:
            res = stats.ttest_rel(arr_a, arr_b)
            test_name = "paired_t_test"
            p_val = float(res.pvalue)
            stat = float(res.statistic)

    return {
        "test_name": test_name,
        "p_value": p_val,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "statistic": stat,
        "raw_significant": bool(p_val < alpha),
    }

def bootstrap_paired_comparison(
    y_true: List[float],
    preds_a: List[float],
    preds_b: List[float],
    metric_func: Any,
    alpha: float = 0.05,
    n_iterations: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """Execute paired test via bootstrap resampling of predictions.
    
    Useful for creating paired samples on a single selection-validation set.
    """
    if len(y_true) != len(preds_a) or len(y_true) != len(preds_b):
        raise ValueError("Lengths of y_true, preds_a, and preds_b must match.")
        
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    
    y_true_np = np.array(y_true)
    preds_a_np = np.array(preds_a)
    preds_b_np = np.array(preds_b)
    
    scores_a = []
    scores_b = []
    
    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        scores_a.append(metric_func(y_true_np[indices], preds_a_np[indices]))
        scores_b.append(metric_func(y_true_np[indices], preds_b_np[indices]))
        
    return paired_model_comparison(scores_a, scores_b, alpha=alpha)
