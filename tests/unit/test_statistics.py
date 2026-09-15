"""Unit tests for paired comparisons and Holm-Bonferroni correction."""

import pytest
from app.statistics.comparison_tests import paired_model_comparison
from app.statistics.correction import MultipleTestingCorrection


def test_paired_comparison_identical():
    """Verify that identical fold scores yield p_value=1.0 and no significance."""
    scores = [0.90, 0.91, 0.92, 0.90, 0.91]
    res = paired_model_comparison(scores, scores)
    assert res["p_value"] == 1.0
    assert res["mean_diff"] == 0.0
    assert not res["raw_significant"]


def test_paired_comparison_significant_difference():
    """Verify that consistent superior performance across folds yields low p-value."""
    scores_a = [0.95, 0.96, 0.94, 0.95, 0.96]
    scores_b = [0.88, 0.89, 0.87, 0.88, 0.89]
    res = paired_model_comparison(scores_a, scores_b)
    assert res["raw_significant"] is True
    assert res["mean_diff"] > 0.05
    assert res["p_value"] < 0.05


def test_holm_bonferroni_multiple_comparisons():
    """Verify Holm-Bonferroni tightens significance thresholds as comparison count grows."""
    tracker = MultipleTestingCorrection(alpha=0.05, min_practical_effect_size=0.003)

    # First comparison: threshold is 0.05 / 1 = 0.05
    res1 = tracker.evaluate_comparison(
        candidate_a="Model_A",
        candidate_b="Model_B",
        raw_p_value=0.03,
        mean_diff=0.01,
    )
    assert res1["effective_threshold"] == 0.05
    assert res1["clears_statistical"] is True
    assert res1["is_significant_winner"] is True

    # After 10 comparisons, threshold is 0.05 / 10 = 0.005
    for i in range(8):
        tracker.evaluate_comparison(f"Cand_{i}", f"Cand_{i+1}", 0.10, 0.001)

    # 10th comparison: p_value 0.03 which cleared comparison 1 now fails!
    res10 = tracker.evaluate_comparison(
        candidate_a="Model_X",
        candidate_b="Model_Y",
        raw_p_value=0.03,
        mean_diff=0.01,
    )
    assert res10["effective_threshold"] == 0.005
    assert res10["clears_statistical"] is False
    assert res10["is_significant_winner"] is False


def test_practical_significance_floor():
    """Verify that statistically significant but tiny differences (<0.3%) fail practical floor."""
    tracker = MultipleTestingCorrection(alpha=0.05, min_practical_effect_size=0.003)

    # p-value is tiny (0.0001), but difference is only 0.0005 (0.05%)
    res = tracker.evaluate_comparison(
        candidate_a="Model_A",
        candidate_b="Model_B",
        raw_p_value=0.0001,
        mean_diff=0.0005,
    )
    assert res["clears_statistical"] is True
    assert res["clears_practical"] is False
    assert res["is_significant_winner"] is False  # Must clear both!
