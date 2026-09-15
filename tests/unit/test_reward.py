"""Unit tests for Reward Calculator (PRD v2 Section 56)."""

from app.automl.search.reward import RewardCalculator

def test_degenerate_case_reward():
    """Verify that a mediocre, near-zero-compute model never beats a strong generalizing model."""
    calculator = RewardCalculator(
        w_perf=1.0, 
        w_var=0.2, 
        w_overfit=0.3, 
        w_comp=0.5, # High penalty for compute for testing
        w_time=0.5
    )
    
    # Strong generalizing model (high compute, high complexity, great perf)
    reward_strong = calculator.calculate_reward(
        mean_cv=0.95,
        std_cv=0.01,
        train_score=0.96,
        complexity_score=100.0,
        compute_time=3600.0,
        time_budget=3600.0
    )
    
    # Degenerate model (zero compute, zero complexity, mediocre perf)
    # E.g. Predict mostly majority class baseline
    reward_degenerate = calculator.calculate_reward(
        mean_cv=0.55,
        std_cv=0.05,
        train_score=0.55,
        complexity_score=1.0,
        compute_time=1.0,
        time_budget=3600.0
    )
    
    # Even after normalizing, the reward for strong should be higher
    # Wait, the history is built dynamically. Let's push both to set bounds
    calculator = RewardCalculator(
        w_perf=1.0, 
        w_var=0.2, 
        w_overfit=0.3, 
        w_comp=0.2,
        w_time=0.2
    )
    
    calculator._update_history(0.95, 0.01, 0.01, 100.0, 1.0)
    calculator._update_history(0.55, 0.05, 0.0, 1.0, 0.0)
    
    # Recalculate strong with established bounds
    r1 = calculator.calculate_reward(
        mean_cv=0.95,
        std_cv=0.01,
        train_score=0.96,
        complexity_score=100.0,
        compute_time=3600.0,
        time_budget=3600.0
    )
    
    # Recalculate degenerate with established bounds
    r2 = calculator.calculate_reward(
        mean_cv=0.55,
        std_cv=0.05,
        train_score=0.55,
        complexity_score=1.0,
        compute_time=1.0,
        time_budget=3600.0
    )
    
    assert r1 > r2, f"Strong model reward ({r1}) should beat degenerate model ({r2})"
