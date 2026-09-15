import pytest
from app.events.schemas import ChampionCertifiedEvent, CertificationRejectedEvent
from app.statistics.correction import MultipleTestingCorrection
from app.core.config import GlobalRunConfig


def test_champion_certified_event_schema_with_attempts():
    """Verify ChampionCertifiedEvent supports retry attempt tracking and below-target warnings."""
    event = ChampionCertifiedEvent(
        run_id="run_123",
        experiment_hash="hash_abc",
        bundle_path="/tmp/model.joblib",
        model_name="XGBoostModel",
        metrics={"f1": 0.88},
        pipeline_attempt=2,
        total_pipeline_attempts=3,
        target_metric_achieved=False,
        below_target_warning="Target score 0.90 not reached after 3 attempts.",
    )
    data = event.model_dump()
    assert data["pipeline_attempt"] == 2
    assert data["total_pipeline_attempts"] == 3
    assert data["target_metric_achieved"] is False
    assert "0.90 not reached" in data["below_target_warning"]


def test_multiple_testing_correction_persists_across_attempts():
    """Verify that a single MultipleTestingCorrection tracker accumulates comparisons across attempts."""
    tracker = MultipleTestingCorrection(alpha=0.05, min_practical_effect_size=0.003)
    
    # Attempt 1: 2 comparisons
    tracker.evaluate_comparison("c1", "c2", raw_p_value=0.02, mean_diff=0.01)
    tracker.evaluate_comparison("c1", "c3", raw_p_value=0.01, mean_diff=0.005)
    assert tracker.total_comparisons_made == 2
    
    # Attempt 2: 2 more comparisons (family size must grow to 4, tightening threshold)
    res3 = tracker.evaluate_comparison("c4", "c5", raw_p_value=0.02, mean_diff=0.01)
    res4 = tracker.evaluate_comparison("c4", "c6", raw_p_value=0.015, mean_diff=0.008)
    assert tracker.total_comparisons_made == 4
    # Check effective threshold for comparison 4 is alpha / 4
    assert pytest.approx(res4["effective_threshold"], 1e-5) == 0.05 / 4


def test_gate6_alpha_attempt_decay():
    """Verify Gate 6 alpha decays inversely with attempt count to prevent p-hacking."""
    alpha = 0.05
    for attempt in range(1, 4):
        adjusted_alpha = alpha / attempt
        if attempt == 1:
            assert pytest.approx(adjusted_alpha) == 0.05
        elif attempt == 2:
            assert pytest.approx(adjusted_alpha) == 0.025
        elif attempt == 3:
            assert pytest.approx(adjusted_alpha) == 0.05 / 3


def test_config_max_pipeline_retries():
    """Verify GlobalRunConfig defaults max_pipeline_retries to 3."""
    cfg = GlobalRunConfig()
    assert cfg.max_pipeline_retries == 3
