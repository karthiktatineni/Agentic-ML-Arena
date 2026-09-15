"""Unit tests for Search Controller (PRD v2 Section 56)."""

from app.automl.search.state import SearchState
from app.automl.search.controller import SearchController

from app.statistics.correction import MultipleTestingCorrection

def test_search_state_updates():
    state = SearchState(total_budget_seconds=100.0)
    tracker = MultipleTestingCorrection(alpha=0.05)
    assert state.budget_exhaustion == 0.0
    
    state.consumed_budget_seconds = 50.0
    assert state.budget_exhaustion == 0.5
    
    state.update_scores("hash1", 0.8, [0.8, 0.8, 0.8], tracker)
    assert state.trials_completed == 1
    assert state.stagnation_count == 0
    
    state.update_scores("hash2", 0.9, [0.9, 0.9, 0.9], tracker)
    assert state.trials_completed == 2
    assert state.stagnation_count == 0
    assert state.top_3_cv_scores == [0.9, 0.8]
    assert state.best_experiment_hash == "hash2"
    
    state.update_scores("hash3", 0.82, [0.82, 0.82, 0.82], tracker)
    assert state.stagnation_count == 1 # Did not beat hash2
    assert tracker.total_comparisons_made == 2

def test_search_controller_stop_conditions():
    controller = SearchController()
    
    state = SearchState(total_budget_seconds=100.0, consumed_budget_seconds=100.0)
    assert controller.select_action(state) == "STOP"
    
    state = SearchState(stagnation_count=21)
    assert controller.select_action(state) == "STOP"
    
def test_search_controller_thompson_sampling():
    controller = SearchController(actions=["A", "B", "C"])
    state = SearchState(trials_completed=5) # skip early rule
    
    # Action A is very successful
    for _ in range(10):
        controller.update("A", 0.9)
    # Action B is failing
    for _ in range(10):
        controller.update("B", 0.1)
        
    action = controller.select_action(state)
    # Due to random chance it might pick B, but A or C is highly likely
    assert action in ["A", "B", "C"]
    assert controller.action_counts["A"] == 10

def test_search_controller_eliminate_candidates():
    controller = SearchController()
    
    # Candidate 1: High current score, flat learning curve
    c1 = {
        "id": "c1",
        "sample_sizes": [1000, 2500, 5000],
        "scores": [0.80, 0.81, 0.81] # Flat
    }
    
    # Candidate 2: Low current score, steep learning curve
    c2 = {
        "id": "c2",
        "sample_sizes": [1000, 2500, 5000],
        "scores": [0.70, 0.76, 0.80] # Steep
    }
    
    # With keep_ratio 0.5, we keep 1 candidate out of 2.
    # Extrapolating to 10000, c2 should overtake c1.
    kept = controller.eliminate_candidates([c1, c2], target_n=10000, keep_ratio=0.5)
    
    assert len(kept) == 1
    assert kept[0] == "c2"
