"""Unit tests for Dashboard."""

from app.api.schemas.experiment import ExperimentObject, ExperimentMetrics
from app.cli.dashboard import Dashboard

from app.events.schemas import ExperimentResultEvent, AgentDecisionEvent

def test_dashboard_leaderboard():
    dash = Dashboard()
    
    e1 = ExperimentResultEvent(
        experiment_hash="hash1",
        run_id="run_1",
        model_family="xgboost",
        state="COMPLETED",
        cv_score=0.95
    )
    
    e2 = ExperimentResultEvent(
        experiment_hash="hash2",
        run_id="run_1",
        model_family="lightgbm",
        state="COMPLETED",
        cv_score=0.90
    )
    
    dash.handle_event(e1)
    dash.handle_event(e2)
    
    table = dash.generate_leaderboard()
    assert table.title == "Leaderboard (Top 10)"
    
    # Check that rows were added
    assert len(table.rows) == 2
    
def test_dashboard_messages():
    dash = Dashboard()
    for i in range(60):
        evt = AgentDecisionEvent(run_id="run_1", agent_name="TestAgent", decision_action=f"Msg {i}", confidence=0.9, reasoning_summary="test")
        dash.handle_event(evt)
    
    assert len(dash.messages) == 15
    assert "Msg 59" in dash.messages[-1]
    assert "Msg 45" in dash.messages[0]
