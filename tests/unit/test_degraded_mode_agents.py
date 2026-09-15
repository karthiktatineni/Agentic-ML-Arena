"""Tests for live LLM failure propagation into degraded mode."""

import pandas as pd

from app.agents.cleaning_agent import CleaningAgent
from app.agents.eda_agent import EDAAgent
from app.events.bus import event_bus
from app.governance.llm_governor import global_governor
from app.llm.base import LLMResponse


class FailingLLM:
    def generate(self, *args, **kwargs):
        return LLMResponse(
            content="",
            provider="test",
            model="failing-model",
            success=False,
            error="401 Unauthorized",
        )


def _drain_event_queue():
    events = []
    while not event_bus._queue.empty():
        events.append(event_bus._queue.get_nowait())
    return events


def test_cleaning_agent_llm_failure_trips_degraded_mode():
    global_governor.reset()
    _drain_event_queue()
    df = pd.DataFrame({"feature": [1, 2, None], "target": [0, 1, 0]})

    cleaned_df, plan = CleaningAgent(FailingLLM(), run_id="run_degraded_cleaning").run(df, "target")
    events = _drain_event_queue()

    assert global_governor.check_governor()
    assert "401 Unauthorized" in global_governor.trip_reason
    assert plan["reasoning"] == "Rule-based fallback (LLM unavailable)"
    assert cleaned_df["feature"].isna().sum() == 0
    assert any(
        e.event_type == "DEGRADED_MODE_ENTERED"
        and e.run_id == "run_degraded_cleaning"
        and e.subsystem == "CleaningAgent"
        for e in events
    )


def test_eda_agent_llm_failure_trips_degraded_mode():
    global_governor.reset()
    _drain_event_queue()
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 0, 0, 1]})

    result = EDAAgent(FailingLLM(), run_id="run_degraded_eda").run(df, "target", True)
    events = _drain_event_queue()

    assert global_governor.check_governor()
    assert "401 Unauthorized" in global_governor.trip_reason
    assert result["recommendations"]["handle_imbalance"] == "class_weights"
    assert any(
        e.event_type == "DEGRADED_MODE_ENTERED"
        and e.run_id == "run_degraded_eda"
        and e.subsystem == "EDAAgent"
        for e in events
    )
