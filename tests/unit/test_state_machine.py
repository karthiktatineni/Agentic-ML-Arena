"""Unit tests for ExperimentStateMachine."""

import pytest
from app.reliability.state_machine import ExperimentStateMachine, ExperimentState, StateMachineError

def test_valid_transitions():
    sm = ExperimentStateMachine()
    assert sm.current_state == ExperimentState.CREATED
    
    sm.transition(ExperimentState.VALIDATING)
    assert sm.current_state == ExperimentState.VALIDATING
    
    sm.transition(ExperimentState.QUEUED)
    sm.transition(ExperimentState.CLAIMED)
    sm.transition(ExperimentState.RUNNING)
    sm.transition(ExperimentState.EVALUATING)
    sm.transition(ExperimentState.COMPLETED)
    sm.transition(ExperimentState.RANKED)
    assert sm.current_state == ExperimentState.RANKED

def test_invalid_transitions():
    sm = ExperimentStateMachine()
    
    with pytest.raises(StateMachineError):
        sm.transition(ExperimentState.RUNNING)

def test_failure_and_retry():
    sm = ExperimentStateMachine(ExperimentState.RUNNING)
    sm.transition(ExperimentState.FAILED)
    assert sm.current_state == ExperimentState.FAILED
    
    sm.transition(ExperimentState.QUEUED)
    assert sm.current_state == ExperimentState.QUEUED
