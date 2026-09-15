"""State Machine for Experiment Lifecycle (PRD v2 Section 36)."""

from typing import List
from app.events.bus import event_bus
from app.events.schemas import PipelineStageUpdatedEvent

class ExperimentState:
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    RANKED = "RANKED"
    FAILED = "FAILED"

class StateMachineError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass

class ExperimentStateMachine:
    """Validates legal state transitions for an experiment's lifecycle."""
    
    VALID_TRANSITIONS = {
        ExperimentState.CREATED: [ExperimentState.VALIDATING, ExperimentState.FAILED],
        ExperimentState.VALIDATING: [ExperimentState.QUEUED, ExperimentState.FAILED],
        ExperimentState.QUEUED: [ExperimentState.CLAIMED, ExperimentState.FAILED],
        ExperimentState.CLAIMED: [ExperimentState.RUNNING, ExperimentState.QUEUED, ExperimentState.FAILED],
        ExperimentState.RUNNING: [ExperimentState.EVALUATING, ExperimentState.FAILED],
        ExperimentState.EVALUATING: [ExperimentState.COMPLETED, ExperimentState.FAILED],
        ExperimentState.COMPLETED: [ExperimentState.RANKED],
        ExperimentState.RANKED: [],
        ExperimentState.FAILED: [ExperimentState.QUEUED], # Allow retry from failure
    }

    def __init__(self, initial_state: str = ExperimentState.CREATED):
        self._current_state = initial_state

    @property
    def current_state(self) -> str:
        return self._current_state

    def transition(self, next_state: str) -> None:
        """Attempt to transition to a new state."""
        allowed = self.VALID_TRANSITIONS.get(self._current_state, [])
        if next_state not in allowed:
            raise StateMachineError(f"Illegal transition: {self._current_state} -> {next_state}")
        
        old_state = self._current_state
        self._current_state = next_state
        
        # Determine status mapping for the UI
        status = "ACTIVE"
        if next_state == ExperimentState.FAILED:
            status = "FAILED"
        elif next_state in (ExperimentState.COMPLETED, ExperimentState.RANKED):
            status = "COMPLETED"
            
        event_bus.publish(PipelineStageUpdatedEvent(
            run_id="unknown_run",  # The runner will inject this if we add it to sm
            stage_name=f"Experiment State: {next_state}",
            status=status,
            message=f"Transitioned from {old_state} to {next_state}"
        ))
