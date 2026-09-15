"""Fault Injection Suite (PRD v2 Section 59)."""

import pytest
import multiprocessing
import time
import os
import signal
from unittest.mock import patch

from app.reliability.state_machine import ExperimentStateMachine, ExperimentState, StateMachineError
from app.reliability.locks import FileLockManager

def worker_simulate_crash(lock_dir: str, experiment_hash: str):
    """Simulates a worker crashing mid-execution after acquiring a lock."""
    lock = FileLockManager(lock_dir=lock_dir)
    lock.acquire(experiment_hash, worker_id="doomed_worker", ttl=300)
    # The process exits abruptly, leaving the lock dangling
    os._exit(1)

def test_lock_holder_crash_mid_experiment(tmp_path):
    """Tests that a dangling lock eventually expires and can be reclaimed."""
    lock_dir = str(tmp_path)
    experiment_hash = "fault_hash"
    
    # 1. Spawn a doomed worker
    p = multiprocessing.Process(target=worker_simulate_crash, args=(lock_dir, experiment_hash))
    p.start()
    p.join(timeout=2)
    
    # 2. Main worker tries to acquire lock
    lock = FileLockManager(lock_dir=lock_dir)
    
    # It should fail because the doomed worker still holds it (TTL 300)
    acquired = lock.acquire(experiment_hash, worker_id="main_worker", ttl=300)
    assert not acquired
    
    # 3. Simulate TTL expiration by manipulating the JSON lock file
    import json
    lock_file = os.path.join(lock_dir, f"lock_{experiment_hash}.json")
    with open(lock_file, "r") as f:
        data = json.load(f)
    
    data["expires_at"] = time.time() - 10 # Expired
    with open(lock_file, "w") as f:
        json.dump(data, f)
        
    # 4. Now main worker should successfully acquire it
    acquired = lock.acquire(experiment_hash, worker_id="main_worker", ttl=300)
    assert acquired

def test_state_machine_invalid_transition():
    """Verify state machine strictly rejects illegal transitions."""
    sm = ExperimentStateMachine(ExperimentState.CREATED)
    
    # Try skipping VALIDATING
    with pytest.raises(StateMachineError):
        sm.transition(ExperimentState.RUNNING)

def test_llm_timeout_budget_exhaustion():
    """Simulate SearchController hitting budget limit (PRD 59)."""
    from app.automl.search.state import SearchState
    from app.automl.search.controller import SearchController
    
    state = SearchState(total_budget_seconds=10.0)
    state.consumed_budget_seconds = 11.0 # Exhausted
    
    controller = SearchController()
    action = controller.select_action(state)
    assert action == "STOP" # Enforces exhaustion gracefully
