"""Unit tests for CheckpointManager."""

import os
import pytest
from app.reliability.checkpoint import CheckpointManager

def test_checkpoint_save_and_load(tmp_path):
    checkpoint_dir = str(tmp_path / ".checkpoints")
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    run_id = "test_run_123"
    state = {"completed_experiments": 5, "current_phase": "HPO"}
    
    manager.save_checkpoint(run_id, state)
    
    loaded_state = manager.load_checkpoint(run_id)
    assert loaded_state == state

def test_load_nonexistent_checkpoint(tmp_path):
    checkpoint_dir = str(tmp_path / ".checkpoints")
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    loaded_state = manager.load_checkpoint("nonexistent")
    assert loaded_state is None
