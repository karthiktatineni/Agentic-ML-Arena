"""Run State Serialization and Checkpointing (PRD v2 Section 36)."""

import os
import json
from typing import Dict, Any, Optional

class CheckpointManager:
    """Manages atomic writing and resuming of global run state."""
    
    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def _checkpoint_file(self, run_id: str) -> str:
        return os.path.join(self.checkpoint_dir, f"run_{run_id}.json")
        
    def save_checkpoint(self, run_id: str, state: Dict[str, Any]) -> None:
        """Atomically save the current state of a run."""
        file_path = self._checkpoint_file(run_id)
        temp_path = file_path + ".tmp"
        
        with open(temp_path, "w") as f:
            json.dump(state, f, indent=2)
            
        os.replace(temp_path, file_path)
        
    def load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load a run state from a checkpoint, or None if it doesn't exist."""
        file_path = self._checkpoint_file(run_id)
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
