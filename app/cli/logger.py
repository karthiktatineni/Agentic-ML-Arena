"""Structured JSONL Logging (PRD v2 Section 8)."""

import json
import os
import time
from typing import Dict, Any

class JSONLLogger:
    def __init__(self, log_dir: str = ".logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Start a new log file per session
        session_id = int(time.time())
        self.log_file = os.path.join(log_dir, f"automl_run_{session_id}.jsonl")
        
    def log_event(self, event_type: str, payload: Dict[str, Any]):
        """Append a structured event to the log."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "payload": payload
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
