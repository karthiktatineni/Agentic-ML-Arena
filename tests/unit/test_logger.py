"""Unit tests for JSONL Logger."""

import os
import json
from app.cli.logger import JSONLLogger

def test_jsonl_logger(tmp_path):
    log_dir = str(tmp_path / ".logs")
    logger = JSONLLogger(log_dir=log_dir)
    
    logger.log_event("TEST_EVENT", {"key": "value"})
    logger.log_event("ANOTHER_EVENT", {"score": 0.95})
    
    assert os.path.exists(logger.log_file)
    
    with open(logger.log_file, 'r') as f:
        lines = f.readlines()
        
    assert len(lines) == 2
    
    event1 = json.loads(lines[0])
    assert event1["event_type"] == "TEST_EVENT"
    assert event1["payload"]["key"] == "value"
    assert "timestamp" in event1
    
    event2 = json.loads(lines[1])
    assert event2["event_type"] == "ANOTHER_EVENT"
    assert event2["payload"]["score"] == 0.95
