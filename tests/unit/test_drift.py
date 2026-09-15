"""Unit tests for Drift Monitor."""

from app.production.drift import DriftMonitor
import numpy as np

def test_drift_monitor():
    np.random.seed(42)
    # Same distribution
    ref = np.random.normal(0, 1, 1000).tolist()
    inc_same = np.random.normal(0, 1, 100).tolist()
    
    result_same = DriftMonitor.evaluate_drift(ref, inc_same)
    assert not result_same["rollback_signal"]
    assert not result_same["auto_rollback_executed"]
    
    # Different distribution
    inc_diff = np.random.normal(2, 1, 100).tolist()
    result_diff = DriftMonitor.evaluate_drift(ref, inc_diff)
    
    assert result_diff["rollback_signal"]
    assert not result_diff["auto_rollback_executed"]
