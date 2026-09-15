"""Unit tests for Progressive Scaler."""

import numpy as np
from app.automl.search.scaling import ProgressiveScaler

def test_power_law_fit():
    # Simulate a learning curve: S(n) = 0.95 - 0.5 * n^(-0.5)
    # n = [1000, 2500, 5000, 10000]
    sample_sizes = [1000, 2500, 5000, 10000]
    
    a_true, b_true, gamma_true = 0.95, 0.5, 0.5
    scores = [
        a_true - b_true * np.power(n, -gamma_true) for n in sample_sizes
    ]
    
    a, b, gamma = ProgressiveScaler.fit_learning_curve(sample_sizes, scores)
    
    # Should fit closely
    assert np.isclose(a, a_true, atol=0.05)
    
def test_extrapolate_score():
    sample_sizes = [1000, 2500, 5000]
    a_true, b_true, gamma_true = 0.95, 0.5, 0.5
    scores = [
        a_true - b_true * np.power(n, -gamma_true) for n in sample_sizes
    ]
    
    predicted_10k = ProgressiveScaler.extrapolate_score(sample_sizes, scores, 10000)
    actual_10k = a_true - b_true * np.power(10000, -gamma_true)
    
    assert np.isclose(predicted_10k, actual_10k, atol=0.05)
    
def test_extrapolate_insufficient_points():
    sample_sizes = [1000, 2500]
    scores = [0.7, 0.8]
    
    predicted = ProgressiveScaler.extrapolate_score(sample_sizes, scores, 5000)
    assert predicted == 0.8 # Falls back to max seen
