"""Progressive Scaling & Learning Curve Extrapolation (PRD v2 Section 37)."""

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Tuple

class ProgressiveScaler:
    """Manages progressive dataset scaling and extrapolates learning curves."""
    
    RUNGS = [0.1, 0.25, 0.5, 1.0]
    
    @staticmethod
    def _power_law(n: np.ndarray, a: float, b: float, gamma: float) -> np.ndarray:
        """S(n) = a - b * n^(-gamma). 
        'a' represents the asymptotic maximum score.
        'b' represents the learning rate penalty.
        'gamma' represents the steepness of the curve.
        """
        # Ensure n > 0 to avoid division by zero or invalid powers
        # Ensure gamma is bounded to avoid overflow
        return a - b * np.power(np.maximum(n, 1e-5), -np.clip(gamma, 1e-5, 10.0))

    @classmethod
    def fit_learning_curve(cls, sample_sizes: List[int], scores: List[float]) -> Tuple[float, float, float]:
        """Fit a power-law curve to extrapolate future performance."""
        if len(sample_sizes) != len(scores) or len(sample_sizes) < 3:
            raise ValueError("Need at least 3 points to fit the learning curve reliably.")
            
        n_array = np.array(sample_sizes, dtype=np.float64)
        s_array = np.array(scores, dtype=np.float64)
        
        try:
            # Bounds: 
            # a (asymptote): [max(scores), 1.0] assuming metric is bounded by 1.0
            # b: [0, inf]
            # gamma: [0.01, 5.0]
            max_score = np.clip(np.max(s_array), 0.0, 1.0)
            p0 = [max_score, 1.0, 0.5]
            
            popt, _ = curve_fit(
                cls._power_law, 
                n_array, 
                s_array, 
                p0=p0,
                bounds=([max_score, 0.0, 0.01], [1.0, np.inf, 5.0]),
                maxfev=2000
            )
            return float(popt[0]), float(popt[1]), float(popt[2])
        except Exception:
            # Fallback if optimization fails to converge
            return 0.0, 0.0, 0.0
            
    @classmethod
    def extrapolate_score(cls, sample_sizes: List[int], scores: List[float], target_n: int) -> float:
        """Predict the score at target_n samples."""
        if len(sample_sizes) < 3:
            # Can't extrapolate, return best seen
            return max(scores) if scores else 0.0
            
        a, b, gamma = cls.fit_learning_curve(sample_sizes, scores)
        if a == 0.0 and b == 0.0 and gamma == 0.0:
            return max(scores)
            
        return float(cls._power_law(np.array([target_n]), a, b, gamma)[0])
