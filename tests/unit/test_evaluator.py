"""Unit tests for immutable EvaluatorEngine."""

import numpy as np
import pytest
from app.automl.evaluation import EvaluatorEngine


def test_evaluator_binary_classification():
    """Verify evaluation metric calculations on binary predictions."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 0])  # 1 false negative
    y_prob = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.9, 0.1],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.8, 0.2],
    ])

    metrics = EvaluatorEngine.evaluate_classification(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 7.0 / 8.0
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["roc_auc"] > 0.8


def test_evaluator_cv_fold_summary():
    """Verify fold aggregation and stability calculation."""
    folds = [
        {"f1": 0.92, "accuracy": 0.91},
        {"f1": 0.94, "accuracy": 0.93},
        {"f1": 0.93, "accuracy": 0.92},
    ]
    summary = EvaluatorEngine.summarize_cv_folds(folds, primary_metric="f1")

    assert round(summary["mean_cv_score"], 2) == 0.93
    assert summary["min_score"] == 0.92
    assert summary["max_score"] == 0.94
    assert summary["std_cv_score"] > 0.0
