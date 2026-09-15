"""Immutable Evaluator Engine (PRD v1/v2 Rule 1)."""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class EvaluatorEngine:
    """Immutable scoring engine.

    Enforces uniform, tamper-proof metric calculations across all experiments.
    """

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute full classification metric dictionary."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) <= 2

        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        if is_binary:
            pos_label = unique_classes[-1]
            prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        else:
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        metrics: Dict[str, Any] = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

        # Probabilistic metrics
        if y_prob is not None:
            try:
                if is_binary:
                    # Probability of positive class
                    prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob
                    metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
                    metrics["brier_score"] = float(brier_score_loss(y_true, prob_pos))
                    metrics["log_loss"] = float(log_loss(y_true, prob_pos))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
                    metrics["log_loss"] = float(log_loss(y_true, y_prob))
            except Exception:
                pass

        # Confusion Matrix
        try:
            cm = confusion_matrix(y_true, y_pred).tolist()
            metrics["confusion_matrix"] = cm
        except Exception:
            pass

        return metrics

    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute full regression metric dictionary."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        return {
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
            "r2": r2,
        }

    @staticmethod
    def summarize_cv_folds(fold_metrics: List[Dict[str, float]], primary_metric: str = "f1") -> Dict[str, Any]:
        """Summarize fold-level evaluations into stability and mean score."""
        scores = [m.get(primary_metric, 0.0) for m in fold_metrics]
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

        return {
            "primary_metric": primary_metric,
            "mean_cv_score": mean_score,
            "std_cv_score": std_score,
            "fold_scores": scores,
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }
