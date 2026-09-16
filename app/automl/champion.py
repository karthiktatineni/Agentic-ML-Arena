"""Champion Selection (PRD v2 Section 34)."""

import pandas as pd
from typing import List, Dict, Any, Optional
from app.api.schemas.experiment import ExperimentObject
from app.statistics.comparison_tests import bootstrap_paired_comparison
from app.statistics.correction import MultipleTestingCorrection

class ChampionSelectionEngine:
    """Selects the statistically proven best model across all candidates."""
    
    @staticmethod
    def select_champion(
        experiments: List[ExperimentObject], 
        y_true_selection_val: List[float],
        metric_func: Any,
        correction_tracker: Optional[MultipleTestingCorrection] = None,
        alpha: float = 0.05
    ) -> Optional[ExperimentObject]:
        """Select champion using paired bootstrap tests on selection-validation predictions."""
        if not experiments:
            return None
            
        if correction_tracker is None:
            correction_tracker = MultipleTestingCorrection(alpha=alpha)
            
        # 1. First preference: experiments with metrics, selection_val_score and predictions
        valid_experiments = [
            e for e in experiments 
            if e.metrics and e.metrics.selection_val_score is not None and e.metrics.selection_val_predictions
        ]
        
        # 2. Fallback: any experiment with metrics and a score
        if not valid_experiments:
            valid_experiments = [
                e for e in experiments 
                if e.metrics and (e.metrics.selection_val_score is not None or e.metrics.mean_cv_score is not None)
            ]

        # 3. Last resort fallback
        if not valid_experiments:
            valid_experiments = [e for e in experiments if e.metrics]
            
        if not valid_experiments:
            return experiments[0] if experiments else None
            
        if len(valid_experiments) == 1:
            return valid_experiments[0]
            
        # Helper to get score safely
        def get_score(exp):
            if not exp.metrics:
                return -float("inf")
            if exp.metrics.selection_val_score is not None:
                return exp.metrics.selection_val_score
            return exp.metrics.mean_cv_score or -float("inf")

        # Sort by raw selection validation score descending (assuming higher is better)
        sorted_experiments = sorted(
            valid_experiments, 
            key=get_score, 
            reverse=True
        )
        
        champion = sorted_experiments[0]
        
        for e in sorted_experiments[1:]:
            try:
                preds_champ = champion.metrics.selection_val_predictions if champion.metrics else []
                preds_challenger = e.metrics.selection_val_predictions if e.metrics else []
                if not preds_champ or not preds_challenger or not y_true_selection_val:
                    continue

                res = bootstrap_paired_comparison(
                    y_true=y_true_selection_val,
                    preds_a=preds_champ,
                    preds_b=preds_challenger,
                    metric_func=metric_func,
                    alpha=alpha,
                    n_iterations=100
                )
                
                # Record this comparison in the global tracker
                eval_result = correction_tracker.evaluate_comparison(
                    candidate_a=champion.experiment_hash,
                    candidate_b=e.experiment_hash,
                    raw_p_value=res.get("p_value", 1.0),
                    mean_diff=res.get("mean_diff", 0.0)
                )
                
                if eval_result.get("mean_diff", 0.0) < 0 and eval_result.get("clears_statistical") and eval_result.get("clears_practical"):
                    champion = e
            except Exception:
                continue
            
        return champion
