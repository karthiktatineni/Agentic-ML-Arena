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
            
        # Filter for completed experiments with valid selection-validation predictions
        valid_experiments = [
            e for e in experiments 
            if e.state == "COMPLETED" and e.metrics and e.metrics.selection_val_score is not None and e.metrics.selection_val_predictions
        ]
        
        if not valid_experiments:
            return None
            
        if len(valid_experiments) == 1:
            return valid_experiments[0]
            
        # Sort by raw selection validation score descending (assuming higher is better)
        sorted_experiments = sorted(
            valid_experiments, 
            key=lambda x: x.metrics.selection_val_score, 
            reverse=True
        )
        
        champion = sorted_experiments[0]
        
        for e in sorted_experiments[1:]:
            res = bootstrap_paired_comparison(
                y_true=y_true_selection_val,
                preds_a=champion.metrics.selection_val_predictions,
                preds_b=e.metrics.selection_val_predictions,
                metric_func=metric_func,
                alpha=alpha,
                n_iterations=100
            )
            
            # Record this comparison in the global tracker
            eval_result = correction_tracker.evaluate_comparison(
                candidate_a=champion.experiment_hash,
                candidate_b=e.experiment_hash,
                raw_p_value=res["p_value"],
                mean_diff=res["mean_diff"]
            )
            
            # If the current champion CAN significantly beat this trailing model,
            # we keep the champion. If it CANNOT, it means they are statistically tied.
            # PRD: "keep the incumbent when tied". The incumbent is the higher-scoring 
            # model (champion). Therefore, we DO NOTHING on a tie. The champion remains.
            # We only demote the champion if a challenger somehow significantly BEATS it,
            # but since they are sorted by score descending, a trailing model's score is lower.
            # Wait, if we want to ensure the top scorer is significantly better than the rest,
            # and if it's NOT, we might prefer a SIMPLER model. But if the rule is strictly
            # "keep the incumbent when tied" where incumbent = highest score, then
            # the champion never loses to a lower-scoring model.
            
            # If the PRD meant "the previously deployed model in registry" as incumbent,
            # that's different. Assuming incumbent = top scorer in current run:
            if eval_result["mean_diff"] < 0 and eval_result["clears_statistical"] and eval_result["clears_practical"]:
                # The trailing model somehow significantly beat the champion (should be impossible if sorted by score)
                champion = e
            
        return champion
