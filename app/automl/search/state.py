"""Search State Representation (PRD v2 Section 56)."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.statistics.comparison_tests import paired_model_comparison
from app.statistics.correction import MultipleTestingCorrection

class SearchState(BaseModel):
    """Contextual state for Bandit/Evolutionary controllers."""
    
    class Config:
        arbitrary_types_allowed = True

    # Meta-features
    dataset_rows: int = 0
    dataset_cols: int = 0
    cat_feature_ratio: float = 0.0
    num_feature_ratio: float = 0.0
    imbalance_ratio: float = 0.0
    
    # Budget Tracking
    total_budget_seconds: float = 3600.0
    consumed_budget_seconds: float = 0.0
    
    # Progress
    top_3_cv_scores: List[float] = Field(default_factory=list)
    best_fold_scores: Optional[List[float]] = None
    best_experiment_hash: Optional[str] = None
    cv_score_variance: float = 0.0
    stagnation_count: int = 0
    trials_completed: int = 0
    
    @property
    def budget_exhaustion(self) -> float:
        if self.total_budget_seconds <= 0:
            return 1.0
        return min(1.0, self.consumed_budget_seconds / self.total_budget_seconds)
        
    def update_scores(
        self, 
        new_experiment_hash: str,
        new_mean_score: float, 
        new_fold_scores: List[float],
        tracker: MultipleTestingCorrection
    ):
        """Update top 3 scores and calculate stagnation via statistical correction."""
        self.trials_completed += 1
        
        # Maintain top 3 scores for variance
        all_scores = sorted(self.top_3_cv_scores + [new_mean_score], reverse=True)
        self.top_3_cv_scores = all_scores[:3]
        
        if len(self.top_3_cv_scores) > 1:
            mean_score = sum(self.top_3_cv_scores) / len(self.top_3_cv_scores)
            self.cv_score_variance = sum((x - mean_score) ** 2 for x in self.top_3_cv_scores) / len(self.top_3_cv_scores)
        
        # Stagnation check
        if self.best_fold_scores is None:
            self.best_fold_scores = new_fold_scores
            self.best_experiment_hash = new_experiment_hash
            self.stagnation_count = 0
            return
            
        # Statistical test against the best so far
        res = paired_model_comparison(
            scores_a=new_fold_scores,
            scores_b=self.best_fold_scores,
            alpha=tracker.alpha
        )
        
        # Evaluate via tracker to update global 'm'
        eval_res = tracker.evaluate_comparison(
            candidate_a=new_experiment_hash,
            candidate_b=self.best_experiment_hash,
            raw_p_value=res["p_value"],
            mean_diff=res["mean_diff"]
        )
        
        if eval_res["is_significant_winner"]:
            # New model is statistically better
            self.best_fold_scores = new_fold_scores
            self.best_experiment_hash = new_experiment_hash
            self.stagnation_count = 0
        else:
            # Stagnation increments if not a significant winner
            self.stagnation_count += 1
