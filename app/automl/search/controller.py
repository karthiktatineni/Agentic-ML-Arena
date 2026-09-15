"""Contextual Bandit & Evolutionary Search Controller (PRD v2 Section 56)."""

import random
import numpy as np
from typing import Dict, Any, List
from app.automl.search.state import SearchState
from app.events.bus import event_bus
from app.events.schemas import ModelEliminatedEvent

ACTIONS = [
    "TRY_MODEL", 
    "ADD_FEATURE", 
    "REMOVE_FEATURE", 
    "TRANSFORM_FEATURE", 
    "CHANGE_IMPUTATION", 
    "CHANGE_ENCODING", 
    "CHANGE_SCALING", 
    "FEATURE_SELECTION", 
    "RESAMPLING", 
    "ENSEMBLE", 
    "STACKING", 
    "STOP"
]

class SearchController:
    """Uses Thompson Sampling to balance exploration and exploitation of AutoML actions."""
    
    def __init__(self, actions: List[str] = ACTIONS, alpha: float = 1.0, beta: float = 1.0):
        self.actions = actions
        
        # Alpha (successes) and Beta (failures) for Beta distribution of each action
        self.action_alpha = {a: alpha for a in actions}
        self.action_beta = {a: beta for a in actions}
        self.action_counts = {a: 0 for a in actions}
        
    def select_action(self, state: SearchState) -> str:
        """Select the next action to perform using Thompson Sampling."""
        
        # Hard stopping rules based on budget or extreme stagnation
        if state.budget_exhaustion >= 1.0:
            return "STOP"
        if state.stagnation_count > 20:
            return "STOP"
            
        # Contextual logic overrides:
        # If we just started, try models
        if state.trials_completed == 0:
            return "TRY_MODEL"
            
        # If we have a lot of stagnation, try an ensemble
        if state.stagnation_count > 10 and "ENSEMBLE" in self.actions:
            # Boost ensemble probability
            self.action_alpha["ENSEMBLE"] += 5
            
        # Sample from Beta distribution for each action
        sampled_theta = {}
        for action in self.actions:
            if action == "STOP":
                continue # Only select STOP via hard rules
            sampled_theta[action] = np.random.beta(
                self.action_alpha[action], 
                self.action_beta[action]
            )
            
        # Select action with highest sampled value (Thompson Sampling)
        best_action = max(sampled_theta.items(), key=lambda x: x[1])[0]
        
        from app.events.schemas import AgentDecisionEvent
        from app.events.bus import event_bus
        event_bus.publish(AgentDecisionEvent(
            run_id="global",
            agent_name="SearchController",
            decision_action=f"Selected search action: {best_action}",
            confidence=sampled_theta[best_action],
            reasoning_summary=f"Thompson Sampling based on {self.action_counts[best_action]} past trials."
        ))
        
        return best_action

    def update(self, action: str, reward: float):
        """Update the Beta distribution for the chosen action."""
        if action not in self.actions or action == "STOP":
            return
            
        self.action_counts[action] += 1
        
        # Assuming reward is normalized [0, 1]
        # Treat reward as probability of success
        self.action_alpha[action] += reward
        self.action_beta[action] += (1.0 - reward)
        
    def crossover_pipelines(self, pipeline_a: Dict[str, Any], pipeline_b: Dict[str, Any]) -> Dict[str, Any]:
        """Evolutionary crossover of preprocessing steps."""
        child = {}
        keys = set(pipeline_a.keys()).union(set(pipeline_b.keys()))
        for k in keys:
            if k in pipeline_a and k in pipeline_b:
                child[k] = random.choice([pipeline_a[k], pipeline_b[k]])
            elif k in pipeline_a:
                child[k] = pipeline_a[k]
            else:
                child[k] = pipeline_b[k]
        return child
        
    def eliminate_candidates(
        self, 
        candidates: List[Dict[str, Any]], 
        target_n: int, 
        keep_ratio: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Evolutionary elimination (PRD v2 Section 44) with Progressive Scaling safeguards (Section 37).
        
        candidates: list of dicts with keys:
            - 'id': str
            - 'sample_sizes': List[int]
            - 'scores': List[float] (must correspond to sample_sizes)
        
        Returns the list of candidate IDs to keep.
        """
        from app.automl.search.scaling import ProgressiveScaler
        
        if not candidates:
            return []
            
        num_keep = max(1, int(len(candidates) * keep_ratio))
        
        # Calculate extrapolated scores for all candidates
        extrapolated = []
        safe_candidates = []
        
        for c in candidates:
            # PRD v2 Section 37: Minimum Sample Floor for Neural Networks
            is_neural = c.get('model_family') in ('neural', 'pytorch', 'mlp')
            max_n = max(c['sample_sizes']) if c['sample_sizes'] else 0
            
            if is_neural and max_n < 5000:
                safe_candidates.append(c['id'])
                continue
                
            if len(c['sample_sizes']) >= 3:
                proj = ProgressiveScaler.extrapolate_score(c['sample_sizes'], c['scores'], target_n)
            else:
                proj = max(c['scores']) if c['scores'] else 0.0
            extrapolated.append((c, proj))
            
        # Sort by extrapolated score descending
        extrapolated.sort(key=lambda x: x[1], reverse=True)
        
        kept = [x[0]['id'] for x in extrapolated[:num_keep]]
        final_kept = list(set(kept + safe_candidates))
        
        # Emit events for eliminated candidates
        for c in candidates:
            if c['id'] not in final_kept:
                event_bus.publish(ModelEliminatedEvent(
                    run_id="unknown_run",
                    experiment_hash=c['id'],
                    reason="Evolutionary elimination (Progressive Scaling cutoff)"
                ))
                
        # Merge safely kept candidates and those kept by extrapolation
        return final_kept
