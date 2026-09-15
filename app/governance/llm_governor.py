"""LLM Governor (PRD v2 Section 58)."""

from typing import Dict, Any

class LLMGovernor:
    """Monitors LLM cost and schema errors. Trips Degraded Mode if limits exceeded."""
    
    def __init__(self, max_budget_usd: float = 1.0, max_consecutive_schema_errors: int = 3):
        self.max_budget_usd = max_budget_usd
        self.max_consecutive_schema_errors = max_consecutive_schema_errors
        
        self.current_spend_usd = 0.0
        self.consecutive_schema_errors = 0
        self.is_tripped = False
        self.trip_reason = ""
        
    def record_cost(self, cost_usd: float):
        self.current_spend_usd += cost_usd
        if self.current_spend_usd >= self.max_budget_usd and not self.is_tripped:
            self.trip(f"Budget exceeded: ${self.current_spend_usd:.2f} >= ${self.max_budget_usd:.2f}")
            
    def record_schema_error(self):
        self.consecutive_schema_errors += 1
        if self.consecutive_schema_errors >= self.max_consecutive_schema_errors and not self.is_tripped:
            self.trip(
                f"Consecutive schema errors exceeded: "
                f"{self.consecutive_schema_errors} >= {self.max_consecutive_schema_errors}"
            )
            
    def record_success(self):
        self.consecutive_schema_errors = 0

    def trip(self, reason: str) -> bool:
        """Enter degraded mode and return True only on the first transition."""
        if self.is_tripped:
            return False
        self.is_tripped = True
        self.trip_reason = reason
        return True

    def reset(self):
        """Reset per-run governor state."""
        self.current_spend_usd = 0.0
        self.consecutive_schema_errors = 0
        self.is_tripped = False
        self.trip_reason = ""
        
    def check_governor(self) -> bool:
        """Returns True if the LLM governor has tripped Degraded Mode."""
        return self.is_tripped

# Global singleton for the run
global_governor = LLMGovernor()
