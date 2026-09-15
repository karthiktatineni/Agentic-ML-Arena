"""Global configuration and Run settings for AutoML Arena."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class FairnessTolerance(BaseModel):
    demographic_parity_diff: float = 0.05
    equalized_odds_gap: float = 0.05


class GlobalRunConfig(BaseModel):
    """Execution parameters governing a complete AutoML Arena run."""
    run_id: Optional[str] = None
    target: str = "target"
    time_column: Optional[str] = None
    group_column: Optional[str] = None
    primary_metric: str = "f1"
    desired_score: float = 0.95
    max_experiments: int = 200
    max_pipeline_retries: int = 3
    max_runtime_minutes: int = 60
    max_parallel_workers: int = 4
    cv_folds: int = 5

    # Splitting strategy
    selection_validation_fraction: float = 0.15
    final_test_fraction: float = 0.18
    nested_cv_row_threshold: int = 5000
    random_seed: int = 42

    # Local compute / HPO budgeting
    max_optuna_trials_per_candidate: int = 1
    max_optuna_time_seconds_per_candidate: int = 180
    optuna_pruner: str = "median"  # median | none

    # Statistical certification & governance
    significance_level: float = 0.05
    multiple_comparison_correction: str = "bonferroni"
    min_practical_effect_size: float = 0.003  # 0.3% practical threshold

    # Concurrency and locking
    lock_backend: str = "file"  # file | redis
    lock_dir: str = ".locks"
    redis_url: str = "redis://localhost:6379/0"
    lock_ttl_seconds: int = 300

    # LLM governance (NVIDIA NIM primary, Ollama/Gemini pluggable)
    llm_provider: str = "nvidia"  # nvidia | gemini | ollama
    llm_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_api_key_env: str = "NVIDIA_API_KEY"
    nvidia_api_key_fallback_env: str = "NVIDIA_API_KEY_FALLBACK"
    llm_max_tokens: int = 2_000_000
    llm_max_requests: int = 500
    llm_schema_error_degraded_mode_threshold: int = 1
    llm_request_timeout_seconds: int = 5
    llm_max_retries: int = 1

    # Fairness & Human Approval Gate
    protected_attributes: List[str] = Field(default_factory=list)
    fairness_tolerance: FairnessTolerance = Field(default_factory=FairnessTolerance)
    require_human_approval: bool = True

    # Model Arena boundaries
    allow_neural_networks: bool = False  # False for Phase 1; True in Phase 4
    allow_ensembles: bool = True
