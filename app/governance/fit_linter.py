"""Detector 4: Structural Fit-Leakage Linter (PRD v2 Section 13).

Inspects experiment preprocessing specifications and pipeline code representations
to verify that no transformer or stateful preprocessor is fit on the full dataset
or outside cross-validation training folds.
"""

from typing import Dict, Any, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class FitLeakageViolation(Exception):
    """Raised when a fit-leakage violation is detected in an experiment configuration."""
    pass


class FitLeakageLinter:
    """Structural validator ensuring transformations obey strict fold-level isolation."""

    STATEFUL_TRANSFORMER_NAMES = {
        "standardscaler", "minmaxscaler", "robustscaler", "simpleimputer",
        "knnimputer", "iterativeimputer", "onehotencoder", "ordinalencoder",
        "targetencoder", "variance_threshold", "pca", "selectkbest",
        "smote", "randomoversampler", "randomundersampler"
    }

    @classmethod
    def lint_experiment_spec(cls, experiment_spec: Dict[str, Any]) -> List[str]:
        """Lint an experiment specification dictionary before execution."""
        violations = []
        preprocessing = experiment_spec.get("preprocessing", {})

        # Check if full-dataset pre-fitting is requested or flagged
        if preprocessing.get("fit_on_full_dataset", False):
            violations.append(
                "Violation: 'fit_on_full_dataset' is True. All stateful transformers must fit exclusively inside CV train folds."
            )

        # Check if target encoding is specified without out-of-fold configuration
        categorical_strategy = preprocessing.get("categorical_encoding", "")
        if "target" in categorical_strategy.lower() and not preprocessing.get("target_encode_oof", True):
            violations.append(
                "Violation: Target encoding specified without out-of-fold (OOF) cross-fitting. Leaks target signal."
            )

        # Check scaling strategy
        numeric_scaling = preprocessing.get("scaling", "")
        if preprocessing.get("global_scaling", False):
            violations.append(
                "Violation: 'global_scaling' is True. Scaling parameters (mean/std/min/max) must not be fit across entire dataset."
            )

        return violations

    @classmethod
    def validate_sklearn_pipeline(cls, pipeline_obj: Any) -> bool:
        """Verify that a scikit-learn estimator/pipeline structure does not pre-fit transformers."""
        if not isinstance(pipeline_obj, (Pipeline, ColumnTransformer)):
            # If not a Pipeline wrapper, individual stateful transformers must not be standalone fitted
            if hasattr(pipeline_obj, "fit") and not hasattr(pipeline_obj, "predict"):
                raise FitLeakageViolation(
                    f"Transformer {type(pipeline_obj).__name__} passed directly rather than encapsulated inside Pipeline."
                )
        return True

    @classmethod
    def assert_no_leakage(cls, experiment_spec: Dict[str, Any]) -> None:
        """Assert experiment spec has no fit leakage; raise FitLeakageViolation if any found."""
        violations = cls.lint_experiment_spec(experiment_spec)
        if violations:
            raise FitLeakageViolation("\n".join(violations))
