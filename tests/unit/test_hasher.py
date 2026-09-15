"""Unit tests for ExperimentHasher."""

from app.experiments.hasher import ExperimentHasher

def test_experiment_hasher_deterministic():
    hash1 = ExperimentHasher.compute_hash(
        dataset_hash="12345",
        model_family="xgboost",
        hyperparameters={"max_depth": 5, "learning_rate": 0.1},
        preprocessing_steps=["standard_scaler"],
        feature_set_version="v2",
        seed=42,
    )
    
    hash2 = ExperimentHasher.compute_hash(
        dataset_hash="12345",
        model_family="xgboost",
        # Swap order of keys in dict to test deterministic serialization
        hyperparameters={"learning_rate": 0.1, "max_depth": 5},
        preprocessing_steps=["standard_scaler"],
        feature_set_version="v2",
        seed=42,
    )
    
    assert hash1 == hash2

def test_experiment_hasher_differs():
    hash1 = ExperimentHasher.compute_hash(
        dataset_hash="12345",
        model_family="xgboost",
        hyperparameters={"max_depth": 5},
        preprocessing_steps=[],
        feature_set_version="v1",
        seed=42,
    )
    
    hash2 = ExperimentHasher.compute_hash(
        dataset_hash="12345",
        model_family="lightgbm",  # different model
        hyperparameters={"max_depth": 5},
        preprocessing_steps=[],
        feature_set_version="v1",
        seed=42,
    )
    
    assert hash1 != hash2
