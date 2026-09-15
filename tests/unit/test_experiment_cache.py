"""Unit tests for atomic cache check-and-claim concurrency control."""

import os
import shutil
import pytest
from app.reliability.locks import FileLockManager
from app.experiments.cache import ExperimentCache, ClaimStatus


@pytest.fixture
def temp_env(tmp_path):
    lock_dir = str(tmp_path / ".test_locks")
    cache_dir = str(tmp_path / ".test_cache")
    yield lock_dir, cache_dir
    shutil.rmtree(lock_dir, ignore_errors=True)
    shutil.rmtree(cache_dir, ignore_errors=True)


def test_atomic_claim_and_cache(temp_env):
    """Verify atomic claim prevents duplicate compute and serves cached results on hit."""
    lock_dir, cache_dir = temp_env
    lock_mgr = FileLockManager(lock_dir=lock_dir)
    cache = ExperimentCache(lock_manager=lock_mgr, cache_dir=cache_dir)

    exp_hash = "sha256_candidate_xgboost_lr01"

    # Worker 1 encounters cold experiment
    status1, data1 = cache.claim_or_get_cached(exp_hash, worker_id="worker_alpha", ttl=10)
    assert status1 == ClaimStatus.CLAIMED
    assert data1 is None

    # Worker 2 simultaneously encounters same experiment before Worker 1 finishes
    status2, data2 = cache.claim_or_get_cached(exp_hash, worker_id="worker_beta", ttl=10)
    assert status2 == ClaimStatus.DUPLICATE_IN_PROGRESS
    assert data2 is None

    # Worker 1 completes execution and writes result
    result_payload = {"cv_mean": 0.952, "best_params": {"max_depth": 5}}
    cache.store_and_release(exp_hash, worker_id="worker_alpha", result_data=result_payload)

    # Worker 3 arrives later -> gets instant cache hit
    status3, data3 = cache.claim_or_get_cached(exp_hash, worker_id="worker_gamma", ttl=10)
    assert status3 == ClaimStatus.CACHED
    assert data3 == result_payload
