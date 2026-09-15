"""Experiment Cache with Atomic Check-and-Claim Concurrency Control (PRD v2 Section 31 & 36)."""

import os
import json
import time
from typing import Dict, Any, Optional, Tuple
from app.reliability.locks import BaseLockManager, FileLockManager


class ClaimStatus:
    CACHED = "CACHED"
    CLAIMED = "CLAIMED"
    DUPLICATE_IN_PROGRESS = "DUPLICATE_IN_PROGRESS"
    LOCK_FAILED = "LOCK_FAILED"


class ExperimentCache:
    """Combines cache lookup with atomic distributed locking to prevent duplicate execution races."""

    def __init__(self, lock_manager: BaseLockManager, cache_dir: str = ".experiment_cache"):
        self.lock_manager = lock_manager
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache: Dict[str, Dict[str, Any]] = {}

    def _cache_file(self, experiment_hash: str) -> str:
        return os.path.join(self.cache_dir, f"cache_{experiment_hash}.json")

    def get(self, experiment_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached experiment result if it exists."""
        if experiment_hash in self._memory_cache:
            return self._memory_cache[experiment_hash]

        cache_path = self._cache_file(experiment_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                self._memory_cache[experiment_hash] = data
                return data
            except Exception:
                return None
        return None

    def claim_or_get_cached(
        self,
        experiment_hash: str,
        worker_id: str,
        ttl: int = 300,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Atomically check cache and claim execution lock if missed.

        Returns:
            (ClaimStatus.CACHED, cached_data) -> Result already cached, skip execution.
            (ClaimStatus.CLAIMED, None) -> Lock acquired; proceed to RUNNING.
            (ClaimStatus.DUPLICATE_IN_PROGRESS, None) -> Another worker is actively computing this.
        """
        # 1. Quick check cache
        cached = self.get(experiment_hash)
        if cached is not None:
            return ClaimStatus.CACHED, cached

        # 2. Atomic lock acquisition
        acquired = self.lock_manager.acquire(experiment_hash, worker_id=worker_id, ttl=ttl)

        if acquired:
            # Double-check cache in case written just before lock acquired
            cached = self.get(experiment_hash)
            if cached is not None:
                self.lock_manager.release(experiment_hash, worker_id=worker_id)
                return ClaimStatus.CACHED, cached
            return ClaimStatus.CLAIMED, None

        # Lock failed to acquire: another worker is running this hash
        # Check if they just finished and wrote cache
        cached = self.get(experiment_hash)
        if cached is not None:
            return ClaimStatus.CACHED, cached

        return ClaimStatus.DUPLICATE_IN_PROGRESS, None

    def store_and_release(
        self,
        experiment_hash: str,
        worker_id: str,
        result_data: Dict[str, Any],
    ) -> None:
        """Store experiment result in cache and atomically release worker's lock."""
        # Write to memory cache
        self._memory_cache[experiment_hash] = result_data

        # Write to disk cache
        cache_path = self._cache_file(experiment_hash)
        try:
            with open(cache_path, "w") as f:
                json.dump(result_data, f)
        except Exception:
            pass

        # Release lock
        self.lock_manager.release(experiment_hash, worker_id=worker_id)

    def abort_claim(self, experiment_hash: str, worker_id: str) -> None:
        """Release lock if execution failed without writing a cache result."""
        self.lock_manager.release(experiment_hash, worker_id=worker_id)
