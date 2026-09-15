"""Distributed and File-based locking system for experiment concurrency control."""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Optional
import portalocker

from app.core.config import GlobalRunConfig


class BaseLockManager(ABC):
    """Abstract interface for atomic experiment locking."""

    @abstractmethod
    def acquire(self, experiment_hash: str, worker_id: str, ttl: int = 300) -> bool:
        """Attempt to atomically acquire a lock for experiment_hash."""
        pass

    @abstractmethod
    def renew(self, experiment_hash: str, worker_id: str, ttl: int = 300) -> bool:
        """Renew an existing lock held by worker_id."""
        pass

    @abstractmethod
    def release(self, experiment_hash: str, worker_id: str) -> bool:
        """Release the lock held by worker_id."""
        pass

    @abstractmethod
    def is_locked(self, experiment_hash: str) -> bool:
        """Check if an experiment_hash is currently locked and not expired."""
        pass


class FileLockManager(BaseLockManager):
    """Cross-platform file-based lock manager for single-node development."""

    def __init__(self, lock_dir: str = ".locks"):
        self.lock_dir = lock_dir
        os.makedirs(self.lock_dir, exist_ok=True)

    def _lock_path(self, experiment_hash: str) -> str:
        return os.path.join(self.lock_dir, f"lock_{experiment_hash}.json")

    def _meta_lock_path(self, experiment_hash: str) -> str:
        return os.path.join(self.lock_dir, f"lock_{experiment_hash}.meta")

    def acquire(self, experiment_hash: str, worker_id: str, ttl: int = 300) -> bool:
        lock_file = self._lock_path(experiment_hash)
        meta_file = self._meta_lock_path(experiment_hash)

        # Use an OS-level file lock on meta_file to protect the read-check-write cycle
        try:
            with portalocker.Lock(meta_file, timeout=2):
                now = time.time()
                if os.path.exists(lock_file):
                    try:
                        with open(lock_file, "r") as f:
                            data = json.load(f)
                        lock_worker = data.get("worker_id")
                        expires_at = data.get("expires_at", 0)

                        if lock_worker == worker_id:
                            # Re-acquire/renew by current worker
                            data["expires_at"] = now + ttl
                            with open(lock_file, "w") as f:
                                json.dump(data, f)
                            return True

                        if now < expires_at:
                            # Actively held by another worker
                            return False
                    except (json.JSONDecodeError, IOError):
                        pass

                # Unlocked or expired: Claim it
                lock_data = {
                    "experiment_hash": experiment_hash,
                    "worker_id": worker_id,
                    "acquired_at": now,
                    "expires_at": now + ttl,
                    "ttl": ttl,
                }
                with open(lock_file, "w") as f:
                    json.dump(lock_data, f)
                return True
        except (portalocker.exceptions.LockException, IOError):
            return False

    def renew(self, experiment_hash: str, worker_id: str, ttl: int = 300) -> bool:
        lock_file = self._lock_path(experiment_hash)
        meta_file = self._meta_lock_path(experiment_hash)

        if not os.path.exists(lock_file):
            return False

        try:
            with portalocker.Lock(meta_file, timeout=2):
                if not os.path.exists(lock_file):
                    return False
                with open(lock_file, "r") as f:
                    data = json.load(f)
                if data.get("worker_id") != worker_id:
                    return False

                data["expires_at"] = time.time() + ttl
                with open(lock_file, "w") as f:
                    json.dump(data, f)
                return True
        except Exception:
            return False

    def release(self, experiment_hash: str, worker_id: str) -> bool:
        lock_file = self._lock_path(experiment_hash)
        meta_file = self._meta_lock_path(experiment_hash)

        if not os.path.exists(lock_file):
            return True

        try:
            with portalocker.Lock(meta_file, timeout=2):
                if os.path.exists(lock_file):
                    try:
                        with open(lock_file, "r") as f:
                            data = json.load(f)
                        if data.get("worker_id") == worker_id:
                            os.remove(lock_file)
                    except Exception:
                        pass
                return True
        except Exception:
            return False

    def is_locked(self, experiment_hash: str) -> bool:
        lock_file = self._lock_path(experiment_hash)
        if not os.path.exists(lock_file):
            return False
        try:
            with open(lock_file, "r") as f:
                data = json.load(f)
            return time.time() < data.get("expires_at", 0)
        except Exception:
            return False


class RedisLockManager(BaseLockManager):
    """Distributed lock manager powered by Redis SETNX."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._redis = None

    def _get_client(self):
        if self._redis is None:
            import redis
            self._redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def acquire(self, experiment_hash: str, worker_id: str, ttl: int = 300) -> bool:
        try:
            r = self._get_client()
            key = f"lock:{experiment_hash}"
            return bool(r.set(key, worker_id, nx=True, ex=ttl))
        except Exception:
            return False

    def renew(self, experiment_hash: str, worker_id: str, ttl: int = 300) -> bool:
        try:
            r = self._get_client()
            key = f"lock:{experiment_hash}"
            if r.get(key) == worker_id:
                r.expire(key, ttl)
                return True
            return False
        except Exception:
            return False

    def release(self, experiment_hash: str, worker_id: str) -> bool:
        try:
            r = self._get_client()
            key = f"lock:{experiment_hash}"
            # Atomic release using Lua script
            lua_release = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            r.eval(lua_release, 1, key, worker_id)
            return True
        except Exception:
            return False

    def is_locked(self, experiment_hash: str) -> bool:
        try:
            r = self._get_client()
            key = f"lock:{experiment_hash}"
            return bool(r.exists(key))
        except Exception:
            return False


def get_lock_manager(config: GlobalRunConfig) -> BaseLockManager:
    """Factory creating configured lock manager backend."""
    if config.lock_backend == "redis":
        return RedisLockManager(redis_url=config.redis_url)
    return FileLockManager(lock_dir=config.lock_dir)
