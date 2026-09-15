"""Unit tests for FileLockManager concurrency and TTL expiration."""

import os
import time
import shutil
import pytest
from app.reliability.locks import FileLockManager


@pytest.fixture
def temp_lock_dir(tmp_path):
    lock_dir = str(tmp_path / ".test_locks")
    yield lock_dir
    if os.path.exists(lock_dir):
        shutil.rmtree(lock_dir, ignore_errors=True)


def test_file_lock_lifecycle(temp_lock_dir):
    """Test basic acquire, renew, is_locked, and release."""
    lock_mgr = FileLockManager(lock_dir=temp_lock_dir)
    hash_id = "exp_hash_12345"

    assert not lock_mgr.is_locked(hash_id)

    # Worker 1 acquires lock
    assert lock_mgr.acquire(hash_id, worker_id="worker_1", ttl=5)
    assert lock_mgr.is_locked(hash_id)

    # Worker 2 attempts to acquire same lock -> should fail
    assert not lock_mgr.acquire(hash_id, worker_id="worker_2", ttl=5)

    # Worker 1 renews
    assert lock_mgr.renew(hash_id, worker_id="worker_1", ttl=10)

    # Worker 2 attempts renewal -> should fail
    assert not lock_mgr.renew(hash_id, worker_id="worker_2", ttl=10)

    # Worker 1 releases
    assert lock_mgr.release(hash_id, worker_id="worker_1")
    assert not lock_mgr.is_locked(hash_id)

    # Worker 2 can now acquire
    assert lock_mgr.acquire(hash_id, worker_id="worker_2", ttl=5)
    assert lock_mgr.is_locked(hash_id)


def test_file_lock_ttl_expiration(temp_lock_dir):
    """Test that an expired lock can be safely claimed by another worker."""
    lock_mgr = FileLockManager(lock_dir=temp_lock_dir)
    hash_id = "exp_hash_expired_test"

    # Worker 1 acquires lock with 1 second TTL
    assert lock_mgr.acquire(hash_id, worker_id="worker_dead", ttl=1)
    assert lock_mgr.is_locked(hash_id)

    # Wait for TTL to expire
    time.sleep(1.2)

    # Lock is now expired
    assert not lock_mgr.is_locked(hash_id)

    # Worker 2 claims expired lock
    assert lock_mgr.acquire(hash_id, worker_id="worker_alive", ttl=5)
    assert lock_mgr.is_locked(hash_id)
