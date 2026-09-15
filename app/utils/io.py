"""Atomic file I/O helpers."""

import os
from pathlib import Path

import joblib


def atomic_joblib_dump(obj: object, path: Path) -> None:
    """Write a joblib bundle atomically via temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp_path)
    os.replace(tmp_path, path)
