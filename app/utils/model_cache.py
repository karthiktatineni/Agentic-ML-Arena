"""Thread-safe LRU cache for in-memory model bundles."""

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import joblib


class ModelBundleCache:
    """Keep the most recently used model bundles in RAM."""

    def __init__(self, max_size: int = 10):
        self._max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()

    def get(self, path: Path) -> Dict[str, Any]:
        key = str(Path(path).resolve())
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        bundle = joblib.load(path)
        with self._lock:
            self._cache[key] = bundle
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        return bundle

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


model_bundle_cache = ModelBundleCache(max_size=10)
