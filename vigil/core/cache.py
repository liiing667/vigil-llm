import hashlib
import json
import threading

from collections import OrderedDict
from typing import Any, Optional


def _canonical(data: Any) -> str:
    """Stable serialization so identical material always hashes identically."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), default=str)


def compute_generation(material: dict) -> str:
    """Fingerprint of the pipeline identity (config, scanner set, thresholds).

    Every request identity, cache entry and response produced by the same
    pipeline configuration shares this generation; any config/scanner change
    yields a new generation and logically invalidates old cache entries.
    """
    return hashlib.sha256(_canonical(material).encode('utf-8')).hexdigest()


def make_cache_key(generation: str, prompt: str, response: Optional[str] = None) -> str:
    """Deterministic cache key bound to a generation and the full request
    context (prompt plus optional response text).

    The key is a pure function of its inputs: a retry after an LRU eviction
    recomputes the exact same key instead of writing to a different one.
    """
    return hashlib.sha256(
        _canonical({
            'generation': generation,
            'prompt': prompt,
            'response': response if response is not None else '',
        }).encode('utf-8')
    ).hexdigest()


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value
