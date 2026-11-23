from collections import OrderedDict
import copy
import threading


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self._lock = threading.RLock()

    def get(self, key: str):
        with self._lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                # Return a deep copy so callers (e.g. the HTTP layer flagging
                # a cache hit) cannot mutate the stored report that later
                # consumers/history rely on.
                return copy.deepcopy(value)
            return None

    def set(self, key: str, value: any):
        with self._lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            # Store an independent copy so subsequent in-place changes made by
            # one request never overwrite another request's cached report.
            self.cache[key] = copy.deepcopy(value)
