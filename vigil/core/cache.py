from collections import OrderedDict
import copy


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            # Return a copy: callers mutate the object for their own request
            # (e.g. adding a "cached" flag or a fresh uuid), which must not
            # leak into the stored entry or other requests.
            return copy.deepcopy(value)
        return None

    def set(self, key: str, value: any):
        # Store an independent copy so the cache never aliases a caller's
        # mutable object.
        value = copy.deepcopy(value)
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
