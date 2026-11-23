"""Tiny numpy stub used only by the stdlib test-suite."""

import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def array(values):
    return list(values)


class _Linalg:
    @staticmethod
    def norm(v):
        return math.sqrt(sum(x * x for x in v))


class _LinalgModule:
    norm = staticmethod(_Linalg.norm)


linalg = _LinalgModule()
