"""Minimal stdlib-only stand-in for pydantic.BaseModel.

Only implements the small subset of the pydantic v1 API used by vigil so
the test-suite can run in environments where the real dependency is not
installed. In production the real pydantic==1.10.x is used unchanged.
"""

import copy
from enum import Enum


class _Missing:
    def __repr__(self):
        return 'MISSING'


MISSING = _Missing()


class FieldInfo:
    def __init__(self, default=MISSING, default_factory=None):
        self.default = default
        self.default_factory = default_factory


def Field(default=MISSING, default_factory=None):
    return FieldInfo(default=default, default_factory=default_factory)


class BaseModel:
    def __init__(self, **data):
        annotations = getattr(type(self), '__annotations__', {})
        for name in annotations:
            if name in data:
                value = data[name]
            else:
                default = getattr(self.__class__, name, MISSING)
                if isinstance(default, FieldInfo):
                    if default.default_factory is not None:
                        value = default.default_factory()
                    elif default.default is not MISSING:
                        value = copy.deepcopy(default.default)
                    else:
                        value = None
                elif default is MISSING:
                    value = None
                else:
                    value = copy.deepcopy(default)
            setattr(self, name, value)

    @staticmethod
    def _convert(value):
        if isinstance(value, BaseModel):
            return value.dict()
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {k: BaseModel._convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [BaseModel._convert(v) for v in value]
        return value

    def dict(self):
        return {
            name: self._convert(getattr(self, name))
            for name in getattr(type(self), '__annotations__', {})
        }
