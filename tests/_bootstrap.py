"""Test bootstrap for environments without Vigil's third-party dependencies."""
import copy
import os
import sys
import types
from enum import Enum

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


class FieldInfoShim:
    def __init__(self, default=None, default_factory=None):
        self.default = default
        self.default_factory = default_factory


def install_stubs():
    loguru = types.ModuleType('loguru')

    class Logger:
        def _log(self, *args, **kwargs):
            return self

        info = warning = warn = error = success = debug = critical = exception = _log

        def add(self, *args, **kwargs):
            return 1

    loguru.logger = Logger()
    sys.modules.setdefault('loguru', loguru)

    pydantic = types.ModuleType('pydantic')

    class BaseModel:
        def __init__(self, **kwargs):
            for field, field_info in type(self).__fields__.items():
                setattr(self, field, kwargs.get(field, resolve_default(field_info[1])))

        def dict(self):
            return {field: plain(getattr(self, field)) for field in type(self).__fields__}

    def Field(default=None, default_factory=None, **kwargs):
        return FieldInfoShim(default, default_factory)

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules.setdefault('pydantic', pydantic)


def resolve_default(info):
    if isinstance(info, FieldInfoShim):
        if info.default_factory is not None:
            value = info.default_factory()
        else:
            value = info.default
    else:
        value = info

    if isinstance(value, (list, dict, set)):
        return copy.deepcopy(value)
    return value


def plain(value):
    from pydantic import BaseModel

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.dict()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def install_vigil_package_stubs():
    # vigil/__init__.py eagerly imports optional scanner dependencies. Tests in
    # this module only need dispatch, schema and common, so bypass that import.
    vigil_pkg = types.ModuleType('vigil')
    vigil_pkg.__path__ = [os.path.join(REPO_ROOT, 'vigil')]
    sys.modules.setdefault('vigil', vigil_pkg)

    core_pkg = types.ModuleType('vigil.core')
    core_pkg.__path__ = [os.path.join(REPO_ROOT, 'vigil', 'core')]
    sys.modules.setdefault('vigil.core', core_pkg)


def collect_pydantic_fields(module):
    from pydantic import BaseModel

    for cls in vars(module).values():
        if not isinstance(cls, type) or not issubclass(cls, BaseModel) or cls is BaseModel:
            continue
        fields = {}
        for klass in reversed(cls.__mro__):
            for name in getattr(klass, '__annotations__', {}):
                default = klass.__dict__.get(name, FieldInfoShim())
                fields[name] = (klass.__annotations__[name], default)
        cls.__fields__ = fields


install_stubs()
install_vigil_package_stubs()
