import importlib


# Built-in scanners self-register through the ``Registration.scanner``
# decorator when their modules are imported. Keep optional/heavy
# dependencies isolated so the registry and core library remain usable even
# if a particular scanner's third-party dependencies are not installed.
_BUILTIN_SCANNERS = (
    'vigil.scanners.sentiment',
    'vigil.scanners.similarity',
    'vigil.scanners.vectordb',
    'vigil.scanners.transformer',
    'vigil.scanners.yara',
)

for _scanner_module in _BUILTIN_SCANNERS:
    try:
        importlib.import_module(_scanner_module)
    except ImportError:
        continue


def load_builtin_scanners(force: bool = False) -> None:
    """Register the built-in scanner plug-ins.

    Modules self-register on first import, so a normal call is a no-op.
    After ``ScannerRegistry.reset_registry()`` pass ``force=True`` to reload
    the modules and re-run their ``Registration.scanner`` decorators against
    the new registry generation.
    """
    for scanner_module in _BUILTIN_SCANNERS:
        try:
            module = importlib.import_module(scanner_module)
            if force:
                importlib.reload(module)
        except ImportError:
            continue


__all__ = ('load_builtin_scanners',)


__version__ = "0.9.7-alpha"
__app__ = "vigil"
__description__ = "LLM security scanner"
