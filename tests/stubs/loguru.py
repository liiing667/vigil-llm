"""Minimal stdlib-only stand-in for loguru.logger used in tests only."""


class _Logger:
    def _noop(self, *args, **kwargs):
        return None

    debug = info = success = warning = warn = error = critical = _noop
    exception = trace = _noop

    def add(self, *args, **kwargs):
        return None


logger = _Logger()
