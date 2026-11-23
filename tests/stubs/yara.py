"""Minimal yara-python stub for tests."""


class _Rules:
    def match(self, data):
        return []


def compile(filepaths=None, source=None):
    return _Rules()
