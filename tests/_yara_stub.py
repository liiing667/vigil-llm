"""Minimal, faithful stub of the yara-python API used by the YARA scanner.

The real ``yara-python`` package requires a compiled libyara extension. These
tests only need to verify vigil's state machine (compile health, scan error
handling, reload recovery, dispatch status), so this stub reproduces the
``yara.compile`` / ``Rules.match`` surface and adds hooks for fault injection.
"""


class Error(Exception):
    pass


class SyntaxError(Error):
    pass


class Match:
    def __init__(self, rule, tags=None, meta=None):
        self.rule = rule
        self.tags = tags or []
        self.meta = meta or {}


# Global runtime fault: when True every Rules.match() raises regardless of
# which compiled instance is used. Simulates a corrupted runtime state that
# persists until rules are recompiled / the fault is cleared.
runtime_fault = False


class Rules:
    def __init__(self, ruleset):
        self._ruleset = ruleset

    def match(self, data=None, filepath=None, **kwargs):
        if runtime_fault:
            raise Error('internal compiler error: rules corrupted')
        text = data or ''
        out = []
        for name, spec in self._ruleset.items():
            needle = spec.get('needle', '')
            if needle and needle in text:
                out.append(Match(name, spec.get('tags', []), spec.get('meta', {})))
        return out


_RULESET = {
    'TestRule': {
        'needle': 'ignore previous instructions',
        'tags': ['injection'],
        'meta': {'category': 'test'},
    }
}

# File basenames that should fail compilation (syntax error injection).
BROKEN_FILES = set()
compile_count = 0


def compile(filepaths=None, source=None, **kwargs):
    global compile_count
    compile_count += 1
    for basename in (filepaths or {}):
        if basename in BROKEN_FILES:
            raise SyntaxError(f'syntax error in {basename}')
    return Rules(_RULESET)


def reset_hooks():
    global compile_count, runtime_fault
    compile_count = 0
    runtime_fault = False
    BROKEN_FILES.clear()
