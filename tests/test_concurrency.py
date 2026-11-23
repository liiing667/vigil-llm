"""
Regression tests for request isolation during concurrent scans.

The production environment ships heavy third-party dependencies (pydantic,
loguru, transformers, ...).  Those are not needed to exercise the dispatch /
cache logic, so minimal stdlib stubs are installed on sys.modules before the
vigil packages are imported.  Run with:

    python3 tests/test_concurrency.py

or:  pytest tests/test_concurrency.py
"""
import copy
import os
import sys
import types
import threading
import time
import unittest
from uuid import UUID, uuid4
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FieldInfoShim:
    def __init__(self, default=None, default_factory=None):
        self.default = default
        self.default_factory = default_factory


# ---------------------------------------------------------------------------
# Stdlib stubs for third-party modules
# ---------------------------------------------------------------------------
def _install_stubs():
    # loguru
    loguru = types.ModuleType('loguru')

    class _Logger:
        def _log(self, *args, **kwargs):
            return self

        info = warning = error = success = debug = critical = exception = _log

        def add(self, *args, **kwargs):
            return 1

    loguru.logger = _Logger()
    sys.modules.setdefault('loguru', loguru)

    # pydantic
    pydantic = types.ModuleType('pydantic')

    class BaseModel:
        def __init__(self, **kwargs):
            for field, value in type(self).__fields__.items():
                default = value[1]
                setattr(self, field, kwargs.get(field, _resolve_default(default)))

        def dict(self):
            out = {}
            for field in type(self).__fields__:
                value = getattr(self, field)
                out[field] = _plain(value)
            return out

    def Field(default=None, default_factory=None, **kwargs):
        return _FieldInfoShim(default, default_factory)

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules.setdefault('pydantic', pydantic)


def _resolve_default(info):
    if isinstance(info, _FieldInfoShim):
        if info.default_factory is not None:
            return info.default_factory()
        default = info.default
    else:
        default = info
    # Mutable defaults (lists/dicts) must be freshly allocated per instance,
    # mirroring pydantic v1 field handling.
    if isinstance(default, (list, dict, set)):
        return copy.deepcopy(default)
    return default


def _plain(value):
    from pydantic import BaseModel

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.dict()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


_install_stubs()

# vigil/__init__.py eagerly imports scanners with heavy third-party deps.
# Pre-register the package pointing at the real directory so submodule imports
# skip that __init__ while still loading production code under test.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_vigil_pkg = types.ModuleType('vigil')
_vigil_pkg.__path__ = [os.path.join(_repo_root, 'vigil')]
sys.modules.setdefault('vigil', _vigil_pkg)
_core_pkg = types.ModuleType('vigil.core')
_core_pkg.__path__ = [os.path.join(_repo_root, 'vigil', 'core')]
sys.modules.setdefault('vigil.core', _core_pkg)

# Patch BaseModel metaclass-less __fields__ collection: schema classes declare
# fields as class annotations; emulate pydantic's field registry by walking
# annotations at class creation time via a base __init_subclass__.
import vigil.schema as schema  # noqa: E402


def _collect_fields(cls):
    fields = {}
    for klass in reversed(cls.__mro__):
        annotations = getattr(klass, '__annotations__', {})
        for name in annotations:
            default = klass.__dict__.get(name, _FieldInfoShim())
            fields[name] = (annotations[name], default)
    cls.__fields__ = fields


for _cls_name in (
    'DatasetEntry', 'ScanModel', 'ResponseModel', 'VectorMatch', 'YaraMatch',
    'ModelMatch', 'SimilarityMatch', 'SentimentMatch'
):
    _cls = getattr(schema, _cls_name, None)
    if _cls is not None:
        _collect_fields(_cls)


from vigil.dispatch import Manager, Scanner  # noqa: E402
from vigil.schema import BaseScanner, ScanModel  # noqa: E402
from vigil.core.cache import LRUCache  # noqa: E402


class _DictMatch:
    """Match object exposing the same .dict() interface as pydantic models."""

    def __init__(self, payload):
        self.payload = payload

    def dict(self):
        return dict(self.payload)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class RecordingScanner(BaseScanner):
    """Scanner that echoes back the (prompt, response, scan_id) it received.

    A small sleep widens the race window so concurrent requests overlap
    deterministically enough to expose shared-state bugs.
    """

    def __init__(self, delay=0.01, fail_first_n=0):
        super().__init__(name='scanner:recorder')
        self.delay = delay
        self.fail_first_n = fail_first_n
        self._calls = 0
        self._lock = threading.Lock()

    def analyze(self, scan_obj, scan_id=None):
        with self._lock:
            self._calls += 1
            call_no = self._calls
        time.sleep(self.delay)
        if call_no <= self.fail_first_n:
            raise RuntimeError(f'simulated failure on call {call_no}')
        scan_obj.results.append(_DictMatch({
            'prompt': scan_obj.prompt,
            'prompt_response': scan_obj.prompt_response,
            'scan_id': str(scan_id),
        }))
        return scan_obj


class SlowScanner(BaseScanner):
    def __init__(self, delay=0.02):
        super().__init__(name='scanner:slow')
        self.delay = delay

    def analyze(self, scan_obj, scan_id=None):
        time.sleep(self.delay)
        scan_obj.results.append(
            _DictMatch({'scan_id': str(scan_id), 'prompt': scan_obj.prompt})
        )
        return scan_obj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestConcurrentRequestIsolation(unittest.TestCase):
    def test_concurrent_scans_have_unique_ids_and_matched_sources(self):
        """Two concurrent /analyze/response-style scans must not mix data."""
        scanner = RecordingScanner(delay=0.02)
        manager = Manager(scanners=[scanner], name='output-test')

        requests = [
            ('prompt-alpha', 'response-alpha'),
            ('prompt-beta', 'response-beta'),
        ]
        results = [None, None]
        barrier = threading.Barrier(2)

        def worker(idx):
            barrier.wait()
            results[idx] = manager.perform_scan(*requests[idx])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        r1, r2 = results

        # unique scan ids
        self.assertNotEqual(r1['uuid'], r2['uuid'])
        UUID(r1['uuid'])
        UUID(r2['uuid'])

        # each report carries its own prompt / response / id end-to-end
        for report, (prompt, response) in zip(results, requests):
            self.assertEqual(report['prompt'], prompt)
            self.assertEqual(report['prompt_response'], response)
            match = report['results']['scanner:recorder']['matches'][0]
            self.assertEqual(match['prompt'], prompt)
            self.assertEqual(match['prompt_response'], response)
            self.assertEqual(match['scan_id'], report['uuid'])
            self.assertEqual(report['status'], 'success')

    def test_retry_after_failure_uses_fresh_scan_id_and_data(self):
        """A failed attempt followed by a retry must not surface old data."""
        scanner = RecordingScanner(delay=0.005, fail_first_n=1)
        manager = Manager(scanners=[scanner], name='retry-test')

        first = manager.perform_scan('retry-prompt', 'retry-response')
        self.assertEqual(first['status'], 'partial_success')
        self.assertTrue(any('simulated failure' in e for e in first['errors']))

        second = manager.perform_scan('retry-prompt', 'retry-response')
        self.assertEqual(second['status'], 'success')
        self.assertNotEqual(first['uuid'], second['uuid'])
        match = second['results']['scanner:recorder']['matches'][0]
        self.assertEqual(match['scan_id'], second['uuid'])
        self.assertEqual(match['prompt'], 'retry-prompt')
        self.assertEqual(match['prompt_response'], 'retry-response')

    def test_many_concurrent_scans_never_share_scan_id(self):
        scanner = SlowScanner(delay=0.005)
        manager = Manager(scanners=[scanner], name='load-test')

        n = 20
        reports = [None] * n
        barrier = threading.Barrier(n)

        def worker(idx):
            barrier.wait()
            reports[idx] = manager.perform_scan(f'prompt-{idx}', f'response-{idx}')

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ids = [r['uuid'] for r in reports]
        self.assertEqual(len(set(ids)), n, 'scan ids must be unique per request')
        for idx, report in enumerate(reports):
            self.assertEqual(report['prompt'], f'prompt-{idx}')
            self.assertEqual(report['prompt_response'], f'response-{idx}')
            match = report['results']['scanner:slow']['matches'][0]
            self.assertEqual(match['scan_id'], report['uuid'])

    def test_empty_response_still_returns_full_report(self):
        """Empty response scans stay usable instead of being dropped."""
        scanner = RecordingScanner(delay=0)
        manager = Manager(scanners=[scanner], name='empty-response')

        report = manager.perform_scan('some prompt', '')
        self.assertEqual(report['status'], 'partial_success')
        self.assertTrue(any('prompt_response value is empty' in e for e in report['errors']))
        self.assertEqual(report['prompt'], 'some prompt')
        self.assertEqual(report['prompt_response'], '')
        self.assertIn('scanner:recorder', report['results'])

    def test_empty_prompt_is_rejected_without_crash(self):
        scanner = RecordingScanner(delay=0)
        manager = Manager(scanners=[scanner], name='empty-prompt')

        report = manager.perform_scan('')
        self.assertEqual(report['status'], 'failed')
        self.assertIn('Input prompt value is empty', report['errors'])
        self.assertEqual(report['prompt_entropy'], 0.0)


class TestLRUCacheIsolation(unittest.TestCase):
    def test_cached_report_is_not_mutated_by_later_consumers(self):
        cache = LRUCache(capacity=4)
        original = {'uuid': 'id-1', 'prompt': 'p', 'results': {'x': [1, 2]}}
        cache.set('p', original)

        hit = cache.get('p')
        hit['cached'] = True
        hit['results']['x'].append(3)

        stored = cache.get('p')
        self.assertNotIn('cached', stored)
        self.assertEqual(stored['results']['x'], [1, 2])
        self.assertEqual(original['results']['x'], [1, 2])

    def test_concurrent_cache_access(self):
        cache = LRUCache(capacity=8)
        errors = []

        def worker(idx):
            try:
                key = f'key-{idx % 4}'
                for _ in range(50):
                    existing = cache.get(key)
                    if existing is None:
                        cache.set(key, {'owner': idx, 'nested': {'v': idx}})
                    else:
                        # mutate the returned copy; must not corrupt storage
                        existing['nested']['v'] = -1
            except Exception as err:  # pragma: no cover
                errors.append(err)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        for i in range(4):
            self.assertIsNotNone(cache.get(f'key-{i}'))


class TestScannerSignatureDefaults(unittest.TestCase):
    def test_base_scanner_default_scan_id_is_unique_per_call(self):
        """Guards against the mutable-default uuid4() regression."""

        class ProbeScanner(BaseScanner):
            def __init__(self):
                super().__init__(name='scanner:probe')
                self.seen = []

            def analyze(self, scan_obj, scan_id=None):
                if scan_id is None:
                    scan_id = uuid4()
                self.seen.append(str(scan_id))
                return scan_obj

        probe = ProbeScanner()
        scan = Scanner([probe])
        scan.run('a', 'id-a')
        scan.run('b', 'id-b')
        self.assertEqual(probe.seen, ['id-a', 'id-b'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
