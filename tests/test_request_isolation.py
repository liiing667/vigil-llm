"""Request-scoped identity tests for canary tokens, scans and cache.

Run with:  python3 tests/run_tests.py

These tests intentionally use only the standard library (plus the tiny
pydantic/loguru stubs shipped under tests/stubs) so they run without the
heavy runtime dependencies (torch, transformers, chromadb, ...).
"""

import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from vigil.core.canary import CanaryTokens  # noqa: E402
from vigil.core.cache import LRUCache  # noqa: E402
from vigil.dispatch import Manager, Scanner  # noqa: E402
from vigil.schema import BaseScanner, ScanModel  # noqa: E402


class OkScanner(BaseScanner):
    """Scanner that always succeeds and records the scan_id it received."""

    def __init__(self):
        super().__init__(name='scanner:ok')
        self.seen_scan_ids = []

    def analyze(self, scan_obj: ScanModel, scan_id: str = None) -> ScanModel:
        self.seen_scan_ids.append(scan_id)
        return scan_obj


class BoomScanner(BaseScanner):
    """Scanner that always raises, simulating a failed scan."""

    def __init__(self):
        super().__init__(name='scanner:boom')

    def analyze(self, scan_obj: ScanModel, scan_id: str = None) -> ScanModel:
        raise RuntimeError('kaboom')


class CanaryTokenIsolationTest(unittest.TestCase):
    def setUp(self):
        self.canary = CanaryTokens()

    def test_token_bound_to_request_id(self):
        rid = str(uuid.uuid4())
        updated = self.canary.add('hello prompt', request_id=rid)
        token = self.canary.get_tokens(rid)[0]
        self.assertIn(token, updated)
        self.assertTrue(self.canary.check(updated, request_id=rid))

    def test_alternating_requests_cannot_reuse_tokens(self):
        # Two interleaved requests, each with its own identity.
        rid_a = str(uuid.uuid4())
        rid_b = str(uuid.uuid4())

        prompt_a = self.canary.add('request A prompt', request_id=rid_a)
        prompt_b = self.canary.add('request B prompt', request_id=rid_b)

        token_a = self.canary.get_tokens(rid_a)[0]
        token_b = self.canary.get_tokens(rid_b)[0]
        self.assertNotEqual(token_a, token_b)

        # Each request only sees its own token.
        self.assertTrue(self.canary.check(prompt_a, request_id=rid_a))
        self.assertTrue(self.canary.check(prompt_b, request_id=rid_b))
        self.assertFalse(self.canary.check(prompt_a, request_id=rid_b))
        self.assertFalse(self.canary.check(prompt_b, request_id=rid_a))

        # An unknown request id never matches any token.
        self.assertFalse(self.canary.check(prompt_a, request_id=str(uuid.uuid4())))

    def test_legacy_global_check_still_works(self):
        # Old callers that do not pass request_id keep using the global list.
        updated = self.canary.add('legacy prompt')
        self.assertTrue(self.canary.check(updated))
        self.assertIn(updated.split('\n')[0].split(' ')[1], self.canary.tokens)

    def test_clear_request_drops_only_that_request(self):
        rid_a = str(uuid.uuid4())
        rid_b = str(uuid.uuid4())
        prompt_a = self.canary.add('a', request_id=rid_a)
        self.canary.add('b', request_id=rid_b)

        self.canary.clear_request(rid_a)
        self.assertEqual(self.canary.get_tokens(rid_a), [])
        self.assertFalse(self.canary.check(prompt_a, request_id=rid_a))
        self.assertEqual(len(self.canary.get_tokens(rid_b)), 1)


class ScanIdentityAndStatusTest(unittest.TestCase):
    def test_scan_id_is_a_string_and_shared_with_scanners(self):
        ok = OkScanner()
        manager = Manager(scanners=[ok], name='input')
        result = manager.perform_scan('some prompt')

        self.assertIsInstance(result['uuid'], str)
        self.assertEqual(ok.seen_scan_ids, [result['uuid']])
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['errors'], [])

    def test_each_request_gets_unique_identity(self):
        ok = OkScanner()
        manager = Manager(scanners=[ok], name='input')
        first = manager.perform_scan('prompt one')
        second = manager.perform_scan('prompt two')
        self.assertNotEqual(first['uuid'], second['uuid'])
        self.assertEqual(ok.seen_scan_ids, [first['uuid'], second['uuid']])

    def test_empty_prompt_is_failed_with_its_own_identity(self):
        ok = OkScanner()
        manager = Manager(scanners=[ok], name='input')
        result = manager.perform_scan('')
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('empty' in e for e in result['errors']))
        self.assertIsInstance(result['uuid'], str)
        # Scanners must not run for an empty prompt.
        self.assertEqual(ok.seen_scan_ids, [])

    def test_all_scanners_failing_marks_request_failed(self):
        manager = Manager(scanners=[BoomScanner()], name='input')
        result = manager.perform_scan('some prompt')
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(len(result['errors']) == 1)

    def test_partial_failure_is_partial_success(self):
        manager = Manager(scanners=[OkScanner(), BoomScanner()], name='input')
        result = manager.perform_scan('some prompt')
        self.assertEqual(result['status'], 'partial_success')
        self.assertIn('scanner:ok', result['results'])
        self.assertEqual(len(result['errors']), 1)

    def test_scanner_run_passes_string_scan_id_not_a_set(self):
        # Regression test for the old `scan_id={resp.uuid}` bug.
        ok = OkScanner()
        scanner = Scanner([ok])
        rid = str(uuid.uuid4())
        scanner.run(prompt='x', scan_id=rid)
        self.assertEqual(ok.seen_scan_ids, [rid])
        self.assertIsInstance(ok.seen_scan_ids[0], str)


class CacheIsolationTest(unittest.TestCase):
    def test_cache_isolates_caller_mutations_from_stored_entry(self):
        cache = LRUCache(capacity=10)
        original = {
            'uuid': str(uuid.uuid4()),
            'status': 'success',
            'results': {'scanner:ok': {'matches': []}},
        }
        cache.set('same prompt', original)

        # The cache hands back independent objects on both get and set, so a
        # request can decorate its own response without corrupting storage.
        served = cache.get('same prompt')
        served['uuid'] = str(uuid.uuid4())
        served['cached'] = True

        another_hit = cache.get('same prompt')
        self.assertNotIn('cached', another_hit)
        self.assertEqual(another_hit['uuid'], original['uuid'])

        original['uuid'] = 'mutated-after-set'
        self.assertNotEqual(cache.get('same prompt')['uuid'], 'mutated-after-set')

    def test_cache_miss_returns_none_and_lru_eviction(self):
        cache = LRUCache(capacity=2)
        cache.set('a', {'uuid': 'a'})
        cache.set('b', {'uuid': 'b'})
        self.assertIsNone(cache.get('missing'))
        cache.get('a')  # touch 'a' so 'b' becomes the LRU entry
        cache.set('c', {'uuid': 'c'})
        self.assertIsNone(cache.get('b'))
        self.assertEqual(cache.get('a')['uuid'], 'a')


if __name__ == '__main__':
    unittest.main(verbosity=2)
