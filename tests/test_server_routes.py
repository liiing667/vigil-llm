"""Route-level isolation tests using a lightweight Flask stub.

These tests load the real ``vigil-server.py`` handlers with a minimal Flask
implementation and a stubbed ``Vigil`` object, then exercise the canary
endpoints and the /analyze/prompt cache behavior through alternating
requests.
"""

import importlib.util
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from vigil.core.cache import LRUCache  # noqa: E402
from vigil.core.canary import CanaryTokens  # noqa: E402
import flask as flask_stub  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_PATH = os.path.join(ROOT, 'vigil-server.py')


class FakeVigil:
    def __init__(self):
        self.canary_tokens = CanaryTokens()

    class _Scanner:
        def perform_scan(self, prompt, prompt_response=None):
            status = 'failed' if not prompt else 'success'
            errors = [] if prompt else ['Input prompt value is empty']
            return {
                'status': status,
                'uuid': 'scan-uuid-for-' + prompt,
                'prompt': prompt,
                'prompt_response': prompt_response,
                'prompt_entropy': 1.0,
                'messages': [],
                'errors': errors,
                'results': {},
            }

    input_scanner = _Scanner()
    output_scanner = _Scanner()


def load_server_module():
    # vigil-server.py is not a package module, load it by path.
    spec = importlib.util.spec_from_file_location('vigil_server_under_test', SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ServerRouteTest(unittest.TestCase):
    def setUp(self):
        self.server = load_server_module()
        self.server.vigil = FakeVigil()
        self.server.lru_cache = LRUCache(capacity=10)

    def _call(self, path, payload):
        flask_stub.request.path = path
        flask_stub.request.json = payload
        handler = self.server.app.routes[path]
        return json.loads(handler())

    def test_canary_add_mints_request_id_and_check_is_scoped(self):
        added = self._call('/canary/add', {'prompt': 'prompt A'})
        self.assertTrue(added['success'])
        rid_a = added['request_id']
        self.assertTrue(rid_a)

        # A different request cannot reuse A's token.
        added_b = self._call('/canary/add', {'prompt': 'prompt B'})
        rid_b = added_b['request_id']
        self.assertNotEqual(rid_a, rid_b)

        check_a = self._call('/canary/check', {
            'prompt': added['result'], 'request_id': rid_a
        })
        self.assertTrue(check_a['result'])

        check_b_scope = self._call('/canary/check', {
            'prompt': added['result'], 'request_id': rid_b
        })
        self.assertFalse(check_b_scope['result'])

        # Explicit client-supplied request_id is honored.
        custom = self._call('/canary/add', {'prompt': 'prompt C', 'request_id': 'fixed-id'})
        self.assertEqual(custom['request_id'], 'fixed-id')
        self.assertTrue(
            self._call('/canary/check', {
                'prompt': custom['result'], 'request_id': 'fixed-id'
            })['result']
        )

    def test_legacy_canary_check_without_request_id_still_works(self):
        added = self._call('/canary/add', {'prompt': 'legacy'})
        legacy = self._call('/canary/check', {'prompt': added['result']})
        self.assertTrue(legacy['result'])
        self.assertIsNone(legacy['request_id'])

    def test_alternating_prompt_requests_get_unique_identities_on_cache_hit(self):
        first = self._call('/analyze/prompt', {'prompt': 'identical prompt'})
        self.assertNotIn('cached', first)

        second = self._call('/analyze/prompt', {'prompt': 'identical prompt'})
        third = self._call('/analyze/prompt', {'prompt': 'identical prompt'})

        self.assertTrue(second.get('cached'))
        self.assertTrue(third.get('cached'))
        self.assertNotEqual(second['uuid'], third['uuid'])
        self.assertNotEqual(second['uuid'], first['uuid'])

        # The stored cache entry must not be mutated by served cached hits.
        stored = self.server.lru_cache.get('identical prompt')
        self.assertNotIn('cached', stored)
        self.assertEqual(stored['uuid'], first['uuid'])

    def test_empty_prompt_still_returns_failed_status(self):
        result = self._call('/analyze/prompt', {'prompt': ''})
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('empty' in e for e in result['errors']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
