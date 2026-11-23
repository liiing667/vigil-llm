# Tests

Run the suite with the standard library only (no third-party packages needed):

```bash
python3 tests/run_tests.py
```

The `stubs/` directory contains tiny, stdlib-only stand-ins for the runtime
dependencies (`pydantic`, `loguru`, `flask`, `numpy`, `nltk`, `transformers`,
`yara`, `chromadb`, ...). They exist solely so these tests can execute in
minimal environments; production continues to use the real packages pinned in
`requirements.txt`. When the real dependencies are installed, the same test
modules can be run with `pytest` as well.

Coverage:

* `test_request_isolation.py` — request-scoped canary tokens, per-request scan
  identity, final status (`success` / `partial_success` / `failed`), and LRU
  cache isolation, including alternating requests.
* `test_server_routes.py` — the real `vigil-server.py` handlers exercised
  through the Flask stub: canary add/check scoping, legacy unscoped calls, and
  unique identities on cache hits.
