"""Shared test bootstrap: make the stdlib stubs importable as pydantic/loguru."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stubs')

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if STUBS not in sys.path:
    sys.path.insert(0, STUBS)
