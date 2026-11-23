"""Minimal Flask stub used only by the stdlib test-suite.

Supports just enough of the API that vigil-server.py relies on:
app routes, a request proxy, jsonify and abort.
"""

import json


class HTTPError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class _RequestProxy:
    def __init__(self):
        self.path = ''
        self.json = {}


def abort(code, message=''):
    raise HTTPError(code, message)


def jsonify(payload):
    return json.dumps(payload)


class Flask:
    def __init__(self, name):
        self.name = name
        self.routes = {}

    def route(self, path, methods=None):
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator

    def run(self, *args, **kwargs):
        return None


request = _RequestProxy()
