"""Minimal chromadb stub for tests."""


class _Collection:
    def add(self, *args, **kwargs):
        return None

    def query(self, *args, **kwargs):
        return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}


def get_or_create_collection(*args, **kwargs):
    return _Collection()
