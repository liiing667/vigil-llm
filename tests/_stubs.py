"""Minimal stdlib-only stubs for third-party dependencies.

The production code targets pydantic 1.x, loguru, nltk, yara, transformers
and the embedding/vector-db stack.  The registry/config generation tests do
not need any of those capabilities, so when a real dependency is missing we
install a permissive stub instead of requiring the full install.  When the
real package is available it is left untouched.
"""
import sys
import types


def _ensure_module(name: str, module: types.ModuleType) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    sys.modules[name] = module
    return module


def install_stubs() -> None:
    # loguru with a logger object that accepts any logging call.
    class _AnyLogger:
        def __getattr__(self, _name):
            def _log(*_args, **_kwargs):
                return None
            return _log

    loguru = types.ModuleType('loguru')
    loguru.logger = _AnyLogger()
    _ensure_module('loguru', loguru)

    # pydantic 1.x style BaseModel / Field.
    pydantic = types.ModuleType('pydantic')

    class BaseModel:
        def __init__(self, **data):
            annotations = getattr(self.__class__, '__annotations__', {})
            for field in annotations:
                setattr(self, field, data.get(field))
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self):
            return {
                key: value
                for key, value in self.__dict__.items()
                if not key.startswith('_')
            }

    def Field(default=None, default_factory=None, **_kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    _ensure_module('pydantic', pydantic)

    # nltk and its sentiment analyzer used at import time by sentiment.py.
    nltk = types.ModuleType('nltk')
    nltk.download = lambda *_args, **_kwargs: None
    nltk_sentiment = types.ModuleType('nltk.sentiment')

    class SentimentIntensityAnalyzer:
        def polarity_scores(self, _text):
            return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

    nltk_sentiment.SentimentIntensityAnalyzer = SentimentIntensityAnalyzer
    nltk.sentiment = nltk_sentiment
    _ensure_module('nltk', nltk)
    _ensure_module('nltk.sentiment', nltk_sentiment)

    # yara-python.
    yara = types.ModuleType('yara')
    yara.compile = lambda **_kwargs: None
    _ensure_module('yara', yara)

    # transformers pipeline.
    transformers = types.ModuleType('transformers')
    transformers.pipeline = lambda *_args, **_kwargs: (
        lambda _text: [{'label': 'safe', 'score': 0.0}]
    )
    _ensure_module('transformers', transformers)

    # Embedding / vector-db stack for similarity and vectordb scanners.
    openai = types.ModuleType('openai')

    class OpenAI:
        def __init__(self, *_, **__):
            pass

    openai.OpenAI = OpenAI
    _ensure_module('openai', openai)

    sentence_transformers = types.ModuleType('sentence_transformers')

    class SentenceTransformer:
        def __init__(self, *_, **__):
            pass

        def encode(self, _text):
            return []

    sentence_transformers.SentenceTransformer = SentenceTransformer
    _ensure_module('sentence_transformers', sentence_transformers)

    chromadb = types.ModuleType('chromadb')
    chromadb.Client = type('Client', (), {})
    chromadb.config = types.ModuleType('chromadb.config')
    chromadb.config.Settings = type('Settings', (), {})
    chromadb.utils = types.ModuleType('chromadb.utils')
    chromadb.utils.embedding_functions = types.ModuleType(
        'chromadb.utils.embedding_functions'
    )
    _ensure_module('chromadb', chromadb)
    _ensure_module('chromadb.config', chromadb.config)
    _ensure_module('chromadb.utils', chromadb.utils)
    _ensure_module(
        'chromadb.utils.embedding_functions',
        chromadb.utils.embedding_functions
    )

    # numpy is imported directly by the embedding module.
    numpy = types.ModuleType('numpy')
    numpy.array = lambda *_, **__: []
    _ensure_module('numpy', numpy)

    # The relevance scanner (not registered by default) pulls in litellm.
    litellm = types.ModuleType('litellm')
    litellm.completion = lambda *_, **__: {}
    _ensure_module('litellm', litellm)

    # PyYAML is used by the optional relevance scanner.
    yaml_mod = types.ModuleType('yaml')
    yaml_mod.safe_load = lambda *_: {}
    _ensure_module('yaml', yaml_mod)
