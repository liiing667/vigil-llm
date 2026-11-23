from vigil.scanners.sentiment import SentimentScanner
from vigil.scanners.similarity import SimilarityScanner
from vigil.scanners.vectordb import VectorScanner
from vigil.scanners.transformer import TransformerScanner
from vigil.scanners.yara import YaraScanner

# The relevance scanner is optional at import time because LiteLLM is only
# required when it is actually configured.  Importing it still matters: the
# registration decorator makes the scanner visible to the registry.
try:
    from vigil.scanners.relevance import RelevanceScanner
except ImportError:
    RelevanceScanner = None


__version__ = "0.9.7-alpha"
__app__ = "vigil"
__description__ = "LLM security scanner"
