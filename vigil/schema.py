from abc import ABC, abstractmethod
from uuid import UUID, uuid4
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from vigil.common import timestamp_str


class StatusEmum(str, Enum):
    SUCCESS = 'success'
    FAILED = 'failed'
    PARTIAL = 'partial_success'


class DatasetEntry(BaseModel):
    text: str = ''
    embeddings: List[float] = []
    metadata: Dict = {'model': 'unknown'}


class ScanModel(BaseModel):
    prompt: str = ''
    prompt_response: Optional[str] = None
    results: List[Dict[str, Any]] = []


class ResponseModel(BaseModel):
    status: StatusEmum = StatusEmum.SUCCESS
    uuid: UUID = Field(default_factory=uuid4)
    timestamp: str = Field(default_factory=timestamp_str)
    prompt: str = ''
    prompt_response: Optional[str] = None
    prompt_entropy: Optional[float] = None
    messages: List[str] = []
    errors: List[str] = []
    results: Dict[str, List[Dict[str, Any]]] = {}


class BaseScanner(ABC):
    def __init__(self, name: str = '') -> None:
        self.name = name

    @abstractmethod
    def analyze(self, scan_obj: ScanModel, scan_id: UUID = None) -> ScanModel:
        # A ``uuid4()`` default is evaluated once when the class is defined,
        # which would make every caller that omits the argument share the same
        # scan id.  Generate a fresh identity per call instead.
        if scan_id is None:
            scan_id = uuid4()
        raise NotImplementedError('This method needs to be overridden in the subclass.')

    def post_init(self):
        """ Optional post-initialization method """
        pass


class VectorMatch(BaseModel):
    text: str = ''
    metadata: Optional[Dict] = {}
    distance: float = 0.0


class YaraMatch(BaseModel):
    rule_name: str = ''
    category: Optional[str] = ''
    tags: List[str] = []


class ModelMatch(BaseModel):
    model_name: str = ''
    score: float = 0.0
    label: str = ''
    threshold: float = 0.0


class SimilarityMatch(BaseModel):
    score: float = 0.0
    threshold: float = 0.0
    message: str = ''


class SentimentMatch(BaseModel):
    negative: float = 0.0
    neutral: float = 0.0
    positive: float = 0.0
    compound: float = 0.0
    threshold: float = 0.0


class RelevanceMatch(BaseModel):
    """Evidence produced by the relevance scanner.

    These are the same fields that the prompt asks the model to return.  The
    scanner, the dispatcher, the response status and every evidence consumer
    therefore all reason about one parsed conclusion instead of a raw model
    string.
    """

    detected: bool = False
    abbreviated_chunks: List[str] = []
    irregular: List[str] = []

    @property
    def chunks(self) -> List[str]:
        """Backwards-compatible alias for ``abbreviated_chunks``."""
        return self.abbreviated_chunks

    def __getitem__(self, key: str):
        """Allow read-only mapping-style access (``match['detected']``)."""
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)
