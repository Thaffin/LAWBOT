"""Lazily constructed application services shared across API requests."""

import logging
from threading import RLock
from typing import Any

from qdrant_client import QdrantClient

from app.config import Settings
from app.embeddings.embedder import LocalEmbedder
from app.llm.llm_client import LLMClient
from app.retrieval.exact_search import ExactSearchIndex
from app.retrieval.retriever import Retriever
from app.retrieval.semantic_search import SemanticSearch
from app.vectordb.collection import collection_health
from app.vectordb.qdrant_client import create_qdrant_client
from app.vectordb.vector_store import QdrantVectorStore


class ServiceUnavailableError(RuntimeError):
    """Raised when a configured external service cannot complete an operation."""

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Create heavyweight or network-backed dependencies only when used."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._client: QdrantClient | None = None
        self._embedder = None
        self._vector_store = None
        self._retriever = None
        self.llm = LLMClient(settings)

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = create_qdrant_client(self.settings)
        return self._client

    @property
    def embedder(self) -> LocalEmbedder:
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    self._embedder = LocalEmbedder(self.settings)
        return self._embedder

    @property
    def vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    self._vector_store = QdrantVectorStore(self.client, self.settings)
        return self._vector_store

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    exact_index = ExactSearchIndex(self.settings.exact_index_file)
                    semantic = SemanticSearch(self.embedder, self.vector_store)
                    self._retriever = Retriever(self.settings, exact_index, semantic)
        return self._retriever

    def qdrant_health(self) -> dict[str, object]:
        try:
            return collection_health(self.client, self.settings)
        except Exception as error:
            logger.warning("Qdrant health operation failed (%s)", type(error).__name__)
            raise ServiceUnavailableError(
                f"Qdrant health check failed ({type(error).__name__})."
            ) from error

    def retrieve(
        self,
        question: str,
        law_name: str | None = None,
        source_file: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        try:
            return self.retriever.retrieve(question, law_name, source_file, top_k)
        except Exception as error:
            logger.warning("Retrieval operation failed (%s)", type(error).__name__)
            raise ServiceUnavailableError(
                f"Retrieval failed ({type(error).__name__})."
            ) from error
