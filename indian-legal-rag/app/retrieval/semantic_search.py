"""Semantic retrieval backed by local BGE-M3 query embeddings and Qdrant."""

from typing import Any

from app.embeddings.embedder import LocalEmbedder
from app.vectordb.vector_store import QdrantVectorStore


class SemanticSearch:
    """Encode a user query once and request filtered top-k evidence."""

    def __init__(self, embedder: LocalEmbedder, vector_store: QdrantVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        question: str,
        top_k: int,
        law_name: str | None = None,
        source_file: str | None = None,
    ) -> list[dict[str, Any]]:
        vector = self.embedder.encode_query(question)
        return self.vector_store.search(vector, top_k, law_name, source_file)
