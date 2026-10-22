"""Batch point writes and filtered vector searches for Qdrant."""

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import Settings


def qdrant_filter(
    law_name: str | None = None, source_file: str | None = None
) -> models.Filter | None:
    conditions = []
    if law_name:
        conditions.append(models.FieldCondition(key="law_name", match=models.MatchValue(value=law_name)))
    if source_file:
        conditions.append(models.FieldCondition(key="source_file", match=models.MatchValue(value=source_file)))
    return models.Filter(must=conditions) if conditions else None


class QdrantVectorStore:
    """Thin, typed wrapper around batched point upserts and nearest-neighbor search."""

    def __init__(self, client: QdrantClient, settings: Settings):
        self.client = client
        self.settings = settings

    def upsert_batch(
        self,
        chunks: Sequence[dict[str, Any]],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector.")
        points = [
            models.PointStruct(
                id=chunk["point_id"],
                vector=list(vector),
                payload=chunk["payload"],
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=points,
                wait=True,
            )

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        law_name: str | None = None,
        source_file: str | None = None,
    ) -> list[dict[str, Any]]:
        response = self.client.query_points(
            collection_name=self.settings.qdrant_collection_name,
            query=list(vector),
            query_filter=qdrant_filter(law_name, source_file),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results = []
        for rank, point in enumerate(response.points, start=1):
            payload = dict(point.payload or {})
            results.append({
                **payload,
                "point_id": str(point.id),
                "score": float(point.score),
                "retrieval_method": "semantic",
                "rank": rank,
            })
        return results

    def collection_count(self) -> int:
        result = self.client.count(
            collection_name=self.settings.qdrant_collection_name,
            exact=True,
        )
        return result.count
