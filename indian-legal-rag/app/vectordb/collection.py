"""Create or validate the Qdrant collection without destructive migration."""

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import Settings
from app.embeddings.embedder import VECTOR_DIMENSIONS

PAYLOAD_INDEX_FIELDS = (
    "law_name",
    "source_file",
    "record_type",
    "section_number",
    "article_number",
    "source_status",
)


class CollectionConfigurationError(RuntimeError):
    """Raised when a pre-existing collection cannot safely accept these vectors."""


def _vector_parameters(info: object) -> tuple[int | None, str | None]:
    config = getattr(info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    if isinstance(vectors, dict):
        vectors = vectors.get("")
        if vectors is None:
            raise CollectionConfigurationError(
                "Collection uses named vectors; configure a compatible unnamed dense vector "
                "or select a separate collection."
            )
    size = getattr(vectors, "size", None)
    distance = getattr(vectors, "distance", None)
    return size, getattr(distance, "value", str(distance) if distance else None)


def ensure_collection(client: QdrantClient, settings: Settings) -> object:
    """Create a missing collection; validate existing settings and preserve its data."""
    name = settings.qdrant_collection_name
    try:
        info = client.get_collection(name)
    except UnexpectedResponse as error:
        status_code = getattr(error, "status_code", None)
        if status_code != 404 and "not found" not in str(error).lower():
            raise RuntimeError("Unable to inspect the configured Qdrant collection.") from error
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=VECTOR_DIMENSIONS,
                distance=models.Distance.COSINE,
            ),
            timeout=30,
        )
        info = client.get_collection(name)
    size, distance = _vector_parameters(info)
    if size != VECTOR_DIMENSIONS or distance != "Cosine":
        raise CollectionConfigurationError(
            f"Collection '{name}' has vector size/distance {size}/{distance}; expected "
            f"{VECTOR_DIMENSIONS}/Cosine. Existing collection was not modified. "
            "Choose a new collection name and re-index to migrate safely."
        )
    for field in PAYLOAD_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    return info


def collection_health(client: QdrantClient, settings: Settings) -> dict[str, object]:
    """Return safe collection health metadata without exposing configuration."""
    info = client.get_collection(settings.qdrant_collection_name)
    size, distance = _vector_parameters(info)
    return {
        "exists": True,
        "status": str(getattr(info, "status", "unknown")),
        "points_count": getattr(info, "points_count", None),
        "vector_size": size,
        "distance": distance,
        "compatible": size == VECTOR_DIMENSIONS and distance == "Cosine",
    }
