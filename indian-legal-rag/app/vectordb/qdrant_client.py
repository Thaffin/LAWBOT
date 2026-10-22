"""Construct a configured Qdrant Cloud client only when explicitly needed."""

from qdrant_client import QdrantClient

from app.config import Settings


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """Return a cloud client or fail without revealing credential values."""
    url, api_key = settings.qdrant_url, settings.qdrant_api_key
    if not url or not api_key:
        raise RuntimeError(
            "Qdrant is not configured; set QDRANT_URL and QDRANT_API_KEY in .env."
        )
    try:
        return QdrantClient(
            url=url,
            api_key=api_key,
            timeout=10,
        )
    except (ValueError, RuntimeError) as error:
        raise RuntimeError("Unable to initialize the configured Qdrant client.") from error
