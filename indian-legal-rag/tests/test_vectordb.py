from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.config import Settings
from app.embeddings.embedder import VECTOR_DIMENSIONS
from app.vectordb.collection import (
    CollectionConfigurationError,
    collection_health,
    ensure_collection,
)


def collection_info(size=VECTOR_DIMENSIONS, distance=models.Distance.COSINE):
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=models.VectorParams(size=size, distance=distance)
            )
        ),
        status="green",
        points_count=7,
    )


class ExistingCollectionClient:
    def __init__(self, info):
        self.info = info
        self.indexed_fields = []
        self.created = False

    def get_collection(self, name):
        return self.info

    def create_collection(self, **kwargs):
        self.created = True

    def create_payload_index(self, **kwargs):
        self.indexed_fields.append(kwargs["field_name"])


def test_existing_compatible_collection_is_kept_and_indexed():
    client = ExistingCollectionClient(collection_info())
    settings = Settings(_env_file=None)
    ensure_collection(client, settings)
    assert client.created is False
    assert "law_name" in client.indexed_fields
    assert collection_health(client, settings)["compatible"] is True


def test_incompatible_existing_collection_is_never_recreated():
    client = ExistingCollectionClient(collection_info(size=768))
    with pytest.raises(CollectionConfigurationError, match="was not modified"):
        ensure_collection(client, Settings(_env_file=None))
    assert client.created is False
    assert client.indexed_fields == []
