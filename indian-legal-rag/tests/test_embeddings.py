import pytest

from app.config import Settings
from app.embeddings.embedder import LocalEmbedder, VECTOR_DIMENSIONS


class StubModel:
    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return [[1.0, *([0.0] * (VECTOR_DIMENSIONS - 1))] for _ in texts]


def test_document_and_query_embeddings_have_1024_normalized_dimensions():
    embedder = LocalEmbedder(Settings(_env_file=None))
    embedder._model = StubModel()
    documents = embedder.encode_documents(["Question: q\nAnswer: a", "Question: x\nAnswer: y"])
    query = embedder.encode_query("a query")
    assert len(documents) == 2
    assert len(documents[0]) == VECTOR_DIMENSIONS
    assert len(query) == VECTOR_DIMENSIONS
    assert sum(value * value for value in query) ** 0.5 == pytest.approx(1.0)


def test_rejects_unexpected_embedding_dimensions():
    class WrongShape:
        def encode(self, texts, **kwargs):
            return [[1.0] * 32 for _ in texts]

    embedder = LocalEmbedder(Settings(_env_file=None))
    embedder._model = WrongShape()
    with pytest.raises(ValueError, match="1024 dimensions"):
        embedder.encode_query("question")
