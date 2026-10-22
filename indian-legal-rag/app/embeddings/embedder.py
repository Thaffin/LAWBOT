"""Local, lazy-loaded BGE-M3 document and query encoder."""

from __future__ import annotations

from collections.abc import Sequence
import math
from threading import Lock
from typing import TYPE_CHECKING

from app.config import Settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

VECTOR_DIMENSIONS = 1024


class LocalEmbedder:
    """Load one SentenceTransformer and reuse it for normalized CPU embeddings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: SentenceTransformer | None = None
        self._lock = Lock()
        self._encode_lock = Lock()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        self._model = SentenceTransformer(
                            self.settings.embedding_model,
                            device=self.settings.embedding_device,
                        )
                    except Exception as error:
                        raise RuntimeError(
                            "Unable to load the configured local embedding model. "
                            "Check model download access and available memory."
                        ) from error
        return self._model

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            model = self._get_model()
            with self._encode_lock:
                vectors = model.encode(
                    list(texts),
                    batch_size=self.settings.embedding_batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        except Exception as error:
            raise RuntimeError(
                "Local embedding failed; check model availability and system memory."
            ) from error
        matrix = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        if len(matrix) != len(texts) or any(
            len(vector) != VECTOR_DIMENSIONS for vector in matrix
        ):
            raise ValueError(
                f"Expected {len(texts)} dense vectors of {VECTOR_DIMENSIONS} dimensions; "
                "received an incompatible vector shape."
            )
        normalized: list[list[float]] = []
        for vector in matrix:
            values = [float(value) for value in vector]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Embedding model returned non-finite vector values.")
            norm = math.sqrt(sum(value * value for value in values))
            if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
                raise ValueError("Embedding vectors are not normalized as expected.")
            normalized.append(values)
        return normalized
