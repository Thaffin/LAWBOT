"""Optional reranking extension point; disabled until a model is selected."""

from typing import Any


class OptionalReranker:
    """Identity behavior keeps reranking optional and adds no model dependency."""

    def rerank(
        self, question: str, candidates: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        return candidates[:limit]
