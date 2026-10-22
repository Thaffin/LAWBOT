"""Orchestrate exact and semantic retrieval and select evidence conservatively."""

from typing import Any

from app.config import Settings
from app.retrieval.exact_search import ExactSearchIndex
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.semantic_search import SemanticSearch


class Retriever:
    """Combine exact provenance-backed matches and Qdrant semantic results."""

    def __init__(
        self,
        settings: Settings,
        exact_index: ExactSearchIndex,
        semantic_search: SemanticSearch,
    ):
        self.settings = settings
        self.exact_index = exact_index
        self.semantic_search = semantic_search

    def retrieve(
        self,
        question: str,
        law_name: str | None = None,
        source_file: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        limit = top_k or self.settings.top_k
        exact = self.exact_index.search(question, law_name, source_file, limit)
        semantic = self.semantic_search.search(
            question, limit, law_name, source_file
        )
        merged = reciprocal_rank_fusion([exact, semantic])
        evidence = merged[: self.settings.final_context_k]
        for rank, item in enumerate(evidence, start=1):
            item["rank"] = rank
            item["retrieval_method"] = (
                "hybrid" if item["record_id"] in {
                    result["record_id"] for result in exact
                } and item["record_id"] in {
                    result.get("record_id") for result in semantic
                }
                else item.get("retrieval_method", "semantic")
            )
        exact_match = bool(exact)
        best_semantic = max(
            (result.get("score", 0.0) for result in semantic),
            default=None,
        )
        threshold = self.settings.similarity_threshold
        insufficient = not evidence or (
            not exact_match
            and (
                threshold is None
                or best_semantic is None
                or best_semantic < threshold
            )
        )
        method = (
            "hybrid" if exact and semantic
            else "exact" if exact
            else "semantic" if semantic
            else "none"
        )
        return {
            "evidence": evidence,
            "retrieval": {
                "method": method,
                "evidence_count": len(evidence),
                "top_k": limit,
                "similarity_threshold": threshold,
            },
            "insufficient_evidence": insufficient,
        }
