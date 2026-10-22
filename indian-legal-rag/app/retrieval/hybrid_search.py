"""Reciprocal-rank fusion for exact and semantic QA retrieval."""

from collections.abc import Sequence
from typing import Any

RRF_CONSTANT = 60


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge result lists using ranks, never comparing incomparable raw scores."""
    merged: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        unique_results = []
        seen_in_list: set[str] = set()
        for result in results:
            record_id = str(result.get("record_id") or result.get("parent_record_id"))
            if not record_id:
                continue
            if record_id in seen_in_list:
                continue
            seen_in_list.add(record_id)
            unique_results.append(result)
        for rank, result in enumerate(unique_results, start=1):
            record_id = str(result.get("record_id") or result.get("parent_record_id"))
            fused_score = 1 / (RRF_CONSTANT + rank)
            method = str(result.get("retrieval_method", ""))
            is_exact = method.startswith(("exact", "explicit_"))
            is_semantic = method == "semantic"
            if record_id not in merged:
                merged[record_id] = {
                    **result,
                    "exact_match": is_exact,
                    "exact_score": result.get("score") if is_exact else None,
                    "semantic_score": result.get("score") if is_semantic else None,
                    "rrf_score": fused_score,
                }
            else:
                merged[record_id]["rrf_score"] += fused_score
                merged[record_id]["exact_match"] |= is_exact
                if is_exact:
                    merged[record_id]["exact_score"] = result.get("score")
                if is_semantic:
                    merged[record_id]["semantic_score"] = result.get("score")
            if merged[record_id]["exact_match"] and merged[record_id]["semantic_score"] is not None:
                merged[record_id]["retrieval_method"] = "hybrid"
            elif merged[record_id]["exact_match"]:
                merged[record_id]["retrieval_method"] = method
            elif is_semantic:
                merged[record_id]["retrieval_method"] = "semantic"
    for result in merged.values():
        result["score"] = result["rrf_score"]
    return sorted(
        merged.values(),
        key=lambda item: (
            not item["exact_match"],
            -item["rrf_score"],
        ),
    )
