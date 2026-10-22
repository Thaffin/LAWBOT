"""FastAPI routes for service health, evidence retrieval, and grounded answers."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.llm.llm_client import (
    LLMConfigurationError,
    LLMServiceError,
    validate_citations,
)
from app.prompts.legal_prompt import build_messages
from app.schemas import (
    AskRequest,
    AskResponse,
    RetrievalRequest,
    RetrievalResponse,
    SourceCitation,
)
from app.runtime import ServiceContainer, ServiceUnavailableError

logger = logging.getLogger(__name__)
router = APIRouter()
DISCLAIMER = (
    "Informational only; based on retrieved QA dataset records. Verify legal provisions "
    "against authoritative sources. Historical IPC/CrPC records do not establish current law."
)
INSUFFICIENT_ANSWER = (
    "The retrieved QA records do not provide sufficiently relevant evidence to answer this question."
)


def _services(request: Request) -> ServiceContainer:
    return request.app.state.services


def _evidence_is_insufficient(result: dict[str, Any]) -> bool:
    return bool(result["insufficient_evidence"])


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "services": {
            "qdrant_configured": settings.qdrant_configured,
            "llm_configured": settings.llm_configured,
            "embedding_model": settings.embedding_model,
            "embedding_loaded": request.app.state.services._embedder is not None,
        },
    }


@router.get("/health/qdrant", response_model=None)
async def qdrant_health(request: Request) -> dict[str, object] | JSONResponse:
    settings = request.app.state.settings
    if not settings.qdrant_configured:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_configured",
                "reachable": False,
                "collection_exists": False,
                "detail": "Set QDRANT_URL and QDRANT_API_KEY in .env.",
            },
        )
    try:
        result = await run_in_threadpool(_services(request).qdrant_health)
    except ServiceUnavailableError as error:
        logger.warning("Qdrant health check failed (%s)", type(error).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "reachable": False,
                "collection_exists": False,
                "detail": "Qdrant could not be reached or the configured collection is missing.",
            },
        )
    return {
        "status": "ok" if result["compatible"] else "incompatible",
        "reachable": True,
        "collection_exists": True,
        "collection": result,
    }


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(body: RetrievalRequest, request: Request) -> RetrievalResponse:
    try:
        result = await run_in_threadpool(
            _services(request).retrieve,
            body.question.strip(),
            body.law_name,
            body.source_file,
            body.top_k,
        )
    except ServiceUnavailableError as error:
        logger.warning("Retrieval failed (%s)", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Retrieval is unavailable. Check Qdrant configuration and embedding model access.",
        ) from error
    return RetrievalResponse(
        question=body.question.strip(),
        evidence=result["evidence"],
        retrieval=result["retrieval"],
        insufficient_evidence=result["insufficient_evidence"],
    )


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    services = _services(request)
    if not request.app.state.settings.llm_configured:
        raise HTTPException(
            status_code=503,
            detail="Answer generation is not configured; set LLM_BASE_URL and LLM_MODEL.",
        )
    try:
        result = await run_in_threadpool(
            services.retrieve,
            body.question.strip(),
            body.law_name,
            None,
            request.app.state.settings.top_k,
        )
    except ServiceUnavailableError as error:
        logger.warning("Retrieval for answer failed (%s)", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Evidence retrieval is unavailable. Check Qdrant and embedding configuration.",
        ) from error
    evidence = result["evidence"]
    if _evidence_is_insufficient(result):
        return AskResponse(
            answer=INSUFFICIENT_ANSWER,
            sources=[],
            retrieval=result["retrieval"],
            insufficient_evidence=True,
            disclaimer=DISCLAIMER,
        )
    try:
        answer = await services.llm.generate(build_messages(body.question.strip(), evidence))
        validate_citations(answer, evidence)
    except LLMConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except LLMServiceError as error:
        logger.warning("Generation failed (%s)", type(error).__name__)
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        logger.error("Generated answer failed citation validation.")
        raise HTTPException(
            status_code=502,
            detail="The generated answer failed source validation and was not returned.",
        ) from error
    citations = [
        SourceCitation(
            record_id=item["record_id"],
            law_name=item["law_name"],
            source_file=item["source_file"],
            question=item["question"],
            source_record_id=item.get("source_record_id"),
            section_number=item.get("section_number"),
            article_number=item.get("article_number"),
            law_version=item.get("law_version"),
            effective_date=item.get("effective_date"),
            source_url=item.get("source_url"),
            source_status=item["source_status"],
        )
        for item in evidence
    ]
    return AskResponse(
        answer=answer,
        sources=citations,
        retrieval=result["retrieval"],
        insufficient_evidence=False,
        disclaimer=DISCLAIMER,
    )
