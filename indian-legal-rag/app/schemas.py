"""Typed records and API request/response contracts."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LegalQARecord(BaseModel):
    """A validated QA pair with dataset provenance."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    question: str
    answer: str
    source_file: str
    law_name: str
    source_record_id: str | None = None
    record_type: str = "qa_pair"
    article_number: str | None = None
    section_number: str | None = None
    chapter: str | None = None
    law_version: str | None = None
    effective_date: str | None = None
    source_url: str | None = None
    source_status: str
    content_hash: str


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    law_name: str | None = Field(default=None, max_length=80)
    source_file: str | None = Field(default=None, max_length=160)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def require_non_whitespace_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must contain non-whitespace text")
        return value


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    law_name: str | None = Field(default=None, max_length=80)

    @field_validator("question")
    @classmethod
    def require_non_whitespace_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must contain non-whitespace text")
        return value


class SourceCitation(BaseModel):
    record_id: str
    law_name: str
    source_file: str
    question: str
    source_record_id: str | None = None
    section_number: str | None = None
    article_number: str | None = None
    law_version: str | None = None
    effective_date: str | None = None
    source_url: str | None = None
    source_status: str


class EvidenceResult(BaseModel):
    record_id: str
    law_name: str
    source_file: str
    question: str
    answer: str
    source_record_id: str | None = None
    section_number: str | None = None
    article_number: str | None = None
    law_version: str | None = None
    effective_date: str | None = None
    source_url: str | None = None
    chapter: str | None = None
    source_status: str
    retrieval_method: str
    rank: int
    score: float | None = None
    rrf_score: float | None = None
    semantic_score: float | None = None
    exact_score: float | None = None


class RetrievalResponse(BaseModel):
    question: str
    evidence: list[EvidenceResult]
    retrieval: dict[str, object]
    insufficient_evidence: bool


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    retrieval: dict[str, object]
    insufficient_evidence: bool
    disclaimer: str
