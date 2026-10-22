"""Normalize QA data, reject incomplete records, and report duplicates safely."""

import json
import logging
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT
from app.ingestion.loader import SOURCE_FILES, SourceDataError, load_json_file
from app.ingestion.metadata import extract_legal_metadata, source_identity
from app.schemas import LegalQARecord
from app.utils.identifiers import content_hash, stable_record_id

logger = logging.getLogger(__name__)


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = []
    for paragraph in re.split(r"\n[ \t]*\n+", normalized):
        cleaned_lines = [
            re.sub(r"[^\S\n]+", " ", line).strip()
            for line in paragraph.split("\n")
        ]
        text = "\n".join(line for line in cleaned_lines if line)
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs).strip()


def normalize_question_key(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def _candidate_key(question: str) -> str:
    return " ".join(sorted(re.findall(r"\w+", normalize_question_key(question))))


def clean_records(
    records: list[dict[str, Any]], source_file: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return accepted records and counts; similar records are warned, not removed."""
    source_name, law_name, source_status = source_identity(source_file)
    accepted: list[dict[str, Any]] = []
    rejected = 0
    exact_seen: set[str] = set()
    duplicate_count = 0
    missing_metadata_count = 0
    warnings: list[str] = []
    normalized_questions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    candidate_questions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    same_answer_questions: dict[str, list[str]] = defaultdict(list)

    for index, raw in enumerate(records, start=1):
        question_raw, answer_raw = raw.get("question"), raw.get("answer")
        if not isinstance(question_raw, str) or not isinstance(answer_raw, str):
            rejected += 1
            logger.warning("%s record %d skipped: question/answer missing or not text", source_name, index)
            continue
        question, answer = normalize_text(question_raw), normalize_text(answer_raw)
        if not question or not answer:
            rejected += 1
            logger.warning("%s record %d skipped: empty question or answer", source_name, index)
            continue

        digest = content_hash(question, answer)
        if digest in exact_seen:
            duplicate_count += 1
            continue
        exact_seen.add(digest)
        record_id = stable_record_id(law_name, digest)
        metadata = extract_legal_metadata(raw, question, answer)
        record = LegalQARecord(
            record_id=record_id,
            question=question,
            answer=answer,
            source_file=source_name,
            law_name=law_name,
            **metadata,
            source_status=source_status,
            content_hash=digest,
        )
        accepted.append(record.model_dump())
        if not any(metadata.get(field) for field in (
            "article_number", "section_number", "chapter"
        )):
            missing_metadata_count += 1

        qkey = normalize_question_key(question)
        ckey = _candidate_key(question)
        for prior_question, prior_answer in normalized_questions[qkey][-100:]:
            if prior_answer != answer:
                warnings.append(
                    f"Record {record_id} shares an exact normalized question with "
                    "another record but has a different answer."
                )
        normalized_questions[qkey].append((question, answer))
        for prior_question, prior_answer in candidate_questions[ckey][-100:]:
            if prior_answer == answer and prior_question != question:
                ratio = SequenceMatcher(None, prior_question.casefold(), question.casefold()).ratio()
                if ratio >= 0.82:
                    warnings.append(
                        f"Likely paraphrase candidate for record {record_id}; "
                        "both records retained for review."
                    )
        candidate_questions[ckey].append((question, answer))
        answer_key = normalize_question_key(answer)
        for prior_question in same_answer_questions[answer_key][-100:]:
            if prior_question != question:
                ratio = SequenceMatcher(
                    None, prior_question.casefold(), question.casefold()
                ).ratio()
                if ratio >= 0.72:
                    warnings.append(
                        f"Likely paraphrase candidate for record {record_id}; "
                        "both records retained for review."
                    )
        same_answer_questions[answer_key].append(question)

    if missing_metadata_count:
        warnings.append(
            f"{missing_metadata_count} accepted records have no explicit or "
            "unambiguous article, section, or chapter metadata."
        )
    report = {
        "source_file": source_name,
        "law_name": law_name,
        "input_count": len(records),
        "accepted_count": len(accepted),
        "rejected_count": rejected,
        "duplicate_count": duplicate_count,
        "missing_metadata_count": missing_metadata_count,
        "warnings": warnings,
    }
    return accepted, report


def clean_dataset(
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Clean the three source datasets without modifying the source files."""
    source_dir = Path(input_dir) if input_dir else PROJECT_ROOT / "data" / "raw"
    cleaned_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "data" / "cleaned"
    reports_dir = Path(report_dir) if report_dir else PROJECT_ROOT / "data" / "reports"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for filename in SOURCE_FILES:
        try:
            original = load_json_file(source_dir / filename)
            accepted, report = clean_records(original, filename)
            target = cleaned_dir / filename
            target.write_text(
                json.dumps(accepted, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            report["output_file"] = str(target.relative_to(PROJECT_ROOT)) if target.is_relative_to(PROJECT_ROOT) else target.name
            reports.append(report)
        except SourceDataError as error:
            errors.append(str(error))
            reports.append({
                "source_file": filename,
                "input_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "duplicate_count": 0,
                "missing_metadata_count": 0,
                "warnings": [],
                "error": str(error),
            })
    aggregate = {
        "report_type": "qa_cleaning",
        "files": reports,
        "errors": errors,
        "input_count": sum(item["input_count"] for item in reports),
        "accepted_count": sum(item["accepted_count"] for item in reports),
        "rejected_count": sum(item["rejected_count"] for item in reports),
        "duplicate_count": sum(item["duplicate_count"] for item in reports),
        "missing_metadata_count": sum(
            item["missing_metadata_count"] for item in reports
        ),
        "warning_count": sum(len(item["warnings"]) for item in reports),
    }
    report_path = reports_dir / "cleaning_report.json"
    report_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports_dir / "cleaning_summary.txt").write_text(
        "\n".join([
            "QA DATA CLEANING SUMMARY",
            f"Input: {aggregate['input_count']}",
            f"Accepted: {aggregate['accepted_count']}",
            f"Rejected: {aggregate['rejected_count']}",
            f"Exact duplicates removed: {aggregate['duplicate_count']}",
            f"Records without explicit legal numbering metadata: {aggregate['missing_metadata_count']}",
            f"Warnings: {aggregate['warning_count']}",
            f"Errors: {len(errors)}",
            *[f"- {error}" for error in errors],
            "",
            "Likely paraphrase candidates are retained; no answers or legal metadata are invented.",
        ]) + "\n",
        encoding="utf-8",
    )
    return aggregate


def validate_sources(input_dir: str | Path | None = None) -> dict[str, Any]:
    """Validate file structure and required QA fields without writing cleaned data."""
    source_dir = Path(input_dir) if input_dir else PROJECT_ROOT / "data" / "raw"
    results = []
    for filename in SOURCE_FILES:
        try:
            records = load_json_file(source_dir / filename)
            accepted, report = clean_records(records, filename)
            report["valid"] = report["rejected_count"] == 0
            report["duplicate_count"] = max(
                0, len(records) - len(accepted) - report["rejected_count"]
            )
            results.append(report)
        except SourceDataError as error:
            results.append({"source_file": filename, "valid": False, "error": str(error)})
    return {"valid": all(item.get("valid", False) for item in results), "files": results}
