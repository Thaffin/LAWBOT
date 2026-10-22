"""Dataset provenance and explicitly present legal metadata extraction."""

import re
from pathlib import Path
from typing import Any

LAW_NAMES = {
    "constitution_qa.json": "constitution",
    "ipc_qa.json": "ipc",
    "crpc_qa.json": "crpc",
}
ARTICLE_PATTERN = re.compile(r"\barticle\s+([0-9]+[A-Za-z]?)\b", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"\bsection\s+([0-9]+[A-Za-z]?)\b", re.IGNORECASE)
CHAPTER_PATTERN = re.compile(
    r"\bchapter\s+([IVXLCDM]+|[0-9]+[A-Za-z]?)\b", re.IGNORECASE
)


def source_identity(source_file: str) -> tuple[str, str, str]:
    """Derive dataset labels from the basename without asserting authenticity."""
    filename = Path(source_file).name
    law_name = LAW_NAMES.get(filename.lower(), Path(filename).stem)
    status = (
        "legacy_dataset_unverified" if law_name in {"ipc", "crpc"}
        else "dataset_unverified"
    )
    return filename, law_name, status


def _explicit_value(raw: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = raw.get(name)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
            return str(value).strip()
    return None


def _extract(pattern: re.Pattern[str], text: str) -> str | None:
    matches = {match.group(1) for match in pattern.finditer(text)}
    return next(iter(matches)) if len(matches) == 1 else None


def extract_legal_metadata(
    raw: dict[str, Any], question: str, answer: str
) -> dict[str, str | None]:
    """Keep explicit metadata or unambiguous legal numbering in the QA text."""
    combined = f"{question}\n{answer}"
    return {
        "source_record_id": _explicit_value(
            raw, ("source_record_id", "original_record_id", "record_id", "id")
        ),
        "article_number": _explicit_value(raw, ("article_number", "article", "article_no"))
        or _extract(ARTICLE_PATTERN, combined),
        "section_number": _explicit_value(raw, ("section_number", "section", "section_no"))
        or _extract(SECTION_PATTERN, combined),
        "chapter": _explicit_value(raw, ("chapter", "chapter_number"))
        or _extract(CHAPTER_PATTERN, combined),
        "law_version": _explicit_value(raw, ("law_version", "version", "amendment")),
        "effective_date": _explicit_value(raw, ("effective_date", "in_force_from")),
        "source_url": _explicit_value(raw, ("source_url", "url")),
    }
