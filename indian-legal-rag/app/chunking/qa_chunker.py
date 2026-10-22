"""Create retrieval documents from QA pairs without treating them as statutes."""

import re
from typing import Any

from app.config import Settings
from app.utils.identifiers import content_hash, stable_point_id

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_long_answer(answer: str, size: int, overlap: int) -> list[str]:
    def split_words(text: str) -> list[str]:
        result: list[str] = []
        current_words: list[str] = []
        current_size = 0
        for word in text.split():
            added_size = len(word) + (1 if current_words else 0)
            if current_words and current_size + added_size > size:
                result.append(" ".join(current_words))
                current_words = [word]
                current_size = len(word)
            else:
                current_words.append(word)
                current_size += added_size
        if current_words:
            result.append(" ".join(current_words))
        return result

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", answer) if part.strip()]
    segments: list[str] = []
    current = ""
    for paragraph in paragraphs or [answer]:
        pieces = SENTENCE_BOUNDARY.split(paragraph)
        for piece in pieces:
            if len(piece) > size:
                clauses = [
                    paragraph_part.strip()
                    for paragraph_part in re.split(r"(?<=[;:])\s+", piece)
                    if paragraph_part.strip()
                ]
                next_pieces = [
                    subpart
                    for clause in clauses
                    for subpart in (
                        [clause] if len(clause) <= size else split_words(clause)
                    )
                ]
            else:
                next_pieces = [piece]
            for sentence in next_pieces:
                candidate = f"{current}\n{sentence}".strip() if current else sentence
                if current and len(candidate) > size:
                    segments.append(current)
                    carry = ""
                    if overlap:
                        cut = max(0, len(current) - overlap)
                        boundary = current.find(" ", cut)
                        carry = (
                            current[boundary + 1:] if boundary >= 0
                            else current[current.rfind(" ") + 1:]
                        )
                    current = f"{carry}\n{sentence}".strip() if carry else sentence
                else:
                    current = candidate
    if current:
        segments.append(current)
    return segments or [answer]


def chunk_record(record: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    """Keep normal QA pairs whole; split only unusually long answers at boundaries."""
    question, answer = record["question"], record["answer"]
    if len(answer) <= settings.chunk_size_chars:
        parts = [answer]
    else:
        parts = _split_long_answer(
            answer,
            settings.chunk_size_chars,
            settings.chunk_overlap_chars,
        )
    chunks = []
    for index, part in enumerate(parts):
        embedding_text = f"Question: {question}\nAnswer: {part}"
        digest = content_hash(question, part)
        chunk_id = stable_point_id(record["record_id"], index)
        payload = {
            **record,
            "answer": answer,
            "chunk_answer": part,
            "parent_record_id": record["record_id"],
            "chunk_index": index,
            "chunk_count": len(parts),
            "chunk_content_hash": digest,
            "embedding_text": embedding_text,
        }
        chunks.append({
            "point_id": chunk_id,
            "record_id": record["record_id"],
            "chunk_index": index,
            "text": embedding_text,
            "payload": payload,
        })
    return chunks


def chunk_records(
    records: list[dict[str, Any]], settings: Settings
) -> list[dict[str, Any]]:
    """Convert cleaned QA records into stable, source-preserving chunks."""
    chunks = []
    for record in records:
        chunks.extend(chunk_record(record, settings))
    return chunks
