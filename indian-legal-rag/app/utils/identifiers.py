"""Stable content identifiers shared by cleaning, chunking, and indexing."""

import hashlib
import uuid

NAMESPACE = uuid.UUID("0d8b143d-b9dd-4bd9-9fc3-e46afeb70dca")


def content_hash(question: str, answer: str) -> str:
    canonical = f"{question.strip()}\0{answer.strip()}".encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def stable_record_id(law_name: str, digest: str) -> str:
    return hashlib.sha256(f"{law_name}\0{digest}".encode("utf-8")).hexdigest()


def stable_point_id(record_id: str, chunk_index: int = 0) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{record_id}:{chunk_index}"))
