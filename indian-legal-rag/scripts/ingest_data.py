"""Explicitly clean, embed, and idempotently upsert the QA collection."""

import json
import sys
from datetime import datetime, timezone

from app.chunking.qa_chunker import chunk_records
from app.config import PROJECT_ROOT, get_settings
from app.embeddings.embedder import LocalEmbedder
from app.ingestion.cleaner import clean_dataset
from app.vectordb.collection import collection_health, ensure_collection
from app.vectordb.qdrant_client import create_qdrant_client
from app.vectordb.vector_store import QdrantVectorStore
from app.retrieval.exact_search import ExactSearchIndex


def main() -> int:
    settings = get_settings()
    report = clean_dataset()
    if report["errors"]:
        print(json.dumps({"status": "aborted", "reason": "source validation failed", **report}, indent=2))
        return 1
    records = []
    for source in sorted((PROJECT_ROOT / "data" / "cleaned").glob("*_qa.json")):
        records.extend(json.loads(source.read_text(encoding="utf-8")))
    chunks = chunk_records(records, settings)
    client = create_qdrant_client(settings)
    ensure_collection(client, settings)
    store = QdrantVectorStore(client, settings)
    embedder = LocalEmbedder(settings)
    before_count = store.collection_count()
    for offset in range(0, len(chunks), settings.upsert_batch_size):
        batch = chunks[offset:offset + settings.upsert_batch_size]
        vectors = embedder.encode_documents([item["text"] for item in batch])
        store.upsert_batch(batch, vectors)
    ExactSearchIndex(settings.exact_index_file).sync(records)
    after_count = store.collection_count()
    collection_status = collection_health(client, settings)
    if not collection_status["compatible"]:
        raise RuntimeError("Qdrant collection became incompatible during ingestion.")
    if after_count < len(chunks):
        raise RuntimeError(
            f"Qdrant reports {after_count} points after indexing {len(chunks)} chunks."
        )
    summary = {
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "chunks_upserted": len(chunks),
        "collection_count_before": before_count,
        "collection_count_after": after_count,
        "collection_status": collection_status["status"],
        "collection_name": settings.qdrant_collection_name,
        "idempotency": "Deterministic point IDs and upserts prevent duplicates; embeddings are regenerated on each run.",
    }
    output = PROJECT_ROOT / "data" / "reports" / "ingestion_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
