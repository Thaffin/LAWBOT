# Module guide

TODO — implementation pending: detailed line-by-line walkthroughs.

## Ingestion and indexing

- `app/ingestion/loader.py` reads each JSON source independently and rejects malformed top-level structures.
- `app/ingestion/cleaner.py` normalizes QA text, rejects unusable pairs, reports exact duplicates and likely paraphrases, and writes cleaned records and reports.
- `app/ingestion/metadata.py` derives dataset provenance and retains only explicit or unambiguous legal metadata.
- `app/chunking/qa_chunker.py` keeps ordinary QA pairs intact and splits only long answers.
- `app/embeddings/embedder.py` lazily loads and reuses the local BGE-M3 Sentence Transformer.
- `app/vectordb/collection.py` checks collection compatibility and creates payload indexes without deleting collections.
- `app/vectordb/vector_store.py` batches Qdrant writes and returns source-bearing nearest neighbors.

## Retrieval and API

- `app/retrieval/exact_search.py` maintains the local SQLite exact question and explicit numbering index.
- `app/retrieval/semantic_search.py` embeds the query and searches Qdrant.
- `app/retrieval/hybrid_search.py` fuses ranked result lists without comparing unrelated raw scores.
- `app/retrieval/retriever.py` prioritizes exact matches and marks empty or below-configured-threshold evidence insufficient.
- `app/prompts/legal_prompt.py` marks evidence as untrusted input and limits generation to retrieved records.
- `app/llm/llm_client.py` calls configured hosted inference and checks that record-ID citations refer to retrieved evidence.
- `app/api/routes.py` exposes health, retrieval-only, and answer routes; `app/main.py` does not run ingestion at startup.

The system does not treat a QA record as a complete legal provision or verify current law.