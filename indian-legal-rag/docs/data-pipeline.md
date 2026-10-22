# Data pipeline

Run from the project root:

1. `python -m scripts.validate_data` independently checks the three expected array-of-object JSON sources.
2. `python -m scripts.clean_data` normalizes Unicode/whitespace, validates question/answer text, removes exact normalized QA duplicates, retains paraphrase candidates with warnings, and writes cleaned records and reports.
3. `python -m scripts.ingest_data` runs cleaning, retains stable IDs, creates or validates the compatible Qdrant collection, chunks QA pairs, embeds in batches, upserts deterministic point IDs, verifies counts, and synchronizes the SQLite exact index.

Raw input files are never overwritten. Invalid/incomplete records are skipped and reported, not repaired with generated text. Version, date, URL, article, or section metadata is preserved only when supplied or unambiguously stated. Filename-based law names are dataset provenance, not authentication of legal source.

Normal QA pairs remain whole; only unusually long answers are split at paragraph/sentence/clause boundaries where practical, carrying the original question and parent record ID. QA-pair embedding is not equivalent to chunking complete legal texts or representing complete statutory sections.

TODO — implementation pending: incremental embedding skip using Qdrant content-hash lookups and stale-record cleanup policy. Current deterministic upserts are safe to repeat but recompute embeddings.
