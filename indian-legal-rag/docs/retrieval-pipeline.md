# Retrieval pipeline

`POST /retrieve` normalizes a bounded query, checks a local SQLite exact-question/article/section index, embeds the query with the same normalized BGE-M3 encoder used for documents, and searches Qdrant with optional law/source filters. Results are merged by Reciprocal Rank Fusion; duplicate parent records are merged and exact matches are prioritized. `score`/`rrf_score` is the fused rank score, while `semantic_score` and `exact_score` retain the method-specific inputs. The response includes stable record IDs, source metadata, method, rank, and available scores.

Qdrant score is not compared to scores from a lexical matcher. BM25 is intentionally not enabled: without a separate indexed lexical corpus and synchronization lifecycle, scanning the entire dataset on each query would not be a reliable scalable implementation. Reranking is an optional pass-through extension point and is disabled.

`SIMILARITY_THRESHOLD` is unset by default. Until it is calibrated with representative questions and evaluation data, semantic-only results are returned by `/retrieve` but `/ask` conservatively abstains; exact question/article/section matches may still be used. Empty retrieval is always insufficient evidence; no legal citation is inferred from a semantic match alone.

TODO — implementation pending: evaluated lexical/BM25 index, calibrated threshold, retrieval quality metrics, and optional local reranker.
