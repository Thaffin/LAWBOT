# Architecture

The active backend is Python/FastAPI. `app/ingestion` validates and cleans independent QA sources; `app/chunking` constructs QA-aware retrieval documents; `app/embeddings` locally loads BGE-M3; `app/vectordb` safely accesses Qdrant; `app/retrieval` fuses exact and semantic results; `app/llm` and `app/prompts` provide optional hosted generation; and `app/api` exposes health, retrieval, and answer routes.

The FastAPI application does not connect to Qdrant, download the model, create a collection, or ingest data at startup. Heavy operations are lazy or explicit CLI actions. Local embedding and remote generation are separate components. The earlier JavaScript scaffold remains preserved but is not used by this backend.

TODO — implementation pending: authentication, operational monitoring, CORS policy, authoritative current-law verification, and deployment-specific hardening.
