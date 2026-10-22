# Indian Legal RAG

A Python/FastAPI retrieval-augmented application scaffold for owner-provided Constitution, IPC, and CrPC question-and-answer datasets. The system treats filenames as dataset provenance, not proof of legal authenticity. IPC and CrPC records are marked as legacy and unverified; they are not evidence of current law.

**Status:** Core modules are implemented, but integrations require configuration and real data validation. No service connection or deployment is claimed. This is informational software, not legal advice.

## Architecture

- `app/ingestion/`: independent JSON loading, field validation, normalization, duplicate handling, provenance, and reports.
- `app/chunking/`: keeps normal QA pairs whole and splits unusually long answers at paragraph, sentence, clause, or word boundaries.
- `app/embeddings/`: lazy local `sentence-transformers` BGE-M3 embeddings; CPU is the default.
- `app/vectordb/`: official Qdrant client, safe collection checks, payload indexes, batched writes, and filtered search.
- `app/retrieval/`: a synchronized SQLite exact index, Qdrant semantic search, rank fusion, optional reranker extension point, and evidence selection.
- `app/prompts/` and `app/llm/`: evidence-only prompt construction and configurable OpenAI-compatible hosted inference.
- `app/api/`: FastAPI `/health`, `/health/qdrant`, `/retrieve`, and `/ask` endpoints.
- `scripts/`: explicit validation, cleaning, Qdrant checking, and ingestion commands. The API never runs ingestion or creates a collection.
- `data/raw/`: original owner-provided inputs. Inputs are never rewritten by cleaning.
- `data/cleaned/`: normalized QA records, separate from raw data.
- `data/reports/`: cleaning, ingestion, and exact-search index output.
- `tests/`: synthetic, non-legal fixtures and unit/API tests; no real cloud credentials needed.

The previous JavaScript/frontend scaffold remains in the repository for preservation, but it is not part of this Python API or its runtime workflow.

## Data flow and legal safeguards

Each of `constitution_qa.json`, `ipc_qa.json`, and `crpc_qa.json` is parsed and validated separately as an array of QA objects. The cleaner normalizes Unicode and whitespace, drops malformed or exact duplicate QA pairs, retains likely paraphrases with a warning, preserves supplied version/date/source metadata, and extracts article/section numbers only when explicitly supplied or unambiguous in the text.

Clean records are kept distinct from raw files. A short QA pair is embedded together as `Question: ... / Answer: ...`; it is not represented as a complete statutory section. Stable IDs derive from source law and normalized QA content. Qdrant stores source metadata with the vectors. Deterministic upserts are idempotent, but the current pipeline recomputes embeddings on every ingestion run.

At query time, the API performs exact and semantic retrieval, merges ranks with Reciprocal Rank Fusion, and passes selected retrieved records only to the hosted generation endpoint. No evidence returns an insufficient-evidence response. Similarity thresholds are unset by default and should be calibrated against representative evaluation queries; until configured, semantic-only `/ask` requests abstain rather than treating every nearest neighbor as relevant.

Historical IPC/CrPC records do not establish present-day law. The Bharatiya Nyaya Sanhita, Bharatiya Nagarik Suraksha Sanhita, and Bharatiya Sakshya Adhiniyam generally took effect on 1 July 2024, subject to applicable transitional provisions and exceptions. This application does not implement authoritative current-law verification or legal comparison. Do not infer that an old provision remains in force.

## Prerequisites

- Python 3.11 or a compatible supported version. Python 3.15 is not currently the recommended setup for the PyTorch/Sentence Transformers dependency stack.
- A Qdrant Cloud endpoint and API key for ingestion and semantic retrieval.
- A configured OpenAI-compatible inference endpoint for `/ask`; health checks, cleaning, and retrieval-only testing do not require the LLM.
- The three real QA JSON files in `data/raw/`. No legal examples or generated records are included.

## macOS setup

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Set values in `.env` locally. Do not commit it. `.env.example` and `config.yaml` provide non-secret defaults; environment variables override YAML configuration. Do not place secrets or legal data in the frontend public directory.

### Qdrant configuration

In the Qdrant Cloud console, open the intended cluster's connection details to obtain its HTTPS endpoint and generate or view an API key with the minimum required permissions. Set those values as `QDRANT_URL` and `QDRANT_API_KEY` in `.env`. Never copy the key into source files, reports, or chat.

The collection defaults to `indian_legal_qa`, uses 1024-dimensional cosine vectors, and is only created/validated by the explicit ingestion command. Existing collections are never deleted or recreated. A mismatch stops ingestion; choose a new collection name and re-index to migrate safely.

### Local embedding and hosted inference

`BAAI/bge-m3` runs locally through Sentence Transformers; no paid embedding API or GPU is required. The model artifacts are multi-gigabyte (budget approximately 2.3 GB for weights, plus runtime/cache; actual download size varies). CPU inference can be slow and memory use depends on batch size. Model download happens on the first embedding call, not at API startup.

Answer generation is a separate hosted inference call configured with `LLM_BASE_URL`, `LLM_MODEL`, and optionally `LLM_API_KEY` if the endpoint requires one. The client expects an OpenAI-compatible `/chat/completions` route; provider-specific differences may require adjustment. Missing LLM configuration leaves health and retrieval available but makes `/ask` return a clear 503 configuration error.

## Data commands

Place the three source files in `data/raw/`. Cleaning never overwrites them.

```bash
python -m scripts.validate_data
python -m scripts.clean_data
python -m scripts.check_qdrant
python -m scripts.ingest_data
```

Validation and cleaning work without Qdrant credentials. The Qdrant check is read-only and does not create a collection. Ingestion explicitly validates/cleans data, creates a missing compatible collection, generates local embeddings in batches, upserts deterministic point IDs, verifies counts, and synchronizes the local exact-search SQLite index. Empty placeholder JSON files are not valid input; supply the actual datasets before running these commands. Ingestion is never started automatically by FastAPI.

An ingestion run re-embeds all records even if the point IDs are unchanged. It will upsert rather than create duplicate points; it does not delete stale points from removed source records.

## Run the API

```bash
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI. Try:

- `GET /health` to see configuration status without exposing credentials.
- `GET /health/qdrant` to check configured Qdrant and collection compatibility.
- `POST /retrieve` with `{"question":"...","law_name":"constitution","top_k":8}` for retrieved evidence without generation.
- `POST /ask` with `{"question":"...","law_name":"constitution"}` for a grounded hosted-model answer and citations.

Qdrant and the local embedding model must be configured/available for semantic retrieval. `/ask` additionally requires a configured LLM. The API does not expose stack traces or secrets.

## Tests

```bash
python -m pytest
```

Tests use synthetic generic QA fixtures and mocked services. They do not require Qdrant credentials, model downloads, or paid LLM calls.

## Troubleshooting and limitations

- **Malformed/missing JSON:** inspect `data/reports/cleaning_report.json`; ensure every source is a JSON array of objects with non-empty string `question` and `answer`.
- **Qdrant not configured/unreachable:** verify endpoint and key in `.env`. `/health/qdrant` and `python -m scripts.check_qdrant` provide safe diagnostics.
- **Incompatible collection:** do not delete it automatically. Use a new collection and re-index to migrate.
- **Model download/memory failure:** ensure Hugging Face model access, free disk space, and RAM; reduce `EMBEDDING_BATCH_SIZE` if needed.
- **No results or poor results:** ensure ingestion succeeded and metadata filters match. Calibrate `SIMILARITY_THRESHOLD` against representative queries; no universal value is assumed.
- **Generation errors:** verify endpoint path, model, authentication requirements, and timeout. Missing generation configuration does not affect health or data cleaning.
- **Current-law questions:** current-law verification and comparison are not implemented. Confirm against authoritative sources and transitional provisions.

## Testing and deployment status

Core automated tests are local-only. No cloud integration is claimed until run with user-provided Qdrant credentials, and no deployment provider has been selected. Deployment hardening, authentication, CORS policy, operational monitoring, and authoritative legal-source verification remain future work.
