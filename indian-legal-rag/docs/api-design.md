# API design

- `GET /health`: application and safe configuration status; no credentials are returned.
- `GET /health/qdrant`: read-only reachability, collection existence, vector compatibility, and count check.
- `POST /retrieve`: bounded question and optional law/source/top-k filters; returns retrieved evidence without LLM generation.
- `POST /ask`: retrieves evidence, abstains on insufficient evidence, otherwise calls configured OpenAI-compatible hosted inference and returns source citations.

FastAPI validates input. The embedding model is lazy and query work runs in a worker thread. Error responses do not include stack traces, prompts, or environment values. Missing LLM configuration yields HTTP 503 for `/ask`; other endpoints remain available. Citation identifiers generated in an answer are checked against the evidence passed to generation.

TODO — implementation pending: authentication, abuse prevention policy, request tracing, and endpoint versioning.
