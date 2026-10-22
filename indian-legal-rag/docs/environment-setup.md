# Environment setup

Create and activate a Python 3.11 virtual environment, then install root `requirements.txt` as shown in the root README. Copy `.env.example` to `.env`; never commit `.env`. `app/config.py` loads `config.yaml` defaults and uses environment variables as overrides. Empty environment values are ignored.

Configure `QDRANT_URL` and `QDRANT_API_KEY` from the owner's Qdrant Cloud cluster. Set `LLM_BASE_URL` and `LLM_MODEL` for an OpenAI-compatible chat-completions service; `LLM_API_KEY` is only required if the endpoint requires authentication. `BAAI/bge-m3` is downloaded and run locally the first time embeddings are requested.

TODO — implementation pending: provider-specific configuration validation and secrets manager integration.
