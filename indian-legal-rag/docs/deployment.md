# Deployment

No hosting provider is selected and no deployment is performed. The API expects persistent storage for the local exact-search SQLite index and cached BGE-M3 artifacts. Configure HTTPS, least-privilege secrets, request limits, CORS, health checks, and resource sizing before public exposure.

Qdrant collection creation and data ingestion are explicit operator-controlled steps, not web-server startup actions. Keep ingestion credentials and raw datasets out of frontend/static deployments. LLM and Qdrant credentials must remain server-side.

TODO — implementation pending: provider-specific container definitions, deployment pipeline, production observability, and security review.
