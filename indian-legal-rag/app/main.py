"""FastAPI application factory and ASGI entry point."""

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.runtime import ServiceContainer
from app.utils.logging_config import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API without loading models or connecting to Qdrant at startup."""
    configured = settings or get_settings()
    configure_logging(configured)
    application = FastAPI(
        title="Indian Legal RAG API",
        description="Informational retrieval over owner-provided legal QA datasets.",
        version="0.1.0",
    )
    application.state.settings = configured
    application.state.services = ServiceContainer(configured)
    application.include_router(router)
    return application


app = create_app()
