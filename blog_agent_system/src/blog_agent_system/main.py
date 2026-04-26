"""
Application entry point — FastAPI app factory with lifespan management.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from blog_agent_system.api.router import router
from blog_agent_system.config.settings import settings
from blog_agent_system.core.orchestrator import BlogOrchestrator
from blog_agent_system.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Setup structured logging
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)

    logger.info("app.starting", env=settings.app_env, version="0.1.0")

    # Initialize orchestrator (lazy + cached)
    app.state.orchestrator = BlogOrchestrator()
    await app.state.orchestrator.initialize()

    logger.info("app.started", orchestrator_ready=True)
    yield

    logger.info("app.shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Blog Agent System",
        description="Agentic AI Blog Writing System — Multi-agent LangGraph pipeline",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.app_env == "development" else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(router, prefix="/api")

    return app


# Default app instance for uvicorn / production
app = create_app()