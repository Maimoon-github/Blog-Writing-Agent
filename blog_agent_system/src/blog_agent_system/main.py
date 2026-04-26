"""
Application entry point — FastAPI app factory.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from blog_agent_system.api.router import router, set_orchestrator
from blog_agent_system.config.settings import settings
from blog_agent_system.core.orchestrator import BlogOrchestrator
from blog_agent_system.utils.logging import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    setup_logging()
    logger = get_logger(__name__)
    logger.info("app.starting", env=settings.app_env)

    # Initialize orchestrator
    orchestrator = BlogOrchestrator()
    set_orchestrator(orchestrator)

    logger.info("app.started")
    yield

    logger.info("app.shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Blog Agent System",
        description="Agentic AI Blog Writing System — Multi-agent pipeline for automated blog generation",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(router)

    return app


# Default app instance for uvicorn
app = create_app()
