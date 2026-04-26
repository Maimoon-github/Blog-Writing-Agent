"""
FastAPI dependencies for DI (orchestrator, DB sessions, etc.).
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from blog_agent_system.core.orchestrator import BlogOrchestrator
from blog_agent_system.persistence.database import get_async_session
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)

# Global orchestrator instance (lazy-initialized)
_orchestrator: BlogOrchestrator | None = None


async def get_orchestrator() -> BlogOrchestrator:
    """Dependency that provides the BlogOrchestrator (singleton)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BlogOrchestrator()
        await _orchestrator.initialize()
        logger.info("orchestrator.dependency_initialized")
    return _orchestrator


async def get_db() -> AsyncSession:
    """FastAPI dependency for async DB session (already defined in database.py)."""
    async for session in get_async_session():
        yield session