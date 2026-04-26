"""
FastAPI dependency injection: DB sessions, Redis, vector store.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from blog_agent_system.persistence.database import get_async_session


async def get_db() -> AsyncSession:
    """Async database session dependency."""
    async for session in get_async_session():
        yield session
