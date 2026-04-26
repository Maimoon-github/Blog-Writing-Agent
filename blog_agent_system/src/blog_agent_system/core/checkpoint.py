"""LangGraph persistence adapter using PostgreSQL."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from blog_agent_system.config.settings import settings
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


async def create_checkpointer() -> AsyncPostgresSaver:
    """Create and initialize PostgreSQL checkpointer."""
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url_sync)
    await checkpointer.setup()
    logger.info("checkpointer.initialized", backend="postgres")
    return checkpointer