"""
Shared state accessor — reads/writes LangGraph checkpoint state.
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from blog_agent_system.core.state import BlogState
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class StateAccessor:
    """Thread-scoped read/write access to LangGraph checkpoint state."""

    def __init__(self, checkpointer: AsyncPostgresSaver):
        self.checkpointer = checkpointer

    async def read(self, thread_id: str) -> BlogState | None:
        """Read current state from checkpoint."""
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await self.checkpointer.aget(config)
        if checkpoint and "channel_values" in checkpoint:
            return BlogState(**checkpoint["channel_values"])
        return None

    async def write(self, thread_id: str, updates: dict) -> None:
        """External state injection (LangGraph handles most via node returns)."""
        config = {"configurable": {"thread_id": thread_id}}
        await self.checkpointer.aput(config, updates)
        logger.debug("shared_state.write", thread_id=thread_id, keys=list(updates.keys()))