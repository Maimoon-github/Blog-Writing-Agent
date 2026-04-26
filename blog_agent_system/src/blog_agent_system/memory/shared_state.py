# src/blog_agent_system/memory/shared_state.py
from langgraph.checkpoint.postgres import PostgresSaver
from blog_agent_system.core.state import BlogState

class StateAccessor:
    def __init__(self, checkpointer: PostgresSaver):
        self.checkpointer = checkpointer
    
    async def read(self, thread_id: str) -> BlogState:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await self.checkpointer.aget(config)
        return BlogState(**checkpoint["channel_values"]) if checkpoint else BlogState()
    
    async def write(self, thread_id: str, updates: dict):
        # LangGraph handles this automatically via node returns,
        # but this is used for external state injection
        config = {"configurable": {"thread_id": thread_id}}
        await self.checkpointer.aput(config, updates)