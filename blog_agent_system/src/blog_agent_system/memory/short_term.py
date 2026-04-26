"""
Short-term memory — Redis-backed conversation buffer with sliding window.
"""

import redis.asyncio as redis
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict
from typing import Sequence

from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class ShortTermMemory:
    """In-context conversation buffer using Redis Lists with TTL eviction."""

    def __init__(self, redis_client: redis.Redis, max_messages: int = 20, ttl: int = 3600):
        self.redis = redis_client
        self.max_messages = max_messages
        self.ttl = ttl

    async def add_messages(self, thread_id: str, messages: Sequence[BaseMessage]) -> None:
        key = f"stm:{thread_id}"
        pipe = self.redis.pipeline()
        for msg in messages:
            import json
            pipe.lpush(key, json.dumps(message_to_dict(msg)))
        pipe.ltrim(key, 0, self.max_messages - 1)
        pipe.expire(key, self.ttl)
        await pipe.execute()

        logger.debug("short_term.added", thread_id=thread_id, count=len(messages))

    async def get_messages(self, thread_id: str) -> list[BaseMessage]:
        key = f"stm:{thread_id}"
        raw = await self.redis.lrange(key, 0, -1)
        if not raw:
            return []
        import json
        message_dicts = [json.loads(m) for m in raw]
        return list(reversed(messages_from_dict(message_dicts)))

    async def clear(self, thread_id: str) -> None:
        """Clear all messages for a thread."""
        await self.redis.delete(f"stm:{thread_id}")