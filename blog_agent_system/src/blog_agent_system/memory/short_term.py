# src/blog_agent_system/memory/short_term.py
import redis
from langchain_core.messages import BaseMessage
from typing import Sequence

class ShortTermMemory:
    def __init__(self, redis_client: redis.Redis, max_messages: int = 20, ttl: int = 3600):
        self.redis = redis_client
        self.max_messages = max_messages
        self.ttl = ttl
    
    async def add_messages(self, thread_id: str, messages: Sequence[BaseMessage]):
        key = f"stm:{thread_id}"
        pipe = self.redis.pipeline()
        for msg in messages:
            pipe.lpush(key, msg.json())
        pipe.ltrim(key, 0, self.max_messages - 1)
        pipe.expire(key, self.ttl)
        await pipe.execute()
    
    async def get_messages(self, thread_id: str) -> list[BaseMessage]:
        key = f"stm:{thread_id}"
        raw = await self.redis.lrange(key, 0, -1)
        return [BaseMessage.parse_json(m) for m in raw]