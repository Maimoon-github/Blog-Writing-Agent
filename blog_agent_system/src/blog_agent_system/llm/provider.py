# src/blog_agent_system/llm/provider.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
from pydantic import BaseModel

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list, tools: list = None, response_format: type[BaseModel] = None) -> str:
        pass
    
    @abstractmethod
    async def stream(self, messages: list) -> AsyncIterator[str]:
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass