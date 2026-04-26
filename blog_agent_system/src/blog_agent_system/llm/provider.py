from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
from pydantic import BaseModel


class LLMProvider(ABC):
    """Abstract base for all LLM providers (Ollama-only in this project)."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        """Generate a complete response."""

    @abstractmethod
    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens for budget management."""