"""
Ollama local LLM provider for cost-free fallback.
"""

from typing import Any, AsyncIterator

import httpx
from pydantic import BaseModel

from blog_agent_system.config.settings import settings
from blog_agent_system.llm.provider import LLMProvider
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""

    def __init__(self, model: str = "llama3.1:70b", temperature: float = 0.7, max_tokens: int = 4096):
        self.base_url = settings.ollama_base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        response_format: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if response_format:
            request_body["format"] = "json"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=request_body)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

    async def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        import json as json_mod

        request_body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=request_body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        chunk = json_mod.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content

    def count_tokens(self, text: str) -> int:
        return len(text) // 4
