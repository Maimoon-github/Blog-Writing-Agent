"""
Anthropic Claude LLM provider implementation.
"""

from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from blog_agent_system.config.settings import settings
from blog_agent_system.llm.provider import LLMProvider
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude model provider."""

    def __init__(self, model: str = "claude-sonnet-4-6", temperature: float = 0.7, max_tokens: int = 4096):
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
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
        # Anthropic uses a separate system parameter
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            else:
                user_messages.append(msg)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": user_messages if user_messages else [{"role": "user", "content": ""}],
        }

        if system_prompt.strip():
            request_kwargs["system"] = system_prompt.strip()

        if tools:
            request_kwargs["tools"] = tools

        response = await self.client.messages.create(**request_kwargs)

        if response.usage:
            logger.info("anthropic.usage", model=self.model,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens)

        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts)

    async def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            else:
                user_messages.append(msg)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": user_messages if user_messages else [{"role": "user", "content": ""}],
        }

        if system_prompt.strip():
            request_kwargs["system"] = system_prompt.strip()

        async with self.client.messages.stream(**request_kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str) -> int:
        return len(text) // 4
