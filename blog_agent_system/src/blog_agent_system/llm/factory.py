"""
LLM Factory — always returns OllamaProvider as per user requirement.
"""

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from blog_agent_system.config.llm_config import AgentLLMConfig
from blog_agent_system.llm.ollama_provider import OllamaProvider
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Factory that always uses local Ollama models."""

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    def create(config: AgentLLMConfig) -> OllamaProvider:
        """Always create an Ollama provider."""
        try:
            provider = OllamaProvider(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            logger.debug("llm_factory.created", model=config.model, provider="ollama")
            return provider
        except Exception as e:
            logger.error("llm_factory.fallback", error=str(e))
            # Ultimate fallback
            return OllamaProvider(model="mistral", temperature=0.7, max_tokens=4096)