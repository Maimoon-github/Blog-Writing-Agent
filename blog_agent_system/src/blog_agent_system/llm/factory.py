# src/blog_agent_system/llm/factory.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMFactory:
    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError))
    )
    def create(config: AgentLLMConfig):
        try:
            if config.provider == "openai":
                return OpenAIProvider(config.model, config.temperature, config.max_tokens)
            elif config.provider == "anthropic":
                return AnthropicProvider(config.model, config.temperature, config.max_tokens)
            elif config.provider == "ollama":
                return OllamaProvider(config.model, config.temperature, config.max_tokens)
        except Exception as e:
            # Fallback to Ollama local model if cloud providers fail
            return OllamaProvider("llama3.1:70b", 0.7, 4096)