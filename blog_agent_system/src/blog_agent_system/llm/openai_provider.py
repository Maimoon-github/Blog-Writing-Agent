# src/blog_agent_system/llm/openai_provider.py
from openai import AsyncOpenAI
from blog_agent_system.llm.provider import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 4096):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def generate(self, messages, tools=None, response_format=None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    def count_tokens(self, text: str) -> int:
        # Use tiktoken for accurate counting
        import tiktoken
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))