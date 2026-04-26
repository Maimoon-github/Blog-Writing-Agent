# src/blog_agent_system/config/llm_config.py
from pydantic import BaseModel
from typing import Optional

class AgentLLMConfig(BaseModel):
    provider: str  # openai, anthropic, ollama
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    retry_attempts: int = 3

AGENT_LLM_CONFIGS = {
    "research": AgentLLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.3, max_tokens=2048),
    "outline": AgentLLMConfig(provider="anthropic", model="claude-3-haiku-20240307", temperature=0.5, max_tokens=2048),
    "writer": AgentLLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022", temperature=0.8, max_tokens=8192),
    "editor": AgentLLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022", temperature=0.4, max_tokens=8192),
    "seo": AgentLLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.2, max_tokens=1024),
    "fact_checker": AgentLLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.1, max_tokens=2048),
    "image": AgentLLMConfig(provider="openai", model="dall-e-3", temperature=1.0, max_tokens=1024),
}