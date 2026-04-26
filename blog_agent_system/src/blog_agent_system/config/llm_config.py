from pydantic import BaseModel

class AgentLLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    retry_attempts: int = 3

# All agents now use local Ollama models
AGENT_LLM_CONFIGS = {
    "research": AgentLLMConfig(
        provider="ollama", 
        model="mistral",           # Fast & good for research
        temperature=0.3, 
        max_tokens=2048
    ),
    "outline": AgentLLMConfig(
        provider="ollama", 
        model="mistral", 
        temperature=0.5, 
        max_tokens=2048
    ),
    "writer": AgentLLMConfig(
        provider="ollama", 
        model="gemma2:9b",         # Better for long-form writing
        temperature=0.8, 
        max_tokens=8192
    ),
    "editor": AgentLLMConfig(
        provider="ollama", 
        model="gemma2:9b", 
        temperature=0.4, 
        max_tokens=8192
    ),
    "seo": AgentLLMConfig(
        provider="ollama", 
        model="mistral", 
        temperature=0.2, 
        max_tokens=1024
    ),
    "fact_checker": AgentLLMConfig(
        provider="ollama", 
        model="mistral", 
        temperature=0.1, 
        max_tokens=2048
    ),
    "image": AgentLLMConfig(
        provider="ollama", 
        model="mistral",           # For image prompt generation
        temperature=1.0, 
        max_tokens=1024
    ),
}