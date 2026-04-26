"""Application settings loaded from environment variables."""
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    app_env: str = Field(default="development")
    app_debug: bool = Field(default=False)

    # LLM Provider API Keys
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")

    # Web Search
    tavily_api_key: str = Field(default="")

    # Database
    database_url: str = Field(default="postgresql+asyncpg://blog_agent:blog_agent_pass@localhost:5432/blog_agent_db")
    database_url_sync: str = Field(default="postgresql+psycopg://blog_agent:blog_agent_pass@localhost:5432/blog_agent_db")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ChromaDB
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="console")

    # Quality Gate
    quality_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_revisions: int = Field(default=3, ge=1, le=10)
    workflow_timeout_seconds: int = Field(default=600, ge=60, le=3600)

    # Blog Defaults
    default_word_count: int = Field(default=1500, ge=500, le=5000)
    default_tone: str = Field(default="informative yet conversational")
    default_audience: str = Field(default="technical professionals")
    default_style_guide: str = Field(default="AP")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return upper

# Singleton
settings = Settings()