"""
Custom exception hierarchy for the blog agent system.
"""

class BlogAgentError(Exception):
    """Base exception for all blog agent system errors."""

    def __init__(self, message: str = "", details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# Orchestration Layer
class WorkflowError(BlogAgentError):
    """Raised when the LangGraph workflow encounters an unrecoverable error."""

    pass


class QualityGateError(BlogAgentError):
    """Raised when the quality gate fails after exhausting all revision attempts."""

    pass


class WorkflowTimeoutError(BlogAgentError):
    """Raised when the workflow exceeds the configured timeout."""

    pass


# Agent Layer
class AgentExecutionError(BlogAgentError):
    """Raised when an agent fails to execute its task."""

    def __init__(self, agent_name: str, message: str = "", details: dict | None = None):
        self.agent_name = agent_name
        super().__init__(f"[{agent_name}] {message}", details)


class AgentConfigError(BlogAgentError):
    """Raised when agent configuration is invalid."""

    pass


# Tool Layer
class ToolExecutionError(BlogAgentError):
    """Raised when a tool invocation fails."""

    def __init__(self, tool_name: str = "", message: str = "", details: dict | None = None):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}': {message}" if tool_name else message, details)


class ToolNotFoundError(BlogAgentError):
    """Raised when a requested tool is not registered."""

    pass


class ToolValidationError(ToolExecutionError):
    """Raised when tool input validation fails."""

    pass


# LLM Layer
class LLMProviderError(BlogAgentError):
    """Raised when an LLM provider call fails."""

    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when an LLM provider rate-limits the request."""

    pass


class LLMConnectionError(LLMProviderError):
    """Raised when an LLM provider is unreachable."""

    pass


class StructuredOutputError(LLMProviderError):
    """Raised when LLM output fails to parse against the expected schema."""

    pass


class TokenBudgetExceededError(LLMProviderError):
    """Raised when input exceeds the model's token budget after truncation."""

    pass


# Persistence Layer
class DatabaseError(BlogAgentError):
    """Raised when a database operation fails."""

    pass


class EntityNotFoundError(DatabaseError):
    """Raised when a requested entity does not exist."""

    def __init__(self, entity_type: str, entity_id: str, details: dict | None = None):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with id '{entity_id}' not found", details)


class VectorStoreError(BlogAgentError):
    """Raised when ChromaDB operations fail."""

    pass