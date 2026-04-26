"""
BaseAgent — foundation for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Any

from blog_agent_system.config.llm_config import AGENT_LLM_CONFIGS
from blog_agent_system.llm.factory import LLMFactory
from blog_agent_system.memory.shared_state import StateAccessor
from blog_agent_system.core.state import BlogState
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents in the pipeline."""

    def __init__(self, role: str):
        self.role = role
        self.config = AGENT_LLM_CONFIGS[role]
        self.llm = LLMFactory.create(self.config)
        self.state = StateAccessor()
        self.tools = []
        logger.debug("agent.initialized", role=role, model=self.config.model)

    def bind_tools(self, tools: list) -> None:
        """Bind tools to the LLM (LangChain compatible)."""
        self.tools = tools
        self.llm = self.llm.bind_tools(tools)  # type: ignore[attr-defined]

    @abstractmethod
    async def execute(self, state: BlogState) -> dict[str, Any]:
        """Execute agent logic and return state deltas (immutability preserved)."""
        pass

    def get_system_prompt(self) -> str:
        """Default system prompt — overridden by concrete agents."""
        return f"You are the {self.role}. Follow your specialized instructions precisely."