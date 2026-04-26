# src/blog_agent_system/agents/base.py
from abc import ABC, abstractmethod
from typing import Any
from blog_agent_system.llm.factory import LLMFactory
from blog_agent_system.config.llm_config import AgentLLMConfig
from blog_agent_system.memory.shared_state import StateAccessor

class BaseAgent(ABC):
    def __init__(self, role: str, config: AgentLLMConfig):
        self.role = role
        self.llm = LLMFactory.create(config)
        self.state = StateAccessor()
        self.tools = []
    
    def bind_tools(self, tools: list):
        self.tools = tools
        self.llm = self.llm.bind_tools(tools)
    
    @abstractmethod
    async def execute(self, state: BlogState) -> dict:
        """Execute agent logic and return state updates."""
        pass
    
    def get_system_prompt(self) -> str:
        return f"You are the {self.role}. Follow your specialized instructions precisely."