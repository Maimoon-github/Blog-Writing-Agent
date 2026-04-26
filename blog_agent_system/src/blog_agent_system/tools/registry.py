"""
Tool registry — registration, discovery, and metadata for all tools.
"""

from typing import Callable, Dict

from blog_agent_system.tools.base import BaseTool
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Central registry for all available tools."""

    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool_instance: BaseTool) -> BaseTool:
        cls._tools[tool_instance.name] = tool_instance
        logger.info("tool_registry.registered", tool=tool_instance.name)
        return tool_instance

    @classmethod
    def get(cls, name: str) -> BaseTool:
        if name not in cls._tools:
            raise KeyError(f"Tool '{name}' not registered. Available: {cls.list_tools()}")
        return cls._tools[name]

    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls._tools.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (for testing)."""
        cls._tools.clear()


def tool(name: str, description: str, args_schema: type):
    """Decorator for tool registration."""
    def decorator(func: Callable):
        tool_instance = BaseTool(
            name=name,
            description=description,
            func=func,
            args_schema=args_schema,
        )
        ToolRegistry.register(tool_instance)
        return func
    return decorator