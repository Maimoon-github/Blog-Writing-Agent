# src/blog_agent_system/tools/registry.py
from typing import Callable, Dict
from blog_agent_system.tools.base import BaseTool

class ToolRegistry:
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, tool: BaseTool):
        cls._tools[tool.name] = tool
        return tool
    
    @classmethod
    def get(cls, name: str) -> BaseTool:
        if name not in cls._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return cls._tools[name]
    
    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls._tools.keys())

def tool(name: str, description: str, args_schema: type):
    """Decorator for tool registration."""
    def decorator(func: Callable):
        tool_instance = BaseTool(
            name=name,
            description=description,
            func=func,
            args_schema=args_schema
        )
        ToolRegistry.register(tool_instance)
        return func
    return decorator