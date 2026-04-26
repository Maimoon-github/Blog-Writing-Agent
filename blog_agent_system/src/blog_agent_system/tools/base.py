# src/blog_agent_system/tools/base.py
from pydantic import BaseModel, ValidationError
from typing import Callable, Any
from blog_agent_system.utils.exceptions import ToolExecutionError

class BaseTool:
    def __init__(self, name: str, description: str, func: Callable, args_schema: type[BaseModel]):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema
    
    def validate_input(self, kwargs: dict) -> BaseModel:
        try:
            return self.args_schema(**kwargs)
        except ValidationError as e:
            raise ToolExecutionError(f"Invalid input for {self.name}: {e}")
    
    async def invoke(self, **kwargs) -> Any:
        validated = self.validate_input(kwargs)
        try:
            result = await self.func(**validated.model_dump())
            return {
                "tool": self.name,
                "status": "success",
                "result": result,
                "input": validated.model_dump()
            }
        except Exception as e:
            return {
                "tool": self.name,
                "status": "error",
                "error": str(e),
                "input": validated.model_dump()
            }