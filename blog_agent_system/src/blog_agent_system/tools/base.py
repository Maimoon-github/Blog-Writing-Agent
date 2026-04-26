"""
Tool base class with input validation and error handling.
"""

from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from blog_agent_system.utils.exceptions import ToolExecutionError
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class BaseTool:
    """Base class for all tools with Pydantic input validation."""

    def __init__(self, name: str, description: str, func: Callable, args_schema: type[BaseModel]):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema

    def validate_input(self, kwargs: dict) -> BaseModel:
        try:
            return self.args_schema(**kwargs)
        except ValidationError as e:
            raise ToolExecutionError(
                tool_name=self.name,
                message=f"Invalid input: {e}",
            )

    async def invoke(self, **kwargs) -> dict[str, Any]:
        validated = self.validate_input(kwargs)
        try:
            result = await self.func(**validated.model_dump())
            logger.info("tool.success", tool=self.name)
            return {
                "tool": self.name,
                "status": "success",
                "result": result,
                "input": validated.model_dump(),
            }
        except Exception as e:
            logger.error("tool.error", tool=self.name, error=str(e))
            return {
                "tool": self.name,
                "status": "error",
                "error": str(e),
                "input": validated.model_dump(),
            }