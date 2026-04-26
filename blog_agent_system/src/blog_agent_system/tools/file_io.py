"""
File I/O tool — read/write local artifacts (markdown, images, JSON).
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from blog_agent_system.tools.registry import tool
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class FileReadInput(BaseModel):
    path: str = Field(..., description="File path to read")
    encoding: str = Field(default="utf-8")


class FileWriteInput(BaseModel):
    path: str = Field(..., description="File path to write")
    content: str = Field(..., description="Content to write")
    format: str = Field(default="text", pattern=r"^(text|json|markdown)$")


@tool(name="file_read", description="Read a local file", args_schema=FileReadInput)
async def file_read(path: str, encoding: str) -> str:
    """Read file contents."""
    file_path = Path(path)
    if not file_path.exists():
        return f"Error: File not found: {path}"
    content = file_path.read_text(encoding=encoding)
    logger.info("file_read.complete", path=path, bytes=len(content))
    return content


@tool(name="file_write", description="Write content to a local file", args_schema=FileWriteInput)
async def file_write(path: str, content: str, format: str) -> dict:
    """Write content to file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    logger.info("file_write.complete", path=path, bytes=len(content))
    return {"success": True, "bytes_written": len(content)}
