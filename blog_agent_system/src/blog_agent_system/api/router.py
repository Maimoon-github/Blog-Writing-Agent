"""
FastAPI router for the blog generation API.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Any

from blog_agent_system.api.schemas import (
    BlogGenerationRequest,
    BlogGenerationResponse,
    TaskStatusResponse,
)
from blog_agent_system.api.dependencies import get_orchestrator, get_db
from blog_agent_system.core.orchestrator import BlogOrchestrator
from blog_agent_system.persistence.repositories.task_repo import TaskRepository
from blog_agent_system.utils.logging import get_logger
from blog_agent_system.utils.exceptions import WorkflowError

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["blog"])


@router.post("/generate", response_model=BlogGenerationResponse)
async def generate_blog(
    request: BlogGenerationRequest,
    background_tasks: BackgroundTasks,
    orchestrator: BlogOrchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """Start a new blog generation workflow."""
    try:
        result = await orchestrator.generate_blog(
            topic=request.topic,
            target_audience=request.target_audience,
            tone=request.tone,
            word_count_target=request.word_count_target,
            include_images=request.include_images,
            style_guide=request.style_guide,
        )
        return result
    except Exception as e:
        logger.error("api.generate.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/status/{thread_id}", response_model=TaskStatusResponse)
async def get_status(
    thread_id: str,
    orchestrator: BlogOrchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """Get real-time status of a running workflow."""
    try:
        status = await orchestrator.get_status(thread_id)
        if status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Thread not found")
        return status
    except Exception as e:
        logger.error("api.status.error", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "healthy"}