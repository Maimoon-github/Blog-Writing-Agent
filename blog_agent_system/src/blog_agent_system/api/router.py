"""
FastAPI route definitions for the blog generation API.
"""

import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from blog_agent_system.core.orchestrator import BlogOrchestrator
from blog_agent_system.models.workflow import TaskRequest, TaskResponse, TaskStatus
from blog_agent_system.persistence.database import get_async_session
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/blog", tags=["blog"])

# Module-level orchestrator (initialized in app lifespan)
_orchestrator: BlogOrchestrator | None = None


def get_orchestrator() -> BlogOrchestrator:
    """Dependency to get the orchestrator instance."""
    if _orchestrator is None:
        return BlogOrchestrator()
    return _orchestrator


def set_orchestrator(orchestrator: BlogOrchestrator) -> None:
    """Set the module-level orchestrator (called during app startup)."""
    global _orchestrator
    _orchestrator = orchestrator


@router.post("/generate", response_model=TaskResponse)
async def generate_blog(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    orchestrator: BlogOrchestrator = Depends(get_orchestrator),
) -> TaskResponse:
    """
    Start a blog generation workflow.

    Returns immediately with a thread_id for status polling.
    The actual generation runs in the background.
    """
    logger.info("api.generate", topic=request.topic)

    # For now, run synchronously. In production, use background tasks + webhook.
    result = await orchestrator.generate_blog(
        topic=request.topic,
        target_audience=request.target_audience,
        tone=request.tone,
        word_count_target=request.word_count_target,
        include_images=request.include_images,
        style_guide=request.style_guide,
    )

    return TaskResponse(**result)


@router.get("/{thread_id}/status", response_model=TaskStatus)
async def get_status(
    thread_id: str,
    orchestrator: BlogOrchestrator = Depends(get_orchestrator),
) -> TaskStatus:
    """Poll the status of a blog generation workflow."""
    result = await orchestrator.get_status(thread_id)
    return TaskStatus(**result)


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "blog-agent-system"}
