"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class BlogGenerationRequest(BaseModel):
    """Request payload for blog generation."""

    topic: str = Field(..., min_length=5, max_length=500)
    target_audience: str = Field(default="technical professionals")
    tone: str = Field(default="informative yet conversational")
    word_count_target: int = Field(default=1500, ge=500, le=5000)
    include_images: bool = Field(default=True)
    style_guide: str = Field(default="AP")


class BlogGenerationResponse(BaseModel):
    """Response payload after blog generation."""

    thread_id: str
    status: str
    final_blog: Optional[str] = None
    quality_score: Optional[float] = None
    revision_count: int = 0
    seo_metadata: Optional[dict] = None
    error: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Status polling response."""

    thread_id: str
    status: str
    current_step: Optional[str] = None
    quality_score: Optional[float] = None
    revision_count: int = 0
    created_at: datetime