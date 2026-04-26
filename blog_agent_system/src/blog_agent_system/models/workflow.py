"""
Pydantic schemas: Task request, status, response, and quality score DTOs for API and orchestration.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class TaskRequest(BaseModel):
    """API request to start a new blog generation task."""

    topic: str = Field(..., min_length=5, max_length=500)
    target_audience: str = Field(default="technical professionals")
    tone: str = Field(default="informative yet conversational")
    word_count_target: int = Field(default=1500, ge=500, le=5000)
    include_images: bool = Field(default=True)
    style_guide: str = Field(default="AP", pattern=r"^(AP|Chicago|MLA)$")


class TaskStatus(BaseModel):
    """Real-time workflow status for polling."""

    thread_id: str
    status: str
    current_step: str = "init"
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    revision_count: int = Field(default=0, ge=0)
    error: Optional[str] = None


class TaskResponse(BaseModel):
    """Final response after workflow completion."""

    thread_id: str
    correlation_id: str = ""
    status: str
    final_blog: str = ""
    draft: str = ""
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    revision_count: int = Field(default=0, ge=0)
    seo_metadata: Optional[dict] = None
    fact_check_results: list[dict] = Field(default_factory=list)
    error: Optional[str] = None


class QualityScore(BaseModel):
    """Detailed quality breakdown (used internally and in responses)."""

    overall: float = Field(ge=0.0, le=1.0)
    readability: float = Field(ge=0.0, le=1.0)
    factual_accuracy: float = Field(ge=0.0, le=1.0)
    seo_compliance: float = Field(ge=0.0, le=1.0)