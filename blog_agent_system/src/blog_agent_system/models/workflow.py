"""
Pydantic schemas: Task request, status, and quality score DTOs.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TaskRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500, description="Blog topic")
    target_audience: str = Field(default="technical professionals")
    tone: str = Field(default="informative yet conversational")
    word_count_target: int = Field(default=1500, ge=500, le=5000)
    include_images: bool = Field(default=True)
    style_guide: str = Field(default="AP", pattern=r"^(AP|Chicago|MLA)$")


class TaskStatus(BaseModel):
    thread_id: str
    status: str
    current_step: str = "init"
    quality_score: float = 0.0
    revision_count: int = 0
    error: Optional[str] = None


class TaskResponse(BaseModel):
    thread_id: str
    correlation_id: str = ""
    status: str
    final_blog: str = ""
    draft: str = ""
    quality_score: float = 0.0
    revision_count: int = 0
    seo_metadata: Optional[dict] = None
    fact_check_results: list[dict] = []
    error: Optional[str] = None


class QualityScore(BaseModel):
    overall: float = Field(ge=0.0, le=1.0)
    readability: float = Field(ge=0.0, le=1.0)
    factual_accuracy: float = Field(ge=0.0, le=1.0)
    seo_compliance: float = Field(ge=0.0, le=1.0)
