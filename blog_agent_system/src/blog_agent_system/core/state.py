from typing import Annotated, Sequence, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
import operator


class Source(BaseModel):
    url: str
    title: str
    snippet: str
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.5)


class Section(BaseModel):
    heading: str
    content: str = ""
    word_count: int = 0
    sources: list[str] = []


class SEOData(BaseModel):
    title_tag: str
    meta_description: str
    keywords: list[str] = []
    readability_score: float = 0.0
    keyword_density: dict[str, float] = {}


class FactCheckResult(BaseModel):
    claim: str
    verified: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    source_ref: Optional[str] = None
    correction: Optional[str] = None


class BlogState(BaseModel):
    """Central state schema for the entire LangGraph workflow."""

    # Messaging
    messages: Annotated[Sequence[BaseMessage], operator.add] = []

    # Input parameters
    topic: str = Field(..., min_length=5, max_length=500)
    target_audience: str = "technical professionals"
    tone: str = "informative yet conversational"
    word_count_target: int = Field(default=1500, ge=500, le=5000)
    include_images: bool = True
    style_guide: str = "AP"

    # Workflow control
    status: str = "pending"
    current_step: str = "init"
    revision_count: int = Field(default=0, ge=0, le=5)
    revision_feedback: str = ""
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_revisions: int = 3

    # Agent outputs
    research_findings: list[Source] = []
    outline: list[Section] = []
    draft_sections: list[Section] = []
    draft: str = ""
    edited_draft: str = ""
    seo_metadata: Optional[SEOData] = None
    fact_check_results: list[FactCheckResult] = []
    cover_image_url: Optional[str] = None
    section_images: dict[str, str] = {}

    # Final output
    final_blog: str = ""
    export_format: str = "markdown"

    class Config:
        arbitrary_types_allowed = True