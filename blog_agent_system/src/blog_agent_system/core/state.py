# src/blog_agent_system/core/state.py
from typing import Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
import operator

class Section(BaseModel):
    heading: str
    content: str = ""
    word_count: int = 0
    sources: list[str] = []

class BlogState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    target_audience: str = "general"
    tone: str = "informative"
    research_findings: list[dict] = []
    outline: list[Section] = []
    draft: str = ""
    draft_sections: list[Section] = []
    seo_metadata: dict = {}
    fact_check_results: list[dict] = []
    quality_score: float = 0.0
    revision_count: int = 0
    max_revisions: int = 3
    final_blog: str = ""
    status: str = "pending"  # pending, researching, outlining, drafting, editing, complete

























# src/blog_agent_system/core/state.py
from typing import Annotated, Sequence, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
import operator

class Source(BaseModel):
    url: str
    title: str
    snippet: str
    credibility_score: float = Field(ge=0.0, le=1.0)

class SEOData(BaseModel):
    title_tag: str
    meta_description: str
    keywords: list[str]
    readability_score: float
    keyword_density: dict[str, float]

class FactCheckResult(BaseModel):
    claim: str
    verified: bool
    confidence: float
    source_ref: Optional[str]
    correction: Optional[str]

class BlogState(BaseModel):
    # Messaging
    messages: Annotated[Sequence[BaseMessage], operator.add] = []
    
    # Input parameters
    topic: str = Field(..., min_length=5, max_length=500)
    target_audience: str = "technical professionals"
    tone: str = "informative yet conversational"
    word_count_target: int = Field(default=1500, ge=500, le=5000)
    include_images: bool = True
    style_guide: str = "AP"  # AP, Chicago, MLA
    
    # Workflow state
    status: str = "pending"
    current_step: str = "init"
    
    # Agent outputs
    research_findings: list[Source] = []
    outline: list[Section] = []
    draft_sections: list[Section] = []
    draft: str = ""
    edited_draft: str = ""
    seo_metadata: Optional[SEOData] = None
    fact_check_results: list[FactCheckResult] = []
    cover_image_url: Optional[str] = None
    section_images: dict[str, str] = {}  # heading -> url
    
    # Quality & loop control
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    revision_count: int = Field(default=0, ge=0, le=5)
    max_revisions: int = 3
    revision_feedback: str = ""
    
    # Final output
    final_blog: str = ""
    export_format: str = "markdown"  # markdown, html, pdf
    
    class Config:
        arbitrary_types_allowed = True