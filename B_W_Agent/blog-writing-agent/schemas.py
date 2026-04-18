"""
Single source of truth for all inter-agent data contracts in the
Autonomous Blog Generation Agent (8-agent parallel pipeline).

These Pydantic models are imported by every agent, tool, graph.py,
state.py, and streamlit_app.py. They serve as strict contracts for
structured output (with_structured_output) and CrewState merging.

References:
- roadmap.html → Phase 2 Step 3 (schemas.py file card)
- idea.md → Section, BlogPlan, ResearchResult, SectionDraft, ImageResult
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class Section(BaseModel):
    """Represents one blog section in the Planner's output.
    Used by Researcher, Writer, Editor, and Image Agent workers.
    """
    id: str
    """Unique section identifier (e.g. 'sec-001')"""
    title: str
    """Section heading"""
    description: str
    """Short description used by Writer/Editor"""
    word_count: int
    """Target word count for this section"""
    search_query: Optional[str] = None
    """Optional search query for Researcher (None = LLM knowledge only)"""
    image_prompt: str
    """Prompt passed to Image Agent for section illustration"""


class BlogPlan(BaseModel):
    """Structured output from Planner node.
    Drives parallel dispatch of all worker crews.
    """
    blog_title: str
    """Full blog post title"""
    feature_image_prompt: str
    """Prompt for the hero/feature image (Image Agent)"""
    sections: List[Section]
    """List of sections (determines parallelism count)"""
    research_required: bool
    """Flag set by Router; controls whether Researcher crews run"""


class ResearchResult(BaseModel):
    """Output of each Researcher worker (one per section)."""
    section_id: str
    """Links result back to its Section"""
    summary: str
    """300–500 word synthesized research summary"""
    sources: List[Dict[str, str]]
    """List of {'url': str, 'title': str, 'snippet': str} for Citation Manager"""
    timestamp: datetime
    """When research was performed"""


class SectionDraft(BaseModel):
    """Output of Writer → Editor pipeline (one per section)."""
    section_id: str
    """Links draft to its original Section"""
    title: str
    """Section title (may be refined by Editor)"""
    content: str
    """Markdown content preserving [citation] and [IMAGE_PLACEHOLDER_{id}] tokens"""
    word_count: int
    """Actual word count after editing"""
    citations: List[str]
    """Raw citation references (resolved later by Citation Manager)"""


class ImageResult(BaseModel):
    """Output of each Image Agent call."""
    section_id: str
    """Links image to its section (or 'feature' for hero image)"""
    prompt: str
    """Exact prompt used"""
    file_path: str
    """Relative path to saved PNG (outputs/images/...)"""
    alt_text: str
    """Generated alt text for Markdown"""
    size: Tuple[int, int]
    """Actual image dimensions (e.g. (512, 512))"""


__all__ = [
    "Section",
    "BlogPlan",
    "ResearchResult",
    "SectionDraft",
    "ImageResult",
]

# Example usage (for reference only — not executed):
# planner_chain = prompt | llm.with_structured_output(BlogPlan)
# result: ResearchResult = researcher_node(...)
# state: CrewState = {"plan": blog_plan, "research_results": [result], ...}