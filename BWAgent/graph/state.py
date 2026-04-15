"""Typed graph state definitions for BWAgent."""

import operator
from typing import Annotated, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class Section(BaseModel):
    id: str = Field(..., description="Unique section identifier")
    title: str = Field(..., description="Section title")
    description: str = Field(..., description="Section content summary")
    word_count: int = Field(..., gt=0, description="Target word count")
    search_query: Optional[str] = Field(None, description="Search query for this section")
    image_prompt: str = Field(..., description="Stable Diffusion prompt for this section image")


class BlogPlan(BaseModel):
    blog_title: str = Field(..., description="Title for the final blog post")
    feature_image_prompt: str = Field(..., description="Prompt for the hero image")
    research_required: bool = Field(..., description="Whether live web research is required")
    sections: List[Section] = Field(..., description="Ordered section planning")


class ResearchResult(BaseModel):
    section_id: str = Field(..., description="ID of the researched section")
    query: str = Field(..., description="Search query used")
    summary: str = Field(..., description="Fact-based summary for the section")
    source_urls: List[str] = Field(default_factory=list, description="Source URLs used for the summary")


class SectionDraft(BaseModel):
    section_id: str = Field(..., description="Section identifier")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Generated Markdown content for the section")
    citation_keys: List[str] = Field(default_factory=list, description="Citation markers found in the draft")


class GeneratedImage(BaseModel):
    section_id: str = Field(..., description="Section ID or 'feature' for the hero image")
    image_path: str = Field(..., description="Filesystem path for the generated image")
    prompt: str = Field(..., description="Prompt used to generate the image")


class GraphState(TypedDict):
    topic: str
    research_required: bool
    blog_plan: Optional[BlogPlan]
    research_results: Annotated[List[ResearchResult], operator.add]
    section_drafts: Annotated[List[SectionDraft], operator.add]
    generated_images: Annotated[List[GeneratedImage], operator.add]
    citation_registry: Dict[str, str]
    final_blog_md: str
    final_blog_html: str
    run_id: str
    error: Optional[str]
