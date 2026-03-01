"""Shared data models and graph state for the blog agent pipeline.

This module defines the complete state schema for the LangGraph workflow,
including all Pydantic models for structured data and the GraphState TypedDict
with reducers for parallel execution. The models include validation constraints
to ensure data integrity throughout the pipeline.

The state uses Annotated types with operator.add reducers to safely accumulate
results from parallel nodes (researchers, writers, image generators) without
conflicts.
"""

import operator
from typing import TypedDict, List, Dict, Optional, Annotated

from pydantic import BaseModel, Field


class Section(BaseModel):
    """Represents a single section in the blog plan.

    Attributes:
        id: Unique identifier for the section.
        title: Section title.
        description: Detailed description of the section's content.
        word_count: Target word count (must be positive).
        search_query: Optional search query for web research; if None, no research.
        image_prompt: Prompt for image generation (can be empty string).
    """

    id: str = Field(..., description="Unique section identifier")
    title: str = Field(..., description="Section title")
    description: str = Field(..., description="Detailed description of the section")
    word_count: int = Field(..., gt=0, description="Target word count (positive integer)")
    search_query: Optional[str] = Field(None, description="Search query for research; None means no research")
    image_prompt: str = Field(..., description="Prompt for image generation")


class BlogPlan(BaseModel):
    """Overall blog plan containing title, feature image prompt, and sections.

    Attributes:
        blog_title: Title of the blog post.
        feature_image_prompt: Prompt for the feature image (can be empty).
        sections: Ordered list of sections.
        research_required: Whether the plan requires web research (global flag).
    """

    blog_title: str = Field(..., description="Title of the blog post")
    feature_image_prompt: str = Field(..., description="Prompt for the feature image")
    sections: List[Section] = Field(..., description="Ordered list of sections")
    research_required: bool = Field(..., description="Whether web research is needed")


class ResearchResult(BaseModel):
    """Result of researching a single section.

    Attributes:
        section_id: ID of the researched section.
        query: The search query used.
        summary: Synthesized research summary.
        source_urls: List of URLs used in research.
    """

    section_id: str = Field(..., description="ID of the researched section")
    query: str = Field(..., description="The search query used")
    summary: str = Field(..., description="Synthesized research summary")
    source_urls: List[str] = Field(
        default_factory=list,
        description="URLs used in research"
    )


class SectionDraft(BaseModel):
    """Draft content for a single section, including citations.

    Attributes:
        section_id: ID of the section.
        title: Section title (repeated for convenience).
        content: Markdown content of the draft.
        citation_keys: List of citation markers found (e.g., ["[SOURCE_1]"]).
    """

    section_id: str = Field(..., description="ID of the section")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Markdown content of the draft")
    citation_keys: List[str] = Field(
        default_factory=list,
        description="List of citation markers found"
    )


class GeneratedImage(BaseModel):
    """Metadata for a generated image.

    Attributes:
        section_id: Section ID this image belongs to, or "feature" for feature image.
        image_path: Filesystem path to the saved image.
        prompt: Prompt used to generate the image.
    """

    section_id: str = Field(..., description="Section ID or 'feature'")
    image_path: str = Field(..., description="Filesystem path to the saved image")
    prompt: str = Field(..., description="Prompt used to generate the image")


class GraphState(TypedDict):
    """Complete state of the LangGraph pipeline.

    The state flows through all nodes, with list fields using reducer functions
    to safely accumulate results from parallel workers. This design follows
    LangGraph best practices for concurrent updates [citation:10].

    Attributes:
        topic: The original user topic.
        research_required: Global flag from router.
        blog_plan: The generated plan (None before planner).
        research_results: Accumulator for parallel researcher outputs.
        section_drafts: Accumulator for parallel writer outputs.
        generated_images: Accumulator for parallel image generation outputs.
        citation_registry: Mapping from citation keys (e.g., "[SOURCE_1]") to URLs.
        final_blog_md: Final Markdown content.
        final_blog_html: Final HTML content.
        run_id: Unique identifier for this run (used for file naming).
        error: Error message if any node fails (allows graceful error handling).
    """

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