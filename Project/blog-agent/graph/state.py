import operator
from typing import TypedDict, List, Dict, Optional, Annotated

from pydantic import BaseModel


class Section(BaseModel):
    id: str
    title: str
    description: str
    word_count: int
    search_query: Optional[str] = None
    image_prompt: str


class BlogPlan(BaseModel):
    blog_title: str
    feature_image_prompt: str
    sections: List[Section]
    research_required: bool


class ResearchResult(BaseModel):
    section_id: str
    query: str
    summary: str
    source_urls: List[str]


class SectionDraft(BaseModel):
    section_id: str
    title: str
    content: str
    citation_keys: List[str]


class GeneratedImage(BaseModel):
    section_id: str
    image_path: str
    prompt: str


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