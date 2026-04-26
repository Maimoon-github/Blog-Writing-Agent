"""
Pydantic schemas: Blog post, section, and metadata DTOs for API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict


class SectionSchema(BaseModel):
    """Single section of a blog post."""

    heading: str
    content: str = ""
    word_count: int = Field(default=0, ge=0)
    sources: list[str] = Field(default_factory=list)


class BlogPostSchema(BaseModel):
    """Complete blog post for API responses."""

    title: str
    sections: list[SectionSchema] = Field(default_factory=list)
    total_word_count: int = Field(default=0, ge=0)
    reading_time_minutes: int = Field(default=0, ge=0)
    cover_image_url: Optional[str] = None
    section_images: Dict[str, str] = Field(default_factory=dict)


class BlogMetadata(BaseModel):
    """SEO and export metadata."""

    title_tag: str = ""
    meta_description: str = ""
    keywords: list[str] = Field(default_factory=list)
    readability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    style_guide: str = "AP"
    export_format: str = "markdown"