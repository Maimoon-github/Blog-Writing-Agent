"""
Pydantic schemas: Blog post, section, and metadata DTOs.
"""

from pydantic import BaseModel, Field
from typing import Optional


class SectionSchema(BaseModel):
    heading: str
    content: str = ""
    word_count: int = 0
    sources: list[str] = []


class BlogPostSchema(BaseModel):
    title: str
    sections: list[SectionSchema] = []
    total_word_count: int = 0
    reading_time_minutes: int = 0
    cover_image_url: Optional[str] = None
    section_images: dict[str, str] = {}


class BlogMetadata(BaseModel):
    title_tag: str = ""
    meta_description: str = ""
    keywords: list[str] = []
    readability_score: float = 0.0
    style_guide: str = "AP"
    export_format: str = "markdown"
