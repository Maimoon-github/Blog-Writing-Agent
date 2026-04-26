"""
Pytest fixtures for the blog agent system test suite.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from blog_agent_system.core.state import BlogState, Source, Section


@pytest.fixture
def sample_state() -> dict:
    """A minimal valid state for testing agents."""
    return {
        "topic": "Introduction to Quantum Computing",
        "target_audience": "software engineers",
        "tone": "technical but accessible",
        "word_count_target": 1500,
        "include_images": False,
        "style_guide": "AP",
        "status": "pending",
        "messages": [],
        "research_findings": [],
        "outline": [],
        "draft_sections": [],
        "draft": "",
        "edited_draft": "",
        "seo_metadata": None,
        "fact_check_results": [],
        "quality_score": 0.0,
        "revision_count": 0,
        "max_revisions": 3,
        "revision_feedback": "",
        "final_blog": "",
        "export_format": "markdown",
        "cover_image_url": None,
        "section_images": {},
        "current_step": "init",
    }


@pytest.fixture
def sample_blog_state(sample_state) -> BlogState:
    """A BlogState instance for testing."""
    return BlogState(**sample_state)


@pytest.fixture
def mock_llm_provider():
    """A mock LLM provider that returns predictable responses."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value="Mock LLM response content.")
    provider.count_tokens = MagicMock(return_value=10)
    return provider


@pytest.fixture
def sample_sources() -> list[Source]:
    """Sample research sources for testing."""
    return [
        Source(url="https://example.com/1", title="Source 1", snippet="First source content", credibility_score=0.9),
        Source(url="https://example.com/2", title="Source 2", snippet="Second source content", credibility_score=0.8),
    ]


@pytest.fixture
def sample_outline() -> list[Section]:
    """Sample outline sections for testing."""
    return [
        Section(heading="Introduction", key_points=["Overview"], word_count=300),
        Section(heading="Main Topic", key_points=["Detail 1", "Detail 2"], word_count=900),
        Section(heading="Conclusion", key_points=["Summary"], word_count=300),
    ]
