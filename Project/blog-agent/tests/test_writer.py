"""Unit tests for the writer node (graph.writer.writer_node)."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from typing import Dict, Any, List

import pytest

from graph.writer import writer_node
from graph.state import SectionDraft, ResearchResult

# -----------------------------------------------------------------------------
# Fixture loading
# -----------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Load sample section draft (expected LLM output)
try:
    with open(FIXTURE_DIR / "sample_section_draft.json") as f:
        SAMPLE_DRAFT = json.load(f)
except FileNotFoundError:
    # Fallback minimal data so tests can run without the actual file
    SAMPLE_DRAFT = {
        "content": (
            "Artificial intelligence is transforming the world. "
            "[SOURCE_1] According to a recent study, AI adoption has doubled. "
            "[SOURCE_2] Another source confirms this trend. "
            "[IMAGE_PLACEHOLDER_section_1] This visual will illustrate the impact."
        )
    }

# Load sample research result (from researcher output)
try:
    with open(FIXTURE_DIR / "sample_research_result.json") as f:
        SAMPLE_RESEARCH_DATA = json.load(f)
except FileNotFoundError:
    SAMPLE_RESEARCH_DATA = {
        "section_id": "section_1",
        "query": "latest AI trends 2025",
        "summary": "AI is advancing rapidly with breakthroughs in generative models and reasoning.",
        "source_urls": ["https://example.com/ai-trends"],
        "sufficient": True,
    }

# Build a ResearchResult instance from the fixture data
research_result = ResearchResult(**SAMPLE_RESEARCH_DATA)

# The content we expect the LLM to return (must contain citations and placeholder)
MOCK_CONTENT = SAMPLE_DRAFT["content"]


# -----------------------------------------------------------------------------
# Base state fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def base_state() -> Dict[str, Any]:
    """Return a minimal GraphState dictionary for testing the writer."""
    return {
        "topic": "test",
        "research_required": True,
        "blog_plan": None,
        "research_results": [research_result],
        "section_drafts": [],
        "generated_images": [],
        "citation_registry": {},
        "final_blog_md": "",
        "final_blog_html": "",
        "run_id": "test",
        "error": None,
        "section_id": "section_1",
        "section_title": "Introduction to AI Trends",
        "section_description": "Overview of current AI trends.",
        "word_count": 400,
        "image_prompt": "Futuristic AI visualization",
    }


# -----------------------------------------------------------------------------
# Mocking helpers
# -----------------------------------------------------------------------------

def configure_mock_llm(mocker, return_content: str = MOCK_CONTENT):
    """
    Configure the mock ChatOllama to return the given content.

    Also patches the synchronous prompt loader to avoid filesystem access.
    """
    # Patch the synchronous prompt loader to return a dummy prompt
    mocker.patch("graph.writer._load_prompt_sync", return_value="You are a blog writer.")

    # Create mock LLM instance
    mock_llm_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.content = return_content
    mock_llm_instance.invoke.return_value = mock_response

    # Patch the ChatOllama class to return our mock instance
    mock_chatollama_class = mocker.patch("graph.writer.ChatOllama")
    mock_chatollama_class.return_value = mock_llm_instance

    return mock_llm_instance


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_writer_returns_draft(mocker, base_state):
    """
    Test that writer_node returns a SectionDraft with correct section_id.
    """
    configure_mock_llm(mocker)

    result = writer_node(base_state)

    assert "section_drafts" in result
    drafts = result["section_drafts"]
    assert len(drafts) == 1
    draft = drafts[0]
    assert isinstance(draft, SectionDraft)
    assert draft.section_id == base_state["section_id"]


def test_citation_keys_extracted(mocker, base_state):
    """
    Test that writer extracts citation keys from the generated content.
    """
    configure_mock_llm(mocker)

    result = writer_node(base_state)
    draft = result["section_drafts"][0]

    assert "[SOURCE_1]" in draft.citation_keys
    assert "[SOURCE_2]" in draft.citation_keys
    # Optionally check exact list (order may vary but extraction preserves order)
    # We'll just check membership


def test_image_placeholder_in_content(mocker, base_state):
    """
    Test that writer preserves the image placeholder in the draft content.
    """
    configure_mock_llm(mocker)

    result = writer_node(base_state)
    draft = result["section_drafts"][0]

    assert "[IMAGE_PLACEHOLDER_section_1]" in draft.content


def test_no_research_uses_knowledge_fallback(mocker, base_state):
    """
    Test that writer still produces a draft when no research is available.
    """
    # Remove research results from state
    base_state["research_results"] = []

    configure_mock_llm(mocker)

    result = writer_node(base_state)

    assert len(result["section_drafts"]) == 1
    draft = result["section_drafts"][0]
    assert draft.section_id == base_state["section_id"]
    # Content may be the same mock content (we don't care exactly what)
    assert draft.content == MOCK_CONTENT


def test_missing_required_fields(mocker, base_state):
    """
    Test that writer returns an error when required fields are missing.
    """
    configure_mock_llm(mocker)

    # Remove section_title
    del base_state["section_title"]

    result = writer_node(base_state)

    assert "error" in result
    assert "Missing required fields" in result["error"]
    assert result["section_drafts"] == []


def test_llm_invocation_failure_returns_placeholder(mocker, base_state):
    """
    Test that when LLM invocation raises an exception, the writer returns a placeholder draft.
    """
    # Patch prompt loader
    mocker.patch("graph.writer._load_prompt_sync", return_value="dummy")
    mock_llm_instance = mocker.MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("LLM service unavailable")
    mock_chatollama_class = mocker.patch("graph.writer.ChatOllama")
    mock_chatollama_class.return_value = mock_llm_instance

    result = writer_node(base_state)

    assert len(result["section_drafts"]) == 1
    draft = result["section_drafts"][0]
    assert draft.content == "*Content generation failed. Please check logs.*"
    assert draft.citation_keys == []


# -----------------------------------------------------------------------------
# Async adaptation note
# -----------------------------------------------------------------------------
# If testing the asynchronous version (writer_node_async), you would:
# - Import writer_node_async from graph.writer
# - Mark tests with @pytest.mark.asyncio
# - Use await writer_node_async(base_state)
# - In configure_mock_llm, patch _load_prompt_async instead of _load_prompt_sync
# - Set up mock_llm_instance.ainvoke instead of .invoke
# Example:
# @pytest.mark.asyncio
# async def test_writer_async_returns_draft(mocker, base_state):
#     mocker.patch("graph.writer._load_prompt_async", return_value="dummy")
#     mock_llm_instance = mocker.MagicMock()
#     mock_llm_instance.ainvoke = mocker.AsyncMock(return_value=MagicMock(content=MOCK_CONTENT))
#     mocker.patch("graph.writer.ChatOllama", return_value=mock_llm_instance)
#     result = await writer_node_async(base_state)
#     assert "section_drafts" in result