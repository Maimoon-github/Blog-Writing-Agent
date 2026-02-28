"""Unit tests for the researcher node (graph.researcher.researcher_node)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

import pytest

from graph.researcher import researcher_node
from graph.state import ResearchResult

# -----------------------------------------------------------------------------
# Fixture loading
# -----------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Load sample research result (expected LLM output)
try:
    with open(FIXTURE_DIR / "sample_research_result.json") as f:
        SAMPLE_RESEARCH = json.load(f)
except FileNotFoundError:
    # Fallback minimal data so tests can run without the actual file
    SAMPLE_RESEARCH = {
        "summary": "AI is transforming industries with breakthroughs in NLP and computer vision.",
        "source_urls": ["https://example.com/ai-trends"],
        "sufficient": True,
    }

# Load sample search results (simulate search engine output)
try:
    with open(FIXTURE_DIR / "sample_search_results.json") as f:
        SAMPLE_SEARCH_RESULTS_DATA = json.load(f)
except FileNotFoundError:
    SAMPLE_SEARCH_RESULTS_DATA = [
        {
            "url": "https://example.com/ai-1",
            "title": "AI in 2025",
            "snippet": "Latest AI trends include generative models...",
        },
        {
            "url": "https://example.com/ai-2",
            "title": "The Future of AI",
            "snippet": "Experts predict breakthroughs in reasoning...",
        },
        {
            "url": "https://example.com/ai-3",
            "title": "AI Impact on Industry",
            "snippet": "How AI is changing business landscapes.",
        },
    ]


def create_mock_search_results(data: list) -> list:
    """Convert list of dicts into list of MagicMock objects with url/title/snippet attrs."""
    results = []
    for item in data:
        mock_res = MagicMock()
        mock_res.url = item["url"]
        mock_res.title = item["title"]
        mock_res.snippet = item.get("snippet", "")
        results.append(mock_res)
    return results


# -----------------------------------------------------------------------------
# Base state fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def base_state() -> Dict[str, Any]:
    """Return a minimal GraphState dictionary for testing the researcher."""
    return {
        "topic": "test topic",
        "research_required": True,
        "blog_plan": None,
        "research_results": [],
        "section_drafts": [],
        "generated_images": [],
        "citation_registry": {},
        "final_blog_md": "",
        "final_blog_html": "",
        "run_id": "test",
        "error": None,
        "section_id": "section_1",
        "search_query": "latest AI trends 2025",
        "section_description": "Introduction to AI trends",
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_hit_skips_search(
    mocker: pytest.MockerFixture,
    base_state: Dict[str, Any],
) -> None:
    """
    Test that when cache returns a result, search is not called.
    """
    # Patch cache.get to return cached data
    mock_cache_get = mocker.patch("graph.researcher.cache.get")
    mock_cache_get.return_value = {
        "summary": SAMPLE_RESEARCH["summary"],
        "source_urls": SAMPLE_RESEARCH["source_urls"],
    }

    # Patch search to verify it is NOT called
    mock_search = mocker.patch("graph.researcher.search")

    # Run the node
    result = await researcher_node(base_state)

    # Assert search not called
    mock_search.assert_not_called()

    # Verify result structure
    assert "research_results" in result
    assert len(result["research_results"]) == 1
    res = result["research_results"][0]
    # If using Pydantic model, we can check attributes; otherwise check dict
    if isinstance(res, ResearchResult):
        assert res.section_id == base_state["section_id"]
        assert res.summary == SAMPLE_RESEARCH["summary"]
        assert res.source_urls == SAMPLE_RESEARCH["source_urls"]
    else:
        # Fallback for if ResearchResult is not used (unlikely)
        assert res["section_id"] == base_state["section_id"]
        assert res["summary"] == SAMPLE_RESEARCH["summary"]
        assert res["source_urls"] == SAMPLE_RESEARCH["source_urls"]


@pytest.mark.asyncio
async def test_successful_research(
    mocker: pytest.MockerFixture,
    base_state: Dict[str, Any],
) -> None:
    """
    Test full research pipeline: cache miss → search → fetch → LLM → store.
    """
    # 1. Patch cache.get → miss
    mock_cache_get = mocker.patch("graph.researcher.cache.get")
    mock_cache_get.return_value = None

    # 2. Patch search → return list of mock objects
    mock_search = mocker.patch("graph.researcher.search")
    mock_search_results = create_mock_search_results(SAMPLE_SEARCH_RESULTS_DATA)
    mock_search.return_value = mock_search_results

    # 3. Patch fetch_page_content → return dummy text
    mock_fetch = mocker.patch("graph.researcher.fetch_page_content")
    mock_fetch.return_value = "Sample extracted text from the webpage."

    # 4. Patch ChatOllama.ainvoke to return a mock with JSON content
    mock_llm_instance = mocker.MagicMock()
    mock_llm_instance.ainvoke = AsyncMock(
        return_value=MagicMock(content=json.dumps(SAMPLE_RESEARCH))
    )
    mock_chatollama_class = mocker.patch("graph.researcher.ChatOllama")
    mock_chatollama_class.return_value = mock_llm_instance

    # 5. Patch cache.set and chroma_store.add_research (they are called via to_thread)
    mock_cache_set = mocker.patch("graph.researcher.cache.set")
    mock_chroma_add = mocker.patch("graph.researcher.chroma_store.add_research")

    # Run the node
    result = await researcher_node(base_state)

    # Assert search was called with expected query
    mock_search.assert_called_once_with(
        query=base_state["search_query"], num_results=5  # MAX_SEARCH_RESULTS constant
    )

    # Assert fetch was called for each URL (top 3 results)
    assert mock_fetch.call_count == len(mock_search_results[:3])

    # Assert LLM was invoked
    mock_llm_instance.ainvoke.assert_awaited_once()

    # Assert storage methods were called
    mock_cache_set.assert_called_once()
    mock_chroma_add.assert_called_once()

    # Verify result
    assert len(result["research_results"]) == 1
    res = result["research_results"][0]
    if isinstance(res, ResearchResult):
        assert res.section_id == base_state["section_id"]
        assert res.summary == SAMPLE_RESEARCH["summary"]
        assert res.source_urls == SAMPLE_RESEARCH["source_urls"]
    else:
        assert res["section_id"] == base_state["section_id"]
        assert res["summary"] == SAMPLE_RESEARCH["summary"]
        assert res["source_urls"] == SAMPLE_RESEARCH["source_urls"]


@pytest.mark.asyncio
async def test_empty_search_query_returns_empty(
    mocker: pytest.MockerFixture,
    base_state: Dict[str, Any],
) -> None:
    """
    Test that when search_query is empty or None, the node returns early with empty list.
    """
    # Patch all external calls to ensure they are NOT called
    mock_cache_get = mocker.patch("graph.researcher.cache.get")
    mock_search = mocker.patch("graph.researcher.search")
    mock_fetch = mocker.patch("graph.researcher.fetch_page_content")
    mock_chatollama = mocker.patch("graph.researcher.ChatOllama")
    mock_cache_set = mocker.patch("graph.researcher.cache.set")
    mock_chroma_add = mocker.patch("graph.researcher.chroma_store.add_research")

    # Case 1: empty string
    base_state["search_query"] = ""
    result = await researcher_node(base_state)
    assert result == {"research_results": []}
    mock_cache_get.assert_not_called()
    mock_search.assert_not_called()
    mock_fetch.assert_not_called()
    mock_chatollama.assert_not_called()
    mock_cache_set.assert_not_called()
    mock_chroma_add.assert_not_called()

    # Case 2: None
    base_state["search_query"] = None
    result = await researcher_node(base_state)
    assert result == {"research_results": []}
    # All mocks remain uncalled


@pytest.mark.asyncio
async def test_search_query_missing_handling(
    mocker: pytest.MockerFixture,
    base_state: Dict[str, Any],
) -> None:
    """
    Test that if search_query is missing (key not present), node handles gracefully.
    (Though state should always have it, but defensive.)
    """
    # Remove search_query from state
    del base_state["search_query"]

    # Patch external calls
    mock_cache_get = mocker.patch("graph.researcher.cache.get")
    mock_search = mocker.patch("graph.researcher.search")

    result = await researcher_node(base_state)

    # Should treat missing as empty and return early
    assert result == {"research_results": []}
    mock_cache_get.assert_not_called()
    mock_search.assert_not_called()


@pytest.mark.asyncio
async def test_search_failure_continues(
    mocker: pytest.MockerFixture,
    base_state: Dict[str, Any],
) -> None:
    """
    Test that if search raises an exception, the node logs error but continues
    (using empty search results) and still tries LLM.
    """
    mock_cache_get = mocker.patch("graph.researcher.cache.get")
    mock_cache_get.return_value = None

    # Make search raise an exception
    mock_search = mocker.patch("graph.researcher.search")
    mock_search.side_effect = Exception("Search API unavailable")

    # Patch fetch (won't be called because search fails)
    mock_fetch = mocker.patch("graph.researcher.fetch_page_content")

    # Patch LLM
    mock_llm_instance = mocker.MagicMock()
    mock_llm_instance.ainvoke = AsyncMock(
        return_value=MagicMock(content=json.dumps(SAMPLE_RESEARCH))
    )
    mock_chatollama_class = mocker.patch("graph.researcher.ChatOllama")
    mock_chatollama_class.return_value = mock_llm_instance

    # Patch storage
    mock_cache_set = mocker.patch("graph.researcher.cache.set")
    mock_chroma_add = mocker.patch("graph.researcher.chroma_store.add_research")

    result = await researcher_node(base_state)

    # Search was called, but fetch should not be called (since search returned empty list)
    mock_search.assert_called_once()
    mock_fetch.assert_not_called()  # No results to fetch

    # LLM should still be invoked with empty content_items
    mock_llm_instance.ainvoke.assert_awaited_once()

    # Verify we still get a result (summary from LLM)
    assert len(result["research_results"]) == 1
    res = result["research_results"][0]
    if isinstance(res, ResearchResult):
        assert res.summary == SAMPLE_RESEARCH["summary"]
    else:
        assert res["summary"] == SAMPLE_RESEARCH["summary"]

    # Storage should still be called
    mock_cache_set.assert_called_once()
    mock_chroma_add.assert_called_once()