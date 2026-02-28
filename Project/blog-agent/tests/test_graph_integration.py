"""End‑to‑end integration tests for the full LangGraph blog generation pipeline.

These tests execute the compiled graph on three representative topics and verify
that the final blog post meets basic expectations. They require Ollama to be
running locally and are marked as `slow` to allow exclusion during quick test runs.

Both synchronous and asynchronous invocation patterns are supported; the async
version is used here for compatibility with async nodes in the graph.
"""

import uuid
import httpx
import pytest
from typing import Dict, Any

from graph.graph_builder import compiled_graph

# -----------------------------------------------------------------------------
# Test data
# -----------------------------------------------------------------------------

TEST_TOPICS = [
    "Latest developments in large language models 2025",   # tech, research likely needed
    "How photosynthesis works",                             # evergreen, maybe less research
    "Quantum computing applications in cryptography"       # research‑heavy
]


# -----------------------------------------------------------------------------
# Fixture: check Ollama availability
# -----------------------------------------------------------------------------

@pytest.fixture(scope="function")
def check_ollama_available() -> None:
    """Skip the test if Ollama is not reachable.

    Attempts to connect to the Ollama API endpoint `/api/tags` with a short
    timeout. If the request fails (exception or non‑200 status), the test is
    skipped with an appropriate message.
    """
    try:
        # Adjust the URL if your Ollama runs on a different host/port
        response = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        response.raise_for_status()
    except Exception as e:
        pytest.skip(f"Ollama not available (error: {e})")


# -----------------------------------------------------------------------------
# Helper: build initial state
# -----------------------------------------------------------------------------

def build_initial_state(topic: str) -> Dict[str, Any]:
    """Create a minimal initial GraphState for the given topic.

    Args:
        topic: The blog topic to start the graph with.

    Returns:
        A dictionary conforming to GraphState with all required fields
        initialized to empty/default values.
    """
    return {
        "topic": topic,
        "run_id": str(uuid.uuid4())[:8],           # short unique identifier
        "research_required": False,                 # will be set by router
        "blog_plan": None,
        "research_results": [],
        "section_drafts": [],
        "generated_images": [],
        "citation_registry": {},
        "final_blog_md": "",
        "final_blog_html": "",
        "error": None,
    }


# -----------------------------------------------------------------------------
# Integration tests (one per topic)
# -----------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_tech_topic(check_ollama_available: None) -> None:
    """Full graph execution on a technology topic.

    Expects:
        - Final Markdown non‑empty
        - At least 4 section drafts
        - At least 1 generated image
        - No error in final state
        - Citation registry non‑empty (research should have been performed)
    """
    topic = TEST_TOPICS[0]
    initial_state = build_initial_state(topic)
    config = {"configurable": {"thread_id": initial_state["run_id"]}}

    # Execute the graph asynchronously (allows parallel worker nodes)
    final_state = await compiled_graph.ainvoke(initial_state, config)

    # Assertions
    assert final_state["final_blog_md"] != "", "Final blog Markdown should not be empty"
    assert len(final_state["section_drafts"]) >= 4, "Expected at least 4 section drafts"
    assert len(final_state["generated_images"]) >= 1, "Expected at least one generated image"
    assert final_state.get("error") is None, f"Graph finished with error: {final_state.get('error')}"
    assert len(final_state["citation_registry"]) > 0, "Expected citations for research‑heavy topic"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_evergreen_topic(check_ollama_available: None) -> None:
    """Full graph execution on an evergreen topic (may not require research).

    Expects:
        - Final Markdown non‑empty
        - At least 4 section drafts
        - At least 1 generated image
        - No error in final state
    """
    topic = TEST_TOPICS[1]
    initial_state = build_initial_state(topic)
    config = {"configurable": {"thread_id": initial_state["run_id"]}}

    final_state = await compiled_graph.ainvoke(initial_state, config)

    assert final_state["final_blog_md"] != "", "Final blog Markdown should not be empty"
    assert len(final_state["section_drafts"]) >= 4, "Expected at least 4 section drafts"
    assert len(final_state["generated_images"]) >= 1, "Expected at least one generated image"
    assert final_state.get("error") is None, f"Graph finished with error: {final_state.get('error')}"
    # No assertion on citations – they may be empty for this topic


@pytest.mark.slow
@pytest.mark.asyncio
async def test_integration_research_topic(check_ollama_available: None) -> None:
    """Full graph execution on a research‑heavy topic.

    Expects:
        - Final Markdown non‑empty
        - At least 4 section drafts
        - At least 1 generated image
        - No error in final state
        - Citation registry non‑empty
    """
    topic = TEST_TOPICS[2]
    initial_state = build_initial_state(topic)
    config = {"configurable": {"thread_id": initial_state["run_id"]}}

    final_state = await compiled_graph.ainvoke(initial_state, config)

    assert final_state["final_blog_md"] != "", "Final blog Markdown should not be empty"
    assert len(final_state["section_drafts"]) >= 4, "Expected at least 4 section drafts"
    assert len(final_state["generated_images"]) >= 1, "Expected at least one generated image"
    assert final_state.get("error") is None, f"Graph finished with error: {final_state.get('error')}"
    assert len(final_state["citation_registry"]) > 0, "Expected citations for research‑heavy topic"


# -----------------------------------------------------------------------------
# Notes on asynchronous execution
# -----------------------------------------------------------------------------
# The graph contains async nodes (researcher_node, writer_node_async, etc.),
# so we use `ainvoke` and mark tests with `@pytest.mark.asyncio`.
# If the graph were fully synchronous, the tests could use `invoke` without
# the asyncio marker.