"""
test_bwa_backend.py – Integration and unit tests for the BWA blog writing agent.
Mocks all external services (Ollama, DuckDuckGo, Hugging Face) to ensure
reliability and speed.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.types import Send
from pydantic import ValidationError

# Import the backend module (adjust path as needed)
from update_bwa_backend_1 import (
    State,
    router_node,
    research_node,
    orchestrator_node,
    fanout,
    worker_node,
    merge_content,
    decide_images,
    generate_images,
    place_images,
    app,
    RouterDecision,
    EvidencePack,
    Plan,
    Task,
    GlobalImagePlan,
    ImageSpec,
    _extract_thinking,
    _hf_generate_image_bytes,
)


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Mock the synchronous ChatOllama instance (llm)."""
    with patch("update_bwa_backend_1.llm") as mock:
        mock.with_structured_output = MagicMock()
        yield mock


@pytest.fixture
def mock_llm_async():
    """Mock the asynchronous ChatOllama instance (llm_async)."""
    with patch("update_bwa_backend_1.llm_async") as mock:
        mock.ainvoke = AsyncMock()
        yield mock


@pytest.fixture
def mock_research_engine():
    """Mock the Web Research Engine (ResearchEngine) if available."""
    with patch("update_bwa_backend_1.ResearchEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        yield mock_engine


@pytest.fixture
def mock_hf_client():
    """Mock the Hugging Face InferenceClient."""
    with patch("update_bwa_backend_1.InferenceClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        # Mock the text_to_image method to return a fake PIL image
        mock_pil = MagicMock()
        mock_client.text_to_image.return_value = mock_pil
        yield mock_client


@pytest.fixture
def sample_state() -> State:
    """A minimal state for tests that require a full State object."""
    return {
        "topic": "Test Topic",
        "as_of": "2025-01-01",
        "router_decision": None,
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "mode": "closed_book",
        "recency_days": 3650,
        "user_research_mode": None,
        "thinking_traces": [],
    }


# ----------------------------------------------------------------------
# Unit tests for helper functions
# ----------------------------------------------------------------------

def test_extract_thinking():
    text = "Some content <think>reasoning 1</think> more <think>reasoning 2</think> end."
    clean, thinking = _extract_thinking(text)
    assert clean == "Some content  more  end."
    assert "reasoning 1" in thinking and "reasoning 2" in thinking

    text = "No tags here."
    clean, thinking = _extract_thinking(text)
    assert clean == text
    assert thinking == ""

    text = "<think>only thinking</think>"
    clean, thinking = _extract_thinking(text)
    assert clean == ""
    assert thinking == "only thinking"


def test_hf_generate_image_bytes_success(mock_hf_client):
    mock_hf_client.text_to_image.return_value.save = MagicMock()
    bytes_ = _hf_generate_image_bytes("a cat")
    assert isinstance(bytes_, bytes)


def test_hf_generate_image_bytes_failure(mock_hf_client):
    mock_hf_client.text_to_image.side_effect = Exception("API error")
    with pytest.raises(RuntimeError, match="Hugging Face image generation failed"):
        _hf_generate_image_bytes("a cat")


# ----------------------------------------------------------------------
# Router node tests
# ----------------------------------------------------------------------

def test_router_node_no_override(mock_llm):
    state: State = {"topic": "AI", "as_of": "2025-01-01"}
    mock_decision = RouterDecision(
        needs_research=True,
        mode="hybrid",
        reason="test",
        queries=["q1", "q2"],
    )
    # Mock the structured output callable
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_decision
    mock_llm.with_structured_output.return_value = mock_structured

    result = router_node(state)
    assert result["needs_research"] is True
    assert result["mode"] == "hybrid"
    assert result["queries"] == ["q1", "q2"]
    assert result["recency_days"] == 45  # hybrid => 45


def test_router_node_with_user_override(mock_llm):
    state: State = {
        "topic": "AI",
        "as_of": "2025-01-01",
        "user_research_mode": "closed_book",
    }
    mock_decision = RouterDecision(
        needs_research=True,
        mode="hybrid",
        reason="test",
        queries=["q1"],
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_decision
    mock_llm.with_structured_output.return_value = mock_structured

    result = router_node(state)
    # Override should force closed_book
    assert result["needs_research"] is False
    assert result["mode"] == "closed_book"
    assert result["recency_days"] == 3650  # default for closed_book


def test_router_node_llm_failure(mock_llm):
    state: State = {"topic": "AI", "as_of": "2025-01-01"}
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("LLM down")
    mock_llm.with_structured_output.return_value = mock_structured

    result = router_node(state)
    assert result["needs_research"] is False
    assert result["mode"] == "closed_book"
    # The reason is stored inside router_decision dict
    assert result["router_decision"]["reason"] == "LLM error; defaulting to closed-book"


# ----------------------------------------------------------------------
# Research node tests
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_research_node_no_queries():
    state: State = {"queries": []}
    result = await research_node(state)
    assert result == {"evidence": []}


@pytest.mark.asyncio
async def test_research_node_wre_unavailable():
    with patch("update_bwa_backend_1.WRE_AVAILABLE", False):
        state: State = {"queries": ["q1"], "topic": "test", "as_of": "2025-01-01", "recency_days": 45}
        result = await research_node(state)
        assert result == {"evidence": []}


# @pytest.mark.asyncio
# async def test_research_node_success(mock_research_engine, mock_llm):
#     # Mock the engine.research to return a report with one extracted content
#     mock_report = MagicMock()
#     mock_content = MagicMock()
#     mock_content.error = None
#     mock_content.word_count = 200
#     mock_content.content = "Sample content " * 100
#     mock_content.title = "Sample Title"
#     mock_content.url = "http://example.com"
#     mock_content.metadata = {"search_query": "q1"}
#     mock_report.extracted_content = [mock_content]
#     mock_research_engine.research.return_value = mock_report

#     # Mock the synthesis LLM to return an EvidencePack
#     mock_structured = MagicMock()
#     mock_structured.invoke.return_value = EvidencePack(
#         evidence=[{"title": "Synth", "url": "http://synth.com", "snippet": "snippet"}]
#     )
#     mock_llm.with_structured_output.return_value = mock_structured

#     state: State = {
#         "queries": ["q1"],
#         "topic": "test",
#         "as_of": "2025-01-01",
#         "recency_days": 45,
#     }
#     result = await research_node(state)

#     assert "evidence" in result
#     assert len(result["evidence"]) == 1
#     assert result["evidence"][0]["url"] == "http://synth.com"


@pytest.mark.asyncio
@patch("web_research_engine.OutputFormatter.to_json")
@patch("web_research_engine.OutputFormatter.to_markdown")
@patch("web_research_engine.OutputFormatter.to_html")
async def test_research_node_success(
    mock_to_html, mock_to_markdown, mock_to_json,
    mock_research_engine, mock_llm, tmp_path
):
    # Mock the engine.research to return a report with one extracted content
    mock_report = MagicMock()
    mock_content = MagicMock()
    mock_content.error = None
    mock_content.word_count = 200
    mock_content.content = "Sample content " * 100
    mock_content.title = "Sample Title"
    mock_content.url = "http://example.com"
    mock_content.metadata = {"search_query": "q1"}
    mock_report.extracted_content = [mock_content]
    mock_research_engine.research.return_value = mock_report

    # Mock the synthesis LLM to return an EvidencePack
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = EvidencePack(
        evidence=[{"title": "Synth", "url": "http://synth.com", "snippet": "snippet"}]
    )
    mock_llm.with_structured_output.return_value = mock_structured

    state: State = {
        "queries": ["q1"],
        "topic": "test",
        "as_of": "2025-01-01",
        "recency_days": 45,
    }

    # Redirect RESEARCH_OUTPUT_DIR to a temporary path
    with patch("update_bwa_backend_1.RESEARCH_OUTPUT_DIR", tmp_path):
        result = await research_node(state)

    assert "evidence" in result
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["url"] == "http://synth.com"
    mock_to_json.assert_called_once()  # now succeeds


@pytest.mark.asyncio
async def test_research_node_synthesis_fallback(mock_research_engine, mock_llm):
    # Mock engine returns raw results
    mock_report = MagicMock()
    mock_content = MagicMock()
    mock_content.error = None
    mock_content.word_count = 200
    mock_content.content = "Sample"
    mock_content.title = "Raw Title"
    mock_content.url = "http://raw.com"
    mock_content.metadata = {}
    mock_report.extracted_content = [mock_content]
    mock_research_engine.research.return_value = mock_report

    # Synthesis LLM fails
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("synthesis error")
    mock_llm.with_structured_output.return_value = mock_structured

    state: State = {
        "queries": ["q1"],
        "topic": "test",
        "as_of": "2025-01-01",
        "recency_days": 45,
    }
    result = await research_node(state)

    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["url"] == "http://raw.com"  # raw fallback


# ----------------------------------------------------------------------
# Orchestrator node tests
# ----------------------------------------------------------------------

def test_orchestrator_node_success(mock_llm):
    state: State = {
        "topic": "AI",
        "mode": "closed_book",
        "as_of": "2025-01-01",
        "recency_days": 3650,
        "evidence": [],
    }
    mock_plan = Plan(
        blog_title="AI Blog",
        audience="devs",
        tone="fun",
        blog_kind="explainer",
        tasks=[
            Task(id=1, title="Intro", goal="g",
                 bullets=["Define AI", "Why it matters", "What you'll learn"],
                 target_words=100)
        ],
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_plan
    mock_llm.with_structured_output.return_value = mock_structured

    result = orchestrator_node(state)
    assert "plan" in result
    plan = result["plan"]
    assert plan.blog_title == "AI Blog"
    assert len(plan.tasks) == 1


def test_orchestrator_node_fallback(mock_llm):
    state: State = {"topic": "AI", "mode": "closed_book"}
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("fail")
    mock_llm.with_structured_output.return_value = mock_structured

    result = orchestrator_node(state)
    plan = result["plan"]
    assert plan.blog_title.startswith("Blog about")
    assert len(plan.tasks) == 1


def test_orchestrator_node_forced_news_roundup(mock_llm):
    state: State = {
        "topic": "AI",
        "mode": "open_book",
        "as_of": "2025-01-01",
        "recency_days": 7,
        "evidence": [],
    }
    mock_plan = Plan(
        blog_title="AI Blog",
        audience="devs",
        tone="fun",
        blog_kind="explainer",  # will be overridden
        tasks=[],
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_plan
    mock_llm.with_structured_output.return_value = mock_structured

    result = orchestrator_node(state)
    assert result["plan"].blog_kind == "news_roundup"


# ----------------------------------------------------------------------
# Fanout test
# ----------------------------------------------------------------------

def test_fanout():
    plan = Plan(
        blog_title="Test",
        audience="",
        tone="",
        blog_kind="explainer",
        tasks=[
            Task(id=1, title="T1", goal="g",
                 bullets=["Point A", "Point B", "Point C"],
                 target_words=100),
            Task(id=2, title="T2", goal="g",
                 bullets=["Item 1", "Item 2", "Item 3"],
                 target_words=100),
        ],
    )
    state: State = {
        "plan": plan,
        "topic": "topic",
        "mode": "closed_book",
        "as_of": "date",
        "recency_days": 3650,
        "evidence": [{"url": "e1"}],
        # other fields not needed
    }
    sends = fanout(state)
    assert len(sends) == 2
    assert all(isinstance(s, Send) for s in sends)
    assert sends[0].node == "worker"
    assert sends[0].arg["task"]["id"] == 1


# ----------------------------------------------------------------------
# Worker node tests
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_node_success(mock_llm_async):
    payload = {
        "task": Task(id=1, title="Intro", goal="g",
                     bullets=["First", "Second", "Third"],
                     target_words=100).model_dump(),
        "plan": Plan(blog_title="B", audience="a", tone="t",
                     blog_kind="explainer", tasks=[]).model_dump(),
        "topic": "t",
        "mode": "m",
        "as_of": "d",
        "recency_days": 3650,
        "evidence": [],
    }
    mock_response = MagicMock()
    mock_response.content = "## Intro\n\nSome content."
    mock_llm_async.ainvoke.return_value = mock_response

    result = await worker_node(payload)
    assert "sections" in result
    assert len(result["sections"]) == 1
    assert result["sections"][0][0] == 1
    assert result["sections"][0][1] == "## Intro\n\nSome content."
    assert result["thinking_traces"] == []


@pytest.mark.asyncio
async def test_worker_node_with_thinking(mock_llm_async):
    payload = {
        "task": Task(id=2, title="Deep", goal="g",
                     bullets=["Insight 1", "Insight 2", "Insight 3"],
                     target_words=100).model_dump(),
        "plan": Plan(blog_title="B", audience="a", tone="t",
                     blog_kind="explainer", tasks=[]).model_dump(),
        "topic": "t",
        "mode": "m",
        "as_of": "d",
        "recency_days": 3650,
        "evidence": [],
    }
    mock_response = MagicMock()
    mock_response.content = "<think>reasoning</think>## Deep\n\nContent."
    mock_llm_async.ainvoke.return_value = mock_response

    result = await worker_node(payload)
    sections = result["sections"]
    assert sections[0][1] == "## Deep\n\nContent."  # clean
    traces = result["thinking_traces"]
    assert len(traces) == 1
    assert traces[0]["task_id"] == 2
    assert traces[0]["thinking"] == "reasoning"


@pytest.mark.asyncio
async def test_worker_node_failure(mock_llm_async):
    payload = {
        "task": Task(id=3, title="Fail", goal="g",
                     bullets=["One", "Two", "Three"],
                     target_words=100).model_dump(),
        "plan": Plan(blog_title="B", audience="a", tone="t",
                     blog_kind="explainer", tasks=[]).model_dump(),
        "topic": "t",
        "mode": "m",
        "as_of": "d",
        "recency_days": 3650,
        "evidence": [],
    }
    mock_llm_async.ainvoke.side_effect = Exception("LLM error")

    result = await worker_node(payload)
    sections = result["sections"]
    assert sections[0][1].startswith("## Fail")
    assert "failed" in sections[0][1]


# ----------------------------------------------------------------------
# Reducer subgraph node tests
# ----------------------------------------------------------------------

def test_merge_content(sample_state):
    sample_state["plan"] = Plan(blog_title="Test Blog", audience="", tone="",
                                 blog_kind="explainer", tasks=[])
    sample_state["sections"] = [(1, "## First\ncontent"), (2, "## Second\nmore")]
    result = merge_content(sample_state)
    merged = result["merged_md"]
    assert merged.startswith("# Test Blog")
    assert "## First" in merged
    assert "## Second" in merged


def test_decide_images_success(mock_llm, sample_state):
    sample_state["plan"] = Plan(blog_title="Test", audience="", tone="",
                                 blog_kind="explainer", tasks=[])
    sample_state["merged_md"] = "# Test\n\nSome text."

    mock_image_plan = GlobalImagePlan(
        md_with_placeholders="# Test\n\n[[IMAGE_1]]",
        images=[
            ImageSpec(placeholder="[[IMAGE_1]]", filename="img.png",
                      alt="alt", caption="cap", prompt="prompt")
        ],
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_image_plan
    mock_llm.with_structured_output.return_value = mock_structured

    result = decide_images(sample_state)
    assert result["md_with_placeholders"] == "# Test\n\n[[IMAGE_1]]"
    assert len(result["image_specs"]) == 1


def test_decide_images_fallback(mock_llm, sample_state):
    sample_state["plan"] = Plan(blog_title="Test", audience="", tone="",
                                 blog_kind="explainer", tasks=[])
    sample_state["merged_md"] = "# Test"

    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("planning fail")
    mock_llm.with_structured_output.return_value = mock_structured

    result = decide_images(sample_state)
    assert result["md_with_placeholders"] == "# Test"
    assert result["image_specs"] == []


def test_generate_images_new(mock_hf_client, tmp_path):
    # Override images_dir to tmp_path for safety
    with patch("update_bwa_backend_1.Path", return_value=tmp_path):
        image_specs = [
            {"placeholder": "[[IMG1]]", "filename": "test.png",
             "alt": "a", "caption": "c", "prompt": "p"}
        ]
        state: State = {"image_specs": image_specs}
        result = generate_images(state)
        updated = result["image_specs"]
        assert len(updated) == 1
        assert "error" not in updated[0]
        assert (tmp_path / "test.png").exists()


def test_generate_images_with_error(mock_hf_client, tmp_path):
    mock_hf_client.text_to_image.side_effect = Exception("HF down")
    with patch("update_bwa_backend_1.Path", return_value=tmp_path):
        image_specs = [
            {"placeholder": "[[IMG1]]", "filename": "test.png",
             "alt": "a", "caption": "c", "prompt": "p"}
        ]
        state: State = {"image_specs": image_specs}
        result = generate_images(state)
        updated = result["image_specs"]
        assert len(updated) == 1
        assert updated[0]["error"] == "Hugging Face image generation failed: HF down"


def test_generate_images_skip_existing(tmp_path):
    (tmp_path / "existing.png").write_text("dummy")
    image_specs = [
        {"placeholder": "[[IMG1]]", "filename": "existing.png",
         "alt": "a", "caption": "c", "prompt": "p"}
    ]
    with patch("update_bwa_backend_1.Path", return_value=tmp_path):
        state: State = {"image_specs": image_specs}
        result = generate_images(state)
        updated = result["image_specs"]
        assert len(updated) == 1
        assert "error" not in updated[0]  # no generation attempted


# def test_place_images_without_errors(sample_state):
#     sample_state["plan"] = Plan(blog_title="Test Blog", audience="", tone="",
#                                  blog_kind="explainer", tasks=[])
#     sample_state["md_with_placeholders"] = "# Test\n\n[[IMG1]] end."
#     sample_state["image_specs"] = [
#         {
#             "placeholder": "[[IMG1]]",
#             "filename": "test.png",
#             "alt": "alt text",
#             "caption": "caption text",
#             "prompt": "prompt",
#         }
#     ]
#     result = place_images(sample_state)
#     final = result["final"]
#     assert "![alt text](images/test.png)" in final
#     assert "*caption text*" in final
#     # Check that the file was written
#     assert Path("test_blog.md").exists()
#     Path("test_blog.md").unlink()  # cleanup



def test_place_images_without_errors(sample_state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # all file writes go to tmp_path

    sample_state["plan"] = Plan(
        blog_title="Test Blog", audience="", tone="",
        blog_kind="explainer", tasks=[]
    )
    sample_state["md_with_placeholders"] = "# Test\n\n[[IMG1]] end."
    sample_state["image_specs"] = [
        {
            "placeholder": "[[IMG1]]",
            "filename": "test.png",
            "alt": "alt text",
            "caption": "caption text",
            "prompt": "prompt",
        }
    ]

    result = place_images(sample_state)

    expected_file = tmp_path / "test_blog.md"
    assert expected_file.exists()

    final = result["final"]
    assert "![alt text](images/test.png)" in final
    assert "*caption text*" in final


# def test_place_images_with_errors(sample_state):
#     sample_state["plan"] = Plan(blog_title="Test Blog", audience="", tone="",
#                                  blog_kind="explainer", tasks=[])
#     sample_state["md_with_placeholders"] = "# Test\n\n[[IMG1]] end."
#     sample_state["image_specs"] = [
#         {
#             "placeholder": "[[IMG1]]",
#             "filename": "test.png",
#             "alt": "alt text",
#             "caption": "caption text",
#             "prompt": "prompt",
#             "error": "Generation failed",
#         }
#     ]
#     result = place_images(sample_state)
#     final = result["final"]
#     assert "> **[IMAGE GENERATION FAILED]**" in final
#     assert "> **Error:** Generation failed" in final



def test_place_images_with_errors(sample_state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    sample_state["plan"] = Plan(
        blog_title="Test Blog", audience="", tone="",
        blog_kind="explainer", tasks=[]
    )
    sample_state["md_with_placeholders"] = "# Test\n\n[[IMG1]] end."
    sample_state["image_specs"] = [
        {
            "placeholder": "[[IMG1]]",
            "filename": "test.png",
            "alt": "alt text",
            "caption": "caption text",
            "prompt": "prompt",
            "error": "Generation failed",
        }
    ]

    result = place_images(sample_state)
    final = result["final"]
    assert "> **[IMAGE GENERATION FAILED]**" in final
    assert "> **Error:** Generation failed" in final

    expected_file = tmp_path / "test_blog.md"
    assert expected_file.exists()   # file is still written


# ----------------------------------------------------------------------
# Full graph integration test (mocked)
# ----------------------------------------------------------------------

# @pytest.mark.asyncio
# async def test_full_graph(mock_llm, mock_llm_async, mock_research_engine, mock_hf_client, tmp_path):
#     """
#     Run the entire compiled graph with all nodes mocked to return plausible data.
#     Verify that the final state contains a non-empty 'final' field.
#     """
#     # Override paths to use tmp_path for file outputs
#     with patch("update_bwa_backend_1.Path", return_value=tmp_path):
#         # --- Router mock ---
#         router_decision = RouterDecision(
#             needs_research=True,
#             mode="hybrid",
#             reason="test",
#             queries=["query1"],
#         )
#         mock_router_structured = MagicMock()
#         mock_router_structured.invoke.return_value = router_decision
#         mock_llm.with_structured_output.return_value = mock_router_structured

#         # --- Research node mocks (WRE) ---
#         mock_report = MagicMock()
#         mock_content = MagicMock()
#         mock_content.error = None
#         mock_content.word_count = 200
#         mock_content.content = "Sample research content"
#         mock_content.title = "Research Title"
#         mock_content.url = "http://example.com"
#         mock_content.metadata = {}
#         mock_report.extracted_content = [mock_content]
#         mock_research_engine.research.return_value = mock_report

#         # --- Research synthesis mocks (LLM) ---
#         mock_synth_structured = MagicMock()
#         mock_synth_structured.invoke.return_value = EvidencePack(
#             evidence=[{"title": "Synth", "url": "http://synth.com", "snippet": "s"}]
#         )
#         # We need to set this after router call, but for simplicity we can chain: the research node will call
#         # llm.with_structured_output again. We'll set it now.
#         mock_llm.with_structured_output.return_value = mock_synth_structured

#         # --- Orchestrator mock ---
#         plan = Plan(
#             blog_title="Integration Test Blog",
#             audience="devs",
#             tone="friendly",
#             blog_kind="explainer",
#             tasks=[
#                 Task(id=1, title="Intro", goal="introduce",
#                      bullets=["What is X", "Why it's important", "What we'll cover"],
#                      target_words=150),
#                 Task(id=2, title="Conclusion", goal="conclude",
#                      bullets=["Summary of points", "Key takeaway", "Next steps"],
#                      target_words=100),
#             ],
#         )
#         mock_orch_structured = MagicMock()
#         mock_orch_structured.invoke.return_value = plan
#         # The orchestrator node will call with_structured_output; set it again.
#         mock_llm.with_structured_output.return_value = mock_orch_structured

#         # --- Worker mocks ---
#         async def worker_side_effect(*args, **kwargs):
#             # Return a different response based on task id (we can inspect payload)
#             # For simplicity, just return a generic section.
#             mock_resp = MagicMock()
#             mock_resp.content = "## Section\n\nContent."
#             return mock_resp
#         mock_llm_async.ainvoke.side_effect = worker_side_effect

#         # --- Decide images mock ---
#         image_plan = GlobalImagePlan(
#             md_with_placeholders="# Integration Test Blog\n\n[[IMAGE_1]]\n\nContent.",
#             images=[
#                 ImageSpec(
#                     placeholder="[[IMAGE_1]]",
#                     filename="hero.png",
#                     alt="hero",
#                     caption="Hero image",
#                     prompt="A hero image",
#                 )
#             ],
#         )
#         mock_img_structured = MagicMock()
#         mock_img_structured.invoke.return_value = image_plan
#         # decide_images will call with_structured_output; set it.
#         mock_llm.with_structured_output.return_value = mock_img_structured

#         # --- Generate images mock (already mocked via HF client) ---
#         mock_hf_client.text_to_image.return_value.save = MagicMock()

#         # --- Initial state ---
#         initial_state: State = {
#             "topic": "Integration test",
#             "as_of": "2025-01-01",
#             "sections": [],
#             "thinking_traces": [],
#             "user_research_mode": None,
#         }

#         # Run the graph
#         final_state = await app.ainvoke(initial_state)

#         # Assertions
#         assert "final" in final_state
#         assert final_state["final"] != ""
#         assert "# Integration Test Blog" in final_state["final"]
#         assert "images/hero.png" in final_state["final"] or "[IMAGE GENERATION FAILED]" in final_state["final"]

#         # Check that the final markdown file was written
#         expected_file = tmp_path / "integration_test_blog.md"
#         assert expected_file.exists()
#         content = expected_file.read_text()
#         assert "Integration Test Blog" in content


@pytest.mark.asyncio
async def test_full_graph(mock_llm, mock_llm_async, mock_research_engine, mock_hf_client, tmp_path):
    """
    Run the entire compiled graph with all nodes mocked to return plausible data.
    Verify that the final state contains a non-empty 'final' field.
    """
    # Patch Path to redirect all file operations into tmp_path
    with patch("update_bwa_backend_1.Path") as mock_path_cls:
        # Make Path(*args) return tmp_path / (*args)
        mock_path_cls.side_effect = lambda *args: tmp_path.joinpath(*args)

        # --- Prepare mocks for each structured output call ---
        # Router
        mock_router_structured = MagicMock()
        router_decision = RouterDecision(
            needs_research=True,
            mode="hybrid",
            reason="test",
            queries=["query1"],
        )
        mock_router_structured.invoke.return_value = router_decision

        # Research synthesis
        mock_synth_structured = MagicMock()
        mock_synth_structured.invoke.return_value = EvidencePack(
            evidence=[{"title": "Synth", "url": "http://synth.com", "snippet": "s"}]
        )

        # Orchestrator
        mock_orch_structured = MagicMock()
        plan = Plan(
            blog_title="Integration Test Blog",
            audience="devs",
            tone="friendly",
            blog_kind="explainer",
            tasks=[
                Task(id=1, title="Intro", goal="introduce",
                     bullets=["What is X", "Why it's important", "What we'll cover"],
                     target_words=150),
                Task(id=2, title="Conclusion", goal="conclude",
                     bullets=["Summary of points", "Key takeaway", "Next steps"],
                     target_words=100),
            ],
        )
        mock_orch_structured.invoke.return_value = plan

        # Decide images
        mock_img_structured = MagicMock()
        image_plan = GlobalImagePlan(
            md_with_placeholders="# Integration Test Blog\n\n[[IMAGE_1]]\n\nContent.",
            images=[
                ImageSpec(
                    placeholder="[[IMAGE_1]]",
                    filename="hero.png",
                    alt="hero",
                    caption="Hero image",
                    prompt="A hero image",
                )
            ],
        )
        mock_img_structured.invoke.return_value = image_plan

        # --- Configure llm.with_structured_output to return the correct mock based on the schema ---
        def with_structured_output_side_effect(schema):
            if schema == RouterDecision:
                return mock_router_structured
            elif schema == EvidencePack:
                return mock_synth_structured
            elif schema == Plan:
                return mock_orch_structured
            elif schema == GlobalImagePlan:
                return mock_img_structured
            else:
                raise ValueError(f"Unexpected schema: {schema}")
        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect

        # --- Research node mocks (WRE) ---
        mock_report = MagicMock()
        mock_content = MagicMock()
        mock_content.error = None
        mock_content.word_count = 200
        mock_content.content = "Sample research content"
        mock_content.title = "Research Title"
        mock_content.url = "http://example.com"
        mock_content.metadata = {}
        mock_report.extracted_content = [mock_content]
        mock_research_engine.research.return_value = mock_report

        # --- Worker mocks ---
        async def worker_side_effect(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.content = "## Section\n\nContent."
            return mock_resp
        mock_llm_async.ainvoke.side_effect = worker_side_effect

        # --- Generate images mock (HF client) ---
        mock_hf_client.text_to_image.return_value.save = MagicMock()

        # --- Initial state ---
        initial_state: State = {
            "topic": "Integration test",
            "as_of": "2025-01-01",
            "sections": [],
            "thinking_traces": [],
            "user_research_mode": None,
        }

        # Run the graph
        final_state = await app.ainvoke(initial_state)

        # Assertions
        assert "final" in final_state
        assert final_state["final"] != ""
        assert "# Integration Test Blog" in final_state["final"]
        assert "images/hero.png" in final_state["final"] or "[IMAGE GENERATION FAILED]" in final_state["final"]

        # Check that the final markdown file was written inside tmp_path
        expected_file = tmp_path / "integration_test_blog.md"
        assert expected_file.exists()
        content = expected_file.read_text()
        assert "Integration Test Blog" in content


# ----------------------------------------------------------------------
# Run tests with: pytest test_bwa_backend.py -v
# ----------------------------------------------------------------------