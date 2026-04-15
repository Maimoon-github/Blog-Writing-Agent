"""LangGraph writer node for parallel section drafting.

This module implements a worker node that generates a complete Markdown draft
for a single blog section, using previously researched content. The node is
designed to be dispatched concurrently via LangGraph's `Send()` for each section.

Both synchronous and asynchronous versions are provided for compatibility with
different execution contexts. The async version uses `asyncio.to_thread` for
file I/O and `ainvoke` for LLM calls to maximize concurrency.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import GraphState, ResearchResult, SectionDraft
from app.config import OLLAMA_MODEL, OLLAMA_BASE_URL

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "writer_prompt.txt"
DEFAULT_WORD_COUNT = 300
PLACEHOLDER_CONTENT = "*Content generation failed. Please check logs.*"


# ---------------------------------------------------------------------------
# Helper functions (shared between sync and async versions)
# ---------------------------------------------------------------------------
def _load_prompt_sync() -> str:
    """Load system prompt from file (synchronous)."""
    if not PROMPT_PATH.exists():
        logger.critical("writer_node.prompt_file_missing", path=str(PROMPT_PATH))
        raise FileNotFoundError(f"Writer prompt not found at '{PROMPT_PATH}'")
    return PROMPT_PATH.read_text(encoding="utf-8")


async def _load_prompt_async() -> str:
    """Load system prompt from file asynchronously (runs in thread)."""
    return await asyncio.to_thread(_load_prompt_sync)


def _extract_citations(content: str) -> List[str]:
    """
    Extract and deduplicate citation keys of the form [SOURCE_\\d+].

    Preserves the order of first appearance.
    """
    # Find all citation patterns
    raw_keys = re.findall(r"\[SOURCE_\d+\]", content)
    # Deduplicate while preserving order
    seen = set()
    unique_keys = []
    for key in raw_keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
    return unique_keys


def _build_user_payload(
    section_id: str,
    section_title: str,
    section_description: str,
    word_count: int,
    research_summary: str,
    source_urls: List[str],
) -> str:
    """Construct the JSON user message payload."""
    payload = {
        "section_id": section_id,
        "section_title": section_title,
        "section_description": section_description,
        "target_word_count": word_count,
        "research_summary": research_summary,
        "source_urls": source_urls,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _find_research_for_section(
    research_results: List[ResearchResult], section_id: str
) -> Optional[ResearchResult]:
    """Find the research result matching the given section ID."""
    return next(
        (r for r in research_results if r.section_id == section_id),
        None,
    )


def _create_placeholder_draft(
    section_id: str, section_title: str, error_msg: str
) -> SectionDraft:
    """Create a minimal draft when generation fails."""
    logger.error(
        "writer_node.using_placeholder",
        section_id=section_id,
        error=error_msg,
    )
    return SectionDraft(
        section_id=section_id,
        title=section_title,
        content=PLACEHOLDER_CONTENT,
        citation_keys=[],
    )


# ---------------------------------------------------------------------------
# Synchronous version
# ---------------------------------------------------------------------------
def writer_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph worker node (synchronous) that writes a blog section draft.

    Called once per section via `Send("writer_node", {...})`. The state passed
    in contains the section's specific fields plus the global `research_results`.

    Args:
        state: GraphState slice with keys:
            - section_id (str)
            - section_title (str)
            - section_description (str)
            - word_count (int, optional)
            - research_results (List[ResearchResult])  # from parent state

    Returns:
        Dictionary with key 'section_drafts' containing a one-element list
        (so the reducer can merge results). In case of unrecoverable error,
        may also include an 'error' key and an empty list.
    """
    # ----------------------------------------------------------------------
    # 1. Extract required fields with validation
    # ----------------------------------------------------------------------
    section_id = state.get("section_id")
    section_title = state.get("section_title")
    section_description = state.get("section_description")
    word_count = state.get("word_count", DEFAULT_WORD_COUNT)
    research_results = state.get("research_results", [])

    # Validate required fields
    missing = []
    if not section_id:
        missing.append("section_id")
    if not section_title:
        missing.append("section_title")
    if not section_description:
        missing.append("section_description")

    if missing:
        logger.warning(
            "writer_node.missing_fields",
            missing_fields=missing,
            section_id=section_id,
        )
        return {
            "error": f"Missing required fields: {', '.join(missing)}",
            "section_drafts": [],
        }

    # ----------------------------------------------------------------------
    # 2. Find matching research (optional)
    # ----------------------------------------------------------------------
    research = _find_research_for_section(research_results, section_id)
    if research:
        research_summary = research.summary
        source_urls = research.source_urls
        logger.debug(
            "writer_node.found_research",
            section_id=section_id,
            source_count=len(source_urls),
        )
    else:
        logger.warning(
            "writer_node.missing_research",
            section_id=section_id,
            msg="No research found for this section; using general knowledge.",
        )
        research_summary = "No web research available. Use general knowledge."
        source_urls = []

    # ----------------------------------------------------------------------
    # 3. Load system prompt (raises FileNotFoundError on failure)
    # ----------------------------------------------------------------------
    try:
        system_prompt = _load_prompt_sync()
    except FileNotFoundError:
        # Re-raise – this is a critical configuration error.
        raise

    # ----------------------------------------------------------------------
    # 4. Build user message and invoke LLM
    # ----------------------------------------------------------------------
    user_message = _build_user_payload(
        section_id,
        section_title,
        section_description,
        word_count,
        research_summary,
        source_urls,
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    logger.info(
        "writer_node.starting",
        section_id=section_id,
        target_word_count=word_count,
    )

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if not content:
            logger.warning(
                "writer_node.empty_response",
                section_id=section_id,
            )
            content = PLACEHOLDER_CONTENT
    except Exception as e:
        logger.error(
            "writer_node.llm_invocation_failed",
            section_id=section_id,
            error=str(e),
        )
        draft = _create_placeholder_draft(section_id, section_title, str(e))
        return {"section_drafts": [draft]}

    # ----------------------------------------------------------------------
    # 5. Extract citations and build draft
    # ----------------------------------------------------------------------
    citation_keys = _extract_citations(content)
    draft = SectionDraft(
        section_id=section_id,
        title=section_title,
        content=content,
        citation_keys=citation_keys,
    )

    # ----------------------------------------------------------------------
    # 6. Log success
    # ----------------------------------------------------------------------
    approx_word_count = len(content.split())
    logger.info(
        "writer_node.completed",
        section_id=section_id,
        approx_word_count=approx_word_count,
        citation_count=len(citation_keys),
    )

    return {"section_drafts": [draft]}


# ---------------------------------------------------------------------------
# Asynchronous version
# ---------------------------------------------------------------------------
async def writer_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Asynchronous LangGraph worker node for writing a blog section draft.

    Equivalent to `writer_node` but uses async I/O and `ainvoke` to avoid
    blocking the event loop. All other behavior is identical.

    Args and return value are the same as `writer_node`.
    """
    # ----------------------------------------------------------------------
    # 1. Extract required fields with validation
    # ----------------------------------------------------------------------
    section_id = state.get("section_id")
    section_title = state.get("section_title")
    section_description = state.get("section_description")
    word_count = state.get("word_count", DEFAULT_WORD_COUNT)
    research_results = state.get("research_results", [])

    # Validate required fields
    missing = []
    if not section_id:
        missing.append("section_id")
    if not section_title:
        missing.append("section_title")
    if not section_description:
        missing.append("section_description")

    if missing:
        logger.warning(
            "writer_node_async.missing_fields",
            missing_fields=missing,
            section_id=section_id,
        )
        return {
            "error": f"Missing required fields: {', '.join(missing)}",
            "section_drafts": [],
        }

    # ----------------------------------------------------------------------
    # 2. Find matching research (optional)
    # ----------------------------------------------------------------------
    research = _find_research_for_section(research_results, section_id)
    if research:
        research_summary = research.summary
        source_urls = research.source_urls
        logger.debug(
            "writer_node_async.found_research",
            section_id=section_id,
            source_count=len(source_urls),
        )
    else:
        logger.warning(
            "writer_node_async.missing_research",
            section_id=section_id,
            msg="No research found for this section; using general knowledge.",
        )
        research_summary = "No web research available. Use general knowledge."
        source_urls = []

    # ----------------------------------------------------------------------
    # 3. Load system prompt asynchronously
    # ----------------------------------------------------------------------
    try:
        system_prompt = await _load_prompt_async()
    except FileNotFoundError:
        raise

    # ----------------------------------------------------------------------
    # 4. Build user message and invoke LLM asynchronously
    # ----------------------------------------------------------------------
    user_message = _build_user_payload(
        section_id,
        section_title,
        section_description,
        word_count,
        research_summary,
        source_urls,
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    logger.info(
        "writer_node_async.starting",
        section_id=section_id,
        target_word_count=word_count,
    )

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        if not content:
            logger.warning(
                "writer_node_async.empty_response",
                section_id=section_id,
            )
            content = PLACEHOLDER_CONTENT
    except Exception as e:
        logger.error(
            "writer_node_async.llm_invocation_failed",
            section_id=section_id,
            error=str(e),
        )
        draft = _create_placeholder_draft(section_id, section_title, str(e))
        return {"section_drafts": [draft]}

    # ----------------------------------------------------------------------
    # 5. Extract citations and build draft
    # ----------------------------------------------------------------------
    citation_keys = _extract_citations(content)
    draft = SectionDraft(
        section_id=section_id,
        title=section_title,
        content=content,
        citation_keys=citation_keys,
    )

    # ----------------------------------------------------------------------
    # 6. Log success
    # ----------------------------------------------------------------------
    approx_word_count = len(content.split())
    logger.info(
        "writer_node_async.completed",
        section_id=section_id,
        approx_word_count=approx_word_count,
        citation_count=len(citation_keys),
    )

    return {"section_drafts": [draft]}


# Note on concurrency and state safety:
# This node is designed to run in parallel for multiple sections. Each worker
# receives its own state slice and returns only its own section draft. The
# parent graph merges all drafts using `operator.add` on the `section_drafts`
# list. Since workers never modify shared state directly, there are no race
# conditions or state conflicts. This aligns with best practices for parallel
# execution in LangGraph [citation:3].