"""LangGraph editor node for refining section drafts.

This module implements a worker node that improves the quality, clarity, SEO,
and tone of a blog section draft. It is designed to run after the writer node,
either in parallel per section or as a sequential refinement step.

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
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "editor_prompt.txt"
DEFAULT_EDITOR_PLACEHOLDER = "*Editing failed – original draft preserved.*"


# ---------------------------------------------------------------------------
# Helper functions (shared between sync and async versions)
# ---------------------------------------------------------------------------
def _load_prompt_sync() -> str:
    """Load system prompt from file (synchronous)."""
    if not PROMPT_PATH.exists():
        logger.critical("editor_node.prompt_file_missing", path=str(PROMPT_PATH))
        raise FileNotFoundError(f"Editor prompt not found at '{PROMPT_PATH}'")
    return PROMPT_PATH.read_text(encoding="utf-8")


async def _load_prompt_async() -> str:
    """Load system prompt from file asynchronously (runs in thread)."""
    return await asyncio.to_thread(_load_prompt_sync)


def _extract_citations(content: str) -> List[str]:
    """
    Extract and deduplicate citation keys of the form [SOURCE_\\d+].

    Preserves the order of first appearance.
    """
    raw_keys = re.findall(r"\[SOURCE_\d+\]", content)
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
    original_content: str,
    research_summary: Optional[str] = None,
) -> str:
    """Construct the JSON user message payload for the editor."""
    payload = {
        "section_id": section_id,
        "section_title": section_title,
        "section_description": section_description,
        "original_content": original_content,
    }
    if research_summary:
        payload["research_summary"] = research_summary
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
    section_id: str, section_title: str, original_content: str, error_msg: str
) -> SectionDraft:
    """Return original draft when editing fails."""
    logger.error(
        "editor_node.using_original",
        section_id=section_id,
        error=error_msg,
    )
    # Preserve original citation keys
    citation_keys = _extract_citations(original_content)
    return SectionDraft(
        section_id=section_id,
        title=section_title,
        content=original_content,
        citation_keys=citation_keys,
    )


# ---------------------------------------------------------------------------
# Asynchronous version (recommended for LangGraph parallel dispatch)
# ---------------------------------------------------------------------------
async def editor_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Asynchronous LangGraph worker node that refines a single section draft.

    Expected state fields (for this parallel instance):
        - section_id: str
        - section_title: str
        - section_description: str
        - section_draft: SectionDraft (the original draft content)
        - research_results: List[ResearchResult] (optional, for context)

    Returns:
        Dictionary with key 'section_drafts' containing a one-element list
        (so the reducer can merge results). In case of error, returns the
        original draft unmodified.

    Note:
        This node is designed to be called concurrently via Send() after
        the writer nodes have completed.
    """
    # 1. Extract required fields
    section_id = state.get("section_id")
    section_title = state.get("section_title")
    section_description = state.get("section_description")
    original_draft = state.get("section_draft")
    research_results = state.get("research_results", [])

    # Validate required fields
    missing = []
    if not section_id:
        missing.append("section_id")
    if not section_title:
        missing.append("section_title")
    if not section_description:
        missing.append("section_description")
    if not original_draft or not isinstance(original_draft, SectionDraft):
        missing.append("section_draft (SectionDraft object)")

    if missing:
        logger.warning(
            "editor_node_async.missing_fields",
            missing_fields=missing,
            section_id=section_id,
        )
        return {
            "error": f"Missing required fields: {', '.join(missing)}",
            "section_drafts": [],
        }

    original_content = original_draft.content
    if not original_content or original_content == "*Content generation failed. Please check logs.*":
        logger.warning(
            "editor_node_async.skipping_empty_draft",
            section_id=section_id,
        )
        # Return original draft unchanged
        return {"section_drafts": [original_draft]}

    # 2. Find matching research (optional, for fact-checking context)
    research = _find_research_for_section(research_results, section_id)
    research_summary = research.summary if research else None

    # 3. Load system prompt
    try:
        system_prompt = await _load_prompt_async()
    except FileNotFoundError:
        raise

    # 4. Build user message and invoke LLM
    user_message = _build_user_payload(
        section_id,
        section_title,
        section_description,
        original_content,
        research_summary,
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    logger.info(
        "editor_node_async.starting",
        section_id=section_id,
        original_length=len(original_content),
    )

    try:
        response = await llm.ainvoke(messages)
        refined_content = response.content.strip()
        if not refined_content:
            logger.warning(
                "editor_node_async.empty_response",
                section_id=section_id,
            )
            refined_content = original_content
    except Exception as e:
        logger.error(
            "editor_node_async.llm_invocation_failed",
            section_id=section_id,
            error=str(e),
        )
        draft = _create_placeholder_draft(
            section_id, section_title, original_content, str(e)
        )
        return {"section_drafts": [draft]}

    # 5. Extract citation keys from refined content (preserve or add)
    citation_keys = _extract_citations(refined_content)

    # 6. Create refined draft
    refined_draft = SectionDraft(
        section_id=section_id,
        title=section_title,
        content=refined_content,
        citation_keys=citation_keys,
    )

    # 7. Log success
    approx_word_count = len(refined_content.split())
    logger.info(
        "editor_node_async.completed",
        section_id=section_id,
        approx_word_count=approx_word_count,
        citation_count=len(citation_keys),
    )

    return {"section_drafts": [refined_draft]}


# ---------------------------------------------------------------------------
# Synchronous version (for non‑async graphs or simple testing)
# ---------------------------------------------------------------------------
def editor_node(state: GraphState) -> Dict[str, Any]:
    """
    Synchronous LangGraph worker node for editing a section draft.

    Equivalent to `editor_node_async` but uses synchronous LLM invocation.
    Suitable for graphs that do not use async execution.

    Args and return value are the same as `editor_node_async`.
    """
    # 1. Extract required fields
    section_id = state.get("section_id")
    section_title = state.get("section_title")
    section_description = state.get("section_description")
    original_draft = state.get("section_draft")
    research_results = state.get("research_results", [])

    # Validate required fields
    missing = []
    if not section_id:
        missing.append("section_id")
    if not section_title:
        missing.append("section_title")
    if not section_description:
        missing.append("section_description")
    if not original_draft or not isinstance(original_draft, SectionDraft):
        missing.append("section_draft (SectionDraft object)")

    if missing:
        logger.warning(
            "editor_node.missing_fields",
            missing_fields=missing,
            section_id=section_id,
        )
        return {
            "error": f"Missing required fields: {', '.join(missing)}",
            "section_drafts": [],
        }

    original_content = original_draft.content
    if not original_content or original_content == "*Content generation failed. Please check logs.*":
        logger.warning(
            "editor_node.skipping_empty_draft",
            section_id=section_id,
        )
        return {"section_drafts": [original_draft]}

    # 2. Find matching research
    research = _find_research_for_section(research_results, section_id)
    research_summary = research.summary if research else None

    # 3. Load system prompt
    try:
        system_prompt = _load_prompt_sync()
    except FileNotFoundError:
        raise

    # 4. Build user message and invoke LLM
    user_message = _build_user_payload(
        section_id,
        section_title,
        section_description,
        original_content,
        research_summary,
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    logger.info(
        "editor_node.starting",
        section_id=section_id,
        original_length=len(original_content),
    )

    try:
        response = llm.invoke(messages)
        refined_content = response.content.strip()
        if not refined_content:
            logger.warning(
                "editor_node.empty_response",
                section_id=section_id,
            )
            refined_content = original_content
    except Exception as e:
        logger.error(
            "editor_node.llm_invocation_failed",
            section_id=section_id,
            error=str(e),
        )
        draft = _create_placeholder_draft(
            section_id, section_title, original_content, str(e)
        )
        return {"section_drafts": [draft]}

    # 5. Extract citation keys
    citation_keys = _extract_citations(refined_content)

    # 6. Create refined draft
    refined_draft = SectionDraft(
        section_id=section_id,
        title=section_title,
        content=refined_content,
        citation_keys=citation_keys,
    )

    # 7. Log success
    approx_word_count = len(refined_content.split())
    logger.info(
        "editor_node.completed",
        section_id=section_id,
        approx_word_count=approx_word_count,
        citation_count=len(citation_keys),
    )

    return {"section_drafts": [refined_draft]}


# ---------------------------------------------------------------------------
# Compatibility alias (the node expected by the main graph)
# ---------------------------------------------------------------------------
# By default, export the async version as the primary node.
# The main graph can import `editor_node` and use it in an async pipeline.
# editor_node = editor_node_async  # Uncomment if graph uses async

# Note on concurrency and state safety:
# This node is designed to run in parallel for multiple sections. Each worker
# receives its own state slice and returns only its own section draft. The
# parent graph merges all drafts using `operator.add` on the `section_drafts`
# list. Since workers never modify shared state directly, there are no race
# conditions or state conflicts.