"""
agents/writer.py
----------------
LangGraph writer worker node for blog-agent.

Dispatched once per blog section via LangGraph Send() for parallel execution.
Each invocation receives a section-specific state slice plus the full
research_results list, writes a draft in Markdown, and returns it for
downstream reducer aggregation.
"""

import json
import re
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from graph.state import GraphState, ResearchResult, SectionDraft

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    """Read the writer system prompt from disk once per call.

    Keeping the read inside the function (rather than at module level) makes
    the worker resilient to hot-reloaded prompt files and avoids stale caches
    during long-running graph sessions.
    """
    prompt_path = Path("prompts/writer_prompt.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Writer prompt not found at '{prompt_path}'. "
            "Ensure prompts/writer_prompt.txt exists relative to the working directory."
        )
    return prompt_path.read_text(encoding="utf-8")


def _dedupe_ordered(items: list[str]) -> list[str]:
    """Return a list with duplicates removed while preserving insertion order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Writer node
# ---------------------------------------------------------------------------

def writer_node(state: GraphState) -> dict:
    """LangGraph worker node: write one blog section draft.

    Called once per section via ``Send("writer_node", {...})``.  The Send()
    dispatcher injects section-specific fields directly into *state*, while
    the shared ``research_results`` list travels along unchanged so every
    worker can look up its own :class:`ResearchResult`.

    Args:
        state: The graph state slice produced by ``Send()``.  Expected keys:
            - ``section_id``          (str)  – stable identifier for this section.
            - ``section_title``       (str)  – human-readable section heading.
            - ``section_description`` (str)  – editorial brief for the LLM.
            - ``word_count``          (int)  – target word count.
            - ``image_prompt``        (str)  – optional visual prompt (carried
                                               through; not used during writing).
            - ``research_results``    (list) – full list of ResearchResult objects.

    Returns:
        A dict with a single key ``"section_drafts"`` containing a one-element
        list so LangGraph's ``operator.add`` reducer can safely merge all
        parallel results into the parent state.
    """

    # ------------------------------------------------------------------
    # 1. Extract section-specific fields from state
    # ------------------------------------------------------------------
    section_id: str = state["section_id"]
    section_title: str = state["section_title"]
    section_description: str = state["section_description"]
    word_count: int = state.get("word_count", 500)
    # image_prompt is carried through but not consumed here
    image_prompt: str = state.get("image_prompt", "")  # noqa: F841

    # ------------------------------------------------------------------
    # 2. Find the matching ResearchResult for this section
    # ------------------------------------------------------------------
    research: ResearchResult | None = next(
        (r for r in state.get("research_results", []) if r.section_id == section_id),
        None,
    )

    # ------------------------------------------------------------------
    # 3. Build research context
    # ------------------------------------------------------------------
    if research is not None:
        research_summary: str = research.summary
        source_urls: list[str] = research.source_urls
    else:
        research_summary = "No web research available. Use general knowledge."
        source_urls = []

    # ------------------------------------------------------------------
    # 4. Load system prompt from disk
    # ------------------------------------------------------------------
    system_prompt: str = _load_system_prompt()

    # ------------------------------------------------------------------
    # 5. Build the user message as a JSON-serialised dict
    # ------------------------------------------------------------------
    user_payload: dict = {
        "section_id": section_id,
        "section_title": section_title,
        "section_description": section_description,
        "target_word_count": word_count,
        "research_summary": research_summary,
        "source_urls": source_urls,
    }
    user_message_text: str = json.dumps(user_payload, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 6. Instantiate the ChatOllama client
    # ------------------------------------------------------------------
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # ------------------------------------------------------------------
    # 7. Invoke the LLM and capture raw Markdown
    # ------------------------------------------------------------------
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message_text),
    ]
    response = llm.invoke(messages)
    content: str = response.content  # raw Markdown string

    # ------------------------------------------------------------------
    # 8. Extract citation keys – deduplicated, insertion-order preserved
    # ------------------------------------------------------------------
    raw_citation_keys: list[str] = re.findall(r"\[SOURCE_\d+\]", content)
    citation_keys: list[str] = _dedupe_ordered(raw_citation_keys)

    # ------------------------------------------------------------------
    # 9. Build the SectionDraft dataclass
    # ------------------------------------------------------------------
    draft = SectionDraft(
        section_id=section_id,
        title=section_title,
        content=content,
        citation_keys=citation_keys,
    )

    # ------------------------------------------------------------------
    # 10. Structured log – section id and approximate word count
    # ------------------------------------------------------------------
    approx_word_count: int = len(content.split())
    logger.info(
        "section_draft_written",
        section_id=section_id,
        approx_word_count=approx_word_count,
        citation_count=len(citation_keys),
    )

    # ------------------------------------------------------------------
    # 11. Return in list form so operator.add reducer can aggregate results
    # ------------------------------------------------------------------
    return {"section_drafts": [draft]}