"""
agents/planner.py
-----------------
LangGraph Planner node for the blog-agent project.
Generates a structured BlogPlan from a topic and research flag,
with ChromaDB context injection and a raw-JSON fallback.
"""

import json
import re
from pathlib import Path

import structlog

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import GraphState, BlogPlan, Section
from app.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from memory.chroma_store import chroma_store

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner_prompt.txt"


def _read_system_prompt() -> str:
    """Read the planner system prompt from disk."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_context_string(prior_research: list) -> str:
    """
    Convert a list of ChromaDB result dicts into a plain-text context block.

    Each item is expected to have at least a ``document`` or ``summary`` key
    (depending on how chroma_store surfaces results).  We handle both.
    """
    if not prior_research:
        return ""

    summaries: list[str] = []
    for item in prior_research:
        # chroma_store may return dicts with 'document' or 'summary'
        text = item.get("document") or item.get("summary") or str(item)
        summaries.append(text.strip())

    context_body = "\n\n".join(summaries)
    return f"Prior research context:\n{context_body}"


def _parse_blog_plan_from_raw(raw_content: str) -> BlogPlan:
    """
    Fallback parser: extract JSON from *raw_content* and build a BlogPlan.

    Tries ``json.loads`` on the full string first; if that fails it uses a
    regex to find the first {...} block and attempts to parse that instead.
    """
    # --- attempt 1: direct parse ---
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        # --- attempt 2: regex extraction ---
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not match:
            raise ValueError(
                "Could not extract a JSON object from the LLM response."
            )
        data = json.loads(match.group())

    # Build Section objects
    sections: list[Section] = [
        Section(**sec) if isinstance(sec, dict) else sec
        for sec in data.get("sections", [])
    ]

    return BlogPlan(
        blog_title=data.get("blog_title", "Untitled Blog Post"),
        sections=sections,
        **{
            k: v
            for k, v in data.items()
            if k not in {"blog_title", "sections"}
        },
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def planner_node(state: GraphState) -> dict:
    """
    LangGraph node: generate a BlogPlan for the given topic.

    Reads the system prompt from ``prompts/planner_prompt.txt``, optionally
    enriches the user message with prior research retrieved from ChromaDB, and
    invokes ChatOllama with structured output.  Falls back to raw JSON parsing
    if structured output raises any exception.

    Args:
        state: The current LangGraph graph state.

    Returns:
        A dict with a single key ``"blog_plan"`` containing the BlogPlan
        instance so that LangGraph can merge it into the shared state.
    """
    # 1. Read system prompt -----------------------------------------------
    system_prompt: str = _read_system_prompt()

    # 2. Query ChromaDB for prior research --------------------------------
    prior_research: list = chroma_store.search_similar(
        state["topic"], n_results=2
    )

    # 3. Build context string ---------------------------------------------
    context_string: str = _build_context_string(prior_research)

    # 4. Instantiate the base LLM -----------------------------------------
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # 5. Attempt structured output ----------------------------------------
    structured_llm = llm.with_structured_output(BlogPlan)

    # 6. Build user message -----------------------------------------------
    user_message: str = (
        f"Topic: {state['topic']}\n"
        f"Research required: {state['research_required']}\n"
        f"{context_string}"
    ).strip()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    # 7. Invoke with structured output; fall back to raw JSON on any error -
    blog_plan: BlogPlan
    try:
        blog_plan = structured_llm.invoke(messages)

    except Exception as structured_exc:  # noqa: BLE001
        logger.warning(
            "structured_output_failed",
            error=str(structured_exc),
            fallback="raw_json_parsing",
        )
        # Raw invocation returns an AIMessage; `.content` is the string body.
        raw_response = llm.invoke(messages)
        raw_content: str = (
            raw_response.content
            if hasattr(raw_response, "content")
            else str(raw_response)
        )
        blog_plan = _parse_blog_plan_from_raw(raw_content)

    # 8. Log plan metadata ------------------------------------------------
    logger.info(
        "blog_plan_generated",
        blog_title=blog_plan.blog_title,
        section_count=len(blog_plan.sections),
    )

    # 9. Return state update ----------------------------------------------
    return {"blog_plan": blog_plan}