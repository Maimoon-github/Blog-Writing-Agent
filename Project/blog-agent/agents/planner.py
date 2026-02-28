# """
# agents/planner.py
# -----------------
# LangGraph Planner node for the blog-agent project.
# Generates a structured BlogPlan from a topic and research flag,
# with ChromaDB context injection and a raw-JSON fallback.
# """

# import json
# import re
# from pathlib import Path

# import structlog

# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage, SystemMessage

# from graph.state import GraphState, BlogPlan, Section
# from app.config import OLLAMA_MODEL, OLLAMA_BASE_URL
# from memory.chroma_store import chroma_store

# logger = structlog.get_logger(__name__)

# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# _PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner_prompt.txt"


# def _read_system_prompt() -> str:
#     """Read the planner system prompt from disk."""
#     return _PROMPT_PATH.read_text(encoding="utf-8")


# def _build_context_string(prior_research: list) -> str:
#     """
#     Convert a list of ChromaDB result dicts into a plain-text context block.

#     Each item is expected to have at least a ``document`` or ``summary`` key
#     (depending on how chroma_store surfaces results).  We handle both.
#     """
#     if not prior_research:
#         return ""

#     summaries: list[str] = []
#     for item in prior_research:
#         # chroma_store may return dicts with 'document' or 'summary'
#         text = item.get("document") or item.get("summary") or str(item)
#         summaries.append(text.strip())

#     context_body = "\n\n".join(summaries)
#     return f"Prior research context:\n{context_body}"


# def _parse_blog_plan_from_raw(raw_content: str) -> BlogPlan:
#     """
#     Fallback parser: extract JSON from *raw_content* and build a BlogPlan.

#     Tries ``json.loads`` on the full string first; if that fails it uses a
#     regex to find the first {...} block and attempts to parse that instead.
#     """
#     # --- attempt 1: direct parse ---
#     try:
#         data = json.loads(raw_content)
#     except json.JSONDecodeError:
#         # --- attempt 2: regex extraction ---
#         match = re.search(r"\{.*\}", raw_content, re.DOTALL)
#         if not match:
#             raise ValueError(
#                 "Could not extract a JSON object from the LLM response."
#             )
#         data = json.loads(match.group())

#     # Build Section objects
#     sections: list[Section] = [
#         Section(**sec) if isinstance(sec, dict) else sec
#         for sec in data.get("sections", [])
#     ]

#     return BlogPlan(
#         blog_title=data.get("blog_title", "Untitled Blog Post"),
#         sections=sections,
#         **{
#             k: v
#             for k, v in data.items()
#             if k not in {"blog_title", "sections"}
#         },
#     )


# # ---------------------------------------------------------------------------
# # Node
# # ---------------------------------------------------------------------------


# def planner_node(state: GraphState) -> dict:
#     """
#     LangGraph node: generate a BlogPlan for the given topic.

#     Reads the system prompt from ``prompts/planner_prompt.txt``, optionally
#     enriches the user message with prior research retrieved from ChromaDB, and
#     invokes ChatOllama with structured output.  Falls back to raw JSON parsing
#     if structured output raises any exception.

#     Args:
#         state: The current LangGraph graph state.

#     Returns:
#         A dict with a single key ``"blog_plan"`` containing the BlogPlan
#         instance so that LangGraph can merge it into the shared state.
#     """
#     # 1. Read system prompt -----------------------------------------------
#     system_prompt: str = _read_system_prompt()

#     # 2. Query ChromaDB for prior research --------------------------------
#     prior_research: list = chroma_store.search_similar(
#         state["topic"], n_results=2
#     )

#     # 3. Build context string ---------------------------------------------
#     context_string: str = _build_context_string(prior_research)

#     # 4. Instantiate the base LLM -----------------------------------------
#     llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

#     # 5. Attempt structured output ----------------------------------------
#     structured_llm = llm.with_structured_output(BlogPlan)

#     # 6. Build user message -----------------------------------------------
#     user_message: str = (
#         f"Topic: {state['topic']}\n"
#         f"Research required: {state['research_required']}\n"
#         f"{context_string}"
#     ).strip()

#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=user_message),
#     ]

#     # 7. Invoke with structured output; fall back to raw JSON on any error -
#     blog_plan: BlogPlan
#     try:
#         blog_plan = structured_llm.invoke(messages)

#     except Exception as structured_exc:  # noqa: BLE001
#         logger.warning(
#             "structured_output_failed",
#             error=str(structured_exc),
#             fallback="raw_json_parsing",
#         )
#         # Raw invocation returns an AIMessage; `.content` is the string body.
#         raw_response = llm.invoke(messages)
#         raw_content: str = (
#             raw_response.content
#             if hasattr(raw_response, "content")
#             else str(raw_response)
#         )
#         blog_plan = _parse_blog_plan_from_raw(raw_content)

#     # 8. Log plan metadata ------------------------------------------------
#     logger.info(
#         "blog_plan_generated",
#         blog_title=blog_plan.blog_title,
#         section_count=len(blog_plan.sections),
#     )

#     # 9. Return state update ----------------------------------------------
#     return {"blog_plan": blog_plan}

























"""LangGraph planner node that generates a blog plan using LLM and prior research context."""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.pregel import RetryPolicy

from graph.state import GraphState, BlogPlan, Section
from app.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from memory.chroma_store import chroma_store

logger = structlog.get_logger(__name__)

# Path to the planner system prompt
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner_prompt.txt"

# Default values for fallback scenarios
DEFAULT_BLOG_TITLE_PREFIX = "Blog post about"
DEFAULT_SECTION_TITLE = "Introduction"
DEFAULT_SECTION_CONTENT = "This section will cover the main aspects of the topic."

# Retry policy for the planner node - handles transient failures
PLANNER_RETRY_POLICY = RetryPolicy(
    initial_interval=1.0,      # Wait 1 second before first retry
    backoff_factor=2.0,         # Double wait time each retry
    max_interval=30.0,          # Cap at 30 seconds
    max_attempts=3,              # Try up to 3 times
    jitter=True,                 # Add randomness to avoid thundering herd
    retry_on=(Exception,),       # Retry on any exception (will be logged separately)
)


def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node that generates a structured blog plan.

    Loads the planner prompt, retrieves relevant prior research from ChromaDB,
    and invokes an LLM to create a BlogPlan. Falls back to robust JSON parsing
    if structured output fails.

    Args:
        state: The current graph state containing at least 'topic' and
               'research_required'. May also contain prior research context.

    Returns:
        A dictionary with either:
        - 'blog_plan': A BlogPlan instance for successful generation
        - 'error': An error message if the node fails unrecoverably

    Raises:
        FileNotFoundError: If the prompt file is missing (critical error).
    """
    # Extract state values with defaults
    topic = state.get("topic", "").strip()
    research_required = state.get("research_required", True)

    if not topic:
        logger.error("planner_node.no_topic_provided")
        return {"error": "No topic provided for planning"}

    # 1. Load system prompt
    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.critical(
            "planner_node.prompt_file_missing",
            path=str(PROMPT_PATH),
        )
        raise  # Cannot function without prompt

    # 2. Retrieve prior research context from ChromaDB
    context_string = ""
    try:
        # Get up to 3 relevant prior research items
        prior_research: List[Dict] = chroma_store.search_similar(topic, n_results=3)
        if prior_research:
            context_string = _build_context_string(prior_research)
            logger.info(
                "planner_node.retrieved_context",
                topic=topic,
                context_count=len(prior_research),
            )
        else:
            logger.info("planner_node.no_prior_research", topic=topic)
    except Exception as e:
        # ChromaDB failure - log warning but continue without context
        logger.warning(
            "planner_node.chromadb_unavailable",
            topic=topic,
            error=str(e),
        )
        context_string = ""

    # 3. Instantiate LLM
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # 4. Build user message with context
    user_message = _build_user_message(topic, research_required, context_string)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    # 5. Attempt structured output with fallback
    try:
        # Primary approach: with_structured_output
        structured_llm = llm.with_structured_output(BlogPlan)
        blog_plan = structured_llm.invoke(messages)
    except Exception as structured_exc:
        # Structured output failed - fall back to raw JSON parsing
        logger.warning(
            "planner_node.structured_output_failed",
            topic=topic,
            error=str(structured_exc),
            fallback="raw_json_parsing",
        )

        # Note: with_structured_output can sometimes cause content loss [citation:5]
        # because it enforces JSON formatting during generation. We use raw parsing
        # as a fallback to ensure complete content.
        try:
            # Raw invocation returns AIMessage with .content
            raw_response = llm.invoke(messages)
            raw_content = (
                raw_response.content
                if hasattr(raw_response, "content")
                else str(raw_response)
            )
            blog_plan = _fallback_parse_blog_plan(raw_content, topic)
        except Exception as parse_exc:
            # Complete failure - use minimal default plan
            logger.error(
                "planner_node.fallback_parsing_failed",
                topic=topic,
                error=str(parse_exc),
            )
            blog_plan = _create_default_plan(topic, research_required)

    # 6. Validate and log the result
    if not blog_plan.blog_title:
        blog_plan.blog_title = f"{DEFAULT_BLOG_TITLE_PREFIX} {topic}"

    section_count = len(blog_plan.sections) if blog_plan.sections else 0
    logger.info(
        "planner_node.plan_generated",
        blog_title=blog_plan.blog_title,
        section_count=section_count,
        topic=topic,
    )

    return {"blog_plan": blog_plan}


def _build_context_string(prior_research: List[Dict]) -> str:
    """
    Convert ChromaDB results into a formatted context string.

    Args:
        prior_research: List of result dicts from chroma_store.search_similar.
                       Each dict may contain 'summary', 'document', and 'source_urls'.

    Returns:
        Formatted context string for inclusion in the prompt.
    """
    if not prior_research:
        return ""

    formatted_items = []
    for i, item in enumerate(prior_research, 1):
        # Extract fields with fallbacks
        summary = (
            item.get("summary") or
            item.get("document") or
            item.get("content") or
            "No summary available"
        )
        sources = item.get("source_urls", [])
        sources_str = f"Sources: {', '.join(sources[:3])}" if sources else ""

        # Format this research item
        item_text = f"Research Item {i}:\nSummary: {summary}"
        if sources_str:
            item_text += f"\n{sources_str}"
        formatted_items.append(item_text)

    return "Prior research context:\n" + "\n---\n".join(formatted_items)


def _build_user_message(topic: str, research_required: bool, context: str) -> str:
    """
    Construct the user message for the LLM.

    Args:
        topic: The blog topic.
        research_required: Whether research is needed.
        context: Optional context string from prior research.

    Returns:
        Formatted user message.
    """
    message = f"Topic: {topic}\nResearch required: {research_required}"
    if context:
        message += f"\n\n{context}"
    return message


def _fallback_parse_blog_plan(raw_content: str, topic: str) -> BlogPlan:
    """
    Robust fallback parser for LLM responses when structured output fails.

    Uses a multi-strategy approach:
    1. Direct JSON parsing
    2. Regex extraction of JSON object
    3. Manual extraction of key fields if all else fails

    Args:
        raw_content: Raw string response from LLM.
        topic: Original topic for logging context.

    Returns:
        A BlogPlan instance (may be minimal/default if parsing completely fails).
    """
    # Strategy 1: Try direct JSON parse
    try:
        data = json.loads(raw_content)
        if isinstance(data, dict):
            return _dict_to_blog_plan(data, topic)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Regex extraction of JSON object
    # Look for {...} with possible nested braces [citation:5]
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return _dict_to_blog_plan(data, topic)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Try to extract sections via regex
    # This handles cases where LLM returns markdown or plain text
    logger.warning(
        "planner_node.json_parse_failed_attempting_text_extraction",
        topic=topic,
        response_snippet=raw_content[:200],
    )

    # Attempt to extract title and sections from markdown/text
    title_match = re.search(r"#\s*(.+?)[\n\r]", raw_content) or re.search(r"Title[:\s]+(.+?)[\n\r]", raw_content, re.IGNORECASE)
    blog_title = title_match.group(1).strip() if title_match else f"{DEFAULT_BLOG_TITLE_PREFIX} {topic}"

    # Look for numbered sections or bullet points
    sections = []
    section_patterns = [
        r"(?:\d+\.|\*)\s*(.+?)[\n\r](.*?)(?=(?:\d+\.|\*)\s|$)",  # Numbered or bullet points
        r"##\s*(.+?)[\n\r](.*?)(?=##|$)",  # Markdown headings
    ]

    for pattern in section_patterns:
        matches = re.findall(pattern, raw_content, re.DOTALL | re.IGNORECASE)
        if matches:
            sections = [
                Section(
                    section_title=title.strip(),
                    content=content.strip() or DEFAULT_SECTION_CONTENT,
                )
                for title, content in matches
            ]
            break

    # If still no sections, create a default one
    if not sections:
        sections = [Section(section_title=DEFAULT_SECTION_TITLE, content=DEFAULT_SECTION_CONTENT)]

    return BlogPlan(blog_title=blog_title, sections=sections)


def _dict_to_blog_plan(data: Dict[str, Any], topic: str) -> BlogPlan:
    """
    Convert a parsed dictionary to a BlogPlan instance with validation.

    Args:
        data: Dictionary parsed from JSON.
        topic: Original topic for fallback title.

    Returns:
        Validated BlogPlan instance.
    """
    # Extract and validate blog_title
    blog_title = data.get("blog_title") or data.get("title") or f"{DEFAULT_BLOG_TITLE_PREFIX} {topic}"
    if not isinstance(blog_title, str):
        blog_title = f"{DEFAULT_BLOG_TITLE_PREFIX} {topic}"

    # Extract and validate sections
    sections_data = data.get("sections", [])
    sections = []

    for sec_data in sections_data:
        if isinstance(sec_data, dict):
            # Handle both Section model fields and alternative names
            title = sec_data.get("section_title") or sec_data.get("title") or sec_data.get("heading") or DEFAULT_SECTION_TITLE
            content = sec_data.get("content") or sec_data.get("body") or sec_data.get("text") or DEFAULT_SECTION_CONTENT
            sections.append(Section(section_title=title, content=content))
        elif isinstance(sec_data, str):
            # If section is just a string, create a default content
            sections.append(Section(section_title=sec_data, content=DEFAULT_SECTION_CONTENT))

    # Ensure at least one section
    if not sections:
        sections = [Section(section_title=DEFAULT_SECTION_TITLE, content=DEFAULT_SECTION_CONTENT)]

    return BlogPlan(blog_title=blog_title, sections=sections)


def _create_default_plan(topic: str, research_required: bool) -> BlogPlan:
    """
    Create a minimal default blog plan when all generation attempts fail.

    Args:
        topic: The blog topic.
        research_required: Whether research is needed.

    Returns:
        A minimal valid BlogPlan instance.
    """
    logger.error("planner_node.using_default_plan", topic=topic)
    return BlogPlan(
        blog_title=f"{DEFAULT_BLOG_TITLE_PREFIX} {topic}",
        sections=[
            Section(
                section_title="Introduction",
                content=f"This blog post explores {topic}. Start with an engaging hook and overview.",
            ),
            Section(
                section_title="Main Content",
                content=f"Cover the key aspects of {topic}. Break down complex ideas into digestible parts.",
            ),
            Section(
                section_title="Conclusion",
                content=f"Summarize the main points about {topic} and end with a thought-provoking conclusion.",
            ),
        ],
    )


# Async version note: To convert this node to async, change the function signature to
# async def planner_node(state: GraphState) -> Dict[str, Any]:
# and replace:
# - structured_llm.invoke() -> await structured_llm.ainvoke()
# - llm.invoke() -> await llm.ainvoke()
# - chroma_store.search_similar() -> await chroma_store.asearch_similar() (if available)
# The rest of the logic remains identical.