"""LangGraph planner node that generates a blog plan using LLM and prior research context.

This node generates a structured BlogPlan (with sections containing search queries and
image prompts) using Mistral via Ollama. It retrieves relevant prior research from
ChromaDB, uses structured output with fallback parsing, and includes retry policies.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.pregel import RetryPolicy

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from graph.state import BlogPlan, GraphState, Section
from memory.chroma_store import chroma_store

logger = structlog.get_logger(__name__)

# Path to the planner system prompt
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "planner_prompt.txt"

# Default values for fallback scenarios
DEFAULT_TITLE_PREFIX = "Blog post about"

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

        # with_structured_output can sometimes cause content loss [citation:5]
        # because it enforces JSON formatting during generation. We use raw parsing
        # as a fallback to ensure complete content.
        try:
            raw_response = llm.invoke(messages)
            raw_content = (
                raw_response.content
                if hasattr(raw_response, "content")
                else str(raw_response)
            )
            blog_plan = _fallback_parse_blog_plan(raw_content, topic, research_required)
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
        blog_plan.blog_title = f"{DEFAULT_TITLE_PREFIX} {topic}"

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


def _fallback_parse_blog_plan(raw_content: str, topic: str, research_required: bool) -> BlogPlan:
    """
    Robust fallback parser for LLM responses when structured output fails.

    Uses a multi-strategy approach:
    1. Direct JSON parsing
    2. Regex extraction of JSON object
    3. Manual extraction of key fields if all else fails

    Args:
        raw_content: Raw string response from LLM.
        topic: Original topic for logging context.
        research_required: Whether research is needed (for default plan).

    Returns:
        A BlogPlan instance (may be minimal/default if parsing completely fails).
    """
    # Strategy 1: Try direct JSON parse
    try:
        data = json.loads(raw_content)
        if isinstance(data, dict):
            return _dict_to_blog_plan(data, topic, research_required)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Regex extraction of JSON object
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return _dict_to_blog_plan(data, topic, research_required)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Fallback to default plan
    logger.warning(
        "planner_node.json_parse_failed_using_default_plan",
        topic=topic,
        response_snippet=raw_content[:200],
    )
    return _create_default_plan(topic, research_required)


def _dict_to_blog_plan(data: Dict[str, Any], topic: str, research_required: bool) -> BlogPlan:
    """
    Convert a parsed dictionary to a BlogPlan instance with rich Section fields.

    Args:
        data: Dictionary parsed from JSON.
        topic: Original topic for fallback title.
        research_required: Whether research is needed (for fallback queries).

    Returns:
        Validated BlogPlan instance.
    """
    # Extract blog title
    blog_title = data.get("blog_title") or data.get("title") or f"{DEFAULT_TITLE_PREFIX} {topic}"
    if not isinstance(blog_title, str):
        blog_title = f"{DEFAULT_TITLE_PREFIX} {topic}"

    # Extract feature image prompt
    feature_image_prompt = data.get("feature_image_prompt", "")

    # Extract sections
    sections_data = data.get("sections", [])
    sections: List[Section] = []

    for sec_data in sections_data:
        if isinstance(sec_data, dict):
            # Build Section with all rich fields, providing defaults where missing
            section_id = sec_data.get("id", f"section_{len(sections)+1}")
            title = sec_data.get("title") or sec_data.get("section_title") or "Untitled Section"
            description = sec_data.get("description") or sec_data.get("content") or ""
            word_count = sec_data.get("word_count", 300)
            search_query = sec_data.get("search_query")
            # If research required but no search_query, generate a default
            if research_required and not search_query:
                search_query = f"{topic} {title.lower()}"
            image_prompt = sec_data.get("image_prompt") or f"Image for {title}"
            sections.append(Section(
                id=section_id,
                title=title,
                description=description,
                word_count=word_count,
                search_query=search_query,
                image_prompt=image_prompt,
            ))
        elif isinstance(sec_data, str):
            # Simple string section - create minimal Section
            sections.append(Section(
                id=f"section_{len(sections)+1}",
                title=sec_data,
                description="",
                word_count=300,
                search_query=f"{topic} {sec_data}" if research_required else None,
                image_prompt=f"Image for {sec_data}",
            ))

    # Ensure at least one section
    if not sections:
        return _create_default_plan(topic, research_required)

    return BlogPlan(
        blog_title=blog_title,
        feature_image_prompt=feature_image_prompt,
        research_required=research_required,
        sections=sections,
    )


def _create_default_plan(topic: str, research_required: bool) -> BlogPlan:
    """
    Create a rich default blog plan when all generation attempts fail.

    Args:
        topic: The blog topic.
        research_required: Whether research is needed.

    Returns:
        A default BlogPlan instance with four sections.
    """
    logger.error("planner_node.using_default_plan", topic=topic)
    return BlogPlan(
        blog_title=f"{DEFAULT_TITLE_PREFIX} {topic}",
        feature_image_prompt=f"Photorealistic hero image for a blog about {topic}, high quality, cinematic lighting, professional photography",
        research_required=research_required,
        sections=[
            Section(
                id="section_1",
                title="Introduction",
                description=f"Introduce the topic and explain why it matters.",
                word_count=350,
                search_query=None if not research_required else f"{topic} overview",
                image_prompt=f"Photorealistic image representing the topic {topic}, clean composition, editorial style",
            ),
            Section(
                id="section_2",
                title="Core concepts",
                description=f"Explain the most important concepts related to {topic}.",
                word_count=400,
                search_query=None if not research_required else f"{topic} key concepts",
                image_prompt=f"Photorealistic conceptual illustration for {topic}, focused detail, soft lighting",
            ),
            Section(
                id="section_3",
                title="Best practices",
                description=f"Describe practical best practices and actionable advice.",
                word_count=400,
                search_query=None if not research_required else f"{topic} best practices",
                image_prompt=f"Professional scene showing practical application of {topic}, crisp detail, engaging composition",
            ),
            Section(
                id="section_4",
                title="Future outlook",
                description=f"Explore future trends and what readers should watch next.",
                word_count=350,
                search_query=None if not research_required else f"future of {topic}",
                image_prompt=f"Future-focused image for {topic}, modern aesthetic, polished editorial photography",
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