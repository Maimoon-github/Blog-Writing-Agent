"""LangGraph router node for topic analysis and safety check."""

import json
import re
from pathlib import Path
from typing import Dict, Any

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import GraphState
from app.config import OLLAMA_MODEL, OLLAMA_BASE_URL

logger = structlog.get_logger(__name__)

# Path to the system prompt for the router
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "router_prompt.txt"

# Default values when parsing fails
DEFAULT_RESEARCH_REQUIRED = True
DEFAULT_SAFE = True


def router_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node that routes the workflow based on topic analysis.

    Loads the router prompt, invokes an LLM to decide if research is needed
    and if the topic is safe. Updates the graph state accordingly.

    Args:
        state: The current graph state containing at least 'topic'.

    Returns:
        A dictionary with keys to update the state. Possible keys:
        - 'research_required': bool (if topic is safe)
        - 'error': str (if topic rejected or LLM fails)

    Raises:
        FileNotFoundError: If the prompt file is missing (critical error).
    """
    topic = state.get("topic", "")
    if not topic:
        logger.error("router_node.no_topic_provided")
        return {"error": "No topic provided", "research_required": False}

    # 1. Load system prompt
    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.critical(
            "router_node.prompt_file_missing",
            path=str(PROMPT_PATH),
        )
        raise  # Re-raise because the node cannot function without a prompt

    # 2. Instantiate LLM
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # 3. Invoke LLM with error handling
    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Topic: {topic}"),
            ]
        )
        raw_content: str = response.content
    except Exception as e:
        logger.error(
            "router_node.llm_invocation_failed",
            error=str(e),
            topic=topic,
        )
        # Fallback: assume research is needed, but signal an error
        return {
            "research_required": DEFAULT_RESEARCH_REQUIRED,
            "error": f"LLM error: {str(e)}",
        }

    # 4. Parse the LLM response
    parsed = _parse_llm_response(raw_content, topic=topic)

    # 5. Safety check
    if not parsed.get("safe", DEFAULT_SAFE):
        logger.warning(
            "router_node.topic_rejected_by_safety_filter",
            topic=topic,
        )
        return {"error": "Topic rejected by safety filter", "research_required": False}

    # 6. Extract research decision
    research_required = parsed.get("research_required", DEFAULT_RESEARCH_REQUIRED)

    logger.info(
        "router_node.routing_decision",
        topic=topic,
        research_required=research_required,
        safe=True,
    )

    return {"research_required": research_required}


def _parse_llm_response(response: str, topic: str = "") -> Dict[str, bool]:
    """
    Attempt to parse a JSON object from the LLM response.

    The expected JSON format: {"research_required": bool, "safe": bool}

    Tries:
      1. Direct json.loads on the entire response.
      2. Regex extraction of the first JSON-like object.
      3. Fallback to default values if both fail.

    Args:
        response: Raw string response from the LLM.
        topic: Topic for logging context (optional).

    Returns:
        Dictionary with at least 'research_required' and 'safe' keys (both bool).
    """
    # Helper to validate and clean parsed dict
    def validate(parsed: Dict[str, Any]) -> Dict[str, bool]:
        research = parsed.get("research_required")
        safe = parsed.get("safe")
        # Ensure both are bools; if missing or wrong type, use defaults
        if not isinstance(research, bool):
            research = DEFAULT_RESEARCH_REQUIRED
        if not isinstance(safe, bool):
            safe = DEFAULT_SAFE
        return {"research_required": research, "safe": safe}

    # Strategy 1: direct JSON parse
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            return validate(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: regex extraction
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return validate(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: fallback defaults
    logger.warning(
        "router_node.parse_fallback_used",
        topic=topic,
        response_snippet=response[:100],  # Log first 100 chars for debugging
    )
    return {
        "research_required": DEFAULT_RESEARCH_REQUIRED,
        "safe": DEFAULT_SAFE,
    }

# Note: To convert this node to async, change the function signature to
# async def router_node(state: GraphState) -> Dict[str, Any]:
# and replace llm.invoke with await llm.ainvoke(...). The rest remains the same.