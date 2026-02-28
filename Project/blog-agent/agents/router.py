import json
import re
from pathlib import Path

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import GraphState
from app.config import OLLAMA_MODEL, OLLAMA_BASE_URL

logger = structlog.get_logger(__name__)


def router_node(state: GraphState) -> dict:
    """
    LangGraph Router node function.

    Loads the system prompt from prompts/router_prompt.txt, invokes a
    ChatOllama LLM with the user's topic, parses the JSON response to
    determine routing decisions, and returns the appropriate state update.

    Args:
        state: The current LangGraph GraphState containing at least 'topic'.

    Returns:
        A dict with 'research_required' (bool), or an 'error' key if the
        topic fails the safety check.
    """
    # 1. Read system prompt from file
    system_prompt = Path("prompts/router_prompt.txt").read_text(encoding="utf-8")

    # 2. Instantiate the ChatOllama LLM
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # 3. Invoke the LLM with system + user messages
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )

    # 4. Extract raw response content
    raw_content: str = response.content

    # 5. Parse JSON with fallback strategies
    parsed_result: dict = {}

    try:
        parsed_result = json.loads(raw_content)
    except (json.JSONDecodeError, ValueError):
        # Fallback: extract the first JSON object from the response via regex
        match = re.search(r"\{.*?\}", raw_content, re.DOTALL)
        if match:
            try:
                parsed_result = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                # Final fallback: safe defaults
                parsed_result = {"research_required": True, "safe": True}
        else:
            parsed_result = {"research_required": True, "safe": True}

    # 6. Safety check — reject the topic if flagged as unsafe
    if parsed_result.get("safe", True) is False:
        logger.warning(
            "router_node.topic_rejected_by_safety_filter",
            topic=state["topic"],
        )
        return {"error": "Topic rejected by safety filter", "research_required": False}

    # 7. Log the routing decision
    research_required: bool = parsed_result.get("research_required", True)
    logger.info(
        "router_node.routing_decision",
        topic=state["topic"],
        research_required=research_required,
    )

    # 8. Return the routing result
    return {"research_required": research_required}