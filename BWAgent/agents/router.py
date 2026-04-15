"""Router node for BWAgent that decides research requirements and safety."""

import json
import re
from typing import Any, Dict

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from graph.state import GraphState

logger = structlog.get_logger(__name__)
PROMPT_PATH = __import__("pathlib").Path(__file__).resolve().parent.parent / "prompts" / "router_prompt.txt"
DEFAULT_RESEARCH_REQUIRED = True
DEFAULT_SAFE = True


def router_node(state: GraphState) -> Dict[str, Any]:
    topic = state.get("topic", "").strip()
    if not topic:
        logger.error("router_node.no_topic")
        return {"error": "No topic provided", "research_required": False}

    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.critical("router_node.prompt_missing", path=str(PROMPT_PATH))
        raise

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {topic}"),
        ])
        raw = response.content
    except Exception as exc:
        logger.error("router_node.llm_error", error=str(exc), topic=topic)
        return {"research_required": DEFAULT_RESEARCH_REQUIRED, "error": f"LLM error: {exc}"}

    parsed = _parse_response(raw, topic=topic)
    if not parsed.get("safe", DEFAULT_SAFE):
        logger.warning("router_node.topic_unsafe", topic=topic)
        return {"research_required": False, "error": "Topic rejected by safety filter"}

    research_required = parsed.get("research_required", DEFAULT_RESEARCH_REQUIRED)
    logger.info("router_node.result", topic=topic, research_required=research_required)
    return {"research_required": research_required}


def _parse_response(response: str, topic: str = "") -> Dict[str, bool]:
    def validate(candidate: Dict[str, Any]) -> Dict[str, bool]:
        research = candidate.get("research_required")
        safe = candidate.get("safe")
        if not isinstance(research, bool):
            research = DEFAULT_RESEARCH_REQUIRED
        if not isinstance(safe, bool):
            safe = DEFAULT_SAFE
        return {"research_required": research, "safe": safe}

    try:
        data = json.loads(response)
        if isinstance(data, dict):
            return validate(data)
    except Exception:
        pass

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return validate(data)
        except Exception:
            pass

    logger.warning("router_node.parse_fallback", topic=topic, snippet=response[:120])
    return {"research_required": DEFAULT_RESEARCH_REQUIRED, "safe": DEFAULT_SAFE}
