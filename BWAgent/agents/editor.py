"""Editor node for BWAgent."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from graph.state import GraphState, SectionDraft

logger = structlog.get_logger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "editor_prompt.txt"


def _extract_citations(content: str) -> List[str]:
    found = re.findall(r"\[SOURCE_\d+\]", content)
    seen = set()
    result: List[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


async def _load_prompt() -> str:
    return await asyncio.to_thread(lambda: PROMPT_PATH.read_text(encoding="utf-8"))


async def editor_node(state: GraphState) -> Dict[str, Any]:
    section_id = state.get("section_id")
    section_title = state.get("section_title")
    section_content = state.get("section_content", "")
    citation_keys = state.get("citation_keys", [])

    if not section_id or not section_title or not section_content:
        logger.error("editor_node.missing_fields", section_id=section_id)
        return {"error": "Missing editor section data", "section_drafts": []}

    try:
        system_prompt = await _load_prompt()
    except FileNotFoundError:
        logger.critical("editor_node.prompt_missing", path=str(PROMPT_PATH))
        raise

    payload = {
        "section_id": section_id,
        "section_title": section_title,
        "section_content": section_content,
        "citation_keys": citation_keys,
    }
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(payload, indent=2))]

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip() or section_content
    except Exception as exc:
        logger.error("editor_node.llm_failed", error=str(exc), section_id=section_id)
        content = section_content

    edited_keys = _extract_citations(content)
    draft = SectionDraft(section_id=section_id, title=section_title, content=content, citation_keys=edited_keys)
    logger.info("editor_node.completed", section_id=section_id, citations=len(edited_keys))
    return {"section_drafts": [draft]}
