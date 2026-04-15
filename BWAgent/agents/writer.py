"""Writer node for BWAgent."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from graph.state import GraphState, ResearchResult, SectionDraft

logger = structlog.get_logger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "writer_prompt.txt"
DEFAULT_WORD_COUNT = 350
PLACEHOLDER = "*Content generation failed. Please check the logs.*"


def _extract_citations(content: str) -> List[str]:
    found = re.findall(r"\[SOURCE_\d+\]", content)
    seen = set()
    result: List[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _find_research(research_results: List[ResearchResult], section_id: str) -> Optional[ResearchResult]:
    return next((item for item in research_results if item.section_id == section_id), None)


async def _load_prompt() -> str:
    return await asyncio.to_thread(lambda: PROMPT_PATH.read_text(encoding="utf-8"))


async def writer_node(state: GraphState) -> Dict[str, Any]:
    section_id = state.get("section_id")
    section_title = state.get("section_title")
    section_description = state.get("section_description")
    word_count = state.get("word_count", DEFAULT_WORD_COUNT)
    research_results = state.get("research_results", [])

    if not section_id or not section_title or not section_description:
        logger.error("writer_node.missing_fields", section_id=section_id)
        return {"error": "Missing required section fields", "section_drafts": []}

    research = _find_research(research_results, section_id)
    if research:
        research_summary = research.summary
        source_urls = research.source_urls
    else:
        research_summary = "No web research available. Use general knowledge."
        source_urls = []

    try:
        system_prompt = await _load_prompt()
    except FileNotFoundError:
        logger.critical("writer_node.prompt_missing", path=str(PROMPT_PATH))
        raise

    payload = {
        "section_id": section_id,
        "section_title": section_title,
        "section_description": section_description,
        "target_word_count": word_count,
        "research_summary": research_summary,
        "source_urls": source_urls,
    }
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(payload, indent=2))]

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip() or PLACEHOLDER
    except Exception as exc:
        logger.error("writer_node.llm_failed", error=str(exc), section_id=section_id)
        content = PLACEHOLDER

    citation_keys = _extract_citations(content)
    draft = SectionDraft(section_id=section_id, title=section_title, content=content, citation_keys=citation_keys)
    logger.info("writer_node.completed", section_id=section_id, citations=len(citation_keys))
    return {"section_drafts": [draft]}
