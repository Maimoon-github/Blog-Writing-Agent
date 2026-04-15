"""Researcher node for BWAgent."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from graph.state import GraphState, ResearchResult
from memory.cache import cache
from memory.chroma_store import chroma_store
from tools.search import SearchResult, search
from tools.scraper import async_fetch_page_content

logger = structlog.get_logger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "researcher_prompt.txt"
MAX_SEARCH_RESULTS = 5
TOP_FETCH = 3


def _parse_llm_response(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
        return {
            "summary": data.get("summary", ""),
            "source_urls": data.get("source_urls", []),
            "sufficient": data.get("sufficient", True),
        }
    except Exception:
        return {"summary": raw.strip(), "source_urls": [], "sufficient": False}


async def _fetch_content_items(results: List[SearchResult]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not results:
        return items

    async def fetch_one(result: SearchResult) -> Dict[str, str]:
        content = await async_fetch_page_content(result.url, max_chars=3000)
        return {
            "title": result.title,
            "url": result.url,
            "extracted_text": content,
        }

    fetch_tasks = [fetch_one(item) for item in results[:TOP_FETCH]]
    return await asyncio.gather(*fetch_tasks)


async def researcher_node(state: GraphState) -> Dict[str, Any]:
    section_id = state.get("section_id")
    search_query = (state.get("search_query") or "").strip()
    section_description = state.get("section_description", "")
    if not section_id:
        logger.error("researcher_node.missing_section_id")
        return {"error": "Missing section_id", "research_results": []}

    if not search_query:
        logger.info("researcher_node.no_search_query", section_id=section_id)
        return {"research_results": []}

    try:
        cached = await cache.get_async(search_query)
    except Exception as exc:
        logger.warning("researcher_node.cache_read_failed", error=str(exc), section_id=section_id)
        cached = None

    if cached:
        logger.info("researcher_node.cache_hit", section_id=section_id, query=search_query)
        return {"research_results": [ResearchResult(section_id=section_id, query=search_query, summary=cached.get("summary", ""), source_urls=cached.get("source_urls", []))]}

    try:
        search_results = await search(search_query, num_results=MAX_SEARCH_RESULTS)
        logger.info("researcher_node.search_completed", section_id=section_id, count=len(search_results))
    except Exception as exc:
        logger.error("researcher_node.search_failed", error=str(exc), section_id=section_id)
        search_results = []

    content_items = await _fetch_content_items(search_results)
    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.critical("researcher_node.prompt_missing", path=str(PROMPT_PATH))
        raise

    payload = {
        "search_query": search_query,
        "section_description": section_description,
        "web_content": content_items,
    }
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(payload, indent=2))]

    summary = ""
    source_urls: List[str] = []
    try:
        response = await llm.ainvoke(messages)
        parsed = _parse_llm_response(response.content)
        summary = parsed["summary"]
        source_urls = parsed.get("source_urls", []) or []
    except Exception as exc:
        logger.error("researcher_node.llm_failed", error=str(exc), section_id=section_id)

    research_result = ResearchResult(
        section_id=section_id,
        query=search_query,
        summary=summary,
        source_urls=source_urls,
    )

    try:
        await cache.set_async(search_query, {"summary": summary, "source_urls": source_urls})
    except Exception as exc:
        logger.warning("researcher_node.cache_write_failed", error=str(exc), section_id=section_id)

    try:
        await chroma_store.add_research_async(section_id, search_query, summary, source_urls)
    except Exception as exc:
        logger.warning("researcher_node.chromadb_failed", error=str(exc), section_id=section_id)

    logger.info("researcher_node.completed", section_id=section_id, urls=len(source_urls))
    return {"research_results": [research_result]}
