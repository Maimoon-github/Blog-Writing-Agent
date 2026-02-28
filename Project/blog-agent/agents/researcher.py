"""LangGraph researcher node for parallel section research.

This module implements an asynchronous worker node that processes a single blog section:
- Checks disk cache for existing research
- Performs web search and fetches page content
- Synthesizes a research summary using an LLM
- Stores results in cache and ChromaDB
- Returns a ResearchResult for merging into the graph state

The node is designed to run concurrently for multiple sections using LangGraph's Send().
All I/O operations are asynchronous to maximize parallelism.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import GraphState, ResearchResult
from app.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from tools.search import search, SearchResult
from tools.web_fetcher import fetch_page_content
from memory.cache import cache
from memory.chroma_store import chroma_store

logger = structlog.get_logger(__name__)

# Constants
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "researcher_prompt.txt"
MAX_SEARCH_RESULTS = 5
TOP_N_FETCH = 3
FETCH_TIMEOUT = 10  # seconds per page


async def researcher_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph worker node that researches a single blog section asynchronously.

    Expected state fields (for this parallel instance):
        - section_id: str
        - search_query: Optional[str]
        - section_description: str

    Args:
        state: GraphState containing data for one section.

    Returns:
        Dictionary with:
            - "research_results": list containing one ResearchResult (or empty)
            - optionally "error" if critical failure occurred.

    Note:
        This node is designed to be called concurrently via Send().
        The returned list is merged into the main state using operator.add.
    """
    # 1. Extract section data
    section_id = state.get("section_id")
    search_query = state.get("search_query", "").strip()
    section_description = state.get("section_description", "")

    if not section_id:
        logger.error("researcher_node.missing_section_id")
        return {"error": "Missing section_id", "research_results": []}

    if not search_query:
        logger.info(
            "researcher_node.no_search_query",
            section_id=section_id,
            reason="No search query provided",
        )
        return {"research_results": []}

    # 2. Check cache (run in thread to avoid blocking)
    try:
        cached = await asyncio.to_thread(cache.get, search_query)
        if cached is not None:
            logger.info(
                "researcher_node.cache_hit",
                section_id=section_id,
                search_query=search_query,
            )
            # Assume cached is a dict with "summary" and "source_urls"
            result = ResearchResult(
                section_id=section_id,
                query=search_query,
                summary=cached.get("summary", ""),
                source_urls=cached.get("source_urls", []),
            )
            return {"research_results": [result]}
    except Exception as e:
        # Cache error – log warning and continue (proceed as if cache miss)
        logger.warning(
            "researcher_node.cache_error",
            section_id=section_id,
            search_query=search_query,
            error=str(e),
        )

    # 3. Perform web search (if search is async, use await; otherwise run in thread)
    search_results: List[SearchResult] = []
    try:
        # Assuming search is synchronous; if it has an async version, replace with await
        search_results = await asyncio.to_thread(
            search, query=search_query, num_results=MAX_SEARCH_RESULTS
        )
        logger.info(
            "researcher_node.search_completed",
            section_id=section_id,
            result_count=len(search_results),
        )
    except Exception as e:
        logger.error(
            "researcher_node.search_failed",
            section_id=section_id,
            search_query=search_query,
            error=str(e),
        )
        # Proceed with empty search results (LLM will use only description)

    # 4. Fetch page content concurrently for top N results
    content_items = await _fetch_content_items(search_results[:TOP_N_FETCH], section_id)

    # 5. Load system prompt
    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.critical(
            "researcher_node.prompt_file_missing",
            path=str(PROMPT_PATH),
        )
        raise  # Cannot function without prompt

    # 6. Build user message as JSON-serializable object
    user_payload = {
        "search_query": search_query,
        "section_description": section_description,
        "content_items": content_items,  # list of dicts with title, url, extracted_text
    }
    user_message = json.dumps(user_payload, indent=2)

    # 7. Invoke LLM asynchronously
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    summary = ""
    source_urls = []
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        )
        raw_content = response.content
        summary, source_urls = _parse_llm_response(raw_content, section_id)
    except Exception as e:
        logger.error(
            "researcher_node.llm_invocation_failed",
            section_id=section_id,
            search_query=search_query,
            error=str(e),
        )
        # summary and source_urls remain empty/default

    # 8. Build ResearchResult
    research_result = ResearchResult(
        section_id=section_id,
        query=search_query,
        summary=summary,
        source_urls=source_urls,
    )

    # 9. Store in cache (run in thread to avoid blocking)
    try:
        await asyncio.to_thread(
            cache.set,
            search_query,
            {"summary": summary, "source_urls": source_urls},
        )
    except Exception as e:
        logger.error(
            "researcher_node.cache_write_failed",
            section_id=section_id,
            search_query=search_query,
            error=str(e),
        )

    # 10. Store in ChromaDB (run in thread to avoid blocking)
    try:
        await asyncio.to_thread(
            chroma_store.add_research,
            section_id=section_id,
            query=search_query,
            summary=summary,
            source_urls=source_urls,
        )
    except Exception as e:
        logger.error(
            "researcher_node.chromadb_write_failed",
            section_id=section_id,
            search_query=search_query,
            error=str(e),
        )

    # 11. Log success
    logger.info(
        "researcher_node.completed",
        section_id=section_id,
        summary_length=len(summary),
        source_count=len(source_urls),
    )

    return {"research_results": [research_result]}


async def _fetch_content_items(
    search_results: List[SearchResult], section_id: str
) -> List[Dict[str, str]]:
    """
    Fetch page content concurrently for the given search results using asyncio.

    Args:
        search_results: List of SearchResult objects.
        section_id: Section ID for logging.

    Returns:
        List of dicts with keys: title, url, extracted_text.
    """
    if not search_results:
        return []

    async def fetch_one(result: SearchResult) -> Dict[str, str]:
        """Fetch a single page, return dict with title, url, extracted_text."""
        try:
            # Assuming fetch_page_content is synchronous; if async exists, replace with await
            extracted_text = await asyncio.to_thread(
                fetch_page_content, result.url, timeout=FETCH_TIMEOUT
            )
            if not extracted_text:
                logger.warning(
                    "researcher_node.fetch_empty",
                    section_id=section_id,
                    url=result.url,
                )
        except Exception as e:
            logger.warning(
                "researcher_node.fetch_failed",
                section_id=section_id,
                url=result.url,
                error=str(e),
            )
            extracted_text = ""
        return {
            "title": result.title,
            "url": result.url,
            "extracted_text": extracted_text,
        }

    # Run all fetch tasks concurrently
    tasks = [fetch_one(result) for result in search_results]
    content_items = await asyncio.gather(*tasks, return_exceptions=False)
    return content_items


def _parse_llm_response(raw_content: str, section_id: str) -> tuple[str, List[str]]:
    """
    Parse LLM response expecting JSON with "summary" and "source_urls" keys.

    Tries:
      1. Direct json.loads
      2. Regex extraction of JSON object
    Falls back to empty summary and empty list if both fail.

    Args:
        raw_content: Raw string from LLM.
        section_id: For logging.

    Returns:
        Tuple (summary, source_urls) where source_urls is a list of strings.
    """
    # Helper to extract from parsed dict
    def extract(data: Dict[str, Any]) -> tuple[str, List[str]]:
        summary = data.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary) if summary is not None else ""
        source_urls = data.get("source_urls", [])
        if not isinstance(source_urls, list):
            source_urls = []
        # Ensure all elements are strings
        source_urls = [str(u) for u in source_urls if u is not None]
        return summary, source_urls

    # Strategy 1: direct JSON parse
    try:
        data = json.loads(raw_content)
        if isinstance(data, dict):
            return extract(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: regex extraction of first JSON object
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return extract(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: fallback
    logger.warning(
        "researcher_node.parse_fallback",
        section_id=section_id,
        response_snippet=raw_content[:200],
    )
    return "", []