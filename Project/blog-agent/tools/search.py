import re
from dataclasses import dataclass
from typing import List

import structlog
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import SearxSearchWrapper

from app.config import SEARXNG_BASE_URL

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def search(query: str, num_results: int = 5) -> List[SearchResult]:
    """
    Search the web using SearxNG, falling back to DuckDuckGo if SearxNG fails.
    Returns a list of SearchResult objects (empty on total failure).
    """
    # Try SearxNG first
    try:
        searx = SearxSearchWrapper(searx_host=SEARXNG_BASE_URL)
        raw_results = searx.results(query, num_results=num_results)

        if raw_results:
            parsed = []
            for item in raw_results[:num_results]:
                # SearxNG result keys: typically 'title', 'url', 'content' (snippet)
                title = item.get("title", "")
                url = item.get("url", "")
                snippet = item.get("content", "")
                parsed.append(SearchResult(title=title, url=url, snippet=snippet))
            if parsed:
                logger.info("search_used_backend", backend="searxng", query=query, count=len(parsed))
                return parsed

        logger.warning("search_searxng_no_results", query=query)
    except Exception as e:
        logger.warning("search_searxng_failed", query=query, error=str(e))

    # Fallback to DuckDuckGo
    try:
        ddg = DuckDuckGoSearchRun()
        raw_text = ddg.run(query)

        # Parse DuckDuckGo plain text output into structured results
        results = _parse_duckduckgo_results(raw_text, num_results)
        if results:
            logger.info("search_used_backend", backend="duckduckgo", query=query, count=len(results))
            return results

        logger.warning("search_duckduckgo_no_results", query=query)
    except Exception as e:
        logger.error("search_duckduckgo_failed", query=query, error=str(e))

    # Total failure
    logger.error("search_all_backends_failed", query=query)
    return []


def _parse_duckduckgo_results(raw_text: str, num_results: int) -> List[SearchResult]:
    """
    Convert DuckDuckGo plain text output into a list of SearchResult objects.
    Each result block is separated by two newlines.
    Within a block, we look for lines starting with "Title:", "Snippet:", "URL:".
    If not found, we treat the whole block as snippet and try to extract any URL.
    """
    blocks = raw_text.strip().split("\n\n")
    results = []

    for block in blocks:
        if len(results) >= num_results:
            break

        lines = block.strip().split("\n")
        title = ""
        snippet = ""
        url = ""

        for line in lines:
            if line.startswith("Title:"):
                title = line[6:].strip()
            elif line.startswith("Snippet:"):
                snippet = line[8:].strip()
            elif line.startswith("URL:"):
                url = line[4:].strip()
            else:
                # If the line doesn't match known prefixes, append to snippet
                if snippet:
                    snippet += " " + line
                else:
                    snippet = line

        # If no explicit URL, try to find one in the text
        if not url:
            url_match = re.search(r"https?://[^\s]+", block)
            if url_match:
                url = url_match.group(0)

        results.append(SearchResult(title=title, url=url, snippet=snippet))

    return results[:num_results]