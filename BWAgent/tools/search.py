"""Unified search tools for BWAgent."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import List

import structlog
from config.settings import SEARXNG_BASE_URL

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def _build_searx_wrapper():
    from langchain_community.utilities import SearxSearchWrapper

    return SearxSearchWrapper(searx_host=SEARXNG_BASE_URL)


def _parse_searx_results(raw: list[dict]) -> list[SearchResult]:
    parsed: list[SearchResult] = []
    for item in raw:
        parsed.append(
            SearchResult(
                title=item.get("title") or "",
                url=item.get("link") or item.get("url") or "",
                snippet=item.get("snippet") or "",
            )
        )
    return parsed


def _search_searx_sync(query: str, num_results: int) -> list[SearchResult]:
    wrapper = _build_searx_wrapper()
    raw = wrapper.results(query, num_results=num_results)
    return _parse_searx_results(raw)


async def _search_searx(query: str, num_results: int, timeout: float = 10.0) -> list[SearchResult]:
    max_attempts = 3
    base_delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(_search_searx_sync, query, num_results),
                timeout=timeout,
            )
            if results:
                logger.info("search.searx_success", query=query, count=len(results), attempt=attempt)
            return results
        except asyncio.TimeoutError:
            logger.warning("search.searx_timeout", query=query, attempt=attempt)
        except Exception as exc:
            logger.warning("search.searx_error", query=query, attempt=attempt, error=str(exc))
        if attempt < max_attempts:
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
    logger.error("search.searx_failed", query=query)
    return []


def _parse_duckduckgo_response(text: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    lines = text.split("\n")
    url_pattern = re.compile(r"https?://[^\s]+")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        urls = url_pattern.findall(line)
        if not urls:
            continue
        url = urls[0]
        text_part = url_pattern.sub("", line, count=1).strip()
        title = ""
        snippet = text_part
        for sep in (" - ", " – ", ": "):
            if sep in text_part:
                parts = text_part.split(sep, 1)
                title = parts[0].strip()
                snippet = parts[1].strip()
                break
        if not title and text_part:
            if len(text_part) > 50:
                title = text_part[:50].rsplit(" ", 1)[0] + "…"
                snippet = text_part[50:].strip()
            else:
                title = text_part
                snippet = ""
        results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results


def _search_duckduckgo_sync(query: str) -> list[SearchResult]:
    from langchain_community.tools import DuckDuckGoSearchRun

    tool = DuckDuckGoSearchRun()
    raw_text = tool.run(query)
    return _parse_duckduckgo_response(raw_text)


async def _search_duckduckgo(query: str, num_results: int, timeout: float = 10.0) -> list[SearchResult]:
    max_attempts = 2
    base_delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(_search_duckduckgo_sync, query),
                timeout=timeout,
            )
            if results:
                logger.info("search.duckduckgo_success", query=query, count=len(results), attempt=attempt)
            return results[:num_results]
        except asyncio.TimeoutError:
            logger.warning("search.duckduckgo_timeout", query=query, attempt=attempt)
        except Exception as exc:
            logger.warning("search.duckduckgo_error", query=query, attempt=attempt, error=str(exc))
        if attempt < max_attempts:
            await asyncio.sleep(base_delay)
    logger.error("search.duckduckgo_failed", query=query)
    return []


async def search(query: str, num_results: int = 5) -> List[SearchResult]:
    if not query or not query.strip():
        logger.debug("search.empty_query")
        return []
    if num_results < 1:
        num_results = 1
    logger.debug("search.start", query=query, num_results=num_results)
    results = await _search_searx(query, num_results)
    if results:
        logger.info("search.backend", backend="searxng", count=len(results))
        return results
    logger.warning("search.fallback", backend="duckduckgo")
    results = await _search_duckduckgo(query, num_results)
    if results:
        logger.info("search.backend", backend="duckduckgo", count=len(results))
    else:
        logger.error("search.failed", query=query)
    return results
