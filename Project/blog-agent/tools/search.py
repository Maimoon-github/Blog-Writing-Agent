"""
tools/search.py
===============
Unified search interface for the blog-agent project.

Primary backend : SearxNG  (langchain_community.utilities.SearxSearchWrapper)
Fallback backend: DuckDuckGo (langchain_community.tools.DuckDuckGoSearchRun)

The public surface is a single async-compatible coroutine:

    results = await search("best async Python frameworks", num_results=5)

Each element of the returned list is a SearchResult dataclass with fields:
    title   – page/article title
    url     – canonical URL of the result
    snippet – short description / excerpt
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import List

import structlog

from app.config import SEARXNG_BASE_URL

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single, normalised search result."""

    title: str
    url: str
    snippet: str


# ---------------------------------------------------------------------------
# Internal: SearxNG backend
# ---------------------------------------------------------------------------


def _build_searx_wrapper():
    """Lazily construct a SearxSearchWrapper to avoid import-time side-effects."""
    from langchain_community.utilities import SearxSearchWrapper  # noqa: PLC0415

    return SearxSearchWrapper(searx_host=SEARXNG_BASE_URL)


def _parse_searx_results(raw: list[dict]) -> list[SearchResult]:
    """Convert raw SearxNG result dicts -> SearchResult objects."""
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
    """Synchronous SearxNG search – runs inside a thread pool."""
    wrapper = _build_searx_wrapper()
    # .results() is synchronous – exactly what the spec requires
    raw = wrapper.results(query, num_results=num_results)
    return _parse_searx_results(raw)


async def _search_searx(
    query: str, num_results: int, timeout: float = 10.0
) -> list[SearchResult]:
    """
    Query SearxNG with retries (up to 3 attempts) and exponential backoff.

    Runs the synchronous .results() call in a thread pool to avoid blocking.
    """
    max_attempts = 3
    base_delay = 1.0

    for attempt in range(1, max_attempts + 1):
        try:
            # Run in thread pool with overall timeout
            results = await asyncio.wait_for(
                asyncio.to_thread(_search_searx_sync, query, num_results),
                timeout=timeout,
            )

            if results:
                logger.info(
                    "searxng_success",
                    query=query,
                    num_results=len(results),
                    attempt=attempt,
                )
            else:
                logger.debug(
                    "searxng_empty",
                    query=query,
                    attempt=attempt,
                )
            return results

        except asyncio.TimeoutError:
            logger.warning(
                "searxng_timeout",
                query=query,
                attempt=attempt,
                timeout=timeout,
                exc_info=True,
            )
        except Exception as e:
            logger.warning(
                "searxng_error",
                query=query,
                attempt=attempt,
                error=str(e),
                exc_info=True,
            )

        # Exponential backoff before next attempt (except last)
        if attempt < max_attempts:
            delay = base_delay * (2 ** (attempt - 1))
            logger.debug("searxng_retry", query=query, next_attempt=attempt + 1, delay=delay)
            await asyncio.sleep(delay)

    # All attempts failed
    logger.error("searxng_failed_all", query=query, max_attempts=max_attempts)
    return []


# ---------------------------------------------------------------------------
# Internal: DuckDuckGo fallback backend
# ---------------------------------------------------------------------------


def _parse_duckduckgo_response(text: str) -> list[SearchResult]:
    """
    Parse the plain‑text output from DuckDuckGoSearchRun into SearchResult objects.

    Heuristic:
    - Split by newline, each non‑empty line is a result.
    - Extract the first URL (http/https) from the line.
    - Remove that URL from the line.
    - The remaining text is split into title and snippet using common separators:
      ' - ', ' – ', ':' . If no separator found, the first 50 characters become the title.
    - Lines without a URL are discarded.
    """
    results: list[SearchResult] = []
    lines = text.split("\n")
    url_pattern = re.compile(r"https?://[^\s]+")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Find all URLs
        urls = url_pattern.findall(line)
        if not urls:
            continue  # discard lines without a URL

        url = urls[0]  # take the first URL
        # Remove the URL from the line (careful: there may be multiple occurrences)
        # We'll replace the first occurrence with empty string.
        text_part = url_pattern.sub("", line, count=1).strip()

        # Heuristic title/snippet split
        title = ""
        snippet = text_part
        # Try common separators
        for sep in (" - ", " – ", ": "):
            if sep in text_part:
                parts = text_part.split(sep, 1)
                title = parts[0].strip()
                snippet = parts[1].strip()
                break
        if not title and text_part:
            # No separator found: take first 50 chars as title, rest as snippet
            if len(text_part) > 50:
                title = text_part[:50].rsplit(" ", 1)[0] + "…"
                snippet = text_part[50:].strip()
            else:
                title = text_part
                snippet = ""

        results.append(SearchResult(title=title, url=url, snippet=snippet))

    return results


def _search_duckduckgo_sync(query: str) -> list[SearchResult]:
    """Synchronous DuckDuckGo search – runs inside a thread pool."""
    from langchain_community.tools import DuckDuckGoSearchRun  # noqa: PLC0415

    tool = DuckDuckGoSearchRun()
    raw_text = tool.run(query)
    return _parse_duckduckgo_response(raw_text)


async def _search_duckduckgo(
    query: str, num_results: int, timeout: float = 10.0
) -> list[SearchResult]:
    """
    Query DuckDuckGo with a single retry and timeout.

    Runs the synchronous .run() call in a thread pool.
    """
    max_attempts = 2
    base_delay = 1.0

    for attempt in range(1, max_attempts + 1):
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(_search_duckduckgo_sync, query),
                timeout=timeout,
            )

            if results:
                logger.info(
                    "duckduckgo_success",
                    query=query,
                    num_results=len(results),
                    attempt=attempt,
                )
            else:
                logger.debug(
                    "duckduckgo_empty",
                    query=query,
                    attempt=attempt,
                )

            # Trim to requested number
            return results[:num_results]

        except asyncio.TimeoutError:
            logger.warning(
                "duckduckgo_timeout",
                query=query,
                attempt=attempt,
                timeout=timeout,
                exc_info=True,
            )
        except Exception as e:
            logger.warning(
                "duckduckgo_error",
                query=query,
                attempt=attempt,
                error=str(e),
                exc_info=True,
            )

        if attempt < max_attempts:
            logger.debug("duckduckgo_retry", query=query, next_attempt=attempt + 1)
            await asyncio.sleep(base_delay)

    logger.error("duckduckgo_failed_all", query=query, max_attempts=max_attempts)
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def search(query: str, num_results: int = 5) -> List[SearchResult]:
    """
    Unified async search function with automatic backend fallback.

    Tries SearxNG first; falls back to DuckDuckGo if SearxNG raises an
    exception OR returns zero results.  Always returns a list (possibly
    empty) and never raises.

    Parameters
    ----------
    query:
        The search query string.
    num_results:
        Maximum number of results to return (default: 5). If less than 1, treated as 1.

    Returns
    -------
    List[SearchResult]
        A (possibly empty) list of normalised search results.
    """
    log = logger.bind(query=query, num_results=num_results)

    # Edge case: empty query
    if not query or not query.strip():
        log.debug("search.empty_query")
        return []

    # Normalise num_results
    if num_results < 1:
        num_results = 1
        log.debug("search.num_results_normalised", num_results=num_results)

    # 1. Try SearxNG (primary)
    log.debug("search.starting_primary", backend="searxng")
    results = await _search_searx(query, num_results)

    if results:
        log.info("search.backend_used", backend="searxng", result_count=len(results))
        return results

    # Primary returned zero or failed – fallback
    log.warning("search.fallback", backend="duckduckgo", reason="SearxNG returned zero or failed")
    results = await _search_duckduckgo(query, num_results)

    if results:
        log.info("search.backend_used", backend="duckduckgo", result_count=len(results))
    else:
        log.error("search.total_failure", msg="Both backends returned no results")
        # Keep empty list

    return results