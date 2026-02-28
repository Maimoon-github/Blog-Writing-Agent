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


async def _search_searx(query: str, num_results: int) -> list[SearchResult]:
    """
    Query SearxNG asynchronously via SearxSearchWrapper.aresults().

    Returns a (possibly empty) list of SearchResult.
    Raises on any error -- caller handles fallback logic.
    """
    wrapper = _build_searx_wrapper()
    # aresults() is the native async method; it uses aiohttp internally.
    raw: list[dict] = await wrapper.aresults(query, num_results=num_results)
    return _parse_searx_results(raw)


# ---------------------------------------------------------------------------
# Internal: DuckDuckGo fallback backend
# ---------------------------------------------------------------------------

# DuckDuckGoSearchRun.run() returns a plain string where each result looks like:
#   [snippet: ..., title: ..., link: https://example.com, date: ..., source: ...]
# We extract each bracket-delimited block and parse the key->value pairs.

_DDG_BLOCK_RE = re.compile(r"\[([^\[\]]+)\]")
_DDG_FIELD_RE = re.compile(
    r"(?:^|,)\s*"                            # field separator
    r"(snippet|title|link|url)"              # known field names
    r":\s*"
    r"(.*?)(?=(?:,\s*(?:snippet|title|link|url|date|source):|$))",
    re.IGNORECASE | re.DOTALL,
)


def _parse_ddg_text(raw_text: str) -> list[SearchResult]:
    """
    Best-effort parse of the unstructured string returned by DuckDuckGoSearchRun.run().

    Each result is enclosed in square brackets; fields are comma-separated
    key: value pairs. Example block:
        [snippet: Some description, title: Page Title, link: https://example.com, ...]
    """
    results: list[SearchResult] = []

    for block_match in _DDG_BLOCK_RE.finditer(raw_text):
        block = block_match.group(1)
        fields: dict[str, str] = {}

        for m in _DDG_FIELD_RE.finditer(block):
            key = m.group(1).strip().lower()
            val = m.group(2).strip().rstrip(",").strip()
            fields[key] = val

        title = fields.get("title", "")
        url = fields.get("link") or fields.get("url") or ""
        snippet = fields.get("snippet", "")

        if url or title or snippet:
            results.append(SearchResult(title=title, url=url, snippet=snippet))

    return results


async def _search_duckduckgo(query: str, num_results: int) -> list[SearchResult]:
    """
    Query DuckDuckGo via DuckDuckGoSearchRun.

    .run() is synchronous, so we offload it to a thread pool executor to
    remain non-blocking inside an async event loop.
    Raises on any error -- the caller handles final failure logic.
    """
    from langchain_community.tools import DuckDuckGoSearchRun  # noqa: PLC0415

    tool = DuckDuckGoSearchRun()
    raw_text: str = await asyncio.to_thread(tool.run, query)
    results = _parse_ddg_text(raw_text)
    return results[:num_results]


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
        Maximum number of results to return (default: 5).

    Returns
    -------
    List[SearchResult]
        A (possibly empty) list of normalised search results.
    """
    log = logger.bind(query=query, num_results=num_results)

    # 1. Try SearxNG (primary)
    try:
        results = await _search_searx(query, num_results)

        if results:
            log.info("search.backend.used", backend="searxng", result_count=len(results))
            return results

        # Empty result set -- treat as soft failure and fall through to DDG.
        log.warning(
            "search.searxng.empty",
            backend="searxng",
            msg="SearxNG returned zero results; triggering DuckDuckGo fallback",
        )

    except Exception as exc:  # noqa: BLE001
        log.warning(
            "search.searxng.error",
            backend="searxng",
            error=str(exc),
            msg="SearxNG raised an exception; triggering DuckDuckGo fallback",
        )

    # 2. Fallback: DuckDuckGo
    try:
        results = await _search_duckduckgo(query, num_results)
        log.info("search.backend.used", backend="duckduckgo", result_count=len(results))
        return results

    except Exception as exc:  # noqa: BLE001
        log.error(
            "search.total_failure",
            error=str(exc),
            msg="Both SearxNG and DuckDuckGo failed; returning empty list",
        )
        return []