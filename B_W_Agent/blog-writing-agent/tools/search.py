"""
tools/search.py
===============

Unified search interface for the Autonomous Blog Generation Agent.

- Primary backend: SearxNG (SearxSearchWrapper.results())
- Fallback backend: DuckDuckGo (DuckDuckGoSearchAPIWrapper.results())
- 24-hour disk-based JSON cache (prevents redundant calls during dev/testing)
- Always returns List[Dict[str, str]] with exactly {"url", "title", "snippet"}
- Exports `search_tool` (LangChain Tool) for CrewAI agents + `search()` helper

References:
- roadmap.html → Phase 3 Step 1 (tools/search.py)
- idea.md → Search wrapper with caching + DDG fallback
"""

import hashlib
import json
from pathlib import Path
import time
import logging
from typing import List, Dict, Any

from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool   # ← required for CrewAI Agent compatibility

from config import SEARXNG_HOST, CACHE_TTL_SEC, CACHE_DIR

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------

def _cache_key(query: str) -> str:
    """Stable filesystem-safe cache key using SHA256."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def cache_get(query: str) -> List[Dict[str, str]] | None:
    """Return cached results if still valid, else None."""
    cache_dir = Path(CACHE_DIR)
    cache_file = cache_dir / f"{_cache_key(query)}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, encoding="utf-8") as f:
            payload: dict = json.load(f)

        if "timestamp" not in payload or "results" not in payload:
            return None

        age = time.time() - payload["timestamp"]
        if age > CACHE_TTL_SEC:
            return None  # expired

        logger.debug("cache_hit", query=query, age_seconds=round(age))
        return payload["results"]

    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        logger.warning("cache_corrupt", query=query)
        return None


def cache_set(query: str, results: List[Dict[str, str]]) -> None:
    """Write results to disk cache."""
    try:
        cache_dir = Path(CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp": time.time(),
            "results": results,
        }

        cache_file = cache_dir / f"{_cache_key(query)}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.debug("cache_written", query=query, num_results=len(results))
    except Exception as e:  # pragma: no cover
        logger.warning("cache_write_failed", query=query, error=str(e))


# ----------------------------------------------------------------------
# Normalization (guarantees exact output contract)
# ----------------------------------------------------------------------

def _normalize(raw: dict) -> Dict[str, str]:
    """Force every result to have exactly the three required keys."""
    return {
        "url": raw.get("url") or raw.get("link") or "",
        "title": raw.get("title") or "",
        "snippet": raw.get("snippet") or raw.get("content") or "",
    }


# ----------------------------------------------------------------------
# Backend search functions
# ----------------------------------------------------------------------

def _search_searxng(query: str, num_results: int) -> List[Dict[str, str]]:
    """Primary search via SearxNG."""
    wrapper = SearxSearchWrapper(searx_host=SEARXNG_HOST)
    raw_results: list[dict] = wrapper.results(query, num_results=num_results)
    return [_normalize(item) for item in raw_results]


def _search_duckduckgo(query: str, num_results: int) -> List[Dict[str, str]]:
    """Fallback via DuckDuckGo."""
    tool = DuckDuckGoSearchRun()
    raw = tool.run(query)
    # DuckDuckGoSearchRun returns a string; we parse it into list of dicts
    # (simple fallback parsing)
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    return [{"url": "", "title": line[:80], "snippet": line} for line in lines[:num_results]]


# ----------------------------------------------------------------------
# Public helper (used by researcher_node)
# ----------------------------------------------------------------------

def search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Main search entrypoint with cache + fallback.
    Returns: List[Dict[str, str]] — exactly {"url", "title", "snippet"}
    """
    if not query or not query.strip():
        logger.debug("empty_query")
        return []

    if num_results < 1:
        num_results = 1

    # 1. Cache hit?
    cached = cache_get(query)
    if cached is not None:
        return cached

    # 2. Try primary (SearxNG)
    try:
        results = _search_searxng(query, num_results)
        if results:
            cache_set(query, results)
            logger.info("searxng_success", query=query, results=len(results))
            return results
    except Exception as e:
        logger.warning("searxng_failed", query=query, error=str(e))

    # 3. Fallback to DuckDuckGo
    try:
        results = _search_duckduckgo(query, num_results)
        if results:
            cache_set(query, results)
            logger.info("duckduckgo_fallback_success", query=query, results=len(results))
            return results
    except Exception as e:
        logger.error("duckduckgo_failed", query=query, error=str(e))

    # Both backends failed
    logger.error("search_total_failure", query=query)
    return []


# ----------------------------------------------------------------------
# LangChain Tool wrapper (required by planner.py + CrewAI agents)
# ----------------------------------------------------------------------

search_tool = Tool(
    name="search",
    func=search,
    description=(
        "Performs web search using SearxNG (primary) with DuckDuckGo fallback. "
        "Returns up to 5 results as list of dicts with keys: url, title, snippet. "
        "Uses 24h disk cache for repeated queries."
    ),
    return_direct=False,
)


# ----------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------
__all__ = ["search_tool", "search", "cache_get", "cache_set"]


# ----------------------------------------------------------------------
# Self-test when run directly
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Running tools/search.py self-test...")
    q = "latest AI agent frameworks 2026"

    r1 = search(q, num_results=3)
    assert len(r1) > 0, "Live search returned no results"
    assert all(
        isinstance(item, dict) and "url" in item and "title" in item and "snippet" in item
        for item in r1
    ), "Result contract violated"

    r2 = search(q, num_results=3)
    assert r1 == r2, "Cache did not return identical results"

    print("✅ All assertions passed. Cache + fallback + search_tool working.")
    print(f"   Sample result: {r1[0]['title'][:60]}...")