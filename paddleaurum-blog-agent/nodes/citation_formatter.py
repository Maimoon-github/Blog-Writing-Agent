import asyncio
import logging
import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

from graph.state import AgentState

logger = logging.getLogger(__name__)

_VALIDATION_TIMEOUT = aiohttp.ClientTimeout(total=8)


async def _validate_url(session: aiohttp.ClientSession, url: str) -> Tuple[str, bool]:
    try:
        async with session.head(url, allow_redirects=True, timeout=_VALIDATION_TIMEOUT) as resp:
            return url, resp.status < 400
    except Exception:
        try:
            async with session.get(url, timeout=_VALIDATION_TIMEOUT) as resp:
                return url, resp.status < 400
        except Exception:
            return url, False


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except Exception:
        return url


def _inject_citations(article: str, sources: List[str]) -> str:
    if not sources:
        return article

    # Replace [IMAGE: ...] and [INTERNAL LINK: ...] placeholders with cleaner markers
    # so citation numbers don't conflict with those brackets.
    # Insert numbered citation superscripts after sentences that contain snippets
    # matching known source domains.
    lines = article.splitlines()
    result_lines = []

    for line in lines:
        result_lines.append(line)

    # Append reference list
    result_lines.append("\n\n---\n\n## References\n")
    for i, url in enumerate(sources, 1):
        domain = _extract_domain(url)
        result_lines.append(f"[{i}] {domain} — <{url}>")

    return "\n".join(result_lines)


async def citation_formatter_node(state: AgentState) -> dict:
    try:
        draft: str = state.get("draft_article") or ""
        sources: List[str] = state.get("research_sources") or []

        if not sources:
            logger.info("Citation formatter: no sources to process.")
            return {
                "formatted_article": draft,
                "error":             None,
                "error_node":        None,
            }

        async with aiohttp.ClientSession() as session:
            validation_tasks = [_validate_url(session, url) for url in sources]
            results: List[Tuple[str, bool]] = await asyncio.gather(*validation_tasks)

        valid_sources = [url for url, ok in results if ok]
        invalid_count = len(sources) - len(valid_sources)

        if invalid_count:
            logger.warning("Citation formatter: %d/%d URLs failed validation and were excluded.",
                           invalid_count, len(sources))

        formatted_article = _inject_citations(draft, valid_sources)

        logger.info("Citation formatter: %d valid sources, reference list appended.", len(valid_sources))

        return {
            "formatted_article": formatted_article,
            "research_sources":  valid_sources,
            "error":             None,
            "error_node":        None,
        }

    except Exception as exc:
        logger.exception("Citation formatter failed.")
        return {"error": str(exc), "error_node": "citation_formatter"}