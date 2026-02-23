# import asyncio
# import logging
# import re
# from typing import List, Optional, Tuple
# from urllib.parse import urlparse

# import aiohttp

# from graph.state import AgentState

# logger = logging.getLogger(__name__)

# _VALIDATION_TIMEOUT = aiohttp.ClientTimeout(total=8)


# async def _validate_url(session: aiohttp.ClientSession, url: str) -> Tuple[str, bool]:
#     try:
#         async with session.head(url, allow_redirects=True, timeout=_VALIDATION_TIMEOUT) as resp:
#             return url, resp.status < 400
#     except Exception:
#         try:
#             async with session.get(url, timeout=_VALIDATION_TIMEOUT) as resp:
#                 return url, resp.status < 400
#         except Exception:
#             return url, False


# def _extract_domain(url: str) -> str:
#     try:
#         return urlparse(url).netloc.removeprefix("www.")
#     except Exception:
#         return url


# def _inject_citations(article: str, sources: List[str]) -> str:
#     if not sources:
#         return article

#     # Replace [IMAGE: ...] and [INTERNAL LINK: ...] placeholders with cleaner markers
#     # so citation numbers don't conflict with those brackets.
#     # Insert numbered citation superscripts after sentences that contain snippets
#     # matching known source domains.
#     lines = article.splitlines()
#     result_lines = []

#     for line in lines:
#         result_lines.append(line)

#     # Append reference list
#     result_lines.append("\n\n---\n\n## References\n")
#     for i, url in enumerate(sources, 1):
#         domain = _extract_domain(url)
#         result_lines.append(f"[{i}] {domain} — <{url}>")

#     return "\n".join(result_lines)


# async def citation_formatter_node(state: AgentState) -> dict:
#     try:
#         draft: str = state.get("draft_article") or ""
#         sources: List[str] = state.get("research_sources") or []

#         if not sources:
#             logger.info("Citation formatter: no sources to process.")
#             return {
#                 "formatted_article": draft,
#                 "error":             None,
#                 "error_node":        None,
#             }

#         async with aiohttp.ClientSession() as session:
#             validation_tasks = [_validate_url(session, url) for url in sources]
#             results: List[Tuple[str, bool]] = await asyncio.gather(*validation_tasks)

#         valid_sources = [url for url, ok in results if ok]
#         invalid_count = len(sources) - len(valid_sources)

#         if invalid_count:
#             logger.warning("Citation formatter: %d/%d URLs failed validation and were excluded.",
#                            invalid_count, len(sources))

#         formatted_article = _inject_citations(draft, valid_sources)

#         logger.info("Citation formatter: %d valid sources, reference list appended.", len(valid_sources))

#         return {
#             "formatted_article": formatted_article,
#             "research_sources":  valid_sources,
#             "error":             None,
#             "error_node":        None,
#         }

#     except Exception as exc:
#         logger.exception("Citation formatter failed.")
#         return {"error": str(exc), "error_node": "citation_formatter"}























# 2##############################################################################



















import asyncio
import logging
import re
from typing import List, Optional, Tuple, Set
from urllib.parse import urlparse
from collections import Counter

import aiohttp

from graph.state import AgentState, ResearchSnippet

logger = logging.getLogger(__name__)

_VALIDATION_TIMEOUT = aiohttp.ClientTimeout(total=8)
_SIMILARITY_THRESHOLD = 0.3  # Jaccard similarity threshold for matching sentences


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


def _extract_sentences(text: str) -> List[str]:
    """Split text into sentences (naive approach)."""
    # Simple sentence splitting on .!? followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def _tokenize(text: str) -> Set[str]:
    """Extract lowercase words for similarity calculation."""
    return set(re.findall(r'\b[a-z0-9]+\b', text.lower()))


def _calculate_similarity(text1: str, text2: str) -> float:
    """Jaccard similarity of word sets."""
    words1 = _tokenize(text1)
    words2 = _tokenize(text2)
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def _inject_citations(article: str, sources: List[str], snippets: List[ResearchSnippet]) -> str:
    """
    Insert inline citation markers like [1] after sentences that match source snippets,
    then append a reference list at the end.
    """
    if not sources:
        return article

    # Build a mapping from URL to snippet text and index
    url_to_snippet = {}
    for snip in snippets:
        if snip["url"] in sources:
            url_to_snippet[snip["url"]] = snip["snippet"]

    sentences = _extract_sentences(article)
    cited_sources: Set[str] = set()
    new_sentences = []

    for sent in sentences:
        # Find sources that match this sentence
        matches = []
        for idx, url in enumerate(sources, start=1):
            if url in cited_sources:
                continue  # already cited somewhere
            snippet = url_to_snippet.get(url, "")
            if not snippet:
                continue
            sim = _calculate_similarity(sent, snippet)
            if sim >= _SIMILARITY_THRESHOLD:
                matches.append((idx, url))

        if matches:
            # Mark these sources as cited
            for idx, url in matches:
                cited_sources.add(url)
            # Insert citation markers (e.g., [1][2])
            citation = "".join(f"[{idx}]" for idx, _ in matches)
            new_sentences.append(f"{sent} {citation}")
        else:
            new_sentences.append(sent)

    # If some sources were not cited, we can still append them to the reference list
    # (the user might have added manual citations, or we can note them as general sources)
    # For simplicity, we include all sources in reference list.

    # Append reference list
    new_sentences.append("\n\n---\n\n## References\n")
    for idx, url in enumerate(sources, 1):
        domain = _extract_domain(url)
        new_sentences.append(f"[{idx}] {domain} — <{url}>")

    return "\n".join(new_sentences)


async def citation_formatter_node(state: AgentState) -> dict:
    try:
        draft: str = state.get("draft_article") or ""
        sources: List[str] = state.get("research_sources") or []
        snippets: List[ResearchSnippet] = state.get("research_snippets") or []

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

        formatted_article = _inject_citations(draft, valid_sources, snippets)

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