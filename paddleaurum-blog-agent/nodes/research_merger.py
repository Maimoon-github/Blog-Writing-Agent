import logging
from typing import List
from urllib.parse import urlparse

from graph.state import AgentState, ResearchSnippet

logger = logging.getLogger(__name__)

_MAX_SNIPPETS_RETAINED = 15


def _relevance_score(snippet: ResearchSnippet, topic: str) -> int:
    topic_words = set(topic.lower().split())
    text = (snippet["title"] + " " + snippet["snippet"]).lower()
    return sum(1 for word in topic_words if word in text)


def _normalise_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    except Exception:
        return url


async def research_merger_node(state: AgentState) -> dict:
    try:
        raw_snippets: List[ResearchSnippet] = state.get("research_snippets", [])
        topic: str = state.get("topic", "")

        if not raw_snippets:
            logger.warning("Research merger received no snippets.")
            return {"research_snippets": [], "research_sources": []}

        seen_urls: set[str] = set()
        deduplicated: List[ResearchSnippet] = []

        for snippet in raw_snippets:
            norm = _normalise_url(snippet.get("url", ""))
            if norm and norm not in seen_urls:
                seen_urls.add(norm)
                deduplicated.append(snippet)

        ranked = sorted(
            deduplicated,
            key=lambda s: _relevance_score(s, topic),
            reverse=True,
        )
        retained = ranked[:_MAX_SNIPPETS_RETAINED]

        sources: List[str] = [s["url"] for s in retained if s.get("url")]

        logger.info(
            "Research merger: %d raw → %d deduplicated → %d retained",
            len(raw_snippets), len(deduplicated), len(retained),
        )

        return {
            "research_snippets": retained,
            "research_sources":  sources,
            "error":             None,
            "error_node":        None,
        }

    except Exception as exc:
        logger.exception("Research merger failed.")
        return {"error": str(exc), "error_node": "research_merger"}