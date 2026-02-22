import asyncio
import logging
from datetime import datetime, timezone
from typing import List

from duckduckgo_search import DDGS

from graph.state import ResearchSnippet

logger = logging.getLogger(__name__)

# NOTE: graph/state.py must declare research_snippets with an operator.add reducer
# for parallel fan-out to accumulate correctly:
#
#   from typing import Annotated
#   import operator
#   research_snippets: Annotated[List[ResearchSnippet], operator.add]
#
# Without this annotation, LangGraph uses last-write-wins for parallel Send results.

_MAX_SNIPPETS_PER_QUERY = 5
_SEARCH_TIMEOUT_SECONDS = 15


def _ddg_search(query: str) -> List[ResearchSnippet]:
    timestamp = datetime.now(timezone.utc).isoformat()
    snippets: List[ResearchSnippet] = []

    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=_MAX_SNIPPETS_PER_QUERY):
            snippets.append(ResearchSnippet(
                query=query,
                url=result.get("href", ""),
                title=result.get("title", ""),
                snippet=result.get("body", ""),
                retrieved_at=timestamp,
            ))

    return snippets


async def research_worker_node(state: dict) -> dict:
    # state is the merged dict from Send({"query": q, **parent_state})
    query: str = (state.get("query") or "").strip()
    if not query:
        return {"research_snippets": []}

    try:
        snippets = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _ddg_search, query),
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )
        logger.info("Research worker: '%s' → %d snippets", query, len(snippets))
        return {"research_snippets": snippets}

    except asyncio.TimeoutError:
        logger.warning("Research worker timed out for query: '%s'", query)
        return {"research_snippets": []}
    except Exception as exc:
        logger.exception("Research worker failed for query: '%s'", query)
        return {"research_snippets": [], "error": str(exc), "error_node": "research_worker"}