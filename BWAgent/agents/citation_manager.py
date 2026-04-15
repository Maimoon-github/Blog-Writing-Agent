"""Citation manager node for BWAgent."""

import re
from typing import Any, Dict, List, Optional

import structlog

from graph.state import GraphState

logger = structlog.get_logger(__name__)


def citation_manager_node(state: GraphState) -> Dict[str, Any]:
    registry: Dict[str, str] = {}
    drafts = state.get("section_drafts", [])
    research_by_section = {result.section_id: result for result in state.get("research_results", [])}

    if not drafts:
        logger.debug("citation_manager.no_drafts")
        return {"citation_registry": registry}

    for draft in drafts:
        research = research_by_section.get(draft.section_id)
        if not research or not research.source_urls:
            logger.debug("citation_manager.no_research", section_id=draft.section_id)
            continue

        for citation_key in draft.citation_keys:
            url = _resolve_citation(citation_key, research.source_urls)
            if url:
                registry[citation_key] = url
            else:
                logger.warning("citation_manager.unresolved", section_id=draft.section_id, key=citation_key)

    logger.info("citation_manager.completed", registry_size=len(registry))
    return {"citation_registry": registry}


def _resolve_citation(citation_key: str, urls: List[str]) -> Optional[str]:
    match = re.search(r"\d+", citation_key)
    if not match:
        return None

    index = int(match.group()) - 1
    if 0 <= index < len(urls):
        return urls[index]
    return None
