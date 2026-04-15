"""LangGraph node that builds a citation registry from section drafts and research results.

This node consolidates citation markers from all drafted sections, mapping each
[SOURCE_N] placeholder to the actual URL from the corresponding research result.
The resulting registry is stored in the graph state and used later to generate
a bibliography. The node handles missing data, malformed keys, and out-of-range
indices gracefully, logging warnings for any issues.
"""

import re
from typing import Dict, Any, List, Optional

import structlog

from graph.state import GraphState

logger = structlog.get_logger(__name__)


def citation_manager_node(state: GraphState) -> Dict[str, Any]:
    """
    Build a flat citation registry mapping [SOURCE_N] markers to URLs.

    For each SectionDraft in state["section_drafts"], the node:
      1. Finds the matching ResearchResult by section_id.
      2. Maps every [SOURCE_N] key → research.source_urls[N-1].
      3. Accumulates results into a deduplicated registry (last write wins).

    Args:
        state: The current graph state, expected to contain:
               - section_drafts: List[SectionDraft]
               - research_results: List[ResearchResult]

    Returns:
        A dictionary with key "citation_registry" containing the mapping.
        In case of critical errors, returns an empty registry.
    """
    registry: Dict[str, str] = {}

    # Fast lookup: section_id → ResearchResult
    research_by_section = {
        r.section_id: r for r in state.get("research_results", [])
    }

    drafts = state.get("section_drafts", [])
    if not drafts:
        logger.debug("citation_manager.no_drafts", registry_size=0)
        return {"citation_registry": registry}

    log = logger.bind(node="citation_manager_node")
    log.debug(
        "starting_citation_resolution",
        num_drafts=len(drafts),
        num_research_results=len(research_by_section),
    )

    # Statistics for logging
    skipped_drafts = 0
    invalid_keys = 0
    out_of_range_keys = 0

    for draft in drafts:
        research = research_by_section.get(draft.section_id)
        if research is None or not research.source_urls:
            log.debug(
                "skipping_draft_no_research",
                section_id=draft.section_id,
                has_research=research is not None,
                has_urls=bool(research and research.source_urls),
            )
            skipped_drafts += 1
            continue

        for citation_key in draft.citation_keys:
            url = _resolve_citation(citation_key, research.source_urls)
            if url is None:
                # Key was malformed or index out of range
                if not re.search(r"\d+", citation_key):
                    invalid_keys += 1
                else:
                    out_of_range_keys += 1
                continue

            registry[citation_key] = url
            log.debug(
                "mapped_citation_key",
                citation_key=citation_key,
                url=url,
                section_id=draft.section_id,
            )

    # Log summary
    unique_urls = set(registry.values())
    log.info(
        "citation_registry_built",
        registry_size=len(registry),
        unique_urls=len(unique_urls),
        skipped_drafts=skipped_drafts,
        invalid_keys=invalid_keys,
        out_of_range_keys=out_of_range_keys,
    )

    return {"citation_registry": registry}


def _resolve_citation(citation_key: str, source_urls: List[str]) -> Optional[str]:
    """
    Extract the numeric index from a citation key and return the corresponding URL.

    The key is expected to be of the form "[SOURCE_N]" where N is a positive integer.
    If the key is malformed or the index is out of range, returns None and logs a
    warning (handled by caller).

    Args:
        citation_key: The citation marker string, e.g., "[SOURCE_3]".
        source_urls: List of URLs from the corresponding research result.

    Returns:
        The URL if the index is valid, otherwise None.
    """
    match = re.search(r"\d+", citation_key)
    if not match:
        return None

    try:
        idx = int(match.group())
    except ValueError:
        return None

    url_index = idx - 1  # SOURCE_N is 1-based; list is 0-based
    if 0 <= url_index < len(source_urls):
        return source_urls[url_index]

    return None


# Note: This node performs no I/O and is CPU‑bound, so no async version is needed.
# It is thread‑safe because it only reads from the state and returns a new dict.