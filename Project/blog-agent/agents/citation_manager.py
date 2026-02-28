"""
Citation Manager Node — blog-agent (LangGraph)
===============================================
Iterates all SectionDraft objects held in graph state, resolves every
[SOURCE_N] marker to the corresponding URL from the matching ResearchResult,
and returns a deduplicated citation_registry dict.
"""

import re
from typing import Dict

import structlog

from graph.state import GraphState

logger = structlog.get_logger(__name__)


def citation_manager_node(state: GraphState) -> dict:
    """LangGraph node: build a flat citation registry from section drafts.

    For each SectionDraft in state["section_drafts"] the node:
      1. Finds the matching ResearchResult by section_id.
      2. Maps every [SOURCE_N] key → research.source_urls[N-1].
      3. Accumulates results into a deduplicated registry.

    Args:
        state: Current LangGraph graph state containing ``section_drafts``
               and ``research_results``.

    Returns:
        dict with a single key ``"citation_registry"`` whose value is
        ``Dict[str, str]`` mapping citation marker → URL.
    """
    citation_registry: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Build a fast section_id → ResearchResult lookup.
    # ------------------------------------------------------------------
    research_by_section: Dict[str, object] = {
        r.section_id: r for r in state.get("research_results", [])
    }

    log = logger.bind(node="citation_manager_node")
    log.debug(
        "starting citation resolution",
        num_drafts=len(state.get("section_drafts", [])),
        num_research_results=len(research_by_section),
    )

    # ------------------------------------------------------------------
    # Iterate every draft and resolve its citation keys.
    # ------------------------------------------------------------------
    for draft in state.get("section_drafts", []):
        research = research_by_section.get(draft.section_id)

        # Skip if there is no matching research result or no URLs to map.
        if research is None or not research.source_urls:
            log.debug(
                "skipping draft — no matching research or empty source_urls",
                section_id=draft.section_id,
            )
            continue

        for citation_key in draft.citation_keys:
            # Extract the integer index from e.g. "[SOURCE_3]" → 3.
            match = re.search(r"\d+", citation_key)
            if match is None:
                log.warning(
                    "could not parse numeric index from citation key",
                    citation_key=citation_key,
                    section_id=draft.section_id,
                )
                continue

            n = int(match.group())
            url_index = n - 1  # SOURCE_N is 1-based; list is 0-based.

            if url_index < len(research.source_urls):
                url = research.source_urls[url_index]
                citation_registry[citation_key] = url
                log.debug(
                    "mapped citation key",
                    citation_key=citation_key,
                    url=url,
                    section_id=draft.section_id,
                )
            else:
                log.warning(
                    "url_index out of range for source_urls",
                    citation_key=citation_key,
                    url_index=url_index,
                    available=len(research.source_urls),
                    section_id=draft.section_id,
                )

    # ------------------------------------------------------------------
    # Summary logging.
    # ------------------------------------------------------------------
    unique_urls = set(citation_registry.values())
    log.info(
        "citation registry built",
        total_citations=len(citation_registry),
        unique_urls=len(unique_urls),
    )

    return {"citation_registry": citation_registry}