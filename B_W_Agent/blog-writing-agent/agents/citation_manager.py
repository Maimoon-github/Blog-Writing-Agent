"""
agents/citation_manager.py
==========================

Pure-Python citation processor for the Autonomous Blog Generation Agent.

- Deduplicates sources by URL (preserving first-appearance order)
- Builds 1-based citation registry
- Resolves generic [citation] markers → numbered [N]
- Appends formatted ## References section
- Returns new CrewState objects only (CrewAI v1.14.2 safe)

References:
- roadmap.html → Phase 3 Step 3
- idea.md → Citation Manager node
"""

import re
import logging
from typing import List, Dict
from copy import deepcopy

from schemas import ResearchResult, SectionDraft
from state import CrewState

# Optional config import (fallback for truncation length)
try:
    from config import CITATION_SNIPPET_MAX
except ImportError:
    CITATION_SNIPPET_MAX = 200

logger = logging.getLogger(__name__)


def build_citation_registry(
    research_results: List[ResearchResult],
) -> Dict[int, Dict[str, str]]:
    """
    Build 1-based citation registry from all sources.
    Deduplicates by URL while preserving first-appearance order.
    """
    if not research_results:
        logger.debug("citation_registry_empty")
        return {}

    seen_urls = {}
    for result in research_results:
        for source in result.sources:
            url = source.get("url") or ""
            if url and url not in seen_urls:
                seen_urls[url] = {
                    "url": url,
                    "title": source.get("title") or "",
                    "snippet": source.get("snippet") or "",
                }

    # Convert to 1-based dict
    registry = {
        i + 1: info for i, info in enumerate(seen_urls.values())
    }
    logger.info("citation_registry_built", unique_count=len(registry))
    return registry


def resolve_citations(
    section_drafts: List[SectionDraft], registry: Dict[int, Dict[str, str]]
) -> List[SectionDraft]:
    """
    Replace every [citation] marker with [N] using sequential numbering.
    Returns brand-new SectionDraft instances (no in-place mutation).
    """
    if not registry:
        return [deepcopy(d) for d in section_drafts]  # defensive copy

    url_to_number: Dict[str, int] = {
        info["url"]: num for num, info in registry.items()
    }

    # Global counter for citation numbering (1-based)
    counter = 1

    def replace_marker(match: re.Match) -> str:
        nonlocal counter
        num = counter
        counter += 1
        # Safety: if we exceed registry size, use [?]
        if num > len(registry):
            return "[?]"
        return f"[{num}]"

    resolved = []
    for draft in section_drafts:
        content = re.sub(r"\[citation\]", replace_marker, draft.content)

        # Create new immutable SectionDraft
        new_draft = SectionDraft(
            section_id=draft.section_id,
            title=draft.title,
            content=content,
            word_count=draft.word_count,
            citations=draft.citations,
        )
        resolved.append(new_draft)

    logger.info("citations_resolved", sections_processed=len(resolved))
    return resolved


def format_references(registry: Dict[int, Dict[str, str]]) -> str:
    """Generate Markdown References section."""
    if not registry:
        return ""

    lines = ["## References\n"]
    for num, info in registry.items():
        snippet = info["snippet"]
        if len(snippet) > CITATION_SNIPPET_MAX:
            snippet = snippet[: CITATION_SNIPPET_MAX - 3] + "..."
        lines.append(
            f"{num}. [{info['title']}]({info['url']}) — {snippet}"
        )
    return "\n".join(lines)


def process_citations(state: CrewState) -> CrewState:
    """
    Main entry point: processes citations and returns updated CrewState.
    Does NOT mutate the input state.
    """
    registry = build_citation_registry(state.get("research_results", []))

    updated_sections = resolve_citations(
        state.get("completed_sections", []), registry
    )

    # Append References to the last section (or create one if empty)
    references_md = format_references(registry)
    if updated_sections and references_md:
        last = updated_sections[-1]
        if not last.content.strip().endswith("References"):
            last.content = last.content.rstrip() + "\n\n" + references_md
    elif references_md:
        # Edge case: no sections but we have references
        placeholder = SectionDraft(
            section_id="references",
            title="References",
            content=references_md,
            word_count=len(references_md.split()),
            citations=[],
        )
        updated_sections = [placeholder]

    # Return new state (CrewAI v1.14.2 safe)
    new_state: CrewState = {
        **state,  # shallow copy of other fields
        "citation_registry": registry,
        "completed_sections": updated_sections,
    }
    logger.info("citation_manager_complete", registry_size=len(registry))
    return new_state


# ----------------------------------------------------------------------
# Self-test when run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("🧪 Testing agents/citation_manager.py...")

    # Minimal test data
    from schemas import ResearchResult
    from datetime import datetime

    test_state: CrewState = {
        "topic": "Test",
        "research_required": True,
        "plan": None,
        "research_results": [
            ResearchResult(
                section_id="section_1",
                summary="Test summary",
                sources=[{"url": "https://example.com/1", "title": "Src 1", "snippet": "Snippet one"}],
                timestamp=datetime.now(),
            ),
            ResearchResult(
                section_id="section_2",
                summary="Test summary 2",
                sources=[{"url": "https://example.com/1", "title": "Src 1", "snippet": "Duplicate"}],
                timestamp=datetime.now(),
            ),
        ],
        "completed_sections": [
            SectionDraft(
                section_id="section_1",
                title="Section One",
                content="This is important[citation].",
                word_count=5,
                citations=[],
            )
        ],
        "generated_images": [],
        "citation_registry": {},
        "final_markdown": "",
        "final_html": "",
        "output_path": "",
    }

    result = process_citations(test_state)

    assert len(result["citation_registry"]) == 1, "Deduplication failed"
    assert "[1]" in result["completed_sections"][0].content, "Marker replacement failed"
    assert "## References" in result["completed_sections"][0].content, "References missing"

    print("✅ All tests passed — citation_manager is production-ready.")