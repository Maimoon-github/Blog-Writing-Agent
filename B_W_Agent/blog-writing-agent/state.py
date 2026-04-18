"""
Single source of truth for the shared CrewState TypedDict in the
Autonomous Blog Generation Agent (8-agent parallel pipeline).

This is the only object passed between agents and graph.py.
It enables async orchestration (asyncio.gather), partial updates,
and SQLite checkpointing (CrewAI v1.14.2).

References:
- roadmap.html → Phase 2 Step 4 (state.py file card)
- idea.md → CrewState as working memory for the full pipeline
"""

from typing import TypedDict, List, Optional
from schemas import BlogPlan, ResearchResult, SectionDraft, ImageResult


class CrewState(TypedDict):
    """Shared state object for the entire pipeline."""
    # Input (Streamlit → Router)
    topic: str
    research_required: bool

    # Router + Planner output
    plan: Optional[BlogPlan]

    # Worker outputs (merged after parallel crews)
    research_results: List[ResearchResult]
    completed_sections: List[SectionDraft]
    generated_images: List[ImageResult]

    # Citation Manager + Reducer outputs
    citation_registry: dict
    final_markdown: str
    final_html: str
    output_path: str


__all__ = ["CrewState"]

# Example partial update pattern used by every agent:
# def some_agent_node(state: CrewState) -> dict:
#     return {"research_results": [new_result]}   # only updated fields
#
# CrewAI v1.14.2 merges these dicts safely and supports
# SQLite checkpoint/resume from any intermediate state.