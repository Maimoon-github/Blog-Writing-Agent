import logging
from typing import Any, Dict

from langgraph.types import interrupt

from graph.state import AgentState

logger = logging.getLogger(__name__)


async def human_review_gate_node(state: AgentState) -> dict:
    """
    Pause the pipeline and surface the article to the human reviewer via Streamlit.

    LangGraph's interrupt() suspends execution at this node and persists the
    current state to the SQLite checkpoint.  The Streamlit UI reads the
    interrupted state, displays the article, and resumes the graph by calling:

        app.invoke(
            Command(resume={"approved": True, "feedback": "Looks great!"}),
            config={"configurable": {"thread_id": session_id}},
        )

    If approved=True  → router sends to publish.
    If approved=False → router sends back to writer with human_feedback in seo_suggestions.
    """
    final_output = state.get("final_output") or {}
    seo_score    = state.get("seo_score") or 0
    is_escalated = state.get("human_review_requested", False)

    review_payload: Dict[str, Any] = {
        "title_tag":         state.get("title_tag", ""),
        "meta_description":  state.get("meta_description", ""),
        "url_slug":          state.get("url_slug", ""),
        "seo_score":         seo_score,
        "word_count":        final_output.get("word_count", 0),
        "citations_count":   len(final_output.get("citations", [])),
        "images_resolved":   sum(1 for img in (state.get("image_manifest") or []) if img.get("url")),
        "revision_iteration": state.get("revision_iteration", 0),
        "escalated_due_to_seo": is_escalated,
        "article_markdown":  final_output.get("markdown", state.get("draft_article", "")),
        "schema_markup":     state.get("schema_markup"),
    }

    logger.info(
        "Human review gate: suspending pipeline (session=%s, seo_score=%d, escalated=%s)",
        state.get("session_id"), seo_score, is_escalated,
    )

    # Suspend here — resumes when the Streamlit UI calls app.invoke(Command(resume=...))
    human_response: Dict[str, Any] = interrupt(review_payload)

    approved: bool   = bool(human_response.get("approved", False))
    feedback: str    = str(human_response.get("feedback", "")).strip()

    result: dict = {
        "approved":      approved,
        "error":         None,
        "error_node":    None,
    }

    if not approved and feedback:
        # Prepend human feedback to seo_suggestions so the writer node sees it
        existing = list(state.get("seo_suggestions") or [])
        human_note = f"[Human reviewer feedback]: {feedback}"
        result["seo_suggestions"]    = [human_note] + existing
        result["revision_iteration"] = 0  # reset loop counter for the manual revision cycle

    logger.info(
        "Human review gate: reviewer decision=%s (session=%s)",
        "APPROVED" if approved else "REVISION REQUESTED",
        state.get("session_id"),
    )

    return result