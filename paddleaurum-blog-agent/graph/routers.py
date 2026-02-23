# # graph/routers.py
# # ──────────────────────────────────────────────────────────────────────────────
# # All conditional routing functions for the PaddleAurum blog pipeline.
# #
# # Every function here is a LangGraph router: it receives the current AgentState,
# # inspects one or more fields, and returns either:
# #   • a string  — the name of the single next node to execute, or
# #   • a list    — [Send(...), ...] objects for parallel fan-out (research workers).
# #
# # Routers are registered in graph_builder.py via `add_conditional_edges`.
# # They must never mutate state — read only.
# # ──────────────────────────────────────────────────────────────────────────────

# from __future__ import annotations

# import logging
# from typing import List, Union

# from langgraph.types import Send

# from .state import AgentState

# logger = logging.getLogger(__name__)

# # ── Threshold constants (mirrored from graph_builder settings) ────────────────
# # Defined here so routers are self-contained and testable without importing
# # the full graph_builder module.

# SEO_PASS_THRESHOLD: int = 85  # minimum score to advance past the Auditor

# # Node name constants — single source of truth to avoid typo-driven routing bugs
# _NODE_PLANNER              = "planner"
# _NODE_RESEARCH_WORKER      = "research_worker"
# _NODE_RESEARCH_MERGER      = "research_merger"
# _NODE_KEYWORD_MAPPER       = "keyword_mapper"
# _NODE_OUTLINE_AGENT        = "outline_agent"
# _NODE_HUMAN_OUTLINE_REVIEW = "human_outline_review"
# _NODE_WRITER               = "writer"
# _NODE_SEO_AUDITOR          = "seo_auditor"
# _NODE_REFLECTION           = "reflection"
# _NODE_IMAGE_SELECTOR       = "image_selector"
# _NODE_HUMAN_REVIEW_GATE    = "human_review_gate"
# _NODE_PUBLISH              = "publish"
# _NODE_ERROR_RECOVERY       = "error_recovery"


# # ── 1. After Input Validator ──────────────────────────────────────────────────


# def route_after_validation(state: AgentState) -> str:
#     """
#     Route after the Input Validator node.

#     Decision
#     --------
#     • Any validation error present  →  error_recovery
#     • Input clean                   →  planner

#     This is the first conditional edge; a hard fail here prevents any downstream
#     node from receiving malformed or missing required fields.
#     """
#     if state.get("error"):
#         logger.warning(
#             "Input validation failed (session=%s): %s",
#             state.get("session_id"),
#             state.get("error"),
#         )
#         return _NODE_ERROR_RECOVERY

#     return _NODE_PLANNER


# # ── 2. After Planner / Orchestrator ──────────────────────────────────────────


# def route_after_planner(
#     state: AgentState,
# ) -> Union[str, List[Send]]:
#     """
#     Route after the Planner node.

#     Decision
#     --------
#     • Any error set by Planner      →  error_recovery  (string)
#     • needs_research is False       →  keyword_mapper   (string, skip research)
#     • needs_research is True        →  [Send(...), ...]  (parallel fan-out)

#     The Send fan-out launches one research_worker instance per sub-query
#     concurrently via LangGraph's async Send API.  Workers write into
#     state.research_snippets[], which the research_merger node later aggregates.

#     Note: when returning a list of Send objects the conditional-edge mapping
#     in graph_builder.py must include "research_worker" as a valid destination.
#     """
#     if state.get("error"):
#         logger.error(
#             "Planner raised an error (session=%s): %s",
#             state.get("session_id"),
#             state.get("error"),
#         )
#         return _NODE_ERROR_RECOVERY

#     sub_queries: List[str] = state.get("sub_queries", [])

#     if not state.get("needs_research", True) or not sub_queries:
#         logger.info(
#             "Research skipped — routing directly to keyword_mapper (session=%s)",
#             state.get("session_id"),
#         )
#         return _NODE_KEYWORD_MAPPER

#     logger.info(
#         "Fanning out %d research workers (session=%s)",
#         len(sub_queries),
#         state.get("session_id"),
#     )
#     return [
#         Send(_NODE_RESEARCH_WORKER, {"query": query, **state})
#         for query in sub_queries
#     ]


# # ── 3. After Research Merger ──────────────────────────────────────────────────


# def route_after_research_merger(state: AgentState) -> str:
#     """
#     Route after the Research Merger node.

#     Decision
#     --------
#     • Error set  →  error_recovery
#     • Success    →  keyword_mapper

#     Kept explicit (rather than a fixed edge) so that future logic — e.g.
#     "retry research if too few snippets were found" — can be added here
#     without touching graph_builder.py.
#     """
#     if state.get("error"):
#         return _NODE_ERROR_RECOVERY

#     return _NODE_KEYWORD_MAPPER


# # ── 4. After Outline Agent ────────────────────────────────────────────────────


# def route_after_outline(state: AgentState, require_outline_approval: bool = False) -> str:
#     """
#     Route after the Outline Agent node.

#     Decision
#     --------
#     • Error set                             →  error_recovery
#     • require_outline_approval is True
#         and outline not yet approved        →  human_outline_review
#     • Otherwise                             →  writer

#     The `require_outline_approval` flag is injected by graph_builder.py at
#     graph-compile time via a partial / closure so routers remain pure functions.

#     Example usage in graph_builder.py::

#         from functools import partial
#         workflow.add_conditional_edges(
#             "outline_agent",
#             partial(route_after_outline, require_outline_approval=REQUIRE_OUTLINE_APPROVAL),
#             {
#                 _NODE_ERROR_RECOVERY:       _NODE_ERROR_RECOVERY,
#                 _NODE_HUMAN_OUTLINE_REVIEW: _NODE_HUMAN_OUTLINE_REVIEW,
#                 _NODE_WRITER:               _NODE_WRITER,
#             },
#         )
#     """
#     if state.get("error"):
#         return _NODE_ERROR_RECOVERY

#     if require_outline_approval and not state.get("outline_approved", False):
#         logger.info(
#             "Outline approval gate active — routing to human_outline_review (session=%s)",
#             state.get("session_id"),
#         )
#         return _NODE_HUMAN_OUTLINE_REVIEW

#     return _NODE_WRITER


# # ── 5. After Human Outline Review (optional gate) ────────────────────────────


# def route_after_outline_review(state: AgentState) -> str:
#     """
#     Route after the optional human outline review gate.

#     Decision
#     --------
#     • outline_approved is True   →  writer           (proceed to draft)
#     • outline_approved is False  →  outline_agent    (revise and re-present)
#     """
#     if state.get("outline_approved", False):
#         logger.info(
#             "Outline approved — routing to writer (session=%s)",
#             state.get("session_id"),
#         )
#         return _NODE_WRITER

#     logger.info(
#         "Outline rejected — routing back to outline_agent for revision (session=%s)",
#         state.get("session_id"),
#     )
#     return _NODE_OUTLINE_AGENT


# # ── 6. After SEO Auditor ──────────────────────────────────────────────────────


# def route_after_seo_audit(state: AgentState) -> str:
#     """
#     Route after the SEO Auditor node.  This is the self-correction decision point.

#     Decision
#     --------
#     • Error set                                            →  error_recovery
#     • seo_score >= SEO_PASS_THRESHOLD (85)                 →  image_selector
#     • seo_score < threshold AND iterations < max_iterations →  reflection
#         (Reflection node sends the draft + issues back to the Writer)
#     • seo_score < threshold AND iterations >= max_iterations →  human_review_gate
#         (Escalate: best attempt flagged for manual review)

#     The revision_iteration counter is incremented by the Reflection node, not here.
#     max_iterations defaults to 3 and is configurable per-run via AgentState.
#     """
#     if state.get("error"):
#         return _NODE_ERROR_RECOVERY

#     score:      int = state.get("seo_score") or 0
#     iterations: int = state.get("revision_iteration", 0)
#     max_iter:   int = state.get("max_iterations", 3)

#     if score >= SEO_PASS_THRESHOLD:
#         logger.info(
#             "SEO score %d/%d passed — routing to image_selector (session=%s)",
#             score,
#             SEO_PASS_THRESHOLD,
#             state.get("session_id"),
#         )
#         return _NODE_IMAGE_SELECTOR

#     if iterations < max_iter:
#         logger.info(
#             "SEO score %d < %d, iteration %d/%d — routing to reflection (session=%s)",
#             score,
#             SEO_PASS_THRESHOLD,
#             iterations,
#             max_iter,
#             state.get("session_id"),
#         )
#         return _NODE_REFLECTION

#     # Exhausted all iterations without passing
#     logger.warning(
#         "SEO score %d never reached %d after %d iterations — escalating to human_review_gate (session=%s)",
#         score,
#         SEO_PASS_THRESHOLD,
#         iterations,
#         state.get("session_id"),
#     )
#     return _NODE_HUMAN_REVIEW_GATE


# # ── 7. After Human Review Gate (final approval) ───────────────────────────────


# def route_after_human_review(state: AgentState) -> str:
#     """
#     Route after the final Human Review Gate (Streamlit UI).

#     Decision
#     --------
#     • approved is True   →  publish   (article goes live on WordPress)
#     • approved is False  →  writer    (human requested a revision)

#     When routing back to writer the human reviewer's inline edits are assumed
#     to have been written into state.draft_article by the human_review_gate node.
#     The revision_iteration counter is also reset there so the SEO loop restarts.
#     """
#     if state.get("approved", False):
#         logger.info(
#             "Article approved — routing to publish (session=%s)",
#             state.get("session_id"),
#         )
#         return _NODE_PUBLISH

#     logger.info(
#         "Revision requested by reviewer — routing back to writer (session=%s)",
#         state.get("session_id"),
#     )
#     return _NODE_WRITER


# # ── 8. After Error Recovery ───────────────────────────────────────────────────


# def route_after_error_recovery(state: AgentState) -> str:
#     """
#     Route after the Error Recovery node.

#     Decision matrix
#     ---------------
#     • retry_count < 3 AND error_node is known
#         →  retry the node that originally failed

#     • retry_count >= 3 AND error_node is NON-CRITICAL
#         (image_selector, citation_formatter, schema_generator)
#         →  skip forward; the recovery node inserts placeholders into state

#     • retry_count >= 3 AND error_node is CRITICAL
#         (planner, writer, seo_auditor, keyword_mapper, outline_agent)
#         →  human_review_gate  (escalate, pipeline cannot self-recover)

#     Non-critical nodes are those whose failure can be gracefully degraded:
#     missing images become "[IMAGE PLACEHOLDER]", missing citations are skipped.
#     Critical nodes produce content the rest of the pipeline depends on.
#     """
#     retry_count: int      = state.get("retry_count", 0)
#     error_node:  str | None = state.get("error_node")

#     _NON_CRITICAL_NODES = {
#         _NODE_IMAGE_SELECTOR,
#         "citation_formatter",
#         "schema_generator",
#     }

#     _CRITICAL_NODES = {
#         _NODE_PLANNER,
#         _NODE_RESEARCH_MERGER,
#         _NODE_KEYWORD_MAPPER,
#         _NODE_OUTLINE_AGENT,
#         _NODE_WRITER,
#         _NODE_SEO_AUDITOR,
#         "final_assembler",
#         "publish",
#     }

#     if retry_count < 3 and error_node:
#         logger.info(
#             "Retrying node '%s' (attempt %d/3, session=%s)",
#             error_node,
#             retry_count + 1,
#             state.get("session_id"),
#         )
#         return error_node  # route back to the failing node

#     if error_node in _NON_CRITICAL_NODES:
#         # Determine next node in the linear sequence after the skipped one
#         _skip_forward_map = {
#             _NODE_IMAGE_SELECTOR:   "citation_formatter",
#             "citation_formatter":   "schema_generator",
#             "schema_generator":     "final_assembler",
#         }
#         next_node = _skip_forward_map.get(error_node, "final_assembler")
#         logger.warning(
#             "Non-critical node '%s' failed after 3 retries — skipping to '%s' (session=%s)",
#             error_node,
#             next_node,
#             state.get("session_id"),
#         )
#         return next_node

#     # Critical failure — escalate
#     logger.error(
#         "Critical node '%s' failed after 3 retries — escalating to human_review_gate (session=%s)",
#         error_node,
#         state.get("session_id"),
#     )
#     return _NODE_HUMAN_REVIEW_GATE


# # ── Public routing map (used by graph_builder.py) ────────────────────────────
# # Maps each source node to its router function and the set of valid destinations.
# # graph_builder.py iterates this dict to register all conditional edges cleanly.

# ROUTING_TABLE = {
#     "input_validator": {
#         "fn":      route_after_validation,
#         "targets": {
#             _NODE_ERROR_RECOVERY: _NODE_ERROR_RECOVERY,
#             _NODE_PLANNER:        _NODE_PLANNER,
#         },
#     },
#     "planner": {
#         "fn":      route_after_planner,
#         "targets": {
#             _NODE_ERROR_RECOVERY:  _NODE_ERROR_RECOVERY,
#             _NODE_KEYWORD_MAPPER:  _NODE_KEYWORD_MAPPER,
#             _NODE_RESEARCH_WORKER: _NODE_RESEARCH_WORKER,
#         },
#     },
#     "research_merger": {
#         "fn":      route_after_research_merger,
#         "targets": {
#             _NODE_ERROR_RECOVERY: _NODE_ERROR_RECOVERY,
#             _NODE_KEYWORD_MAPPER: _NODE_KEYWORD_MAPPER,
#         },
#     },
#     "seo_auditor": {
#         "fn":      route_after_seo_audit,
#         "targets": {
#             _NODE_ERROR_RECOVERY:   _NODE_ERROR_RECOVERY,
#             _NODE_IMAGE_SELECTOR:   _NODE_IMAGE_SELECTOR,
#             _NODE_REFLECTION:       _NODE_REFLECTION,
#             _NODE_HUMAN_REVIEW_GATE: _NODE_HUMAN_REVIEW_GATE,
#         },
#     },
#     "human_review_gate": {
#         "fn":      route_after_human_review,
#         "targets": {
#             _NODE_PUBLISH: _NODE_PUBLISH,
#             _NODE_WRITER:  _NODE_WRITER,
#         },
#     },
#     "error_recovery": {
#         "fn":      route_after_error_recovery,
#         # Targets are dynamic (any node can be retried), so graph_builder.py
#         # registers error_recovery edges manually rather than using this map.
#         "targets": {},
#     },
# }



















# @##################################################################






























# graph/routers.py
# ──────────────────────────────────────────────────────────────────────────────
# All conditional routing functions for the PaddleAurum blog pipeline.
#
# Every function here is a LangGraph router: it receives the current AgentState,
# inspects one or more fields, and returns either:
#   • a string  — the name of the single next node to execute, or
#   • a list    — [Send(...), ...] objects for parallel fan-out (research workers).
#
# Routers are registered in graph_builder.py via `add_conditional_edges`.
# They must never mutate state — read only.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import List, Union

from langgraph.types import Send

from .state import AgentState
# ── Import configuration constants from central settings ─────────────────────
from config.settings import SEO_THRESHOLD as SEO_PASS_THRESHOLD

logger = logging.getLogger(__name__)

# Node name constants — single source of truth to avoid typo-driven routing bugs
_NODE_PLANNER              = "planner"
_NODE_RESEARCH_WORKER      = "research_worker"
_NODE_RESEARCH_MERGER      = "research_merger"
_NODE_KEYWORD_MAPPER       = "keyword_mapper"
_NODE_OUTLINE_AGENT        = "outline_agent"
_NODE_HUMAN_OUTLINE_REVIEW = "human_outline_review"
_NODE_WRITER               = "writer"
_NODE_SEO_AUDITOR          = "seo_auditor"
_NODE_REFLECTION           = "reflection"
_NODE_IMAGE_SELECTOR       = "image_selector"
_NODE_HUMAN_REVIEW_GATE    = "human_review_gate"
_NODE_PUBLISH              = "publish"
_NODE_ERROR_RECOVERY       = "error_recovery"


# ── 1. After Input Validator ──────────────────────────────────────────────────


def route_after_validation(state: AgentState) -> str:
    """
    Route after the Input Validator node.

    Decision
    --------
    • Any validation error present  →  error_recovery
    • Input clean                   →  planner

    This is the first conditional edge; a hard fail here prevents any downstream
    node from receiving malformed or missing required fields.
    """
    if state.get("error"):
        logger.warning(
            "Input validation failed (session=%s): %s",
            state.get("session_id"),
            state.get("error"),
        )
        return _NODE_ERROR_RECOVERY

    return _NODE_PLANNER


# ── 2. After Planner / Orchestrator ──────────────────────────────────────────


def route_after_planner(
    state: AgentState,
) -> Union[str, List[Send]]:
    """
    Route after the Planner node.

    Decision
    --------
    • Any error set by Planner      →  error_recovery  (string)
    • needs_research is False       →  keyword_mapper   (string, skip research)
    • needs_research is True        →  [Send(...), ...]  (parallel fan-out)

    The Send fan-out launches one research_worker instance per sub-query
    concurrently via LangGraph's async Send API.  Workers write into
    state.research_snippets[], which the research_merger node later aggregates.

    Note: when returning a list of Send objects the conditional-edge mapping
    in graph_builder.py must include "research_worker" as a valid destination.
    """
    if state.get("error"):
        logger.error(
            "Planner raised an error (session=%s): %s",
            state.get("session_id"),
            state.get("error"),
        )
        return _NODE_ERROR_RECOVERY

    sub_queries: List[str] = state.get("sub_queries", [])

    if not state.get("needs_research", True) or not sub_queries:
        logger.info(
            "Research skipped — routing directly to keyword_mapper (session=%s)",
            state.get("session_id"),
        )
        return _NODE_KEYWORD_MAPPER

    logger.info(
        "Fanning out %d research workers (session=%s)",
        len(sub_queries),
        state.get("session_id"),
    )
    return [
        Send(_NODE_RESEARCH_WORKER, {"query": query, **state})
        for query in sub_queries
    ]


# ── 3. After Research Merger ──────────────────────────────────────────────────


def route_after_research_merger(state: AgentState) -> str:
    """
    Route after the Research Merger node.

    Decision
    --------
    • Error set  →  error_recovery
    • Success    →  keyword_mapper

    Kept explicit (rather than a fixed edge) so that future logic — e.g.
    "retry research if too few snippets were found" — can be added here
    without touching graph_builder.py.
    """
    if state.get("error"):
        return _NODE_ERROR_RECOVERY

    return _NODE_KEYWORD_MAPPER


# ── 4. After Outline Agent ────────────────────────────────────────────────────


def route_after_outline(state: AgentState, require_outline_approval: bool = False) -> str:
    """
    Route after the Outline Agent node.

    Decision
    --------
    • Error set                             →  error_recovery
    • require_outline_approval is True
        and outline not yet approved        →  human_outline_review
    • Otherwise                             →  writer

    The `require_outline_approval` flag is injected by graph_builder.py at
    graph-compile time via a partial / closure so routers remain pure functions.

    Example usage in graph_builder.py::

        from functools import partial
        workflow.add_conditional_edges(
            "outline_agent",
            partial(route_after_outline, require_outline_approval=REQUIRE_OUTLINE_APPROVAL),
            {
                _NODE_ERROR_RECOVERY:       _NODE_ERROR_RECOVERY,
                _NODE_HUMAN_OUTLINE_REVIEW: _NODE_HUMAN_OUTLINE_REVIEW,
                _NODE_WRITER:               _NODE_WRITER,
            },
        )
    """
    if state.get("error"):
        return _NODE_ERROR_RECOVERY

    if require_outline_approval and not state.get("outline_approved", False):
        logger.info(
            "Outline approval gate active — routing to human_outline_review (session=%s)",
            state.get("session_id"),
        )
        return _NODE_HUMAN_OUTLINE_REVIEW

    return _NODE_WRITER


# ── 5. After Human Outline Review (optional gate) ────────────────────────────


def route_after_outline_review(state: AgentState) -> str:
    """
    Route after the optional human outline review gate.

    Decision
    --------
    • outline_approved is True   →  writer           (proceed to draft)
    • outline_approved is False  →  outline_agent    (revise and re-present)
    """
    if state.get("outline_approved", False):
        logger.info(
            "Outline approved — routing to writer (session=%s)",
            state.get("session_id"),
        )
        return _NODE_WRITER

    logger.info(
        "Outline rejected — routing back to outline_agent for revision (session=%s)",
        state.get("session_id"),
    )
    return _NODE_OUTLINE_AGENT


# ── 6. After SEO Auditor ──────────────────────────────────────────────────────


def route_after_seo_audit(state: AgentState) -> str:
    """
    Route after the SEO Auditor node.  This is the self-correction decision point.

    Decision
    --------
    • Error set                                            →  error_recovery
    • seo_score >= SEO_PASS_THRESHOLD (85)                 →  image_selector
    • seo_score < threshold AND iterations < max_iterations →  reflection
        (Reflection node sends the draft + issues back to the Writer)
    • seo_score < threshold AND iterations >= max_iterations →  human_review_gate
        (Escalate: best attempt flagged for manual review)

    The revision_iteration counter is incremented by the Reflection node, not here.
    max_iterations defaults to 3 and is configurable per-run via AgentState.
    """
    if state.get("error"):
        return _NODE_ERROR_RECOVERY

    score:      int = state.get("seo_score") or 0
    iterations: int = state.get("revision_iteration", 0)
    max_iter:   int = state.get("max_iterations", 3)

    if score >= SEO_PASS_THRESHOLD:
        logger.info(
            "SEO score %d/%d passed — routing to image_selector (session=%s)",
            score,
            SEO_PASS_THRESHOLD,
            state.get("session_id"),
        )
        return _NODE_IMAGE_SELECTOR

    if iterations < max_iter:
        logger.info(
            "SEO score %d < %d, iteration %d/%d — routing to reflection (session=%s)",
            score,
            SEO_PASS_THRESHOLD,
            iterations,
            max_iter,
            state.get("session_id"),
        )
        return _NODE_REFLECTION

    # Exhausted all iterations without passing
    logger.warning(
        "SEO score %d never reached %d after %d iterations — escalating to human_review_gate (session=%s)",
        score,
        SEO_PASS_THRESHOLD,
        iterations,
        state.get("session_id"),
    )
    return _NODE_HUMAN_REVIEW_GATE


# ── 7. After Human Review Gate (final approval) ───────────────────────────────


def route_after_human_review(state: AgentState) -> str:
    """
    Route after the final Human Review Gate (Streamlit UI).

    Decision
    --------
    • approved is True   →  publish   (article goes live on WordPress)
    • approved is False  →  writer    (human requested a revision)

    When routing back to writer the human reviewer's inline edits are assumed
    to have been written into state.draft_article by the human_review_gate node.
    The revision_iteration counter is also reset there so the SEO loop restarts.
    """
    if state.get("approved", False):
        logger.info(
            "Article approved — routing to publish (session=%s)",
            state.get("session_id"),
        )
        return _NODE_PUBLISH

    logger.info(
        "Revision requested by reviewer — routing back to writer (session=%s)",
        state.get("session_id"),
    )
    return _NODE_WRITER


# ── 8. After Error Recovery ───────────────────────────────────────────────────


def route_after_error_recovery(state: AgentState) -> str:
    """
    Route after the Error Recovery node.

    Decision matrix
    ---------------
    • retry_count < 3 AND error_node is known
        →  retry the node that originally failed

    • retry_count >= 3 AND error_node is NON-CRITICAL
        (image_selector, citation_formatter, schema_generator)
        →  skip forward; the recovery node inserts placeholders into state

    • retry_count >= 3 AND error_node is CRITICAL
        (planner, writer, seo_auditor, keyword_mapper, outline_agent)
        →  human_review_gate  (escalate, pipeline cannot self-recover)

    Non-critical nodes are those whose failure can be gracefully degraded:
    missing images become "[IMAGE PLACEHOLDER]", missing citations are skipped.
    Critical nodes produce content the rest of the pipeline depends on.
    """
    retry_count: int      = state.get("retry_count", 0)
    error_node:  str | None = state.get("error_node")

    _NON_CRITICAL_NODES = {
        _NODE_IMAGE_SELECTOR,
        "citation_formatter",
        "schema_generator",
    }

    _CRITICAL_NODES = {
        _NODE_PLANNER,
        _NODE_RESEARCH_MERGER,
        _NODE_KEYWORD_MAPPER,
        _NODE_OUTLINE_AGENT,
        _NODE_WRITER,
        _NODE_SEO_AUDITOR,
        "final_assembler",
        "publish",
    }

    if retry_count < 3 and error_node:
        logger.info(
            "Retrying node '%s' (attempt %d/3, session=%s)",
            error_node,
            retry_count + 1,
            state.get("session_id"),
        )
        return error_node  # route back to the failing node

    if error_node in _NON_CRITICAL_NODES:
        # Determine next node in the linear sequence after the skipped one
        _skip_forward_map = {
            _NODE_IMAGE_SELECTOR:   "citation_formatter",
            "citation_formatter":   "schema_generator",
            "schema_generator":     "final_assembler",
        }
        next_node = _skip_forward_map.get(error_node, "final_assembler")
        logger.warning(
            "Non-critical node '%s' failed after 3 retries — skipping to '%s' (session=%s)",
            error_node,
            next_node,
            state.get("session_id"),
        )
        return next_node

    # Critical failure — escalate
    logger.error(
        "Critical node '%s' failed after 3 retries — escalating to human_review_gate (session=%s)",
        error_node,
        state.get("session_id"),
    )
    return _NODE_HUMAN_REVIEW_GATE


# ── Public routing map (used by graph_builder.py) ────────────────────────────
# Maps each source node to its router function and the set of valid destinations.
# graph_builder.py iterates this dict to register all conditional edges cleanly.

ROUTING_TABLE = {
    "input_validator": {
        "fn":      route_after_validation,
        "targets": {
            _NODE_ERROR_RECOVERY: _NODE_ERROR_RECOVERY,
            _NODE_PLANNER:        _NODE_PLANNER,
        },
    },
    "planner": {
        "fn":      route_after_planner,
        "targets": {
            _NODE_ERROR_RECOVERY:  _NODE_ERROR_RECOVERY,
            _NODE_KEYWORD_MAPPER:  _NODE_KEYWORD_MAPPER,
            _NODE_RESEARCH_WORKER: _NODE_RESEARCH_WORKER,
        },
    },
    "research_merger": {
        "fn":      route_after_research_merger,
        "targets": {
            _NODE_ERROR_RECOVERY: _NODE_ERROR_RECOVERY,
            _NODE_KEYWORD_MAPPER: _NODE_KEYWORD_MAPPER,
        },
    },
    "seo_auditor": {
        "fn":      route_after_seo_audit,
        "targets": {
            _NODE_ERROR_RECOVERY:   _NODE_ERROR_RECOVERY,
            _NODE_IMAGE_SELECTOR:   _NODE_IMAGE_SELECTOR,
            _NODE_REFLECTION:       _NODE_REFLECTION,
            _NODE_HUMAN_REVIEW_GATE: _NODE_HUMAN_REVIEW_GATE,
        },
    },
    "human_review_gate": {
        "fn":      route_after_human_review,
        "targets": {
            _NODE_PUBLISH: _NODE_PUBLISH,
            _NODE_WRITER:  _NODE_WRITER,
        },
    },
    "error_recovery": {
        "fn":      route_after_error_recovery,
        # Targets are dynamic (any node can be retried), so graph_builder.py
        # registers error_recovery edges manually rather than using this map.
        "targets": {},
    },
}