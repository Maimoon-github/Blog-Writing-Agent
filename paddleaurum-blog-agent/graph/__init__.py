# graph/__init__.py
# ──────────────────────────────────────────────────────────────────────────────
# Public interface for the `graph` module.
#
# External callers (main.py, scheduler.py, tests) import exclusively from here.
# This keeps internal module paths private and makes refactoring safe.
#
# Usage
# -----
#
#   # Build and run the pipeline
#   from graph import build_graph, run_pipeline, make_initial_state
#
#   app, tracer = build_graph()
#   result = await run_pipeline(topic="pickleball kitchen rules", session_id=uuid4().hex)
#
#   # Work with state types directly
#   from graph import AgentState, IntentType, Tone, SEOIssue, KeywordMap
#
#   # Inspect routing logic in tests
#   from graph import route_after_seo_audit, route_after_human_review
#
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# ── Graph assembly & execution ─────────────────────────────────────────────────
from .graph_builder import (
    build_graph,
    run_pipeline,
    CredentialManager,
    SEO_PASS_THRESHOLD,
    REQUIRE_OUTLINE_APPROVAL,
)

# ── State types ────────────────────────────────────────────────────────────────
from .state import (
    # Primary container
    AgentState,

    # Enumerations
    IntentType,
    Severity,
    Tone,

    # Sub-types
    ResearchSnippet,
    KeywordMap,
    ContentOutline,
    SEOIssue,
    ImageSlot,
    SchemaMarkup,
    FinalOutput,

    # Factory
    make_initial_state,
)

# ── Routing functions (exposed for unit testing) ───────────────────────────────
from .routers import (
    route_after_validation,
    route_after_planner,
    route_after_research_merger,
    route_after_outline,
    route_after_outline_review,
    route_after_seo_audit,
    route_after_human_review,
    route_after_error_recovery,
    ROUTING_TABLE,
)

# ── Module metadata ────────────────────────────────────────────────────────────

__version__: str = "1.0.0"
__author__:  str = "PaddleAurum Agentic Workflow Architect"
__all__: list[str] = [
    # Graph
    "build_graph",
    "run_pipeline",
    "CredentialManager",
    "SEO_PASS_THRESHOLD",
    "REQUIRE_OUTLINE_APPROVAL",

    # State
    "AgentState",
    "IntentType",
    "Severity",
    "Tone",
    "ResearchSnippet",
    "KeywordMap",
    "ContentOutline",
    "SEOIssue",
    "ImageSlot",
    "SchemaMarkup",
    "FinalOutput",
    "make_initial_state",

    # Routers
    "route_after_validation",
    "route_after_planner",
    "route_after_research_merger",
    "route_after_outline",
    "route_after_outline_review",
    "route_after_seo_audit",
    "route_after_human_review",
    "route_after_error_recovery",
    "ROUTING_TABLE",
]