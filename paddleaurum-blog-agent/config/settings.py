# config/settings.py
# Centralised configuration for the paddleaurum.com autonomous blog pipeline.
#
# Every constant here is consumed by at least one other module.
# Import pattern used across the codebase:
#
#     from config.settings import SEO_THRESHOLD, MAX_ITERATIONS, DEFAULT_TONE
#
# Environment variables override infra defaults at runtime; pipeline behavior
# constants (SEO_THRESHOLD, MAX_ITERATIONS, etc.) are code-level decisions and
# are not overridden by env vars.

import os

# ── Pipeline behavior ─────────────────────────────────────────────────────────
# Consumed by: graph/state.py (make_initial_state), graph/routers.py,
#              nodes/input_validator.py, nodes/error_recovery.py

MAX_ITERATIONS: int = 3
# Maximum number of Writer → SEO Auditor → Reflection cycles before the
# pipeline escalates to the human review gate regardless of SEO score.
# Matches AgentState.max_iterations default and ErrorRecoveryNode retry cap.

SEO_THRESHOLD: int = 85
# Minimum SEO audit score (0–100) required for the pipeline to advance past
# the Auditor node without triggering a reflection loop.
# Consumed by graph/routers.py (route_after_seo_audit).

DEFAULT_TONE: str = "coach"
# Default authorial tone injected into AgentState when none is supplied.
# Valid values: "coach" | "expert" | "beginner-friendly"  (see graph/state.Tone).
# Consumed by graph/state.py (make_initial_state) and nodes/input_validator.py.

DEFAULT_WORD_COUNT: int = 1500
# Default target article length in words when no word_count_goal is supplied.
# Consumed by graph/state.py (make_initial_state) and nodes/input_validator.py.

REQUIRE_OUTLINE_APPROVAL: bool = (
    os.getenv("REQUIRE_OUTLINE_APPROVAL", "false").lower() == "true"
)
# When True, the pipeline pauses at an optional human gate after the Outline
# Agent node, allowing a reviewer to approve or revise the heading structure
# before the Coach Writer runs.
# Consumed by graph/graph_builder.py and graph/routers.py (route_after_outline).

# ── LLM ──────────────────────────────────────────────────────────────────────
# Consumed by: graph/graph_builder.py (_build_llm)

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Vector store & embeddings ─────────────────────────────────────────────────
# Consumed by: graph/graph_builder.py (_build_vectordb)

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
CHROMADB_PERSIST_DIR: str = os.getenv("CHROMADB_PERSIST_DIR", "./chromadb_store")

# ── Persistence ───────────────────────────────────────────────────────────────
# Consumed by: graph/graph_builder.py (SqliteSaver)

CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "./checkpoints/checkpoints.db")

# ── Observability ─────────────────────────────────────────────────────────────
# Consumed by: graph/graph_builder.py (_configure_langsmith)

LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "paddleaurum-blog-agent")