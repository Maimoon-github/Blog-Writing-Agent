# # graph/graph_builder.py
# # ──────────────────────────────────────────────────────────────────────────────
# # Assembles the complete PaddleAurum blog-generation LangGraph graph.
# #
# # Responsibilities
# # ----------------
# #   1. Load and validate credentials from environment variables.
# #   2. Initialise shared infrastructure (Ollama LLM, ChromaDB, LangSmith).
# #   3. Register every node (imported from nodes/).
# #   4. Wire deterministic (fixed) edges.
# #   5. Register conditional edges using router functions from routers.py.
# #   6. Attach the SQLite checkpointer for resumable runs.
# #   7. Compile and return the runnable app.
# #
# # Node implementations live in nodes/.  This file only assembles the graph —
# # it contains no business logic.
# # ──────────────────────────────────────────────────────────────────────────────

# from __future__ import annotations

# import logging
# import os
# from functools import partial
# from typing import Optional

# from dotenv import load_dotenv
# from langchain.callbacks.tracers import LangChainTracer
# from langchain_community.llms import Ollama
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph import END, StateGraph

# from .state import AgentState
# from .routers import (
#     ROUTING_TABLE,
#     route_after_outline,
#     route_after_outline_review,
#     route_after_error_recovery,
#     SEO_PASS_THRESHOLD,
#     _NODE_ERROR_RECOVERY,
#     _NODE_HUMAN_OUTLINE_REVIEW,
#     _NODE_OUTLINE_AGENT,
#     _NODE_WRITER,
# )

# # Node imports — one module per node in nodes/
# from nodes.input_validator   import input_validator_node
# from nodes.planner           import planner_node
# from nodes.research_worker   import research_worker_node
# from nodes.research_merger   import research_merger_node
# from nodes.keyword_mapper    import keyword_mapper_node
# from nodes.outline_agent     import outline_agent_node
# from nodes.human_outline_review import human_outline_review_node
# from nodes.coaching_writer   import coaching_writer_node
# from nodes.seo_auditor       import seo_auditor_node
# from nodes.reflection        import reflection_node
# from nodes.image_selector    import image_selector_node
# from nodes.citation_formatter import citation_formatter_node
# from nodes.schema_generator  import schema_generator_node
# from nodes.final_assembler   import final_assembler_node
# from nodes.human_review_gate import human_review_gate_node
# from nodes.error_recovery    import error_recovery_node
# from nodes.publish           import publish_node

# logger = logging.getLogger(__name__)

# # ─────────────────────────────────────────────────────────────────────────────
# # Configuration constants
# # These values can be overridden via environment variables where appropriate.
# # ─────────────────────────────────────────────────────────────────────────────

# REQUIRE_OUTLINE_APPROVAL: bool = (
#     os.getenv("REQUIRE_OUTLINE_APPROVAL", "false").lower() == "true"
# )
# CHECKPOINT_DB_PATH: str  = os.getenv("CHECKPOINT_DB_PATH", "./checkpoints/checkpoints.db")
# EMBEDDING_MODEL:    str  = "sentence-transformers/all-MiniLM-L6-v2"
# CHROMADB_PERSIST:   str  = os.getenv("CHROMADB_PERSIST_DIR", "./chromadb_store")
# LANGSMITH_PROJECT:  str  = os.getenv("LANGSMITH_PROJECT", "paddleaurum-blog-agent")
# OLLAMA_MODEL:       str  = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
# OLLAMA_BASE_URL:    str  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# # ─────────────────────────────────────────────────────────────────────────────
# # Credential management
# # ─────────────────────────────────────────────────────────────────────────────


# class CredentialManager:
#     """
#     Loads and validates all required environment variables at startup.

#     Credentials are read once here and injected into tool constructors.
#     They are never written into AgentState or logged by LangSmith.

#     Call `CredentialManager.load()` before building the graph.
#     """

#     _loaded: bool = False

#     # Required credentials — build raises EnvironmentError if absent
#     REQUIRED: tuple[str, ...] = (
#         "LANGSMITH_API_KEY",
#         "UNSPLASH_ACCESS_KEY",
#         "PEXELS_API_KEY",
#         "WORDPRESS_URL",
#         "WORDPRESS_USER",
#         "WORDPRESS_APP_PASSWORD",
#     )

#     @classmethod
#     def load(cls) -> None:
#         """Load .env file and validate all required credentials are present."""
#         load_dotenv()
#         missing = [k for k in cls.REQUIRED if not os.getenv(k)]
#         if missing:
#             raise EnvironmentError(
#                 f"Missing required environment variables: {', '.join(missing)}\n"
#                 "Copy .env.example to .env and fill in all values."
#             )
#         cls._loaded = True
#         logger.info("All credentials loaded successfully.")

#     @staticmethod
#     def get(key: str) -> str:
#         """Retrieve a single env var; raises if not set."""
#         value = os.getenv(key)
#         if not value:
#             raise EnvironmentError(f"Environment variable '{key}' is not set.")
#         return value

#     @staticmethod
#     def get_unsplash_key()          -> str: return CredentialManager.get("UNSPLASH_ACCESS_KEY")
#     @staticmethod
#     def get_pexels_key()            -> str: return CredentialManager.get("PEXELS_API_KEY")
#     @staticmethod
#     def get_langsmith_key()         -> str: return CredentialManager.get("LANGSMITH_API_KEY")
#     @staticmethod
#     def get_wordpress_url()         -> str: return CredentialManager.get("WORDPRESS_URL")
#     @staticmethod
#     def get_wordpress_user()        -> str: return CredentialManager.get("WORDPRESS_USER")
#     @staticmethod
#     def get_wordpress_app_password() -> str: return CredentialManager.get("WORDPRESS_APP_PASSWORD")


# # ─────────────────────────────────────────────────────────────────────────────
# # Shared infrastructure
# # ─────────────────────────────────────────────────────────────────────────────


# def _build_llm() -> Ollama:
#     """Initialise the local Ollama LLM.  No API key required — runs fully offline."""
#     logger.info("Initialising Ollama LLM: model=%s url=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
#     return Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


# def _build_vectordb() -> Chroma:
#     """
#     Initialise the ChromaDB persistent vector store.

#     The store is shared across all CrewAI agents via the rag_retrieval_tool.
#     Each collection (pickleball_rules, coaching_materials, seo_guidelines, etc.)
#     is accessed by collection name at query time.
#     """
#     logger.info("Initialising ChromaDB at: %s", CHROMADB_PERSIST)
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#     return Chroma(
#         persist_directory=CHROMADB_PERSIST,
#         embedding_function=embeddings,
#     )


# def _configure_langsmith() -> LangChainTracer:
#     """
#     Configure LangSmith tracing.

#     Sets the required environment variables (LangChain picks these up globally)
#     and returns a LangChainTracer to pass as a callback when invoking the graph.

#     Each pipeline run produces a trace in the LangSmith dashboard showing:
#       - node execution order and timing
#       - token usage per LLM call
#       - RAG retrieval queries and results
#       - SEO score per iteration
#       - full error stack traces
#     """
#     os.environ["LANGCHAIN_TRACING_V2"] = "true"
#     os.environ["LANGCHAIN_API_KEY"]    = CredentialManager.get_langsmith_key()
#     os.environ["LANGCHAIN_PROJECT"]    = LANGSMITH_PROJECT

#     logger.info("LangSmith tracing enabled for project: %s", LANGSMITH_PROJECT)
#     return LangChainTracer(project_name=LANGSMITH_PROJECT)


# # ─────────────────────────────────────────────────────────────────────────────
# # Graph builder
# # ─────────────────────────────────────────────────────────────────────────────


# def build_graph(credentials_already_loaded: bool = False):
#     """
#     Assemble, compile, and return the runnable LangGraph application.

#     Parameters
#     ----------
#     credentials_already_loaded
#         Pass True if `CredentialManager.load()` was already called externally
#         (e.g. in a test fixture).  Avoids double-loading .env.

#     Returns
#     -------
#     app
#         A compiled LangGraph `CompiledGraph` instance.  Run with::

#             result = await app.ainvoke(
#                 initial_state,
#                 config={
#                     "callbacks": [tracer],
#                     "configurable": {"thread_id": session_id},
#                 },
#             )
#     tracer
#         The `LangChainTracer` to pass in `config["callbacks"]`.
#     """
#     # ── Step 1: Credentials & infrastructure ─────────────────────────────────
#     if not credentials_already_loaded:
#         CredentialManager.load()

#     tracer  = _configure_langsmith()
#     llm     = _build_llm()
#     vectordb = _build_vectordb()

#     logger.info(
#         "Building PaddleAurum blog graph | SEO threshold=%d | outline_approval=%s",
#         SEO_PASS_THRESHOLD,
#         REQUIRE_OUTLINE_APPROVAL,
#     )

#     # ── Step 2: Initialise the StateGraph ────────────────────────────────────
#     workflow = StateGraph(AgentState)

#     # ── Step 3: Register all nodes ───────────────────────────────────────────
#     #
#     # Each node function is imported from nodes/.  Infrastructure objects
#     # (llm, vectordb) are injected via partial() where needed so node modules
#     # remain independently testable without a live LLM or DB.
#     #
#     # Node naming convention: lowercase_snake_case matching the node name used
#     # in all conditional-edge target maps.

#     workflow.add_node("input_validator",    input_validator_node)
#     workflow.add_node("planner",            partial(planner_node,         llm=llm))
#     workflow.add_node("research_worker",    research_worker_node)           # stateless tool call
#     workflow.add_node("research_merger",    research_merger_node)           # pure aggregation
#     workflow.add_node("keyword_mapper",     partial(keyword_mapper_node,   llm=llm, vectordb=vectordb))
#     workflow.add_node("outline_agent",      partial(outline_agent_node,    llm=llm, vectordb=vectordb))
#     workflow.add_node("human_outline_review", human_outline_review_node)   # Streamlit gate (optional)
#     workflow.add_node("writer",             partial(coaching_writer_node,  llm=llm, vectordb=vectordb))
#     workflow.add_node("seo_auditor",        seo_auditor_node)               # rule-based + LLM hybrid
#     workflow.add_node("reflection",         partial(reflection_node,       llm=llm))
#     workflow.add_node("image_selector",     partial(
#                                                 image_selector_node,
#                                                 unsplash_key=CredentialManager.get_unsplash_key(),
#                                                 pexels_key=CredentialManager.get_pexels_key(),
#                                             ))
#     workflow.add_node("citation_formatter", citation_formatter_node)
#     workflow.add_node("schema_generator",   schema_generator_node)
#     workflow.add_node("final_assembler",    final_assembler_node)
#     workflow.add_node("human_review_gate",  human_review_gate_node)        # Streamlit review + approve
#     workflow.add_node("error_recovery",     error_recovery_node)
#     workflow.add_node("publish",            partial(
#                                                 publish_node,
#                                                 wp_url=CredentialManager.get_wordpress_url(),
#                                                 wp_user=CredentialManager.get_wordpress_user(),
#                                                 wp_password=CredentialManager.get_wordpress_app_password(),
#                                             ))

#     # ── Step 4: Entry point ───────────────────────────────────────────────────
#     workflow.set_entry_point("input_validator")

#     # ── Step 5: Fixed (deterministic) edges ───────────────────────────────────
#     #
#     # These edges fire unconditionally — the source node always transitions
#     # to the same destination regardless of state.

#     workflow.add_edge("research_merger",    "keyword_mapper")    # after merge → keyword analysis
#     workflow.add_edge("keyword_mapper",     "outline_agent")     # after keywords → build outline
#     workflow.add_edge("reflection",         "writer")            # reflection always loops to writer
#     workflow.add_edge("citation_formatter", "schema_generator")  # sequential: cite → schema
#     workflow.add_edge("schema_generator",   "final_assembler")   # sequential: schema → assemble
#     workflow.add_edge("final_assembler",    "human_review_gate") # always needs human sign-off
#     workflow.add_edge("publish",            END)                 # publish is the terminal node

#     # ── Step 6: Conditional edges ─────────────────────────────────────────────
#     #
#     # Registered using ROUTING_TABLE from routers.py where possible.
#     # Edges with injected config (outline approval, error recovery) are
#     # registered explicitly below.

#     # 6a. Nodes covered by ROUTING_TABLE
#     for source_node, config in ROUTING_TABLE.items():
#         if source_node == "error_recovery":
#             continue  # handled separately below (dynamic targets)
#         if config["targets"]:
#             workflow.add_conditional_edges(
#                 source_node,
#                 config["fn"],
#                 config["targets"],
#             )

#     # 6b. Outline gate — requires runtime REQUIRE_OUTLINE_APPROVAL flag
#     workflow.add_conditional_edges(
#         "outline_agent",
#         partial(route_after_outline, require_outline_approval=REQUIRE_OUTLINE_APPROVAL),
#         {
#             _NODE_ERROR_RECOVERY:        _NODE_ERROR_RECOVERY,
#             _NODE_HUMAN_OUTLINE_REVIEW:  _NODE_HUMAN_OUTLINE_REVIEW,
#             _NODE_WRITER:                _NODE_WRITER,
#         },
#     )

#     # 6c. Human outline review gate (only active when REQUIRE_OUTLINE_APPROVAL=True)
#     workflow.add_conditional_edges(
#         "human_outline_review",
#         route_after_outline_review,
#         {
#             _NODE_WRITER:        _NODE_WRITER,
#             _NODE_OUTLINE_AGENT: _NODE_OUTLINE_AGENT,
#         },
#     )

#     # 6d. Error recovery — targets are dynamic (any node can be retried).
#     #     Build the complete target map from all registered node names.
#     all_node_names = {
#         "input_validator", "planner", "research_worker", "research_merger",
#         "keyword_mapper", "outline_agent", "human_outline_review", "writer",
#         "seo_auditor", "reflection", "image_selector", "citation_formatter",
#         "schema_generator", "final_assembler", "human_review_gate", "publish",
#         "error_recovery",
#     }
#     workflow.add_conditional_edges(
#         "error_recovery",
#         route_after_error_recovery,
#         {node: node for node in all_node_names},  # identity map: name → name
#     )

#     # ── Step 7: Image selector → citation formatter (post-SEO parallel branch) ──
#     #
#     # After the SEO Auditor routes to image_selector, image selection and
#     # citation formatting run sequentially (image_selector → citation_formatter).
#     # This edge is intentionally fixed; both nodes are non-blocking async calls
#     # so wall-clock latency is minimal.
#     workflow.add_edge("image_selector", "citation_formatter")

#     # ── Step 8: Compile with SQLite checkpointer ──────────────────────────────
#     #
#     # SqliteSaver enables resumable pipelines: if a node fails mid-run, the
#     # next invocation with the same thread_id (session_id) will resume from
#     # the last successfully committed node rather than restarting from scratch.

#     logger.info("Attaching SQLite checkpointer: %s", CHECKPOINT_DB_PATH)
#     memory = SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
#     app    = workflow.compile(checkpointer=memory)

#     logger.info("Graph compiled successfully — %d nodes registered.", len(all_node_names))
#     return app, tracer


# # ─────────────────────────────────────────────────────────────────────────────
# # Convenience runner
# # ─────────────────────────────────────────────────────────────────────────────


# async def run_pipeline(
#     topic: str,
#     session_id: str,
#     target_keyword: Optional[str] = None,
#     word_count_goal: int = 1_500,
#     max_iterations: int = 3,
# ) -> dict:
#     """
#     High-level async entry point for running a single blog-generation pipeline.

#     Builds the graph, constructs the initial state, invokes the pipeline, and
#     returns the final AgentState dict.

#     Parameters
#     ----------
#     topic           : Blog topic (e.g. "pickleball kitchen rules for beginners").
#     session_id      : UUID; used for LangSmith trace ID and SQLite thread ID.
#     target_keyword  : Optional keyword override.
#     word_count_goal : Target word count (default 1 500).
#     max_iterations  : Max Writer → SEO Auditor → Reflection cycles (default 3).

#     Returns
#     -------
#     dict
#         Final AgentState after the pipeline completes (or is paused at a
#         human review gate).
#     """
#     from .state import make_initial_state, Tone  # local import avoids circular deps

#     app, tracer = build_graph()

#     initial_state = make_initial_state(
#         topic=topic,
#         session_id=session_id,
#         target_keyword=target_keyword,
#         tone=Tone.COACH,
#         word_count_goal=word_count_goal,
#         max_iterations=max_iterations,
#     )

#     logger.info("Starting pipeline for topic='%s' (session=%s)", topic, session_id)

#     result = await app.ainvoke(
#         initial_state,
#         config={
#             "callbacks": [tracer],
#             "configurable": {"thread_id": session_id},
#         },
#     )

#     logger.info(
#         "Pipeline completed (session=%s) | SEO score=%s | approved=%s",
#         session_id,
#         result.get("seo_score"),
#         result.get("approved"),
#     )
#     return result






















# @################################################################################






















# graph/graph_builder.py
# ──────────────────────────────────────────────────────────────────────────────
# Assembles the complete PaddleAurum blog-generation LangGraph graph.
#
# Responsibilities
# ----------------
#   1. Load and validate credentials from environment variables.
#   2. Initialise shared infrastructure (Ollama LLM, ChromaDB, LangSmith).
#   3. Register every node (imported from nodes/).
#   4. Wire deterministic (fixed) edges.
#   5. Register conditional edges using router functions from routers.py.
#   6. Attach the SQLite checkpointer for resumable runs.
#   7. Compile and return the runnable app.
#
# Node implementations live in nodes/.  This file only assembles the graph —
# it contains no business logic.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

# ── Import configuration constants from central settings ─────────────────────
from config.settings import (
    REQUIRE_OUTLINE_APPROVAL,
    CHECKPOINT_DB_PATH,
    EMBEDDING_MODEL,
    CHROMADB_PERSIST_DIR,
    LANGSMITH_PROJECT,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)

from .state import AgentState
from .routers import (
    ROUTING_TABLE,
    route_after_outline,
    route_after_outline_review,
    route_after_error_recovery,
    SEO_PASS_THRESHOLD,
    _NODE_ERROR_RECOVERY,
    _NODE_HUMAN_OUTLINE_REVIEW,
    _NODE_OUTLINE_AGENT,
    _NODE_WRITER,
)

# Node imports — one module per node in nodes/
from nodes.input_validator   import input_validator_node
from nodes.planner           import planner_node
from nodes.research_worker   import research_worker_node
from nodes.research_merger   import research_merger_node
from nodes.keyword_mapper    import keyword_mapper_node
from nodes.outline_agent     import outline_agent_node
from nodes.human_outline_review import human_outline_review_node
from nodes.coaching_writer   import coaching_writer_node
from nodes.seo_auditor       import seo_auditor_node
from nodes.reflection        import reflection_node
from nodes.image_selector    import image_selector_node
from nodes.citation_formatter import citation_formatter_node
from nodes.schema_generator  import schema_generator_node
from nodes.final_assembler   import final_assembler_node
from nodes.human_review_gate import human_review_gate_node
from nodes.error_recovery    import error_recovery_node
from nodes.publish           import publish_node

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Credential management
# ─────────────────────────────────────────────────────────────────────────────


class CredentialManager:
    """
    Loads and validates all required environment variables at startup.

    Credentials are read once here and injected into tool constructors.
    They are never written into AgentState or logged by LangSmith.

    Call `CredentialManager.load()` before building the graph.
    """

    _loaded: bool = False

    # Required credentials — build raises EnvironmentError if absent
    REQUIRED: tuple[str, ...] = (
        "LANGSMITH_API_KEY",
        "UNSPLASH_ACCESS_KEY",
        "PEXELS_API_KEY",
        "WORDPRESS_URL",
        "WORDPRESS_USER",
        "WORDPRESS_APP_PASSWORD",
    )

    @classmethod
    def load(cls) -> None:
        """Load .env file and validate all required credentials are present."""
        load_dotenv()
        missing = [k for k in cls.REQUIRED if not os.getenv(k)]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example to .env and fill in all values."
            )
        cls._loaded = True
        logger.info("All credentials loaded successfully.")

    @staticmethod
    def get(key: str) -> str:
        """Retrieve a single env var; raises if not set."""
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(f"Environment variable '{key}' is not set.")
        return value

    @staticmethod
    def get_unsplash_key()          -> str: return CredentialManager.get("UNSPLASH_ACCESS_KEY")
    @staticmethod
    def get_pexels_key()            -> str: return CredentialManager.get("PEXELS_API_KEY")
    @staticmethod
    def get_langsmith_key()         -> str: return CredentialManager.get("LANGSMITH_API_KEY")
    @staticmethod
    def get_wordpress_url()         -> str: return CredentialManager.get("WORDPRESS_URL")
    @staticmethod
    def get_wordpress_user()        -> str: return CredentialManager.get("WORDPRESS_USER")
    @staticmethod
    def get_wordpress_app_password() -> str: return CredentialManager.get("WORDPRESS_APP_PASSWORD")


# ─────────────────────────────────────────────────────────────────────────────
# Shared infrastructure
# ─────────────────────────────────────────────────────────────────────────────


def _build_llm() -> Ollama:
    """Initialise the local Ollama LLM.  No API key required — runs fully offline."""
    logger.info("Initialising Ollama LLM: model=%s url=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
    return Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


def _build_vectordb() -> Chroma:
    """
    Initialise the ChromaDB persistent vector store.

    The store is shared across all CrewAI agents via the rag_retrieval_tool.
    Each collection (pickleball_rules, coaching_materials, seo_guidelines, etc.)
    is accessed by collection name at query time.
    """
    logger.info("Initialising ChromaDB at: %s", CHROMADB_PERSIST_DIR)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMADB_PERSIST_DIR,
        embedding_function=embeddings,
    )


def _configure_langsmith() -> LangChainTracer:
    """
    Configure LangSmith tracing.

    Sets the required environment variables (LangChain picks these up globally)
    and returns a LangChainTracer to pass as a callback when invoking the graph.

    Each pipeline run produces a trace in the LangSmith dashboard showing:
      - node execution order and timing
      - token usage per LLM call
      - RAG retrieval queries and results
      - SEO score per iteration
      - full error stack traces
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = CredentialManager.get_langsmith_key()
    os.environ["LANGCHAIN_PROJECT"]    = LANGSMITH_PROJECT

    logger.info("LangSmith tracing enabled for project: %s", LANGSMITH_PROJECT)
    return LangChainTracer(project_name=LANGSMITH_PROJECT)


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────


def build_graph(credentials_already_loaded: bool = False):
    """
    Assemble, compile, and return the runnable LangGraph application.

    Parameters
    ----------
    credentials_already_loaded
        Pass True if `CredentialManager.load()` was already called externally
        (e.g. in a test fixture).  Avoids double-loading .env.

    Returns
    -------
    app
        A compiled LangGraph `CompiledGraph` instance.  Run with::

            result = await app.ainvoke(
                initial_state,
                config={
                    "callbacks": [tracer],
                    "configurable": {"thread_id": session_id},
                },
            )
    tracer
        The `LangChainTracer` to pass in `config["callbacks"]`.
    """
    # ── Step 1: Credentials & infrastructure ─────────────────────────────────
    if not credentials_already_loaded:
        CredentialManager.load()

    tracer  = _configure_langsmith()
    llm     = _build_llm()
    vectordb = _build_vectordb()

    logger.info(
        "Building PaddleAurum blog graph | SEO threshold=%d | outline_approval=%s",
        SEO_PASS_THRESHOLD,
        REQUIRE_OUTLINE_APPROVAL,
    )

    # ── Step 2: Initialise the StateGraph ────────────────────────────────────
    workflow = StateGraph(AgentState)

    # ── Step 3: Register all nodes ───────────────────────────────────────────
    #
    # Each node function is imported from nodes/.  Infrastructure objects
    # (llm, vectordb) are injected via partial() where needed so node modules
    # remain independently testable without a live LLM or DB.
    #
    # Node naming convention: lowercase_snake_case matching the node name used
    # in all conditional-edge target maps.

    workflow.add_node("input_validator",    input_validator_node)
    workflow.add_node("planner",            partial(planner_node,         llm=llm))
    workflow.add_node("research_worker",    research_worker_node)           # stateless tool call
    workflow.add_node("research_merger",    research_merger_node)           # pure aggregation
    workflow.add_node("keyword_mapper",     partial(keyword_mapper_node,   llm=llm, vectordb=vectordb))
    workflow.add_node("outline_agent",      partial(outline_agent_node,    llm=llm, vectordb=vectordb))
    workflow.add_node("human_outline_review", human_outline_review_node)   # Streamlit gate (optional)
    workflow.add_node("writer",             partial(coaching_writer_node,  llm=llm, vectordb=vectordb))
    workflow.add_node("seo_auditor",        seo_auditor_node)               # rule-based + LLM hybrid
    workflow.add_node("reflection",         partial(reflection_node,       llm=llm))
    workflow.add_node("image_selector",     partial(
                                                image_selector_node,
                                                unsplash_key=CredentialManager.get_unsplash_key(),
                                                pexels_key=CredentialManager.get_pexels_key(),
                                            ))
    workflow.add_node("citation_formatter", citation_formatter_node)
    workflow.add_node("schema_generator",   schema_generator_node)
    workflow.add_node("final_assembler",    final_assembler_node)
    workflow.add_node("human_review_gate",  human_review_gate_node)        # Streamlit review + approve
    workflow.add_node("error_recovery",     error_recovery_node)
    workflow.add_node("publish",            partial(
                                                publish_node,
                                                wp_url=CredentialManager.get_wordpress_url(),
                                                wp_user=CredentialManager.get_wordpress_user(),
                                                wp_password=CredentialManager.get_wordpress_app_password(),
                                            ))

    # ── Step 4: Entry point ───────────────────────────────────────────────────
    workflow.set_entry_point("input_validator")

    # ── Step 5: Fixed (deterministic) edges ───────────────────────────────────
    #
    # These edges fire unconditionally — the source node always transitions
    # to the same destination regardless of state.

    workflow.add_edge("research_merger",    "keyword_mapper")    # after merge → keyword analysis
    workflow.add_edge("keyword_mapper",     "outline_agent")     # after keywords → build outline
    workflow.add_edge("reflection",         "writer")            # reflection always loops to writer
    workflow.add_edge("citation_formatter", "schema_generator")  # sequential: cite → schema
    workflow.add_edge("schema_generator",   "final_assembler")   # sequential: schema → assemble
    workflow.add_edge("final_assembler",    "human_review_gate") # always needs human sign-off
    workflow.add_edge("publish",            END)                 # publish is the terminal node

    # ── Step 6: Conditional edges ─────────────────────────────────────────────
    #
    # Registered using ROUTING_TABLE from routers.py where possible.
    # Edges with injected config (outline approval, error recovery) are
    # registered explicitly below.

    # 6a. Nodes covered by ROUTING_TABLE
    for source_node, config in ROUTING_TABLE.items():
        if source_node == "error_recovery":
            continue  # handled separately below (dynamic targets)
        if config["targets"]:
            workflow.add_conditional_edges(
                source_node,
                config["fn"],
                config["targets"],
            )

    # 6b. Outline gate — requires runtime REQUIRE_OUTLINE_APPROVAL flag
    workflow.add_conditional_edges(
        "outline_agent",
        partial(route_after_outline, require_outline_approval=REQUIRE_OUTLINE_APPROVAL),
        {
            _NODE_ERROR_RECOVERY:        _NODE_ERROR_RECOVERY,
            _NODE_HUMAN_OUTLINE_REVIEW:  _NODE_HUMAN_OUTLINE_REVIEW,
            _NODE_WRITER:                _NODE_WRITER,
        },
    )

    # 6c. Human outline review gate (only active when REQUIRE_OUTLINE_APPROVAL=True)
    workflow.add_conditional_edges(
        "human_outline_review",
        route_after_outline_review,
        {
            _NODE_WRITER:        _NODE_WRITER,
            _NODE_OUTLINE_AGENT: _NODE_OUTLINE_AGENT,
        },
    )

    # 6d. Error recovery — targets are dynamic (any node can be retried).
    #     Build the complete target map from all registered node names.
    all_node_names = {
        "input_validator", "planner", "research_worker", "research_merger",
        "keyword_mapper", "outline_agent", "human_outline_review", "writer",
        "seo_auditor", "reflection", "image_selector", "citation_formatter",
        "schema_generator", "final_assembler", "human_review_gate", "publish",
        "error_recovery",
    }
    workflow.add_conditional_edges(
        "error_recovery",
        route_after_error_recovery,
        {node: node for node in all_node_names},  # identity map: name → name
    )

    # ── Step 7: Image selector → citation formatter (post-SEO parallel branch) ──
    #
    # After the SEO Auditor routes to image_selector, image selection and
    # citation formatting run sequentially (image_selector → citation_formatter).
    # This edge is intentionally fixed; both nodes are non-blocking async calls
    # so wall-clock latency is minimal.
    workflow.add_edge("image_selector", "citation_formatter")

    # ── Step 8: Compile with SQLite checkpointer ──────────────────────────────
    #
    # SqliteSaver enables resumable pipelines: if a node fails mid-run, the
    # next invocation with the same thread_id (session_id) will resume from
    # the last successfully committed node rather than restarting from scratch.

    logger.info("Attaching SQLite checkpointer: %s", CHECKPOINT_DB_PATH)
    memory = SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
    app    = workflow.compile(checkpointer=memory)

    logger.info("Graph compiled successfully — %d nodes registered.", len(all_node_names))
    return app, tracer


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────


async def run_pipeline(
    topic: str,
    session_id: str,
    target_keyword: Optional[str] = None,
    word_count_goal: int = 1_500,
    max_iterations: int = 3,
) -> dict:
    """
    High-level async entry point for running a single blog-generation pipeline.

    Builds the graph, constructs the initial state, invokes the pipeline, and
    returns the final AgentState dict.

    Parameters
    ----------
    topic           : Blog topic (e.g. "pickleball kitchen rules for beginners").
    session_id      : UUID; used for LangSmith trace ID and SQLite thread ID.
    target_keyword  : Optional keyword override.
    word_count_goal : Target word count (default 1 500).
    max_iterations  : Max Writer → SEO Auditor → Reflection cycles (default 3).

    Returns
    -------
    dict
        Final AgentState after the pipeline completes (or is paused at a
        human review gate).
    """
    from .state import make_initial_state, Tone  # local import avoids circular deps

    app, tracer = build_graph()

    initial_state = make_initial_state(
        topic=topic,
        session_id=session_id,
        target_keyword=target_keyword,
        tone=Tone.COACH,
        word_count_goal=word_count_goal,
        max_iterations=max_iterations,
    )

    logger.info("Starting pipeline for topic='%s' (session=%s)", topic, session_id)

    result = await app.ainvoke(
        initial_state,
        config={
            "callbacks": [tracer],
            "configurable": {"thread_id": session_id},
        },
    )

    logger.info(
        "Pipeline completed (session=%s) | SEO score=%s | approved=%s",
        session_id,
        result.get("seo_score"),
        result.get("approved"),
    )
    return result