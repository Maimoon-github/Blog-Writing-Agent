# """
# graph/graph_builder.py

# Defines and compiles the complete LangGraph StateGraph for the blog-agent project.
# Imports all node functions and state, defines the graph topology, dispatch functions,
# and exports the compiled graph with a SqliteSaver checkpointer.
# """

# from typing import List

# from langgraph.graph import END, START, StateGraph
# from langgraph.types import Send

# from agents.citation_manager import citation_manager_node
# from agents.image_agent import image_agent_node
# from agents.planner import planner_node
# from agents.reducer import reducer_node
# from agents.researcher import researcher_node
# from agents.router import router_node
# from agents.writer import writer_node
# from graph.checkpointer import get_checkpointer
# from graph.state import GraphState


# # ---------------------------------------------------------------------------
# # Dispatch functions
# # ---------------------------------------------------------------------------

# def dispatch_researchers(state: GraphState) -> List[Send]:
#     """
#     Fan-out: spawn one researcher_node per section that has a search_query.
#     If no sections need research, skip straight to writer dispatch.
#     """
#     sends = [
#         Send(
#             "researcher_node",
#             {
#                 "topic": state["topic"],
#                 "run_id": state["run_id"],
#                 "research_results": [],
#                 "section_drafts": [],
#                 "generated_images": [],
#                 "section_id": section.id,
#                 "search_query": section.search_query,
#                 "section_description": section.description,
#                 "blog_plan": state["blog_plan"],
#                 "citation_registry": {},
#                 "research_required": state["research_required"],
#                 "final_blog_md": "",
#                 "final_blog_html": "",
#                 "error": None,
#             },
#         )
#         for section in state["blog_plan"].sections
#         if section.search_query is not None
#     ]

#     # If nothing needs researching, jump directly to the writer barrier node.
#     if not sends:
#         return [Send("writer_dispatch", state)]

#     return sends


# def dispatch_writers(state: GraphState) -> List[Send]:
#     """
#     Fan-out: spawn one writer_node per section in the blog plan.
#     """
#     return [
#         Send(
#             "writer_node",
#             {
#                 **state,
#                 "section_id": section.id,
#                 "section_title": section.title,
#                 "section_description": section.description,
#                 "word_count": section.word_count,
#                 "image_prompt": section.image_prompt,
#             },
#         )
#         for section in state["blog_plan"].sections
#     ]


# def dispatch_image_agents(state: GraphState) -> List[Send]:
#     """
#     Fan-out: spawn one image_agent_node for the feature image plus one per section.
#     """
#     sends: List[Send] = []

#     # Feature image
#     sends.append(
#         Send(
#             "image_agent_node",
#             {
#                 **state,
#                 "section_id": "feature",
#                 "image_prompt": state["blog_plan"].feature_image_prompt,
#             },
#         )
#     )

#     # Per-section images
#     for section in state["blog_plan"].sections:
#         sends.append(
#             Send(
#                 "image_agent_node",
#                 {
#                     **state,
#                     "section_id": section.id,
#                     "image_prompt": section.image_prompt,
#                 },
#             )
#         )

#     return sends


# # ---------------------------------------------------------------------------
# # No-op barrier / join nodes
# # ---------------------------------------------------------------------------

# def _noop(state: GraphState) -> GraphState:
#     """Pass-through node used as a fan-in synchronisation barrier."""
#     return state


# # ---------------------------------------------------------------------------
# # Graph construction
# # ---------------------------------------------------------------------------

# builder = StateGraph(GraphState)

# # --- Worker nodes ---
# builder.add_node("router_node", router_node)
# builder.add_node("planner_node", planner_node)
# builder.add_node("researcher_node", researcher_node)
# builder.add_node("writer_node", writer_node)
# builder.add_node("image_agent_node", image_agent_node)
# builder.add_node("citation_manager_node", citation_manager_node)
# builder.add_node("reducer_node", reducer_node)

# # --- Barrier / join nodes ---
# builder.add_node("research_complete", _noop)   # fan-in for all researcher_node outputs
# builder.add_node("writer_dispatch", _noop)      # fallback target when no research needed
# builder.add_node("writer_complete", _noop)      # fan-in for all writer_node outputs
# builder.add_node("image_complete", _noop)       # fan-in for all image_agent_node outputs

# # --- Fixed edges ---
# builder.add_edge(START, "router_node")
# builder.add_edge("router_node", "planner_node")

# # researcher_node outputs converge at research_complete
# builder.add_edge("researcher_node", "research_complete")

# # writer_dispatch (no-research shortcut) also feeds into the writer fan-out
# builder.add_edge("writer_dispatch", "writer_complete")  # bypass: go straight to writer barrier
# # NOTE: Because writer_dispatch is a _noop that receives the full state we use a
# #       conditional edge from it so it runs dispatch_writers the same way.

# # writer_node outputs converge at writer_complete
# builder.add_edge("writer_node", "writer_complete")

# # image_agent_node outputs converge at image_complete
# builder.add_edge("image_agent_node", "image_complete")

# # Sequential tail of the pipeline
# builder.add_edge("image_complete", "citation_manager_node")
# builder.add_edge("citation_manager_node", "reducer_node")
# builder.add_edge("reducer_node", END)

# # --- Conditional (fan-out) edges ---

# # planner → dispatch researchers (or skip to writers if none needed)
# builder.add_conditional_edges("planner_node", dispatch_researchers)

# # research_complete → dispatch writers
# builder.add_conditional_edges("research_complete", dispatch_writers)

# # writer_dispatch (no-research path) → dispatch writers
# builder.add_conditional_edges("writer_dispatch", dispatch_writers)

# # writer_complete → dispatch image agents
# builder.add_conditional_edges("writer_complete", dispatch_image_agents)

# # ---------------------------------------------------------------------------
# # Compilation
# # ---------------------------------------------------------------------------

# checkpointer = get_checkpointer()
# compiled_graph = builder.compile(checkpointer=checkpointer)












































"""LangGraph graph builder for the blog agent workflow.

This module constructs the complete StateGraph for the blog generation pipeline.
It defines the topology, dispatch functions for parallel execution, and compiles
the graph with SQLite persistence for crash recovery and state checkpointing.

The workflow follows a map-reduce pattern:
1. Router → Planner → (conditional) → Parallel research → Join → Parallel writing → Join → Parallel image generation → Join → Citation management → Reducer → END
"""

import structlog
from typing import List, Union, Sequence

from langgraph.graph import StateGraph, END, START, Send
from langgraph.types import Command

from graph.state import GraphState
from agents.router import router_node
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from agents.image_agent import image_agent_node
from agents.citation_manager import citation_manager_node
from agents.reducer import reducer_node
from graph.checkpointer import get_checkpointer

logger = structlog.get_logger(__name__)

# Node name constants (prevents typos)
ROUTER = "router_node"
PLANNER = "planner_node"
RESEARCHER = "researcher_node"
WRITER = "writer_node"
IMAGE_AGENT = "image_agent_node"
CITATION = "citation_manager_node"
REDUCER = "reducer_node"
RESEARCH_JOIN = "research_complete"
WRITER_JOIN = "writer_complete"
IMAGE_JOIN = "image_complete"

# ---------------------------------------------------------------------------
# Join Nodes (Fan-in barriers)
# ---------------------------------------------------------------------------

def research_complete(state: GraphState) -> GraphState:
    """No-op join node for researcher fan-in.
    
    All researcher_node outputs converge here before proceeding to writer dispatch.
    """
    logger.debug("Research complete barrier reached", run_id=state.get("run_id"))
    return state


def writer_complete(state: GraphState) -> GraphState:
    """No-op join node for writer fan-in.
    
    All writer_node outputs converge here before proceeding to image dispatch.
    """
    logger.debug("Writer complete barrier reached", run_id=state.get("run_id"))
    return state


def image_complete(state: GraphState) -> GraphState:
    """No-op join node for image agent fan-in.
    
    All image_agent_node outputs converge here before proceeding to citation manager.
    """
    logger.debug("Image complete barrier reached", run_id=state.get("run_id"))
    return state


# ---------------------------------------------------------------------------
# Dispatch Functions (Fan-out using Send)
# ---------------------------------------------------------------------------

def dispatch_researchers(state: GraphState) -> List[Send]:
    """Fan-out: spawn one researcher_node per section with a search_query.
    
    Args:
        state: Current graph state containing blog_plan with sections
        
    Returns:
        List of Send objects targeting researcher_node, or a Send to writer_dispatch
        if no research is required.
        
    Note:
        Each researcher gets minimal state: topic, run_id, and section-specific fields.
        This avoids passing unnecessary data to parallel workers.
    """
    # Validate required state fields
    if not state.get("blog_plan"):
        logger.error("Cannot dispatch researchers: missing blog_plan", run_id=state.get("run_id"))
        # Fallback: send to writer dispatch to continue workflow
        return [Send(WRITER_JOIN, state)]
    
    if not hasattr(state["blog_plan"], "sections"):
        logger.error("blog_plan missing sections attribute", run_id=state.get("run_id"))
        return [Send(WRITER_JOIN, state)]
    
    sends = []
    sections = state["blog_plan"].sections
    
    for section in sections:
        if section.search_query:  # Only spawn researchers for sections that need research
            sends.append(
                Send(
                    RESEARCHER,
                    {
                        # Global fields
                        "topic": state["topic"],
                        "run_id": state["run_id"],
                        # Section-specific fields
                        "section_id": section.id,
                        "search_query": section.search_query,
                        "section_description": section.description,
                        "blog_plan": state["blog_plan"],
                        # Initialize result containers
                        "research_results": [],
                        "section_drafts": [],
                        "generated_images": [],
                        "citation_registry": {},
                        "research_required": state.get("research_required", True),
                        "final_blog_md": "",
                        "final_blog_html": "",
                        "error": None,
                    }
                )
            )
    
    logger.info(
        "Dispatched researchers",
        count=len(sends),
        total_sections=len(sections),
        run_id=state.get("run_id")
    )
    
    # If no research needed, go directly to writer dispatch via the join node
    if not sends:
        return [Send(WRITER_JOIN, state)]
    
    return sends


def dispatch_writers(state: GraphState) -> List[Send]:
    """Fan-out: spawn one writer_node per section.
    
    Args:
        state: Current graph state (either from research_complete or writer_dispatch)
        
    Returns:
        List of Send objects targeting writer_node, one per section.
    """
    if not state.get("blog_plan") or not hasattr(state["blog_plan"], "sections"):
        logger.error("Cannot dispatch writers: missing blog_plan sections", run_id=state.get("run_id"))
        return [Send(IMAGE_JOIN, state)]  # Skip to next stage if possible
    
    sends = []
    sections = state["blog_plan"].sections
    
    for section in sections:
        # Each writer gets the full state plus section overrides
        sends.append(
            Send(
                WRITER,
                {
                    **state,
                    "section_id": section.id,
                    "section_title": section.title,
                    "section_description": section.description,
                    "word_count": section.word_count,
                    "image_prompt": section.image_prompt,
                }
            )
        )
    
    logger.info(
        "Dispatched writers",
        count=len(sends),
        run_id=state.get("run_id")
    )
    
    return sends


def dispatch_image_agents(state: GraphState) -> List[Send]:
    """Fan-out: spawn image agents for feature image and per-section images.
    
    Args:
        state: Current graph state from writer_complete
        
    Returns:
        List of Send objects targeting image_agent_node, one for each image prompt.
        If no prompts exist, returns a Send to image_complete to continue workflow.
    """
    sends = []
    run_id = state.get("run_id")
    
    # Validate blog_plan exists
    if not state.get("blog_plan"):
        logger.error("Cannot dispatch image agents: missing blog_plan", run_id=run_id)
        return [Send(IMAGE_JOIN, state)]
    
    # Feature image
    feature_prompt = state["blog_plan"].feature_image_prompt
    if feature_prompt:
        sends.append(
            Send(
                IMAGE_AGENT,
                {
                    **state,
                    "section_id": "feature",
                    "image_prompt": feature_prompt,
                }
            )
        )
    
    # Section images
    sections = state["blog_plan"].sections
    for section in sections:
        if section.image_prompt:  # Only spawn if prompt exists
            sends.append(
                Send(
                    IMAGE_AGENT,
                    {
                        **state,
                        "section_id": section.id,
                        "image_prompt": section.image_prompt,
                    }
                )
            )
    
    logger.info(
        "Dispatched image agents",
        feature_image=bool(feature_prompt),
        section_images=len(sends) - (1 if feature_prompt else 0),
        total=len(sends),
        run_id=run_id
    )
    
    # If no images to generate, proceed directly to next stage
    if not sends:
        return [Send(IMAGE_JOIN, state)]
    
    return sends


# ---------------------------------------------------------------------------
# Conditional Routing Functions
# ---------------------------------------------------------------------------

def route_after_planner(state: GraphState) -> Union[str, List[Send]]:
    """Determine next step after planner based on research requirements.
    
    Args:
        state: Graph state after planner execution
        
    Returns:
        - If research is required and sections need research: list of Send to researchers
        - If no research needed: name of join node to trigger writer dispatch
    """
    research_required = state.get("research_required", True)
    
    if not research_required:
        logger.info("No research required, proceeding to writers", run_id=state.get("run_id"))
        return RESEARCH_JOIN  # Will then dispatch writers via conditional edge
    
    # Check if any sections actually need research
    if state.get("blog_plan") and hasattr(state["blog_plan"], "sections"):
        has_research = any(section.search_query for section in state["blog_plan"].sections)
        if has_research:
            return dispatch_researchers(state)
    
    # No sections need research despite flag being true
    logger.warning(
        "Research required but no sections have search_queries, skipping to writers",
        run_id=state.get("run_id")
    )
    return RESEARCH_JOIN


def route_after_research(state: GraphState) -> List[Send]:
    """After research completes, always dispatch writers.
    
    Args:
        state: Graph state from research_complete join node
        
    Returns:
        List of Send objects to writer_node
    """
    return dispatch_writers(state)


def route_after_writers(state: GraphState) -> List[Send]:
    """After writing completes, dispatch image agents.
    
    Args:
        state: Graph state from writer_complete join node
        
    Returns:
        List of Send objects to image_agent_node
    """
    return dispatch_image_agents(state)


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct the complete StateGraph with all nodes and edges.
    
    Returns:
        StateGraph instance (not yet compiled)
    """
    builder = StateGraph(GraphState)
    
    # Add all nodes
    builder.add_node(ROUTER, router_node)
    builder.add_node(PLANNER, planner_node)
    builder.add_node(RESEARCHER, researcher_node)
    builder.add_node(WRITER, writer_node)
    builder.add_node(IMAGE_AGENT, image_agent_node)
    builder.add_node(CITATION, citation_manager_node)
    builder.add_node(REDUCER, reducer_node)
    builder.add_node(RESEARCH_JOIN, research_complete)
    builder.add_node(WRITER_JOIN, writer_complete)
    builder.add_node(IMAGE_JOIN, image_complete)
    
    logger.info("Added all nodes to graph", node_count=len(builder.nodes))
    
    # -----------------------------------------------------------------------
    # Linear start
    # -----------------------------------------------------------------------
    builder.add_edge(START, ROUTER)
    builder.add_edge(ROUTER, PLANNER)
    
    # -----------------------------------------------------------------------
    # Conditional after planner
    # -----------------------------------------------------------------------
    builder.add_conditional_edges(PLANNER, route_after_planner)
    
    # -----------------------------------------------------------------------
    # Research fan-in
    # -----------------------------------------------------------------------
    builder.add_edge(RESEARCHER, RESEARCH_JOIN)
    
    # After research, dispatch writers
    builder.add_conditional_edges(RESEARCH_JOIN, route_after_research)
    
    # -----------------------------------------------------------------------
    # Writer fan-in
    # -----------------------------------------------------------------------
    builder.add_edge(WRITER, WRITER_JOIN)
    
    # After writers, dispatch image agents
    builder.add_conditional_edges(WRITER_JOIN, route_after_writers)
    
    # -----------------------------------------------------------------------
    # Image fan-in
    # -----------------------------------------------------------------------
    builder.add_edge(IMAGE_AGENT, IMAGE_JOIN)
    
    # -----------------------------------------------------------------------
    # Linear finish
    # -----------------------------------------------------------------------
    builder.add_edge(IMAGE_JOIN, CITATION)
    builder.add_edge(CITATION, REDUCER)
    builder.add_edge(REDUCER, END)
    
    logger.info("Graph topology constructed")
    
    return builder


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_graph() -> StateGraph:
    """Build and compile the graph with persistence.
    
    Returns:
        Compiled StateGraph ready for invocation
    """
    builder = build_graph()
    
    # Initialize checkpointer for persistence
    try:
        checkpointer = get_checkpointer()
        logger.info("Checkpointer obtained successfully")
    except Exception as e:
        logger.error("Failed to initialize checkpointer", error=str(e))
        raise RuntimeError(f"Cannot compile graph without checkpointer: {e}") from e
    
    # Compile with checkpointer
    compiled = builder.compile(checkpointer=checkpointer)
    
    logger.info(
        "Graph compiled successfully",
        node_count=len(builder.nodes),
        edge_count=len(builder.edges) if hasattr(builder, "edges") else "unknown"
    )
    
    return compiled


# Export compiled graph as the primary interface
compiled_graph = compile_graph()

__all__ = [
    "compiled_graph",
    "build_graph",
    "compile_graph",
    "dispatch_researchers",
    "dispatch_writers",
    "dispatch_image_agents",
]