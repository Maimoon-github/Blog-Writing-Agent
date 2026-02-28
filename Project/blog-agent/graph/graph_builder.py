"""
graph/graph_builder.py

Defines and compiles the complete LangGraph StateGraph for the blog-agent project.
Imports all node functions and state, defines the graph topology, dispatch functions,
and exports the compiled graph with a SqliteSaver checkpointer.
"""

from typing import List

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.citation_manager import citation_manager_node
from agents.image_agent import image_agent_node
from agents.planner import planner_node
from agents.reducer import reducer_node
from agents.researcher import researcher_node
from agents.router import router_node
from agents.writer import writer_node
from graph.checkpointer import get_checkpointer
from graph.state import GraphState


# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

def dispatch_researchers(state: GraphState) -> List[Send]:
    """
    Fan-out: spawn one researcher_node per section that has a search_query.
    If no sections need research, skip straight to writer dispatch.
    """
    sends = [
        Send(
            "researcher_node",
            {
                "topic": state["topic"],
                "run_id": state["run_id"],
                "research_results": [],
                "section_drafts": [],
                "generated_images": [],
                "section_id": section.id,
                "search_query": section.search_query,
                "section_description": section.description,
                "blog_plan": state["blog_plan"],
                "citation_registry": {},
                "research_required": state["research_required"],
                "final_blog_md": "",
                "final_blog_html": "",
                "error": None,
            },
        )
        for section in state["blog_plan"].sections
        if section.search_query is not None
    ]

    # If nothing needs researching, jump directly to the writer barrier node.
    if not sends:
        return [Send("writer_dispatch", state)]

    return sends


def dispatch_writers(state: GraphState) -> List[Send]:
    """
    Fan-out: spawn one writer_node per section in the blog plan.
    """
    return [
        Send(
            "writer_node",
            {
                **state,
                "section_id": section.id,
                "section_title": section.title,
                "section_description": section.description,
                "word_count": section.word_count,
                "image_prompt": section.image_prompt,
            },
        )
        for section in state["blog_plan"].sections
    ]


def dispatch_image_agents(state: GraphState) -> List[Send]:
    """
    Fan-out: spawn one image_agent_node for the feature image plus one per section.
    """
    sends: List[Send] = []

    # Feature image
    sends.append(
        Send(
            "image_agent_node",
            {
                **state,
                "section_id": "feature",
                "image_prompt": state["blog_plan"].feature_image_prompt,
            },
        )
    )

    # Per-section images
    for section in state["blog_plan"].sections:
        sends.append(
            Send(
                "image_agent_node",
                {
                    **state,
                    "section_id": section.id,
                    "image_prompt": section.image_prompt,
                },
            )
        )

    return sends


# ---------------------------------------------------------------------------
# No-op barrier / join nodes
# ---------------------------------------------------------------------------

def _noop(state: GraphState) -> GraphState:
    """Pass-through node used as a fan-in synchronisation barrier."""
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

builder = StateGraph(GraphState)

# --- Worker nodes ---
builder.add_node("router_node", router_node)
builder.add_node("planner_node", planner_node)
builder.add_node("researcher_node", researcher_node)
builder.add_node("writer_node", writer_node)
builder.add_node("image_agent_node", image_agent_node)
builder.add_node("citation_manager_node", citation_manager_node)
builder.add_node("reducer_node", reducer_node)

# --- Barrier / join nodes ---
builder.add_node("research_complete", _noop)   # fan-in for all researcher_node outputs
builder.add_node("writer_dispatch", _noop)      # fallback target when no research needed
builder.add_node("writer_complete", _noop)      # fan-in for all writer_node outputs
builder.add_node("image_complete", _noop)       # fan-in for all image_agent_node outputs

# --- Fixed edges ---
builder.add_edge(START, "router_node")
builder.add_edge("router_node", "planner_node")

# researcher_node outputs converge at research_complete
builder.add_edge("researcher_node", "research_complete")

# writer_dispatch (no-research shortcut) also feeds into the writer fan-out
builder.add_edge("writer_dispatch", "writer_complete")  # bypass: go straight to writer barrier
# NOTE: Because writer_dispatch is a _noop that receives the full state we use a
#       conditional edge from it so it runs dispatch_writers the same way.

# writer_node outputs converge at writer_complete
builder.add_edge("writer_node", "writer_complete")

# image_agent_node outputs converge at image_complete
builder.add_edge("image_agent_node", "image_complete")

# Sequential tail of the pipeline
builder.add_edge("image_complete", "citation_manager_node")
builder.add_edge("citation_manager_node", "reducer_node")
builder.add_edge("reducer_node", END)

# --- Conditional (fan-out) edges ---

# planner → dispatch researchers (or skip to writers if none needed)
builder.add_conditional_edges("planner_node", dispatch_researchers)

# research_complete → dispatch writers
builder.add_conditional_edges("research_complete", dispatch_writers)

# writer_dispatch (no-research path) → dispatch writers
builder.add_conditional_edges("writer_dispatch", dispatch_writers)

# writer_complete → dispatch image agents
builder.add_conditional_edges("writer_complete", dispatch_image_agents)

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

checkpointer = get_checkpointer()
compiled_graph = builder.compile(checkpointer=checkpointer)