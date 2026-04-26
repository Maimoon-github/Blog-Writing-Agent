from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from blog_agent_system.core.state import BlogState
from blog_agent_system.agents.research_agent import research_node
from blog_agent_system.agents.outline_agent import outline_node
from blog_agent_system.agents.writer_agent import writer_node
from blog_agent_system.agents.editor_agent import editor_node
from blog_agent_system.agents.seo_agent import seo_node
from blog_agent_system.agents.fact_checker_agent import fact_checker_node
from blog_agent_system.agents.image_agent import image_node


def quality_gate_node(state: BlogState) -> dict:
    """Quality gate with revision decision."""
    if state.quality_score >= 0.85 or state.revision_count >= state.max_revisions:
        return {"status": "complete", "current_step": "quality_gate"}
    return {
        "status": "revising",
        "current_step": "quality_gate",
        "revision_count": state.revision_count + 1,
        "revision_feedback": "Improve clarity, transitions, and factual accuracy based on previous feedback.",
    }


def create_blog_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Create the complete LangGraph supervisor workflow."""
    builder = StateGraph(BlogState)

    # Nodes
    builder.add_node("research", research_node)
    builder.add_node("outline", outline_node)
    builder.add_node("write", writer_node)
    builder.add_node("edit", editor_node)
    builder.add_node("seo", seo_node)
    builder.add_node("fact_check", fact_checker_node)
    builder.add_node("image_gen", image_node)
    builder.add_node("quality_gate", quality_gate_node)

    # Sequential pipeline
    builder.set_entry_point("research")
    builder.add_edge("research", "outline")
    builder.add_edge("outline", "write")
    builder.add_edge("write", "edit")

    # Parallel fork
    builder.add_edge("edit", "seo")
    builder.add_edge("edit", "fact_check")

    # Join parallel branches
    builder.add_edge("seo", "quality_gate")
    builder.add_edge("fact_check", "quality_gate")

    # Conditional revision loop
    builder.add_conditional_edges(
        "quality_gate",
        lambda s: "accept" if s.quality_score >= 0.85 or s.revision_count >= s.max_revisions else "revise",
        {"accept": END, "revise": "write"}
    )

    # Conditional image generation
    builder.add_conditional_edges(
        "quality_gate",
        lambda s: "generate" if s.include_images else "skip",
        {"generate": "image_gen", "skip": END}
    )
    builder.add_edge("image_gen", END)

    return builder.compile(checkpointer=checkpointer)