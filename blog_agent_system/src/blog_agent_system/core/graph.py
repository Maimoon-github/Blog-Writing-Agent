# src/blog_agent_system/core/graph.py (simplified)
from langgraph.graph import StateGraph, END
from blog_agent_system.agents import (
    ResearchAgent, OutlineAgent, WriterAgent, 
    EditorAgent, SEOAgent, FactCheckerAgent
)

def create_blog_graph():
    builder = StateGraph(BlogState)
    
    # Register nodes
    builder.add_node("research", ResearchAgent().execute)
    builder.add_node("outline", OutlineAgent().execute)
    builder.add_node("write", WriterAgent().execute)
    builder.add_node("edit", EditorAgent().execute)
    builder.add_node("seo", SEOAgent().execute)
    builder.add_node("fact_check", FactCheckerAgent().execute)
    builder.add_node("quality_gate", quality_gate_node)
    
    # Sequential edges
    builder.set_entry_point("research")
    builder.add_edge("research", "outline")
    builder.add_edge("outline", "write")
    builder.add_edge("write", "edit")
    
    # Parallel fork after edit
    builder.add_edge("edit", "seo")
    builder.add_edge("edit", "fact_check")
    
    # Join parallel branches
    builder.add_edge("seo", "quality_gate")
    builder.add_edge("fact_check", "quality_gate")
    
    # Conditional loopback
    builder.add_conditional_edges(
        "quality_gate",
        lambda state: "accept" if state.quality_score >= 0.85 or state.revision_count >= 3 else "revise",
        {"accept": END, "revise": "write"}
    )
    
    return builder.compile()









# src/blog_agent_system/core/graph.py
from typing import Literal

def should_generate_images(state: BlogState) -> Literal["generate", "skip"]:
    """Conditional entry for image generation based on user preference."""
    return "generate" if state.get("include_images", False) else "skip"

def parallel_join_condition(state: BlogState) -> Literal["quality_gate", "wait"]:
    """Ensures both SEO and FactChecker complete before quality gate."""
    required_keys = ["seo_metadata", "fact_check_results"]
    if all(k in state and state[k] for k in required_keys):
        return "quality_gate"
    return "wait"

def revision_router(state: BlogState) -> Literal["accept", "revise"]:
    """Routes to END or back to writer based on quality score."""
    if state.quality_score >= 0.85:
        return "accept"
    if state.revision_count >= state.max_revisions:
        return "accept"  # Force accept after max iterations
    return "revise"