#!/usr/bin/env python
"""Entry point for paddleaurum.com autonomous blog generation pipeline."""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langsmith import Client

# Import node functions (each defined in its own module under nodes/)
from nodes import (
    input_validator_node,
    planner_node,
    research_worker_node,
    research_merger_node,
    keyword_mapper_node,
    outline_agent_node,
    coaching_writer_node,
    seo_auditor_node,
    reflection_node,
    image_selector_node,
    citation_formatter_node,
    schema_generator_node,
    final_assembler_node,
    human_review_gate_node,
    error_recovery_node,
    publish_node,
)
from state.schema import AgentState

# Load environment variables
load_dotenv()

# LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "paddleaurum-blog-agent")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

# Router functions (defined inline for simplicity; could be imported from graph.routers)
def route_after_validation(state: AgentState) -> str:
    if state.get("error"):
        return "error_recovery"
    return "planner"

def route_after_planner(state: AgentState):
    from langgraph.types import Send
    if not state.get("needs_research", False):
        return "keyword_mapper"
    # Fan-out to parallel research workers
    return [
        Send("research_worker", {"state": state, "query": q})
        for q in state.get("sub_queries", [])
    ]

def route_after_seo_audit(state: AgentState) -> str:
    score = state.get("seo_score", 0)
    iterations = state.get("revision_iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if score >= 85:
        return "image_selector"
    elif iterations < max_iter:
        return "reflection"
    else:
        return "human_review_gate"

def route_after_human_review(state: AgentState) -> str:
    if state.get("approved"):
        return "publish"
    return "writer"

# Build the graph
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("input_validator", input_validator_node)
workflow.add_node("planner", planner_node)
workflow.add_node("research_worker", research_worker_node)
workflow.add_node("research_merger", research_merger_node)
workflow.add_node("keyword_mapper", keyword_mapper_node)
workflow.add_node("outline_agent", outline_agent_node)
workflow.add_node("writer", coaching_writer_node)
workflow.add_node("seo_auditor", seo_auditor_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("image_selector", image_selector_node)
workflow.add_node("citation_formatter", citation_formatter_node)
workflow.add_node("schema_generator", schema_generator_node)
workflow.add_node("final_assembler", final_assembler_node)
workflow.add_node("human_review_gate", human_review_gate_node)
workflow.add_node("error_recovery", error_recovery_node)
workflow.add_node("publish", publish_node)

# Set entry point
workflow.set_entry_point("input_validator")

# Deterministic edges
workflow.add_edge("research_merger", "keyword_mapper")
workflow.add_edge("keyword_mapper", "outline_agent")
workflow.add_edge("reflection", "writer")
workflow.add_edge("citation_formatter", "schema_generator")
workflow.add_edge("schema_generator", "final_assembler")
workflow.add_edge("final_assembler", "human_review_gate")
workflow.add_edge("publish", END)

# Conditional edges
workflow.add_conditional_edges(
    "input_validator",
    route_after_validation,
    {"error_recovery": "error_recovery", "planner": "planner"},
)
workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {"keyword_mapper": "keyword_mapper", "research_worker": "research_worker"},
)
workflow.add_conditional_edges(
    "seo_auditor",
    route_after_seo_audit,
    {
        "image_selector": "image_selector",
        "reflection": "reflection",
        "human_review_gate": "human_review_gate",
    },
)
workflow.add_conditional_edges(
    "human_review_gate",
    route_after_human_review,
    {"publish": "publish", "writer": "writer"},
)

# Parallel branches after SEO approval: image selector and citation formatter run concurrently
workflow.add_edge("image_selector", "citation_formatter")

# Set up checkpointing (SQLite for production)
checkpointer = SqliteSaver.from_conn_string("./checkpoints/checkpoints.db")
app = workflow.compile(checkpointer=checkpointer)


async def run_pipeline(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the graph with the given initial state and return final state."""
    session_id = initial_state.get("session_id", str(uuid.uuid4()))
    config = {
        "configurable": {"thread_id": session_id},
    }
    # Add LangSmith tracing if desired
    if LANGSMITH_API_KEY:
        from langchain.callbacks.tracers import LangChainTracer
        tracer = LangChainTracer(project_name=os.environ["LANGCHAIN_PROJECT"])
        config["callbacks"] = [tracer]

    final_state = await app.ainvoke(initial_state, config=config)
    return final_state


def main(topic: str, target_keyword: str = None, tone: str = "coach", word_count_goal: int = 1500):
    """Synchronous entry point for CLI usage."""
    initial_state: AgentState = {
        "topic": topic,
        "target_keyword": target_keyword,
        "tone": tone,
        "word_count_goal": word_count_goal,
        "session_id": str(uuid.uuid4()),
        "needs_research": False,  # will be set by planner
        "sub_queries": [],
        "research_snippets": [],
        "research_sources": [],
        "keyword_map": None,
        "content_outline": None,
        "faq_candidates": [],
        "internal_link_placeholders": [],
        "draft_article": None,
        "revision_iteration": 0,
        "max_iterations": 3,
        "seo_score": None,
        "seo_issues": [],
        "seo_suggestions": [],
        "image_manifest": [],
        "formatted_article": None,
        "schema_markup": None,
        "title_tag": None,
        "meta_description": None,
        "url_slug": None,
        "final_output": None,
        "approved": False,
        "human_review_requested": False,
        "error": None,
        "error_node": None,
        "retry_count": 0,
    }

    result = asyncio.run(run_pipeline(initial_state))
    print("Pipeline completed. Final state keys:", result.keys())
    if result.get("final_output"):
        print("Article generated. Output:", result["final_output"])
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run autonomous blog generation.")
    parser.add_argument("--topic", required=True, help="Article topic")
    parser.add_argument("--target-keyword", help="Primary keyword (optional)")
    parser.add_argument("--tone", default="coach", choices=["coach", "expert", "beginner-friendly"])
    parser.add_argument("--word-count", type=int, default=1500, help="Target word count")
    args = parser.parse_args()

    main(
        topic=args.topic,
        target_keyword=args.target_keyword,
        tone=args.tone,
        word_count_goal=args.word_count,
    )