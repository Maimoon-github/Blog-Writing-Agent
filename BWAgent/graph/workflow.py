"""LangGraph workflow orchestration for BWAgent."""

import structlog
from typing import Dict, List, Union

from langgraph.graph import END, Send, START, StateGraph

from agents.citation_manager import citation_manager_node
from agents.editor import editor_node
from agents.image_agent import image_agent_node
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.reducer import reducer_node
from agents.router import router_node
from agents.writer import writer_node
from graph.checkpointer import get_checkpointer
from graph.state import GraphState

logger = structlog.get_logger(__name__)

ROUTER = "router_node"
PLANNER = "planner_node"
RESEARCHER = "researcher_node"
WRITER = "writer_node"
EDITOR = "editor_node"
IMAGE_AGENT = "image_agent_node"
CITATION = "citation_manager_node"
REDUCER = "reducer_node"
RESEARCH_JOIN = "research_complete"
WRITER_JOIN = "writer_complete"
EDITOR_JOIN = "editor_complete"
IMAGE_JOIN = "image_complete"


def research_complete(state: GraphState) -> GraphState:
    logger.debug("Research complete barrier reached", run_id=state.get("run_id"))
    return state


def writer_complete(state: GraphState) -> GraphState:
    logger.debug("Writer complete barrier reached", run_id=state.get("run_id"))
    return state


def editor_complete(state: GraphState) -> GraphState:
    logger.debug("Editor complete barrier reached", run_id=state.get("run_id"))
    return state


def image_complete(state: GraphState) -> GraphState:
    logger.debug("Image complete barrier reached", run_id=state.get("run_id"))
    return state


def dispatch_researchers(state: GraphState) -> List[Send]:
    if not state.get("blog_plan"):
        logger.error("dispatch_researchers.missing_blog_plan", run_id=state.get("run_id"))
        return [Send(WRITER_JOIN, state)]

    sections = state["blog_plan"].sections
    sends: List[Send] = []

    for section in sections:
        if section.search_query:
            sends.append(
                Send(
                    RESEARCHER,
                    {
                        "topic": state["topic"],
                        "run_id": state["run_id"],
                        "section_id": section.id,
                        "search_query": section.search_query,
                        "section_description": section.description,
                        "blog_plan": state["blog_plan"],
                        "research_results": [],
                        "section_drafts": [],
                        "generated_images": [],
                        "citation_registry": {},
                        "final_blog_md": "",
                        "final_blog_html": "",
                        "research_required": state.get("research_required", True),
                        "error": None,
                    },
                )
            )

    if not sends:
        logger.info("dispatch_researchers.no_sections_needing_research", run_id=state.get("run_id"))
        return [Send(WRITER_JOIN, state)]

    logger.info(
        "dispatch_researchers.dispatched",
        count=len(sends),
        total_sections=len(sections),
        run_id=state.get("run_id"),
    )
    return sends


def dispatch_writers(state: GraphState) -> List[Send]:
    if not state.get("blog_plan"):
        logger.error("dispatch_writers.missing_blog_plan", run_id=state.get("run_id"))
        return [Send(EDITOR_JOIN, state)]

    sends: List[Send] = []
    for section in state["blog_plan"].sections:
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
                },
            )
        )

    logger.info("dispatch_writers.dispatched", count=len(sends), run_id=state.get("run_id"))
    return sends


def dispatch_editors(state: GraphState) -> List[Send]:
    drafts = state.get("section_drafts", [])
    if not drafts:
        logger.warning("dispatch_editors.no_section_drafts", run_id=state.get("run_id"))
        return [Send(IMAGE_JOIN, state)]

    sends: List[Send] = []
    for draft in drafts:
        sends.append(
            Send(
                EDITOR,
                {
                    **state,
                    "section_id": draft.section_id,
                    "section_title": draft.title,
                    "section_content": draft.content,
                    "citation_keys": draft.citation_keys,
                },
            )
        )

    logger.info("dispatch_editors.dispatched", count=len(sends), run_id=state.get("run_id"))
    return sends


def dispatch_image_agents(state: GraphState) -> List[Send]:
    if not state.get("blog_plan"):
        logger.error("dispatch_image_agents.missing_blog_plan", run_id=state.get("run_id"))
        return [Send(IMAGE_JOIN, state)]

    sends: List[Send] = []
    feature_prompt = state["blog_plan"].feature_image_prompt
    if feature_prompt:
        sends.append(
            Send(
                IMAGE_AGENT,
                {
                    **state,
                    "section_id": "feature",
                    "image_prompt": feature_prompt,
                },
            )
        )

    for section in state["blog_plan"].sections:
        if section.image_prompt:
            sends.append(
                Send(
                    IMAGE_AGENT,
                    {
                        **state,
                        "section_id": section.id,
                        "image_prompt": section.image_prompt,
                    },
                )
            )

    if not sends:
        logger.info("dispatch_image_agents.no_images", run_id=state.get("run_id"))
        return [Send(IMAGE_JOIN, state)]

    logger.info(
        "dispatch_image_agents.dispatched",
        count=len(sends),
        run_id=state.get("run_id"),
    )
    return sends


def route_after_planner(state: GraphState) -> Union[str, List[Send]]:
    if state.get("research_required") and state.get("blog_plan"):
        has_research = any(section.search_query for section in state["blog_plan"].sections)
        if has_research:
            return dispatch_researchers(state)

    return [Send(WRITER_JOIN, state)]


def route_after_research(state: GraphState) -> List[Send]:
    return dispatch_writers(state)


def route_after_writers(state: GraphState) -> List[Send]:
    return dispatch_editors(state)


def route_after_editors(state: GraphState) -> List[Send]:
    return dispatch_image_agents(state)


def build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    builder.add_node(ROUTER, router_node)
    builder.add_node(PLANNER, planner_node)
    builder.add_node(RESEARCHER, researcher_node)
    builder.add_node(WRITER, writer_node)
    builder.add_node(EDITOR, editor_node)
    builder.add_node(IMAGE_AGENT, image_agent_node)
    builder.add_node(CITATION, citation_manager_node)
    builder.add_node(REDUCER, reducer_node)
    builder.add_node(RESEARCH_JOIN, research_complete)
    builder.add_node(WRITER_JOIN, writer_complete)
    builder.add_node(EDITOR_JOIN, editor_complete)
    builder.add_node(IMAGE_JOIN, image_complete)

    builder.add_edge(START, ROUTER)
    builder.add_edge(ROUTER, PLANNER)
    builder.add_conditional_edges(PLANNER, route_after_planner)
    builder.add_edge(RESEARCHER, RESEARCH_JOIN)
    builder.add_conditional_edges(RESEARCH_JOIN, route_after_research)
    builder.add_edge(WRITER, WRITER_JOIN)
    builder.add_conditional_edges(WRITER_JOIN, route_after_writers)
    builder.add_edge(EDITOR, EDITOR_JOIN)
    builder.add_conditional_edges(EDITOR_JOIN, route_after_editors)
    builder.add_edge(IMAGE_AGENT, IMAGE_JOIN)
    builder.add_edge(IMAGE_JOIN, CITATION)
    builder.add_edge(CITATION, REDUCER)
    builder.add_edge(REDUCER, END)

    logger.info("Graph topology constructed", node_count=len(builder.nodes))
    return builder


def compile_graph() -> StateGraph:
    builder = build_graph()
    try:
        checkpointer = get_checkpointer()
    except Exception as e:
        logger.error("compile_graph.checkpointer_failed", error=str(e))
        raise
    compiled = builder.compile(checkpointer=checkpointer)
    logger.info("Graph compiled successfully")
    return compiled

compiled_graph = compile_graph()

__all__ = [
    "compiled_graph",
    "build_graph",
    "compile_graph",
]
