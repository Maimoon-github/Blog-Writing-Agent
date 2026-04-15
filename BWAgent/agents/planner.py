"""Planner node for BWAgent that generates a structured BlogPlan."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from graph.state import BlogPlan, GraphState, Section
from memory.chroma_store import chroma_store

logger = structlog.get_logger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "planner_prompt.txt"
DEFAULT_TITLE_PREFIX = "Blog post about"


def planner_node(state: GraphState) -> Dict[str, Any]:
    topic = state.get("topic", "").strip()
    research_required = state.get("research_required", True)
    if not topic:
        logger.error("planner_node.no_topic")
        return {"error": "No topic provided"}

    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.critical("planner_node.prompt_missing", path=str(PROMPT_PATH))
        raise

    context_text = _build_context(state.get("topic", ""))
    user_message = f"Topic: {topic}\nResearch required: {research_required}"
    if context_text:
        user_message += f"\n\n{context_text}"

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

    try:
        structured = llm.with_structured_output(BlogPlan)
        blog_plan = structured.invoke(messages)
    except Exception as exc:
        logger.warning("planner_node.structured_output_failed", error=str(exc))
        try:
            raw = llm.invoke(messages).content
            blog_plan = _parse_fallback(raw, topic, research_required)
        except Exception as parse_exc:
            logger.error("planner_node.fallback_failed", error=str(parse_exc))
            blog_plan = _default_plan(topic, research_required)

    if not blog_plan.blog_title:
        blog_plan.blog_title = f"{DEFAULT_TITLE_PREFIX} {topic}"

    logger.info(
        "planner_node.generated",
        topic=topic,
        section_count=len(blog_plan.sections),
        research_required=blog_plan.research_required,
    )
    return {"blog_plan": blog_plan}


def _build_context(topic: str) -> str:
    try:
        results = chroma_store.search_similar(topic, n_results=3)
    except Exception as exc:
        logger.warning("planner_node.chromadb_error", error=str(exc))
        return ""

    if not results:
        return ""

    items = []
    for idx, item in enumerate(results, start=1):
        sources = item.get("source_urls", [])
        items.append(
            f"Research Item {idx}:\nSummary: {item.get('summary', '')}\nSources: {', '.join(sources[:3])}"
        )

    return "Prior research context:\n" + "\n---\n".join(items)


def _parse_fallback(raw: str, topic: str, research_required: bool) -> BlogPlan:
    try:
        data = json.loads(raw)
        return _dict_to_blog_plan(data, topic, research_required)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return _dict_to_blog_plan(data, topic, research_required)
            except Exception:
                pass

    logger.warning("planner_node.raw_parse_failed", topic=topic)
    return _default_plan(topic, research_required)


def _dict_to_blog_plan(data: Dict[str, Any], topic: str, research_required: bool) -> BlogPlan:
    sections_raw = data.get("sections", [])
    sections: List[Section] = []
    for item in sections_raw:
        try:
            sections.append(Section(
                id=item.get("id", "section_1"),
                title=item.get("title", ""),
                description=item.get("description", ""),
                word_count=int(item.get("word_count", 300)),
                search_query=item.get("search_query"),
                image_prompt=item.get("image_prompt", ""),
            ))
        except Exception:
            continue

    return BlogPlan(
        blog_title=data.get("blog_title", f"Blog post about {topic}"),
        feature_image_prompt=data.get("feature_image_prompt", ""),
        research_required=data.get("research_required", research_required),
        sections=sections,
    )


def _default_plan(topic: str, research_required: bool) -> BlogPlan:
    return BlogPlan(
        blog_title=f"Blog post about {topic}",
        feature_image_prompt=f"Photorealistic hero image for a blog about {topic}, high quality, cinematic lighting, professional photography",
        research_required=research_required,
        sections=[
            Section(
                id="section_1",
                title="Introduction",
                description=f"Introduce the topic and explain why it matters.",
                word_count=350,
                search_query=None if not research_required else f"{topic} overview",
                image_prompt=f"Photorealistic image representing the topic {topic}, clean composition, editorial style",
            ),
            Section(
                id="section_2",
                title="Core concepts",
                description=f"Explain the most important concepts related to {topic}.",
                word_count=400,
                search_query=None if not research_required else f"{topic} key concepts",
                image_prompt=f"Photorealistic conceptual illustration for {topic}, focused detail, soft lighting",
            ),
            Section(
                id="section_3",
                title="Best practices",
                description=f"Describe practical best practices and actionable advice.",
                word_count=400,
                search_query=None if not research_required else f"{topic} best practices",
                image_prompt=f"Professional scene showing practical application of {topic}, crisp detail, engaging composition",
            ),
            Section(
                id="section_4",
                title="Future outlook",
                description=f"Explore future trends and what readers should watch next.",
                word_count=350,
                search_query=None if not research_required else f"future of {topic}",
                image_prompt=f"Future-focused image for {topic}, modern aesthetic, polished editorial photography",
            ),
        ],
    )
