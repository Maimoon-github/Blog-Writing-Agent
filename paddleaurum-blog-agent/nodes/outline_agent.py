# import json
# import logging
# from typing import Any, Dict, List

# from crewai import Agent, Crew, Process, Task
# from langchain_community.llms import Ollama
# from langchain_community.vectorstores import Chroma
# from langchain_core.tools import tool

# from graph.state import AgentState, ContentOutline

# logger = logging.getLogger(__name__)


# def _make_rag_tool(vectordb: Chroma):
#     @tool("content_pattern_retrieval")
#     def content_pattern_retrieval(query: str) -> str:
#         """Retrieve content structure patterns and competitor outlines from the knowledge base."""
#         docs = vectordb.similarity_search(query, k=4)
#         return "\n\n".join(doc.page_content for doc in docs) if docs else "No results found."
#     return content_pattern_retrieval


# async def outline_agent_node(state: AgentState, *, llm: Ollama, vectordb: Chroma) -> dict:
#     try:
#         topic: str = state["topic"]
#         keyword_map = state.get("keyword_map") or {}
#         primary_kw: str = keyword_map.get("primary", topic)
#         intent_type: str = str(keyword_map.get("intent_type", "informational"))
#         task_plan: dict = state.get("task_plan") or {}
#         key_sections: List[str] = task_plan.get("key_sections", [])

#         rag_tool = _make_rag_tool(vectordb)

#         agent = Agent(
#             role="Content Strategist & Outline Architect",
#             goal="Build perfectly structured outlines for pickleball coaching articles that match SERP intent and maximise topical authority.",
#             backstory=(
#                 "You analyse top-ranking content and build comprehensive outlines that beat competitors. "
#                 "You understand topical clustering, entity coverage, and FAQ schema opportunities for pickleball content."
#             ),
#             llm=llm,
#             tools=[rag_tool],
#             memory=True,
#             verbose=False,
#             max_iter=3,
#         )

#         key_sections_hint = (
#             f"Suggested sections from planner: {', '.join(key_sections)}" if key_sections else ""
#         )

#         task_description = (
#             f"Topic: {topic}\n"
#             f"Primary keyword: {primary_kw}\n"
#             f"Search intent: {intent_type}\n"
#             f"{key_sections_hint}\n\n"
#             "Use content_pattern_retrieval to check patterns for high-ranking pickleball articles, "
#             "then build a complete content outline. "
#             "Return ONLY valid JSON — no markdown fences:\n"
#             "{\n"
#             '  "headings": [\n'
#             '    {"level": "H1", "text": "..."},\n'
#             '    {"level": "H2", "text": "..."},\n'
#             '    {"level": "H3", "text": "..."},\n'
#             '    ...\n'
#             "  ],\n"
#             '  "faq_candidates": [\n'
#             '    "Question 1?",\n'
#             '    "Question 2?",\n'
#             '    ...\n'
#             "  ],\n"
#             '  "internal_link_placeholders": [\n'
#             '    "Anchor text for link 1",\n'
#             '    "Anchor text for link 2",\n'
#             '    "Anchor text for link 3"\n'
#             "  ]\n"
#             "}\n\n"
#             "Rules:\n"
#             "- Exactly one H1 (the article title, containing the primary keyword).\n"
#             "- 5-8 H2 sections covering the topic comprehensively.\n"
#             "- 2-3 H3s under each H2 where appropriate.\n"
#             "- Minimum 4 FAQ candidates (questions a reader would ask).\n"
#             "- Minimum 3 internal_link_placeholders (anchor texts linking to other paddleaurum.com articles)."
#         )

#         task = Task(
#             description=task_description,
#             agent=agent,
#             expected_output="Valid JSON content outline with headings, faq_candidates, internal_link_placeholders.",
#         )

#         crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
#         raw_result: str = crew.kickoff()

#         cleaned = raw_result.strip().strip("```json").strip("```").strip()
#         parsed: dict = json.loads(cleaned)

#         headings: List[Dict[str, Any]] = [
#             {"level": str(h.get("level", "H2")), "text": str(h.get("text", ""))}
#             for h in parsed.get("headings", [])
#             if h.get("text")
#         ]
#         faq_candidates: List[str] = [str(q) for q in parsed.get("faq_candidates", []) if q]
#         internal_links: List[str] = [str(a) for a in parsed.get("internal_link_placeholders", []) if a]

#         content_outline = ContentOutline(
#             headings=headings,
#             faq_candidates=faq_candidates,
#             internal_link_placeholders=internal_links,
#         )

#         logger.info(
#             "Outline agent: %d headings, %d FAQs, %d internal links",
#             len(headings), len(faq_candidates), len(internal_links),
#         )

#         return {
#             "content_outline":            content_outline,
#             "faq_candidates":             faq_candidates,
#             "internal_link_placeholders": internal_links,
#             "error":                      None,
#             "error_node":                 None,
#         }

#     except Exception as exc:
#         logger.exception("Outline agent failed.")
#         return {"error": str(exc), "error_node": "outline_agent"}

















# @################################################################





















import json
import logging
from typing import Any, Dict, List

from agents.content_strategist import run as run_content_strategist
from graph.state import AgentState, ContentOutline

logger = logging.getLogger(__name__)


async def outline_agent_node(state: AgentState, *, llm, vectordb) -> dict:
    try:
        topic: str = state["topic"]
        keyword_map = state.get("keyword_map") or {}
        primary_kw: str = keyword_map.get("primary", topic)
        intent_type: str = str(keyword_map.get("intent_type", "informational"))
        task_plan: dict = state.get("task_plan") or {}
        key_sections: List[str] = task_plan.get("key_sections", [])

        raw_result: str = run_content_strategist(
            llm=llm,
            vectordb=vectordb,
            topic=topic,
            primary_kw=primary_kw,
            intent_type=intent_type,
            key_sections=key_sections,
        )

        cleaned = raw_result.strip().strip("```json").strip("```").strip()
        parsed: dict = json.loads(cleaned)

        headings: List[Dict[str, Any]] = [
            {"level": str(h.get("level", "H2")), "text": str(h.get("text", ""))}
            for h in parsed.get("headings", [])
            if h.get("text")
        ]
        faq_candidates: List[str] = [str(q) for q in parsed.get("faq_candidates", []) if q]
        internal_links: List[str] = [str(a) for a in parsed.get("internal_link_placeholders", []) if a]

        content_outline = ContentOutline(
            headings=headings,
            faq_candidates=faq_candidates,
            internal_link_placeholders=internal_links,
        )

        logger.info(
            "Outline agent: %d headings, %d FAQs, %d internal links",
            len(headings), len(faq_candidates), len(internal_links),
        )

        return {
            "content_outline":            content_outline,
            "faq_candidates":             faq_candidates,
            "internal_link_placeholders": internal_links,
            "error":                      None,
            "error_node":                 None,
        }

    except Exception as exc:
        logger.exception("Outline agent failed.")
        return {"error": str(exc), "error_node": "outline_agent"}