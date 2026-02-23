# import json
# import logging
# from typing import List

# from crewai import Agent, Crew, Process, Task
# from langchain_community.llms import Ollama
# from langchain_community.vectorstores import Chroma
# from langchain_core.tools import tool

# from graph.state import AgentState, IntentType, KeywordMap

# logger = logging.getLogger(__name__)


# def _make_rag_tool(vectordb: Chroma):
#     @tool("seo_knowledge_retrieval")
#     def seo_knowledge_retrieval(query: str) -> str:
#         """Retrieve SEO guidelines and keyword best practices from the knowledge base."""
#         docs = vectordb.similarity_search(query, k=4)
#         return "\n\n".join(doc.page_content for doc in docs) if docs else "No results found."
#     return seo_knowledge_retrieval


# async def keyword_mapper_node(state: AgentState, *, llm: Ollama, vectordb: Chroma) -> dict:
#     try:
#         topic: str = state["topic"]
#         target_kw: str = state.get("target_keyword") or ""
#         research_context = "\n".join(
#             f"- {s['title']}: {s['snippet']}"
#             for s in (state.get("research_snippets") or [])[:8]
#         )

#         rag_tool = _make_rag_tool(vectordb)

#         agent = Agent(
#             role="Senior SEO Strategist",
#             goal=(
#                 "Extract the most effective keyword strategy for pickleball content "
#                 "that will rank on Google for paddleaurum.com."
#             ),
#             backstory=(
#                 "You have 8 years of SEO experience specialising in sports and coaching content. "
#                 "You deeply understand Google's E-E-A-T signals, semantic search, and topical authority. "
#                 "You know paddleaurum.com's content library and its competitive landscape."
#             ),
#             llm=llm,
#             tools=[rag_tool],
#             memory=True,
#             verbose=False,
#             max_iter=3,
#         )

#         task_description = (
#             f"Topic: {topic}\n"
#             f"Provided keyword hint: {target_kw or 'none'}\n\n"
#             f"Research context:\n{research_context}\n\n"
#             "Using the seo_knowledge_retrieval tool to check SEO guidelines, produce a keyword map. "
#             "Return ONLY valid JSON with this exact structure — no markdown fences:\n"
#             "{\n"
#             '  "primary": "single best-ranking keyword phrase",\n'
#             '  "secondary": ["kw1", "kw2", "kw3", "kw4"],\n'
#             '  "lsi": ["related_term1", "related_term2", "related_term3", "related_term4", "related_term5"],\n'
#             '  "entities": ["entity1", "entity2", "entity3"],\n'
#             '  "intent_type": "informational|commercial|navigational"\n'
#             "}"
#         )

#         task = Task(
#             description=task_description,
#             agent=agent,
#             expected_output="Valid JSON keyword map with primary, secondary, lsi, entities, intent_type.",
#         )

#         crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
#         raw_result: str = crew.kickoff()

#         cleaned = raw_result.strip().strip("```json").strip("```").strip()
#         parsed: dict = json.loads(cleaned)

#         intent_map = {i.value: i for i in IntentType}
#         intent_type = intent_map.get(
#             parsed.get("intent_type", "informational").lower(),
#             IntentType.INFORMATIONAL,
#         )

#         keyword_map = KeywordMap(
#             primary=str(parsed.get("primary", topic)).lower().strip(),
#             secondary=[str(k).lower() for k in parsed.get("secondary", [])],
#             lsi=[str(k).lower() for k in parsed.get("lsi", [])],
#             entities=[str(e) for e in parsed.get("entities", [])],
#             intent_type=intent_type,
#         )

#         logger.info("Keyword mapper: primary='%s' secondary=%d lsi=%d",
#                     keyword_map["primary"], len(keyword_map["secondary"]), len(keyword_map["lsi"]))

#         return {
#             "keyword_map":  keyword_map,
#             "intent_type":  intent_type,
#             "error":        None,
#             "error_node":   None,
#         }

#     except Exception as exc:
#         logger.exception("Keyword mapper failed.")
#         return {"error": str(exc), "error_node": "keyword_mapper"}












# @##################################################################################



















import json
import logging
from typing import List

from agents.seo_strategist import run as run_seo_strategist
from graph.state import AgentState, IntentType, KeywordMap

logger = logging.getLogger(__name__)


async def keyword_mapper_node(state: AgentState, *, llm, vectordb) -> dict:
    try:
        topic: str = state["topic"]
        target_kw: str = state.get("target_keyword") or ""
        research_context = "\n".join(
            f"- {s['title']}: {s['snippet']}"
            for s in (state.get("research_snippets") or [])[:8]
        )

        raw_result: str = run_seo_strategist(
            llm=llm,
            vectordb=vectordb,
            topic=topic,
            target_kw=target_kw,
            research_context=research_context,
        )

        cleaned = raw_result.strip().strip("```json").strip("```").strip()
        parsed: dict = json.loads(cleaned)

        intent_map = {i.value: i for i in IntentType}
        intent_type = intent_map.get(
            parsed.get("intent_type", "informational").lower(),
            IntentType.INFORMATIONAL,
        )

        keyword_map = KeywordMap(
            primary=str(parsed.get("primary", topic)).lower().strip(),
            secondary=[str(k).lower() for k in parsed.get("secondary", [])],
            lsi=[str(k).lower() for k in parsed.get("lsi", [])],
            entities=[str(e) for e in parsed.get("entities", [])],
            intent_type=intent_type,
        )

        logger.info("Keyword mapper: primary='%s' secondary=%d lsi=%d",
                    keyword_map["primary"], len(keyword_map["secondary"]), len(keyword_map["lsi"]))

        return {
            "keyword_map":  keyword_map,
            "intent_type":  intent_type,
            "error":        None,
            "error_node":   None,
        }

    except Exception as exc:
        logger.exception("Keyword mapper failed.")
        return {"error": str(exc), "error_node": "keyword_mapper"}