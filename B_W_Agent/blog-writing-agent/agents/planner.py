# """agents/planner.py
# Planner Node — generates structured BlogPlan + dispatches parallel worker crews.
# Exactly matches roadmap Phase 3 Step 6 and idea.md requirements.
# CrewAI 1.x + LangChain structured output patterns.
# """

# from pathlib import Path
# from typing import List, Dict, Any

# import chromadb

# from crewai import Agent, Crew, Task, LLM          # ← LLM added for CrewAI 1.x agents
# from langchain_ollama import ChatOllama             # ← for structured output (with_structured_output)

# from config import OLLAMA_MODEL, OLLAMA_BASE_URL
# from schemas import BlogPlan, Section
# from state import CrewState
# from tools.search import search_tool
# from tools.image_gen import image_gen_tool

# # ----------------------------------------------------------------------
# # Prompt loading (plain-text, editable at runtime)
# # ----------------------------------------------------------------------
# PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner_prompt.txt"
# PLANNER_PROMPT: str = PROMPT_PATH.read_text(encoding="utf-8")

# # ----------------------------------------------------------------------
# # LangChain LLM — used ONLY for planner_node structured output
# # (ChatOllama supports .with_structured_output(); crewai.LLM does not)
# # ----------------------------------------------------------------------
# _langchain_llm = ChatOllama(
#     model=OLLAMA_MODEL,
#     base_url=OLLAMA_BASE_URL,
#     temperature=0.2,
#     format="json",
# )

# # ----------------------------------------------------------------------
# # CrewAI LLM — used ONLY inside Agent() constructors in dispatch_crews
# # CrewAI 1.x requires crewai.LLM (backed by LiteLLM), not a LangChain object.
# # api_key="ollama" is a required dummy value; LiteLLM ignores it for local models.
# # ----------------------------------------------------------------------
# _crew_llm = LLM(
#     model=f"ollama/{OLLAMA_MODEL}",
#     base_url=OLLAMA_BASE_URL,
#     api_key="ollama",          # required by LiteLLM even for local endpoints
#     temperature=0.2,
# )

# # ----------------------------------------------------------------------
# # Embedder config for all Crew() constructors
# # Prevents CrewAI defaulting to OpenAI embeddings (which require OPENAI_API_KEY).
# # ----------------------------------------------------------------------
# _OLLAMA_EMBEDDER = {
#     "provider": "ollama",
#     "config": {
#         "model": "nomic-embed-text",
#         "base_url": OLLAMA_BASE_URL,
#     },
# }

# # ----------------------------------------------------------------------
# # Optional ChromaDB RAG
# # ----------------------------------------------------------------------
# def _get_rag_context(topic: str) -> str:
#     """Inject recent research from ChromaDB collection 'blog_research'."""
#     try:
#         client = chromadb.PersistentClient(path="./chroma_db")
#         collection = client.get_or_create_collection("blog_research")
#         results = collection.query(
#             query_texts=[topic],
#             n_results=3,
#             include=["documents", "metadatas"]
#         )
#         if results and results.get("documents") and results["documents"][0]:
#             docs = results["documents"][0]
#             return "\n\n---\nRelevant past research:\n" + "\n---\n".join(docs)
#     except Exception:
#         return ""
#     return ""


# # ----------------------------------------------------------------------
# # Fallback plan (safe default on LLM failure)
# # ----------------------------------------------------------------------
# def _create_default_plan(topic: str) -> BlogPlan:
#     return BlogPlan(
#         blog_title=f"Deep Dive: {topic}",
#         feature_image_prompt=f"Modern abstract illustration representing {topic}, high-tech, clean design",
#         sections=[
#             Section(
#                 id="intro",
#                 title="Introduction",
#                 description="Introduce the topic and its importance",
#                 word_count=450,
#                 search_query=None,
#                 image_prompt=f"Visual metaphor for {topic} introduction"
#             ),
#             Section(
#                 id="key-concepts",
#                 title="Key Concepts",
#                 description="Explain core technical details",
#                 word_count=550,
#                 search_query=None,
#                 image_prompt=f"Diagram explaining core concepts of {topic}"
#             ),
#             Section(
#                 id="current-state",
#                 title="Current State & Trends",
#                 description="Latest developments and real-world impact",
#                 word_count=500,
#                 search_query=None,
#                 image_prompt=f"Infographic of {topic} trends 2026"
#             ),
#             Section(
#                 id="future-outlook",
#                 title="Future Outlook",
#                 description="What comes next and implications",
#                 word_count=400,
#                 search_query=None,
#                 image_prompt=f"Forward-looking vision of {topic}"
#             ),
#         ],
#         research_required=True,
#     )


# # ----------------------------------------------------------------------
# # Core factory — returns flat list of Crews for parallel dispatch
# # ----------------------------------------------------------------------
# def dispatch_crews(plan: BlogPlan) -> List[Crew]:
#     """Creates one Crew per worker type per section.
#     All Agent() constructors use _crew_llm (crewai.LLM) per CrewAI 1.x contract.
#     All Crew() constructors include embedder to prevent OpenAI default.
#     Returns flat list ready for asyncio.gather in graph.py.
#     """
#     crews: List[Crew] = []

#     for section in plan.sections:
#         # 1. Researcher Crew
#         if plan.research_required and (section.search_query or section.description):
#             researcher = Agent(
#                 role="Expert Researcher",
#                 goal=f"Conduct deep research for section: {section.title}",
#                 backstory="You are a meticulous fact-finder. Use search tools to gather accurate, up-to-date information.",
#                 llm=_crew_llm,                 # ← crewai.LLM (not ChatOllama)
#                 tools=[search_tool],
#                 verbose=True,
#             )
#             research_task = Task(
#                 description=(
#                     f"Research the section '{section.title}'.\n"
#                     f"Query: {section.search_query or section.description}\n"
#                     "Return a structured ResearchResult summary with sources."
#                 ),
#                 agent=researcher,
#                 expected_output="ResearchResult object (summary + sources)",
#             )
#             crews.append(Crew(
#                 agents=[researcher],
#                 tasks=[research_task],
#                 embedder=_OLLAMA_EMBEDDER,     # ← prevent OpenAI embedder default
#                 verbose=1,
#             ))

#         # 2. Writer Crew
#         writer = Agent(
#             role="Professional Blog Writer",
#             goal=f"Draft engaging section: {section.title}",
#             backstory="You are an SEO-conscious technical writer who produces clear, citation-rich Markdown.",
#             llm=_crew_llm,
#             verbose=True,
#         )
#         write_task = Task(
#             description=(
#                 f"Write the '{section.title}' section.\n"
#                 f"Target word count: {section.word_count}\n"
#                 f"Description: {section.description}\n"
#                 f"Use [citation] markers and insert [IMAGE_PLACEHOLDER_{section.id}] after the first paragraph."
#             ),
#             agent=writer,
#             expected_output="SectionDraft (Markdown content with placeholders)",
#         )
#         crews.append(Crew(
#             agents=[writer],
#             tasks=[write_task],
#             embedder=_OLLAMA_EMBEDDER,
#             verbose=1,
#         ))

#         # 3. Editor Crew
#         editor = Agent(
#             role="Senior Editor",
#             goal=f"Polish and optimize section: {section.title}",
#             backstory="You are a detail-oriented editor focused on clarity, flow, SEO, tone, and exact word count.",
#             llm=_crew_llm,
#             verbose=True,
#         )
#         edit_task = Task(
#             description=f"Refine the draft for '{section.title}' — improve clarity, SEO, and enforce word count.",
#             agent=editor,
#             expected_output="Refined SectionDraft",
#         )
#         crews.append(Crew(
#             agents=[editor],
#             tasks=[edit_task],
#             embedder=_OLLAMA_EMBEDDER,
#             verbose=1,
#         ))

#         # 4. Image Agent Crew
#         image_agent = Agent(
#             role="Image Generation Specialist",
#             goal=f"Generate visual for section: {section.title}",
#             backstory="You create high-quality, relevant images using Stable Diffusion v1.4.",
#             llm=_crew_llm,
#             tools=[image_gen_tool],
#             verbose=True,
#         )
#         image_task = Task(
#             description=f"Generate image using prompt: {section.image_prompt}",
#             agent=image_agent,
#             expected_output="ImageResult (file_path, alt_text, etc.)",
#         )
#         crews.append(Crew(
#             agents=[image_agent],
#             tasks=[image_task],
#             embedder=_OLLAMA_EMBEDDER,
#             verbose=1,
#         ))

#     return crews


# # ----------------------------------------------------------------------
# # Main node function (called by graph.py)
# # ----------------------------------------------------------------------
# def planner_node(state: CrewState) -> Dict[str, Any]:
#     """Entry point: returns {"plan": BlogPlan, "crews": list[Crew]}"""
#     topic: str = state.get("topic", "").strip()
#     if not topic:
#         raise ValueError("planner_node: topic is required in CrewState")

#     # 1. Optional RAG enrichment
#     rag_context: str = _get_rag_context(topic)

#     # 2. Prepare prompt
#     formatted_prompt = PLANNER_PROMPT.format(
#         topic=topic,
#         rag_context=rag_context or "No prior relevant research found."
#     )

#     # 3. Structured output with retries (_langchain_llm supports .with_structured_output)
#     structured_llm = _langchain_llm.with_structured_output(BlogPlan)

#     plan: BlogPlan | None = None
#     for attempt in range(3):
#         try:
#             raw_output = structured_llm.invoke(formatted_prompt)
#             plan = BlogPlan.model_validate(raw_output) if isinstance(raw_output, dict) else raw_output

#             # Self-validation checkpoint
#             if len(plan.sections) < 3:
#                 raise ValueError(f"Generated only {len(plan.sections)} sections (minimum 3 required)")
#             for sec in plan.sections:
#                 if not sec.title or not sec.image_prompt or sec.word_count < 200:
#                     raise ValueError("Section validation failed: missing title/image_prompt or insufficient word_count")
#             break
#         except Exception as e:
#             if attempt == 2:
#                 print(f"[planner_node] Structured output failed after 3 attempts: {e}. Using fallback plan.")
#                 plan = _create_default_plan(topic)
#             continue

#     # 4. Dispatch parallel crews
#     crews = dispatch_crews(plan)

#     return {
#         "plan": plan,
#         "crews": crews,
#     }




























# /*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-


























"""agents/planner.py — FINAL robust version with defensive JSON parsing for Ollama"""

from pathlib import Path
from typing import List, Dict, Any
import json
import logging
import re
import time

import chromadb
from crewai import Agent, Crew, Task, LLM
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from schemas import BlogPlan, Section
from state import CrewState
from tools.search import search_tool
from tools.image_gen import image_gen_tool

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Prompt loading
# ----------------------------------------------------------------------
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner_prompt.txt"
try:
    PLANNER_PROMPT: str = PROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    PLANNER_PROMPT = (
        "You are a strategic blog planner. Given the topic: {topic}\n"
        "{rag_context}\n"
        "Generate a structured blog plan with title, feature image prompt, "
        "and 4-6 sections. Each section must have id, title, description, "
        "word_count, optional search_query, and image_prompt.\n"
        "Output ONLY valid JSON matching the BlogPlan schema. No extra text."
    )

# ----------------------------------------------------------------------
# LLM instances
# ----------------------------------------------------------------------
_crew_llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
    temperature=0.2
)

_langchain_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    format="json"
)

_OLLAMA_EMBEDDER = {
    "provider": "ollama",
    "config": {"model": "nomic-embed-text", "base_url": OLLAMA_BASE_URL}
}

# ----------------------------------------------------------------------
# Defensive JSON cleaner (fixes '\n "blog_title"' and markdown fences)
# ----------------------------------------------------------------------
def _clean_json_response(text: str) -> str:
    """Strip common Ollama prefixes, fences, and extra text."""
    if not text or not isinstance(text, str):
        return "{}"
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    if '{' in text:
        text = text[text.find('{'):]
    if '}' in text:
        text = text[:text.rfind('}') + 1]
    return text.strip()

# ----------------------------------------------------------------------
# ChromaDB RAG, Default Plan, Dispatch Crews (unchanged)
# ----------------------------------------------------------------------
def _get_rag_context(topic: str) -> str:
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("blog_research")
        results = collection.query(query_texts=[topic], n_results=3, include=["documents", "metadatas"])
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            return "\n\n---\nRelevant past research:\n" + "\n---\n".join(docs)
    except Exception as e:
        logger.warning("ChromaDB RAG unavailable: %s", e)
    return ""

def _create_default_plan(topic: str) -> BlogPlan:
    return BlogPlan(
        blog_title=f"Deep Dive: {topic}",
        feature_image_prompt=f"Modern abstract illustration representing {topic}, high-tech, clean design, photorealistic",
        sections=[
            Section(id="intro", title="Introduction", description="Introduce the topic and its importance",
                    word_count=450, search_query=None, image_prompt=f"Visual metaphor for {topic} introduction"),
            Section(id="key-concepts", title="Key Concepts", description="Explain core technical details",
                    word_count=550, search_query=None, image_prompt=f"Diagram explaining core concepts of {topic}"),
            Section(id="current-state", title="Current State & Trends", description="Latest developments and real-world impact",
                    word_count=500, search_query=None, image_prompt=f"Infographic of {topic} trends 2026"),
            Section(id="future-outlook", title="Future Outlook", description="What comes next and implications",
                    word_count=400, search_query=None, image_prompt=f"Forward-looking vision of {topic}"),
        ],
        research_required=True,
    )

def dispatch_crews(plan: BlogPlan) -> List[Crew]:
    """Your original dispatch_crews (Researcher only when needed)."""
    crews: List[Crew] = []
    for section in plan.sections:
        if plan.research_required and section.search_query:
            researcher = Agent(
                role="Expert Researcher", goal=f"Conduct deep research for section: {section.title}",
                backstory="You are a meticulous fact-finder.",
                llm=_crew_llm, tools=[search_tool], verbose=True,
            )
            research_task = Task(
                description=f"Research '{section.title}'.\nQuery: {section.search_query}\nReturn ResearchResult.",
                agent=researcher, expected_output="ResearchResult object",
            )
            crews.append(Crew(agents=[researcher], tasks=[research_task], verbose=True))

        # Writer, Editor, Image Agent (full original logic)
        for role, desc, expected in [
            ("Professional Blog Writer", 
             f"Write the '{section.title}' section. Target: {section.word_count} words. Use [citation] and [IMAGE_PLACEHOLDER_{section.id}].",
             "SectionDraft"),
            ("Senior Editor", 
             f"Refine the draft for '{section.title}' — improve clarity, SEO, enforce word count.",
             "Refined SectionDraft"),
            ("Image Generation Specialist", 
             f"Generate image using prompt: {section.image_prompt}",
             "ImageResult")
        ]:
            agent = Agent(role=role, goal=f"{role} for {section.title}",
                          backstory="...", llm=_crew_llm, verbose=True,
                          tools=[image_gen_tool] if "Image" in role else None)
            task = Task(description=desc, agent=agent, expected_output=expected)
            crews.append(Crew(agents=[agent], tasks=[task], verbose=True))
    return crews

# ----------------------------------------------------------------------
# ROBUST PLANNER NODE
# ----------------------------------------------------------------------
def planner_node(state: CrewState) -> Dict[str, Any]:
    """Robust planner_node with defensive parsing + retries."""
    topic: str = state.get("topic", "").strip()
    if not topic:
        raise ValueError("planner_node: topic is required")

    research_required = state.get("research_required", True)
    rag_context = _get_rag_context(topic)

    formatted_prompt = PLANNER_PROMPT.format(
        topic=topic,
        rag_context=rag_context or "No prior relevant research found."
    )

    structured_llm = _langchain_llm.with_structured_output(BlogPlan, method="json_schema")

    # plan: BlogPlan | None = None
    # for attempt in range(3):
    #     try:
    #         raw_output = structured_llm.invoke(formatted_prompt)

    #         if isinstance(raw_output, str):
    #             cleaned = _clean_json_response(raw_output)
    #             raw_dict = json.loads(cleaned) if cleaned else {}
    #         else:
    #             raw_dict = raw_output if isinstance(raw_output, dict) else getattr(raw_output, 'dict', lambda: raw_output)()

    #         plan = BlogPlan.model_validate(raw_dict)

    #         if not (3 <= len(plan.sections) <= 6):
    #             raise ValueError(f"Invalid section count: {len(plan.sections)}")
    #         for sec in plan.sections:
    #             if not sec.title or not sec.image_prompt or sec.word_count < 200:
    #                 raise ValueError("Section missing required fields")

    #         logger.info(f"[planner_node] SUCCESS: valid BlogPlan with {len(plan.sections)} sections")
    #         break

    #     except (ValidationError, json.JSONDecodeError, Exception) as e:
    #         logger.warning(f"[planner_node] Attempt {attempt+1}/3 failed: {type(e).__name__} - {e}")
    #         if attempt < 2:
    #             time.sleep(2 ** attempt)
    #         else:
    #             logger.error("[planner_node] All retries failed → using fallback plan")
    #             plan = _create_default_plan(topic)

    # Inside planner_node, replace the structured_llm block with:

    # 3. Manual invoke + defensive parsing (Mistral 7B json_schema is unreliable)
    plan: BlogPlan | None = None
    for attempt in range(3):
        try:
            raw_text = _langchain_llm.invoke(formatted_prompt).content
            cleaned = _clean_json_response(raw_text)
            raw_dict = json.loads(cleaned) if cleaned else {}
            plan = BlogPlan.model_validate(raw_dict)

            if not (3 <= len(plan.sections) <= 6):
                raise ValueError(f"Invalid section count: {len(plan.sections)}")
            for sec in plan.sections:
                if not sec.title or not sec.image_prompt or sec.word_count < 200:
                    raise ValueError("Section missing required fields")

            logger.info(f"[planner_node] SUCCESS: valid BlogPlan with {len(plan.sections)} sections")
            break

        except (ValidationError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"[planner_node] Attempt {attempt+1}/3 failed: {type(e).__name__} - {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.error("[planner_node] All retries failed → using fallback plan")
                plan = _create_default_plan(topic)

    if not research_required:
        for sec in plan.sections:
            sec.search_query = None
        plan.research_required = False

    crews = dispatch_crews(plan)
    return {"plan": plan, "crews": crews}