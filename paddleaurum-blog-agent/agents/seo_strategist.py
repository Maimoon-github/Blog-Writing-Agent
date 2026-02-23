# """
# agents/seo_strategist.py

# Defines the SEO Strategist CrewAI agent responsible for producing a structured
# keyword map (primary, secondary, LSI, entities, intent_type) from a given topic
# and research context.

# Calling contract
# ----------------
# Nodes import and call the top-level `run()` function:

#     from agents.seo_strategist import run as run_seo_strategist

#     raw_json: str = run_seo_strategist(
#         llm=llm,
#         vectordb=vectordb,
#         topic="pickleball kitchen rules for beginners",
#         target_kw="pickleball kitchen rules",        # "" if not provided
#         research_context="- Title: ...\n  Snippet: ...",
#     )

# `run()` returns a raw string.  The calling node is responsible for JSON parsing,
# type coercion into KeywordMap, and writing results back to AgentState.
# """

# from __future__ import annotations

# from crewai import Agent, Crew, Process, Task
# from langchain_community.llms import Ollama
# from langchain_community.vectorstores import Chroma
# from langchain_core.tools import tool

# _ROLE      = "Senior SEO Strategist"
# _GOAL      = (
#     "Extract the most effective keyword strategy for pickleball content "
#     "that will rank on Google for paddleaurum.com."
# )
# _BACKSTORY = (
#     "You have 8 years of SEO experience specialising in sports and coaching content. "
#     "You deeply understand Google's E-E-A-T signals, semantic search, and topical authority. "
#     "You know paddleaurum.com's content library and its competitive landscape inside out. "
#     "You never guess — you always verify keyword choices against your knowledge base first."
# )

# _EXPECTED_OUTPUT = (
#     "Valid JSON keyword map containing: primary (string), secondary (list of strings), "
#     "lsi (list of strings), entities (list of strings), intent_type "
#     "(one of: informational | commercial | navigational)."
# )

# _JSON_SCHEMA = """\
# {
#   "primary": "single best-ranking keyword phrase",
#   "secondary": ["kw1", "kw2", "kw3", "kw4"],
#   "lsi": ["related_term1", "related_term2", "related_term3", "related_term4", "related_term5"],
#   "entities": ["entity1", "entity2", "entity3"],
#   "intent_type": "informational|commercial|navigational"
# }"""

# _TASK_RULES = """\
# Rules:
# - primary: the single highest-value keyword phrase that best represents the user's search query.
# - secondary: 3-5 closely related keywords the article should also rank for.
# - lsi: 4-6 latent semantic indexing terms (conceptually related, not exact matches).
# - entities: named entities — players, equipment brands, organisations (e.g. USAPA), venues.
# - intent_type: commercial if the query implies buying/comparing; navigational if brand-specific; \
# otherwise informational.
# - Return ONLY valid JSON — no markdown fences, no preamble, no commentary."""


# def _make_rag_tool(vectordb: Chroma) -> tool:
#     @tool("seo_knowledge_retrieval")
#     def seo_knowledge_retrieval(query: str) -> str:
#         """
#         Retrieve SEO guidelines, keyword best practices, and topical authority
#         patterns from the paddleaurum.com knowledge base.
#         Use this before selecting any keyword to verify it aligns with site strategy.
#         """
#         docs = vectordb.similarity_search(query, k=4)
#         return "\n\n".join(doc.page_content for doc in docs) if docs else "No results found."

#     return seo_knowledge_retrieval


# def build_agent(llm: Ollama, vectordb: Chroma) -> Agent:
#     """Instantiate the SEO Strategist agent with its RAG tool bound to the given vectordb."""
#     return Agent(
#         role=_ROLE,
#         goal=_GOAL,
#         backstory=_BACKSTORY,
#         llm=llm,
#         tools=[_make_rag_tool(vectordb)],
#         memory=True,
#         verbose=False,
#         max_iter=3,
#     )


# def build_task(agent: Agent, *, topic: str, target_kw: str, research_context: str) -> Task:
#     """
#     Build the keyword-mapping task for a specific article topic.

#     Parameters
#     ----------
#     agent           : The SEO Strategist agent instance.
#     topic           : Raw article topic string.
#     target_kw       : Optional keyword override from the user; empty string if absent.
#     research_context: Pre-formatted research snippets (title + snippet lines) from
#                       the Research Merger node.
#     """
#     description = (
#         f"Topic: {topic}\n"
#         f"Provided keyword hint: {target_kw or 'none — derive the best primary keyword yourself'}\n\n"
#         f"Research context (use to inform keyword choices):\n{research_context}\n\n"
#         "Step 1: Call seo_knowledge_retrieval with the topic to check site keyword strategy "
#         "and any existing coverage on paddleaurum.com.\n"
#         "Step 2: Using the retrieved guidelines and the research context, produce a keyword map.\n\n"
#         f"Return ONLY this JSON structure — no markdown fences:\n{_JSON_SCHEMA}\n\n"
#         f"{_TASK_RULES}"
#     )
#     return Task(
#         description=description,
#         agent=agent,
#         expected_output=_EXPECTED_OUTPUT,
#     )


# def run(
#     llm: Ollama,
#     vectordb: Chroma,
#     *,
#     topic: str,
#     target_kw: str,
#     research_context: str,
# ) -> str:
#     """
#     Build agent + task + crew, execute, and return the raw output string.

#     The caller (nodes/keyword_mapper.py) is responsible for JSON parsing
#     and writing the result into AgentState.
#     """
#     agent = build_agent(llm, vectordb)
#     task  = build_task(agent, topic=topic, target_kw=target_kw, research_context=research_context)
#     crew  = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
#     return str(crew.kickoff()).strip()
















# @##########################################################################################






















# agents/seo_strategist.py
"""
agents/seo_strategist.py

Defines the SEO Strategist CrewAI agent responsible for producing a structured
keyword map (primary, secondary, LSI, entities, intent_type) from a given topic
and research context.

Calling contract
----------------
Nodes import and call the top-level `run()` function:

    from agents.seo_strategist import run as run_seo_strategist

    raw_json: str = run_seo_strategist(
        llm=llm,
        vectordb=vectordb,
        topic="pickleball kitchen rules for beginners",
        target_kw="pickleball kitchen rules",        # "" if not provided
        research_context="- Title: ...\n  Snippet: ...",
    )

`run()` returns a raw string.  The calling node is responsible for JSON parsing,
type coercion into KeywordMap, and writing results back to AgentState.
"""

from __future__ import annotations

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool


def _load_prompt(filename: str) -> str:
    """Load a prompt text file from config/prompts/."""
    path = Path(__file__).parent.parent / "config" / "prompts" / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(f"Prompt file {path} not found. Please ensure the file exists.")


_ROLE      = "Senior SEO Strategist"
_GOAL      = (
    "Extract the most effective keyword strategy for pickleball content "
    "that will rank on Google for paddleaurum.com."
)
_BACKSTORY = _load_prompt("seo_strategist_backstory.txt")

_EXPECTED_OUTPUT = (
    "Valid JSON keyword map containing: primary (string), secondary (list of strings), "
    "lsi (list of strings), entities (list of strings), intent_type "
    "(one of: informational | commercial | navigational)."
)

_JSON_SCHEMA = """\
{
  "primary": "single best-ranking keyword phrase",
  "secondary": ["kw1", "kw2", "kw3", "kw4"],
  "lsi": ["related_term1", "related_term2", "related_term3", "related_term4", "related_term5"],
  "entities": ["entity1", "entity2", "entity3"],
  "intent_type": "informational|commercial|navigational"
}"""

_TASK_RULES = """\
Rules:
- primary: the single highest-value keyword phrase that best represents the user's search query.
- secondary: 3-5 closely related keywords the article should also rank for.
- lsi: 4-6 latent semantic indexing terms (conceptually related, not exact matches).
- entities: named entities — players, equipment brands, organisations (e.g. USAPA), venues.
- intent_type: commercial if the query implies buying/comparing; navigational if brand-specific; \
otherwise informational.
- Return ONLY valid JSON — no markdown fences, no preamble, no commentary."""


def _make_rag_tool(vectordb: Chroma) -> tool:
    @tool("seo_knowledge_retrieval")
    def seo_knowledge_retrieval(query: str) -> str:
        """
        Retrieve SEO guidelines, keyword best practices, and topical authority
        patterns from the paddleaurum.com knowledge base.
        Use this before selecting any keyword to verify it aligns with site strategy.
        """
        docs = vectordb.similarity_search(query, k=4)
        return "\n\n".join(doc.page_content for doc in docs) if docs else "No results found."

    return seo_knowledge_retrieval


def build_agent(llm: Ollama, vectordb: Chroma) -> Agent:
    """Instantiate the SEO Strategist agent with its RAG tool bound to the given vectordb."""
    return Agent(
        role=_ROLE,
        goal=_GOAL,
        backstory=_BACKSTORY,
        llm=llm,
        tools=[_make_rag_tool(vectordb)],
        memory=True,
        verbose=False,
        max_iter=3,
    )


def build_task(agent: Agent, *, topic: str, target_kw: str, research_context: str) -> Task:
    """
    Build the keyword-mapping task for a specific article topic.

    Parameters
    ----------
    agent           : The SEO Strategist agent instance.
    topic           : Raw article topic string.
    target_kw       : Optional keyword override from the user; empty string if absent.
    research_context: Pre-formatted research snippets (title + snippet lines) from
                      the Research Merger node.
    """
    description = (
        f"Topic: {topic}\n"
        f"Provided keyword hint: {target_kw or 'none — derive the best primary keyword yourself'}\n\n"
        f"Research context (use to inform keyword choices):\n{research_context}\n\n"
        "Step 1: Call seo_knowledge_retrieval with the topic to check site keyword strategy "
        "and any existing coverage on paddleaurum.com.\n"
        "Step 2: Using the retrieved guidelines and the research context, produce a keyword map.\n\n"
        f"Return ONLY this JSON structure — no markdown fences:\n{_JSON_SCHEMA}\n\n"
        f"{_TASK_RULES}"
    )
    return Task(
        description=description,
        agent=agent,
        expected_output=_EXPECTED_OUTPUT,
    )


def run(
    llm: Ollama,
    vectordb: Chroma,
    *,
    topic: str,
    target_kw: str,
    research_context: str,
) -> str:
    """
    Build agent + task + crew, execute, and return the raw output string.

    The caller (nodes/keyword_mapper.py) is responsible for JSON parsing
    and writing the result into AgentState.
    """
    agent = build_agent(llm, vectordb)
    task  = build_task(agent, topic=topic, target_kw=target_kw, research_context=research_context)
    crew  = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff()).strip()