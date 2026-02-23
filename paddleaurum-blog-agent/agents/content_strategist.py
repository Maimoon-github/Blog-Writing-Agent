# """
# agents/content_strategist.py

# Defines the Content Strategist CrewAI agent responsible for producing a complete
# H1 → H2 → H3 content outline aligned with SERP intent, including FAQ candidates
# and internal linking placeholders.

# Calling contract
# ----------------
# Nodes import and call the top-level `run()` function:

#     from agents.content_strategist import run as run_content_strategist

#     raw_json: str = run_content_strategist(
#         llm=llm,
#         vectordb=vectordb,
#         topic="pickleball kitchen rules for beginners",
#         primary_kw="pickleball kitchen rules",
#         intent_type="informational",
#         key_sections=["What Is the Kitchen?", "NVZ Violations", "Beginner Mistakes"],
#     )

# `run()` returns a raw string.  The calling node is responsible for JSON parsing,
# construction of ContentOutline, and writing results back to AgentState.
# """

# from __future__ import annotations

# from typing import List

# from crewai import Agent, Crew, Process, Task
# from langchain_community.llms import Ollama
# from langchain_community.vectorstores import Chroma
# from langchain_core.tools import tool

# _ROLE      = "Content Strategist & Outline Architect"
# _GOAL      = (
#     "Build perfectly structured content outlines for pickleball coaching articles "
#     "that match SERP intent and maximise topical authority for paddleaurum.com."
# )
# _BACKSTORY = (
#     "You analyse top-ranking pickleball content and build comprehensive outlines that beat competitors. "
#     "You understand topical clustering, entity coverage, and FAQ schema opportunities. "
#     "Every outline you create satisfies Google's quality signals: one clear H1, logical H2 sections, "
#     "granular H3 breakdowns, FAQ sections that target featured snippets, and internal links "
#     "that strengthen paddleaurum.com's topical authority graph."
# )

# _EXPECTED_OUTPUT = (
#     "Valid JSON outline containing: headings (list of {level, text} objects), "
#     "faq_candidates (list of question strings), "
#     "internal_link_placeholders (list of anchor text strings)."
# )

# _JSON_SCHEMA = """\
# {
#   "headings": [
#     {"level": "H1", "text": "..."},
#     {"level": "H2", "text": "..."},
#     {"level": "H3", "text": "..."}
#   ],
#   "faq_candidates": [
#     "Question 1?",
#     "Question 2?"
#   ],
#   "internal_link_placeholders": [
#     "Anchor text for link 1",
#     "Anchor text for link 2",
#     "Anchor text for link 3"
#   ]
# }"""

# _TASK_RULES = """\
# Rules:
# - Exactly one H1 — the article title, must contain the primary keyword.
# - 5-8 H2 sections providing comprehensive topic coverage.
# - 2-3 H3 sub-sections under each H2 where granularity adds value.
# - Minimum 4 faq_candidates — natural questions a reader would ask about this topic.
#   Phrase them as full questions ending in "?".
# - Minimum 3 internal_link_placeholders — anchor texts for links to other paddleaurum.com articles
#   (e.g. "best pickleball paddles for beginners", "how to score in pickleball").
# - Align heading structure with the search intent:
#     informational  → educational H2s ("What Is...", "How to...", "Common Mistakes...")
#     commercial     → comparison H2s ("Best... for...", "X vs Y", "How to Choose...")
#     navigational   → brand/site-specific H2s
# - Return ONLY valid JSON — no markdown fences, no preamble, no commentary."""


# def _make_rag_tool(vectordb: Chroma) -> tool:
#     @tool("content_pattern_retrieval")
#     def content_pattern_retrieval(query: str) -> str:
#         """
#         Retrieve content structure patterns, competitor outline analysis, and
#         topical coverage guides from the paddleaurum.com knowledge base.
#         Use this to check what sections high-ranking pickleball articles include.
#         """
#         docs = vectordb.similarity_search(query, k=4)
#         return "\n\n".join(doc.page_content for doc in docs) if docs else "No results found."

#     return content_pattern_retrieval


# def build_agent(llm: Ollama, vectordb: Chroma) -> Agent:
#     """Instantiate the Content Strategist agent with its RAG tool bound to the given vectordb."""
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


# def build_task(
#     agent: Agent,
#     *,
#     topic: str,
#     primary_kw: str,
#     intent_type: str,
#     key_sections: List[str],
# ) -> Task:
#     """
#     Build the outline-generation task for a specific article topic.

#     Parameters
#     ----------
#     agent         : The Content Strategist agent instance.
#     topic         : Raw article topic string.
#     primary_kw    : Primary keyword from the Keyword Mapper.
#     intent_type   : Classified search intent ("informational" | "commercial" | "navigational").
#     key_sections  : Suggested H2 titles from the Planner task_plan (may be empty).
#     """
#     sections_hint = (
#         f"Suggested sections from the planner (use as a starting point, improve as needed):\n"
#         + "\n".join(f"  - {s}" for s in key_sections)
#         if key_sections
#         else "No section suggestions provided — derive the best structure yourself."
#     )

#     description = (
#         f"Topic: {topic}\n"
#         f"Primary keyword: {primary_kw}\n"
#         f"Search intent: {intent_type}\n\n"
#         f"{sections_hint}\n\n"
#         "Step 1: Call content_pattern_retrieval with the topic and primary keyword to check "
#         "what sections and structures high-ranking pickleball articles use.\n"
#         "Step 2: Build a complete content outline that covers the topic comprehensively, "
#         f"matches the '{intent_type}' search intent, and includes the primary keyword in the H1.\n\n"
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
#     primary_kw: str,
#     intent_type: str,
#     key_sections: List[str],
# ) -> str:
#     """
#     Build agent + task + crew, execute, and return the raw output string.

#     The caller (nodes/outline_agent.py) is responsible for JSON parsing,
#     construction of ContentOutline, and writing the result into AgentState.
#     """
#     agent = build_agent(llm, vectordb)
#     task  = build_task(
#         agent,
#         topic=topic,
#         primary_kw=primary_kw,
#         intent_type=intent_type,
#         key_sections=key_sections,
#     )
#     crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
#     return str(crew.kickoff()).strip()


























# @################################################################################




















# agents/content_strategist.py
"""
agents/content_strategist.py

Defines the Content Strategist CrewAI agent responsible for producing a complete
H1 → H2 → H3 content outline aligned with SERP intent, including FAQ candidates
and internal linking placeholders.

Calling contract
----------------
Nodes import and call the top-level `run()` function:

    from agents.content_strategist import run as run_content_strategist

    raw_json: str = run_content_strategist(
        llm=llm,
        vectordb=vectordb,
        topic="pickleball kitchen rules for beginners",
        primary_kw="pickleball kitchen rules",
        intent_type="informational",
        key_sections=["What Is the Kitchen?", "NVZ Violations", "Beginner Mistakes"],
    )

`run()` returns a raw string.  The calling node is responsible for JSON parsing,
construction of ContentOutline, and writing results back to AgentState.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

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


_ROLE      = "Content Strategist & Outline Architect"
_GOAL      = (
    "Build perfectly structured content outlines for pickleball coaching articles "
    "that match SERP intent and maximise topical authority for paddleaurum.com."
)
_BACKSTORY = _load_prompt("content_strategist_backstory.txt")

_EXPECTED_OUTPUT = (
    "Valid JSON outline containing: headings (list of {level, text} objects), "
    "faq_candidates (list of question strings), "
    "internal_link_placeholders (list of anchor text strings)."
)

_JSON_SCHEMA = """\
{
  "headings": [
    {"level": "H1", "text": "..."},
    {"level": "H2", "text": "..."},
    {"level": "H3", "text": "..."}
  ],
  "faq_candidates": [
    "Question 1?",
    "Question 2?"
  ],
  "internal_link_placeholders": [
    "Anchor text for link 1",
    "Anchor text for link 2",
    "Anchor text for link 3"
  ]
}"""

_TASK_RULES = """\
Rules:
- Exactly one H1 — the article title, must contain the primary keyword.
- 5-8 H2 sections providing comprehensive topic coverage.
- 2-3 H3 sub-sections under each H2 where granularity adds value.
- Minimum 4 faq_candidates — natural questions a reader would ask about this topic.
  Phrase them as full questions ending in "?".
- Minimum 3 internal_link_placeholders — anchor texts for links to other paddleaurum.com articles
  (e.g. "best pickleball paddles for beginners", "how to score in pickleball").
- Align heading structure with the search intent:
    informational  → educational H2s ("What Is...", "How to...", "Common Mistakes...")
    commercial     → comparison H2s ("Best... for...", "X vs Y", "How to Choose...")
    navigational   → brand/site-specific H2s
- Return ONLY valid JSON — no markdown fences, no preamble, no commentary."""


def _make_rag_tool(vectordb: Chroma) -> tool:
    @tool("content_pattern_retrieval")
    def content_pattern_retrieval(query: str) -> str:
        """
        Retrieve content structure patterns, competitor outline analysis, and
        topical coverage guides from the paddleaurum.com knowledge base.
        Use this to check what sections high-ranking pickleball articles include.
        """
        docs = vectordb.similarity_search(query, k=4)
        return "\n\n".join(doc.page_content for doc in docs) if docs else "No results found."

    return content_pattern_retrieval


def build_agent(llm: Ollama, vectordb: Chroma) -> Agent:
    """Instantiate the Content Strategist agent with its RAG tool bound to the given vectordb."""
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


def build_task(
    agent: Agent,
    *,
    topic: str,
    primary_kw: str,
    intent_type: str,
    key_sections: List[str],
) -> Task:
    """
    Build the outline-generation task for a specific article topic.

    Parameters
    ----------
    agent         : The Content Strategist agent instance.
    topic         : Raw article topic string.
    primary_kw    : Primary keyword from the Keyword Mapper.
    intent_type   : Classified search intent ("informational" | "commercial" | "navigational").
    key_sections  : Suggested H2 titles from the Planner task_plan (may be empty).
    """
    sections_hint = (
        f"Suggested sections from the planner (use as a starting point, improve as needed):\n"
        + "\n".join(f"  - {s}" for s in key_sections)
        if key_sections
        else "No section suggestions provided — derive the best structure yourself."
    )

    description = (
        f"Topic: {topic}\n"
        f"Primary keyword: {primary_kw}\n"
        f"Search intent: {intent_type}\n\n"
        f"{sections_hint}\n\n"
        "Step 1: Call content_pattern_retrieval with the topic and primary keyword to check "
        "what sections and structures high-ranking pickleball articles use.\n"
        "Step 2: Build a complete content outline that covers the topic comprehensively, "
        f"matches the '{intent_type}' search intent, and includes the primary keyword in the H1.\n\n"
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
    primary_kw: str,
    intent_type: str,
    key_sections: List[str],
) -> str:
    """
    Build agent + task + crew, execute, and return the raw output string.

    The caller (nodes/outline_agent.py) is responsible for JSON parsing,
    construction of ContentOutline, and writing the result into AgentState.
    """
    agent = build_agent(llm, vectordb)
    task  = build_task(
        agent,
        topic=topic,
        primary_kw=primary_kw,
        intent_type=intent_type,
        key_sections=key_sections,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff()).strip()