# """
# agents/coach_writer.py

# Defines the Pickleball Coach Writer CrewAI agent (persona: Aurum) responsible for
# producing a complete, publish-ready Markdown article that satisfies Google's
# E-E-A-T signals.

# Calling contract
# ----------------
# Nodes import and call the top-level `run()` function:

#     from agents.coach_writer import run as run_coach_writer

#     markdown: str = run_coach_writer(
#         llm=llm,
#         vectordb=vectordb,
#         topic="pickleball kitchen rules for beginners",
#         tone=Tone.COACH,
#         word_count_goal=1500,
#         primary_kw="pickleball kitchen rules",
#         secondary=["NVZ rules", "non-volley zone", "volley fault"],
#         lsi=["dink", "erne", "USAPA", "kitchen line"],
#         outline_text="H1: ...\nH2: ...\nH3: ...",
#         research_context="[1] Title (url)\n    Snippet...",
#         revision_context="",            # "" on first pass; numbered list on revisions
#         internal_links_count=3,
#     )

# `run()` returns the raw Markdown string.  The calling node writes it directly
# into AgentState["draft_article"].
# """

# from __future__ import annotations

# from typing import List

# from crewai import Agent, Crew, Process, Task
# from langchain_community.llms import Ollama
# from langchain_community.vectorstores import Chroma
# from langchain_core.tools import tool

# from graph.state import Tone

# _ROLE = "Senior Pickleball Coach & Content Writer"
# _GOAL = (
#     "Write authoritative, engaging, SEO-optimised pickleball articles that reflect "
#     "real coaching expertise and satisfy Google's E-E-A-T standards."
# )
# _BACKSTORY_BASE = (
#     "You are Aurum — a Certified Pickleball Coach with 5+ years of playing and teaching. "
#     "You write exclusively for paddleaurum.com. Every article you produce includes concrete "
#     "examples: specific drills, scoring scenarios, equipment comparisons, and player tips. "
#     "You always verify rules and technique claims using your knowledge base before writing "
#     "about them — accuracy is non-negotiable. "
#     "Your writing earns Google's trust because it reflects genuine, first-hand experience."
# )

# # Tone-specific backstory extensions — appended to _BACKSTORY_BASE
# _TONE_EXTENSIONS: dict[Tone, str] = {
#     Tone.COACH: (
#         "Your voice is authoritative, enthusiastic, and practical. "
#         "You use 'you' and direct imperatives. You share personal coaching insights "
#         "and drills you have actually used on court."
#     ),
#     Tone.EXPERT: (
#         "Your audience consists of advanced players who already know the basics. "
#         "You use precise technical terminology, analyse strategy at a high level, "
#         "and assume the reader understands standard pickleball mechanics."
#     ),
#     Tone.BEGINNER_FRIENDLY: (
#         "Your audience is complete beginners. You define every pickleball term on first use, "
#         "write in short sentences, use numbered steps for procedures, "
#         "and adopt an encouraging, patient tone throughout."
#     ),
# }

# # Tone-specific writing instructions injected into the task description
# _TONE_INSTRUCTIONS: dict[Tone, str] = {
#     Tone.COACH: (
#         "Write as an experienced coach sharing hard-won knowledge. "
#         "Use 'we', 'you', and direct imperatives. "
#         "Include at least one named drill, one scoring scenario, and one equipment tip."
#     ),
#     Tone.EXPERT: (
#         "Write for advanced players. Use correct technical terminology throughout. "
#         "Focus on strategy, mechanics optimisation, and competitive edge. "
#         "Skip basic definitions — assume full familiarity with pickleball fundamentals."
#     ),
#     Tone.BEGINNER_FRIENDLY: (
#         "Write for complete beginners. Define every term in parentheses on first use. "
#         "Use simple sentences and numbered step-by-step instructions wherever applicable. "
#         "Be encouraging — never condescending."
#     ),
# }

# _EXPECTED_OUTPUT_TEMPLATE = "Complete Markdown article of approximately {word_count} words."


# def _make_rag_tool(vectordb: Chroma) -> tool:
#     @tool("pickleball_knowledge_retrieval")
#     def pickleball_knowledge_retrieval(query: str) -> str:
#         """
#         Retrieve official pickleball rules (USAPA), coaching techniques, drills,
#         equipment guides, and strategy content from the paddleaurum.com knowledge base.
#         ALWAYS call this tool before writing about any rule, technique, or equipment claim
#         to ensure factual accuracy.
#         """
#         docs = vectordb.similarity_search(query, k=5)
#         return "\n\n".join(doc.page_content for doc in docs) if docs else "No results found."

#     return pickleball_knowledge_retrieval


# def _build_backstory(tone: Tone) -> str:
#     extension = _TONE_EXTENSIONS.get(tone, _TONE_EXTENSIONS[Tone.COACH])
#     return f"{_BACKSTORY_BASE} {extension}"


# def build_agent(llm: Ollama, vectordb: Chroma, tone: Tone = Tone.COACH) -> Agent:
#     """
#     Instantiate the Coach Writer agent.

#     The agent's backstory is extended with tone-specific personality so the
#     persona remains consistent throughout an iterative revision session.
#     """
#     return Agent(
#         role=_ROLE,
#         goal=_GOAL,
#         backstory=_build_backstory(tone),
#         llm=llm,
#         tools=[_make_rag_tool(vectordb)],
#         memory=True,
#         verbose=False,
#         max_iter=5,
#     )


# def build_task(
#     agent: Agent,
#     *,
#     topic: str,
#     tone: Tone,
#     word_count_goal: int,
#     primary_kw: str,
#     secondary: List[str],
#     lsi: List[str],
#     outline_text: str,
#     research_context: str,
#     revision_context: str,
#     internal_links_count: int,
# ) -> Task:
#     """
#     Build the article-writing task with all runtime context injected.

#     Parameters
#     ----------
#     agent               : The Coach Writer agent instance.
#     topic               : Raw article topic string.
#     tone                : Authorial tone (Tone enum).
#     word_count_goal     : Target word count.
#     primary_kw          : Primary keyword — must appear in first 100 words.
#     secondary           : Secondary keywords to embed naturally.
#     lsi                 : LSI terms to weave in throughout.
#     outline_text        : Pre-formatted heading skeleton ("H1: ...\nH2: ...").
#     research_context    : Pre-formatted snippets ("[N] Title (url)\n    Snippet").
#     revision_context    : Empty string on first pass; numbered revision instructions
#                           from the Reflection node on subsequent iterations.
#     internal_links_count: Minimum number of [INTERNAL LINK: ...] placeholders required.
#     """
#     tone_instruction = _TONE_INSTRUCTIONS.get(tone, _TONE_INSTRUCTIONS[Tone.COACH])
#     revision_block   = (
#         f"\n\nREVISION INSTRUCTIONS — apply every item before writing:\n{revision_context}"
#         if revision_context.strip()
#         else ""
#     )
#     secondary_str = ", ".join(secondary) if secondary else "none"
#     lsi_str       = ", ".join(lsi) if lsi else "none"

#     description = (
#         f"Write a complete, publish-ready blog article for paddleaurum.com.\n\n"
#         f"Topic: {topic}\n"
#         f"Target word count: {word_count_goal} words (±10% is acceptable)\n"
#         f"Primary keyword: {primary_kw}\n"
#         f"Secondary keywords: {secondary_str}\n"
#         f"LSI terms (weave in naturally): {lsi_str}\n\n"
#         f"Tone: {tone_instruction}\n\n"
#         f"Content outline to follow exactly:\n{outline_text}\n\n"
#         f"Research to draw from (cite inline as [1], [2], etc.):\n{research_context}"
#         f"{revision_block}\n\n"
#         "Writing rules — follow all of these without exception:\n"
#         f"1. Primary keyword '{primary_kw}' must appear naturally within the first 100 words.\n"
#         "2. Secondary keywords and LSI terms must be embedded naturally — never keyword-stuffed.\n"
#         "3. Verify any rules, technique claims, or equipment facts using "
#         "pickleball_knowledge_retrieval before writing about them.\n"
#         "4. Mark every image placement with: [IMAGE: descriptive alt text here]\n"
#         f"5. Include at least {internal_links_count} internal link markers formatted as: "
#         "[INTERNAL LINK: anchor text → target article topic]\n"
#         "6. Keyword density for the primary keyword must be 1-2% of total word count.\n"
#         "7. Output clean Markdown only — no preamble, no meta-commentary, no fences.\n"
#         "8. The article must satisfy Google's E-E-A-T signals: demonstrate real experience, "
#         "verifiable expertise, and trustworthy sourcing throughout."
#     )

#     return Task(
#         description=description,
#         agent=agent,
#         expected_output=_EXPECTED_OUTPUT_TEMPLATE.format(word_count=word_count_goal),
#     )


# def run(
#     llm: Ollama,
#     vectordb: Chroma,
#     *,
#     topic: str,
#     tone: Tone,
#     word_count_goal: int,
#     primary_kw: str,
#     secondary: List[str],
#     lsi: List[str],
#     outline_text: str,
#     research_context: str,
#     revision_context: str,
#     internal_links_count: int,
# ) -> str:
#     """
#     Build agent + task + crew, execute, and return the raw Markdown string.

#     The caller (nodes/coaching_writer.py) writes the result directly into
#     AgentState["draft_article"].
#     """
#     agent = build_agent(llm, vectordb, tone)
#     task  = build_task(
#         agent,
#         topic=topic,
#         tone=tone,
#         word_count_goal=word_count_goal,
#         primary_kw=primary_kw,
#         secondary=secondary,
#         lsi=lsi,
#         outline_text=outline_text,
#         research_context=research_context,
#         revision_context=revision_context,
#         internal_links_count=internal_links_count,
#     )
#     crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
#     return str(crew.kickoff()).strip()


























# @############################################################################################





























# agents/coach_writer.py
"""
agents/coach_writer.py

Defines the Pickleball Coach Writer CrewAI agent (persona: Aurum) responsible for
producing a complete, publish-ready Markdown article that satisfies Google's
E-E-A-T signals.

Calling contract
----------------
Nodes import and call the top-level `run()` function:

    from agents.coach_writer import run as run_coach_writer

    markdown: str = run_coach_writer(
        llm=llm,
        vectordb=vectordb,
        topic="pickleball kitchen rules for beginners",
        tone=Tone.COACH,
        word_count_goal=1500,
        primary_kw="pickleball kitchen rules",
        secondary=["NVZ rules", "non-volley zone", "volley fault"],
        lsi=["dink", "erne", "USAPA", "kitchen line"],
        outline_text="H1: ...\nH2: ...\nH3: ...",
        research_context="[1] Title (url)\n    Snippet...",
        revision_context="",            # "" on first pass; numbered list on revisions
        internal_links_count=3,
    )

`run()` returns the raw Markdown string.  The calling node writes it directly
into AgentState["draft_article"].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from crewai import Agent, Crew, Process, Task
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool

from graph.state import Tone


def _load_prompt(filename: str) -> str:
    """Load a prompt text file from config/prompts/."""
    path = Path(__file__).parent.parent / "config" / "prompts" / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(f"Prompt file {path} not found. Please ensure the file exists.")


_ROLE = "Senior Pickleball Coach & Content Writer"
_GOAL = (
    "Write authoritative, engaging, SEO-optimised pickleball articles that reflect "
    "real coaching expertise and satisfy Google's E-E-A-T standards."
)
_BACKSTORY_BASE = _load_prompt("coach_writer_backstory_base.txt")

# Tone-specific backstory extensions — appended to _BACKSTORY_BASE
_TONE_EXTENSIONS: dict[Tone, str] = {
    Tone.COACH: (
        "Your voice is authoritative, enthusiastic, and practical. "
        "You use 'you' and direct imperatives. You share personal coaching insights "
        "and drills you have actually used on court."
    ),
    Tone.EXPERT: (
        "Your audience consists of advanced players who already know the basics. "
        "You use precise technical terminology, analyse strategy at a high level, "
        "and assume the reader understands standard pickleball mechanics."
    ),
    Tone.BEGINNER_FRIENDLY: (
        "Your audience is complete beginners. You define every pickleball term on first use, "
        "write in short sentences, use numbered steps for procedures, "
        "and adopt an encouraging, patient tone throughout."
    ),
}

# Tone-specific writing instructions injected into the task description
_TONE_INSTRUCTIONS: dict[Tone, str] = {
    Tone.COACH: (
        "Write as an experienced coach sharing hard-won knowledge. "
        "Use 'we', 'you', and direct imperatives. "
        "Include at least one named drill, one scoring scenario, and one equipment tip."
    ),
    Tone.EXPERT: (
        "Write for advanced players. Use correct technical terminology throughout. "
        "Focus on strategy, mechanics optimisation, and competitive edge. "
        "Skip basic definitions — assume full familiarity with pickleball fundamentals."
    ),
    Tone.BEGINNER_FRIENDLY: (
        "Write for complete beginners. Define every term in parentheses on first use. "
        "Use simple sentences and numbered step-by-step instructions wherever applicable. "
        "Be encouraging — never condescending."
    ),
}

_EXPECTED_OUTPUT_TEMPLATE = "Complete Markdown article of approximately {word_count} words."


def _make_rag_tool(vectordb: Chroma) -> tool:
    @tool("pickleball_knowledge_retrieval")
    def pickleball_knowledge_retrieval(query: str) -> str:
        """
        Retrieve official pickleball rules (USAPA), coaching techniques, drills,
        equipment guides, and strategy content from the paddleaurum.com knowledge base.
        ALWAYS call this tool before writing about any rule, technique, or equipment claim
        to ensure factual accuracy.
        """
        docs = vectordb.similarity_search(query, k=5)
        return "\n\n".join(doc.page_content for doc in docs) if docs else "No results found."

    return pickleball_knowledge_retrieval


def _build_backstory(tone: Tone) -> str:
    extension = _TONE_EXTENSIONS.get(tone, _TONE_EXTENSIONS[Tone.COACH])
    return f"{_BACKSTORY_BASE} {extension}"


def build_agent(llm: Ollama, vectordb: Chroma, tone: Tone = Tone.COACH) -> Agent:
    """
    Instantiate the Coach Writer agent.

    The agent's backstory is extended with tone-specific personality so the
    persona remains consistent throughout an iterative revision session.
    """
    return Agent(
        role=_ROLE,
        goal=_GOAL,
        backstory=_build_backstory(tone),
        llm=llm,
        tools=[_make_rag_tool(vectordb)],
        memory=True,
        verbose=False,
        max_iter=5,
    )


def build_task(
    agent: Agent,
    *,
    topic: str,
    tone: Tone,
    word_count_goal: int,
    primary_kw: str,
    secondary: List[str],
    lsi: List[str],
    outline_text: str,
    research_context: str,
    revision_context: str,
    internal_links_count: int,
) -> Task:
    """
    Build the article-writing task with all runtime context injected.

    Parameters
    ----------
    agent               : The Coach Writer agent instance.
    topic               : Raw article topic string.
    tone                : Authorial tone (Tone enum).
    word_count_goal     : Target word count.
    primary_kw          : Primary keyword — must appear in first 100 words.
    secondary           : Secondary keywords to embed naturally.
    lsi                 : LSI terms to weave in throughout.
    outline_text        : Pre-formatted heading skeleton ("H1: ...\nH2: ...").
    research_context    : Pre-formatted snippets ("[N] Title (url)\n    Snippet").
    revision_context    : Empty string on first pass; numbered revision instructions
                          from the Reflection node on subsequent iterations.
    internal_links_count: Minimum number of [INTERNAL LINK: ...] placeholders required.
    """
    tone_instruction = _TONE_INSTRUCTIONS.get(tone, _TONE_INSTRUCTIONS[Tone.COACH])
    revision_block   = (
        f"\n\nREVISION INSTRUCTIONS — apply every item before writing:\n{revision_context}"
        if revision_context.strip()
        else ""
    )
    secondary_str = ", ".join(secondary) if secondary else "none"
    lsi_str       = ", ".join(lsi) if lsi else "none"

    description = (
        f"Write a complete, publish-ready blog article for paddleaurum.com.\n\n"
        f"Topic: {topic}\n"
        f"Target word count: {word_count_goal} words (±10% is acceptable)\n"
        f"Primary keyword: {primary_kw}\n"
        f"Secondary keywords: {secondary_str}\n"
        f"LSI terms (weave in naturally): {lsi_str}\n\n"
        f"Tone: {tone_instruction}\n\n"
        f"Content outline to follow exactly:\n{outline_text}\n\n"
        f"Research to draw from (cite inline as [1], [2], etc.):\n{research_context}"
        f"{revision_block}\n\n"
        "Writing rules — follow all of these without exception:\n"
        f"1. Primary keyword '{primary_kw}' must appear naturally within the first 100 words.\n"
        "2. Secondary keywords and LSI terms must be embedded naturally — never keyword-stuffed.\n"
        "3. Verify any rules, technique claims, or equipment facts using "
        "pickleball_knowledge_retrieval before writing about them.\n"
        "4. Mark every image placement with: [IMAGE: descriptive alt text here]\n"
        f"5. Include at least {internal_links_count} internal link markers formatted as: "
        "[INTERNAL LINK: anchor text → target article topic]\n"
        "6. Keyword density for the primary keyword must be 1-2% of total word count.\n"
        "7. Output clean Markdown only — no preamble, no meta-commentary, no fences.\n"
        "8. The article must satisfy Google's E-E-A-T signals: demonstrate real experience, "
        "verifiable expertise, and trustworthy sourcing throughout."
    )

    return Task(
        description=description,
        agent=agent,
        expected_output=_EXPECTED_OUTPUT_TEMPLATE.format(word_count=word_count_goal),
    )


def run(
    llm: Ollama,
    vectordb: Chroma,
    *,
    topic: str,
    tone: Tone,
    word_count_goal: int,
    primary_kw: str,
    secondary: List[str],
    lsi: List[str],
    outline_text: str,
    research_context: str,
    revision_context: str,
    internal_links_count: int,
) -> str:
    """
    Build agent + task + crew, execute, and return the raw Markdown string.

    The caller (nodes/coaching_writer.py) writes the result directly into
    AgentState["draft_article"].
    """
    agent = build_agent(llm, vectordb, tone)
    task  = build_task(
        agent,
        topic=topic,
        tone=tone,
        word_count_goal=word_count_goal,
        primary_kw=primary_kw,
        secondary=secondary,
        lsi=lsi,
        outline_text=outline_text,
        research_context=research_context,
        revision_context=revision_context,
        internal_links_count=internal_links_count,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff()).strip()