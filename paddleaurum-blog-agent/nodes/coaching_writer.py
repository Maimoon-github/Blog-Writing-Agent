import logging
from typing import List

from crewai import Agent, Crew, Process, Task
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool

from graph.state import AgentState, Tone

logger = logging.getLogger(__name__)

_TONE_INSTRUCTIONS = {
    Tone.COACH: (
        "Write as an experienced pickleball coach sharing hard-won knowledge. "
        "Use 'we', 'you', direct imperatives. Include specific drills, scoring scenarios, "
        "equipment comparisons, and player tips. Voice: authoritative, enthusiastic, practical."
    ),
    Tone.EXPERT: (
        "Write for advanced players. Use technical terminology correctly. "
        "Assume the reader understands basic rules. Focus on strategy, mechanics, and optimisation."
    ),
    Tone.BEGINNER_FRIENDLY: (
        "Write for complete beginners. Define every term on first use. "
        "Use simple sentences, numbered steps, and encouraging language. Avoid jargon."
    ),
}


def _make_rag_tool(vectordb: Chroma):
    @tool("pickleball_knowledge_retrieval")
    def pickleball_knowledge_retrieval(query: str) -> str:
        """Retrieve pickleball rules, coaching techniques, and equipment guides from the knowledge base."""
        docs = vectordb.similarity_search(query, k=5)
        return "\n\n".join(doc.page_content for doc in docs) if docs else "No results found."
    return pickleball_knowledge_retrieval


def _build_outline_text(state: AgentState) -> str:
    outline = state.get("content_outline") or {}
    headings = outline.get("headings", [])
    if not headings:
        return f"H1: {state['topic']}"
    return "\n".join(f"{h['level']}: {h['text']}" for h in headings)


def _build_research_context(state: AgentState) -> str:
    snippets = (state.get("research_snippets") or [])[:10]
    if not snippets:
        return "No external research available."
    lines = [f"[{i+1}] {s['title']} ({s['url']})\n    {s['snippet']}" for i, s in enumerate(snippets)]
    return "\n\n".join(lines)


def _build_revision_context(state: AgentState) -> str:
    suggestions = state.get("seo_suggestions") or []
    iteration = state.get("revision_iteration", 0)
    if not suggestions or iteration == 0:
        return ""
    return (
        f"\n\nREVISION INSTRUCTIONS (iteration {iteration}):\n"
        + "\n".join(f"- {s}" for s in suggestions)
    )


async def coaching_writer_node(state: AgentState, *, llm: Ollama, vectordb: Chroma) -> dict:
    try:
        topic: str = state["topic"]
        tone: Tone = state.get("tone") or Tone.COACH
        word_count_goal: int = state.get("word_count_goal", 1500)

        keyword_map = state.get("keyword_map") or {}
        primary_kw: str = keyword_map.get("primary", topic)
        secondary: List[str] = keyword_map.get("secondary", [])
        lsi: List[str] = keyword_map.get("lsi", [])
        internal_links: List[str] = state.get("internal_link_placeholders", [])

        outline_text = _build_outline_text(state)
        research_context = _build_research_context(state)
        revision_context = _build_revision_context(state)
        tone_instruction = _TONE_INSTRUCTIONS.get(tone, _TONE_INSTRUCTIONS[Tone.COACH])

        rag_tool = _make_rag_tool(vectordb)

        agent = Agent(
            role="Senior Pickleball Coach & Content Writer",
            goal="Write authoritative, engaging, SEO-optimised pickleball articles reflecting real coaching expertise.",
            backstory=(
                "You are Aurum — a Certified Pickleball Coach with 5+ years of playing and teaching. "
                "You write for paddleaurum.com. You always include concrete examples: specific drills, "
                "scoring scenarios, equipment comparisons, and player tips. "
                "Your writing satisfies Google's E-E-A-T standards because it reflects genuine experience."
            ),
            llm=llm,
            tools=[rag_tool],
            memory=True,
            verbose=False,
            max_iter=5,
        )

        task_description = (
            f"Write a complete, publish-ready blog article for paddleaurum.com.\n\n"
            f"Topic: {topic}\n"
            f"Target word count: {word_count_goal} words (±10%)\n"
            f"Primary keyword: {primary_kw}\n"
            f"Secondary keywords: {', '.join(secondary)}\n"
            f"LSI terms to weave in naturally: {', '.join(lsi)}\n\n"
            f"Tone instructions: {tone_instruction}\n\n"
            f"Content outline to follow:\n{outline_text}\n\n"
            f"Research to draw from (cite inline as [1], [2], etc.):\n{research_context}\n"
            f"{revision_context}\n\n"
            "Writing rules:\n"
            f"1. The primary keyword '{primary_kw}' must appear in the first 100 words.\n"
            "2. Use secondary keywords and LSI terms naturally — never stuffed.\n"
            f"3. Include image alt text placeholders formatted as: [IMAGE: descriptive alt text here]\n"
            f"4. Include at least {len(internal_links)} internal link placeholders formatted as: "
            "[INTERNAL LINK: anchor text → target article topic]\n"
            "5. Use the pickleball_knowledge_retrieval tool to verify any rules, "
            "scores, or technique claims before writing about them.\n"
            "6. Output the article in clean Markdown only — no preamble, no commentary."
        )

        task = Task(
            description=task_description,
            agent=agent,
            expected_output=f"Complete Markdown article of approximately {word_count_goal} words.",
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        draft: str = crew.kickoff().strip()

        logger.info(
            "Coach writer: draft produced, ~%d words (iteration %d)",
            len(draft.split()),
            state.get("revision_iteration", 0),
        )

        return {
            "draft_article": draft,
            "error":         None,
            "error_node":    None,
        }

    except Exception as exc:
        logger.exception("Coaching writer failed.")
        return {"error": str(exc), "error_node": "writer"}