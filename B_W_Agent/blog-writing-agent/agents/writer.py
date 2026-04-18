"""agents/writer.py
Lightweight Writer Node (roadmap Step 8).
Pure Python + flat asyncio.gather compatible.
Inserts [IMAGE_PLACEHOLDER_{id}] after first paragraph + generates [citation] markers.
CrewAI 1.x + LangChain json_mode pattern.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from schemas import Section, ResearchResult, SectionDraft

# ----------------------------------------------------------------------
# Prompt loading
# ----------------------------------------------------------------------
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "writer_prompt.txt"
WRITER_PROMPT: str = PROMPT_PATH.read_text(encoding="utf-8")

# ----------------------------------------------------------------------
# LLM (structured output)
# Note: langchain_ollama.ChatOllama is used here (not crewai.LLM) because
# writer_node is called as a pure Python function, not a CrewAI Agent.
# ----------------------------------------------------------------------
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
    format="json",
)
structured_llm = llm.with_structured_output(SectionDraft, method="json_mode")

# ----------------------------------------------------------------------
# Main node
# ----------------------------------------------------------------------
def writer_node(section: Section, research: Optional[ResearchResult] = None) -> Dict[str, Any]:
    """writer_node(section, research) → {"completed_sections": [SectionDraft]}"""
    if not isinstance(section, Section):
        raise ValueError("writer_node: input must be a valid Section object")

    research_summary = research.summary if research else "[NO_RESEARCH_AVAILABLE]"
    sources_str = "\n".join(
        f"[{i+1}] {s.get('title', 'N/A')} → {s.get('url', 'N/A')}"
        for i, s in enumerate(research.sources) if research and research.sources
    ) or "No sources available."

    formatted_prompt = WRITER_PROMPT.format(
        section_title=section.title,
        section_description=section.description,
        target_word_count=section.word_count,
        research_summary=research_summary,
        sources=sources_str
    )

    draft: SectionDraft | None = None
    for attempt in range(3):
        try:
            raw = structured_llm.invoke(formatted_prompt)
            draft = SectionDraft.model_validate(raw) if isinstance(raw, dict) else raw
            break
        except Exception:
            if attempt == 2:
                draft = SectionDraft(
                    section_id=section.id,
                    title=section.title,
                    content=f"# {section.title}\n\n[RESEARCH_FAILED] Draft could not be generated.",
                    word_count=section.word_count,
                    citations=[]
                )

    # Post-process: insert image placeholder after first paragraph
    if draft:
        paragraphs = draft.content.split("\n\n")
        if len(paragraphs) >= 1:
            paragraphs[0] = paragraphs[0].strip() + f"\n\n[IMAGE_PLACEHOLDER_{section.id}]"
        draft.content = "\n\n".join(paragraphs)
        draft.word_count = len(draft.content.split())

    return {"completed_sections": [draft] if draft else []}