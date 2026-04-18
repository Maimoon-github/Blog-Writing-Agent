"""agents/editor.py
Lightweight Editor Node (roadmap Step 8).
Pure Python + flat asyncio.gather compatible.
Strictly preserves [IMAGE_PLACEHOLDER_...] and [citation] tokens.
CrewAI 1.x + LangChain json_mode pattern.
"""

from pathlib import Path
from typing import Dict, Any

from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from schemas import SectionDraft

# ----------------------------------------------------------------------
# Prompt loading
# ----------------------------------------------------------------------
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "editor_prompt.txt"
EDITOR_PROMPT: str = PROMPT_PATH.read_text(encoding="utf-8")

# ----------------------------------------------------------------------
# LLM (structured output)
# Note: langchain_ollama.ChatOllama is used here (not crewai.LLM) because
# editor_node is called as a pure Python function, not as a CrewAI Agent.
# .with_structured_output() is a LangChain method not available on crewai.LLM.
# ----------------------------------------------------------------------
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    format="json",
)
structured_llm = llm.with_structured_output(SectionDraft, method="json_mode")

# ----------------------------------------------------------------------
# Main node
# ----------------------------------------------------------------------
def editor_node(draft: SectionDraft) -> Dict[str, Any]:
    """editor_node(draft) → {"completed_sections": [SectionDraft]}"""
    if not isinstance(draft, SectionDraft):
        raise ValueError("editor_node: input must be a valid SectionDraft object")

    formatted_prompt = EDITOR_PROMPT.format(
        section_title=draft.title,
        current_content=draft.content,
        target_word_count=draft.word_count
    )

    # LLM call with 2 retries
    edited: SectionDraft | None = None
    for attempt in range(3):
        try:
            raw = structured_llm.invoke(formatted_prompt)
            edited = SectionDraft.model_validate(raw) if isinstance(raw, dict) else raw
            break
        except Exception:
            if attempt == 2:
                edited = draft  # Tier-2 fallback: return original unchanged

    # Preserve section_id in case the LLM changed it
    if edited and edited.section_id != draft.section_id:
        edited.section_id = draft.section_id

    return {"completed_sections": [edited] if edited else []}
