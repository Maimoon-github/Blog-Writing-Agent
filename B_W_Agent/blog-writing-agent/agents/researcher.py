"""agents/researcher.py
Lightweight Researcher Node (per roadmap Step 7 + idea.md).
Pure Python function for flat asyncio.gather in graph.py.
Uses direct tools.search.search() with 3-tier retry + 2–3 varied queries.
CrewAI 1.x + LangChain structured output (json_mode) pattern.
"""

from pathlib import Path
from typing import Dict, Any, List
import time
from datetime import datetime

from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from schemas import Section, ResearchResult
from tools.search import search

# ----------------------------------------------------------------------
# Prompt loading (plain-text, editable at runtime)
# ----------------------------------------------------------------------
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "researcher_prompt.txt"
RESEARCHER_PROMPT: str = PROMPT_PATH.read_text(encoding="utf-8")

# ----------------------------------------------------------------------
# LLM instance (shared, json_mode for Mistral stability)
# Note: langchain_ollama.ChatOllama is used here (not crewai.LLM) because
# researcher_node is called as a pure Python function, not a CrewAI Agent.
# ----------------------------------------------------------------------
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    format="json",
)

structured_llm = llm.with_structured_output(ResearchResult, method="json_mode")

# ----------------------------------------------------------------------
# Retry wrapper for tools.search.search() (3-tier exponential backoff)
# ----------------------------------------------------------------------
def _search_with_retry(query: str, max_retries: int = 3) -> List[Dict]:
    """Direct call to tools.search.search() with 1s/2s/4s backoff."""
    for attempt in range(max_retries):
        try:
            results = search(query)
            return results[:8]
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[researcher_node] Search failed after {max_retries} attempts for: {query} | {e}")
                return []
            time.sleep(2 ** attempt)
    return []

# ----------------------------------------------------------------------
# Generate 2–3 varied queries
# ----------------------------------------------------------------------
def _generate_varied_queries(section: Section) -> List[str]:
    base = (section.search_query or section.description or section.title).strip()
    if not base:
        base = section.title
    queries = [base]
    if section.search_query:
        queries.extend([
            f"latest {base} 2026",
            f"{base} key facts statistics trends"
        ])
    return queries[:3]

# ----------------------------------------------------------------------
# Main lightweight node (called in parallel via asyncio.gather)
# ----------------------------------------------------------------------
def researcher_node(section: Section) -> Dict[str, Any]:
    """researcher_node(section: Section) -> {"research_results": [ResearchResult]}"""
    if not isinstance(section, Section):
        raise ValueError("researcher_node: input must be a valid Section object")

    queries = _generate_varied_queries(section)

    all_results: List[Dict] = []
    for q in queries:
        all_results.extend(_search_with_retry(q))

    search_results_str = "\n\n---\n".join(
        f"Title: {r.get('title', 'N/A')}\n"
        f"Snippet: {r.get('snippet', 'N/A')}\n"
        f"URL: {r.get('url', 'N/A')}"
        for r in all_results[:12]
    )

    formatted_prompt = RESEARCHER_PROMPT.format(
        section_title=section.title,
        section_description=section.description,
        search_results=search_results_str or "No search results available.",
        target_word_count=400
    )

    research_result: ResearchResult | None = None
    for attempt in range(3):
        try:
            raw = structured_llm.invoke(formatted_prompt)
            research_result = ResearchResult.model_validate(raw) if isinstance(raw, dict) else raw

            word_count = len(research_result.summary.split())
            if not (300 <= word_count <= 500) or not research_result.sources:
                raise ValueError(f"Validation failed: {word_count} words, {len(research_result.sources)} sources")
            break
        except Exception as e:
            if attempt == 2:
                print(f"[researcher_node] Structured output failed for section {section.id}: {e}")
                research_result = None

    if research_result is None:
        research_result = ResearchResult(
            section_id=section.id,
            summary="[RESEARCH_FAILED] — Could not retrieve or summarize web results.",
            sources=[],
            timestamp=datetime.now()
        )

    return {"research_results": [research_result]}