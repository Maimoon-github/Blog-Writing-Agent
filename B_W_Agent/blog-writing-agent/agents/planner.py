"""agents/planner.py
Planner Node — generates structured BlogPlan + dispatches parallel worker crews.
Exactly matches roadmap Phase 3 Step 6 and idea.md requirements.
Tested against CrewAI v1.14.2 + LangChain structured output patterns.
"""

from pathlib import Path
from typing import List, Dict, Any

import chromadb

from crewai import Agent, Crew, Task
from langchain_community.chat_models import ChatOllama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from schemas import BlogPlan, Section, CrewState
from tools.search import search_tool          # exported LangChain-compatible tool
from tools.image_gen import image_gen_tool    # exported LangChain-compatible tool

# ----------------------------------------------------------------------
# Prompt loading (plain-text, editable at runtime)
# ----------------------------------------------------------------------
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner_prompt.txt"
PLANNER_PROMPT: str = PROMPT_PATH.read_text(encoding="utf-8")

# ----------------------------------------------------------------------
# LLM instance (shared)
# ----------------------------------------------------------------------
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    format="json",          # helps Mistral produce valid structured output
)

# ----------------------------------------------------------------------
# Optional ChromaDB RAG (native manual access – 2026 best practice)
# ----------------------------------------------------------------------
def _get_rag_context(topic: str) -> str:
    """Inject recent research from ChromaDB collection 'blog_research'."""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("blog_research")
        results = collection.query(
            query_texts=[topic],
            n_results=3,
            include=["documents", "metadatas"]
        )
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            return "\n\n---\nRelevant past research:\n" + "\n---\n".join(docs)
    except Exception:
        return ""  # silently skip if collection missing/empty
    return ""


# ----------------------------------------------------------------------
# Fallback plan (safe default on LLM failure)
# ----------------------------------------------------------------------
def _create_default_plan(topic: str) -> BlogPlan:
    return BlogPlan(
        blog_title=f"Deep Dive: {topic}",
        feature_image_prompt=f"Modern abstract illustration representing {topic}, high-tech, clean design",
        sections=[
            Section(
                id="intro",
                title="Introduction",
                description="Introduce the topic and its importance",
                word_count=450,
                search_query=None,
                image_prompt=f"Visual metaphor for {topic} introduction"
            ),
            Section(
                id="key-concepts",
                title="Key Concepts",
                description="Explain core technical details",
                word_count=550,
                search_query=None,
                image_prompt=f"Diagram explaining core concepts of {topic}"
            ),
            Section(
                id="current-state",
                title="Current State & Trends",
                description="Latest developments and real-world impact",
                word_count=500,
                search_query=None,
                image_prompt=f"Infographic of {topic} trends 2025"
            ),
            Section(
                id="future-outlook",
                title="Future Outlook",
                description="What comes next and implications",
                word_count=400,
                search_query=None,
                image_prompt=f"Forward-looking vision of {topic}"
            ),
        ],
        research_required=True,
    )


# ----------------------------------------------------------------------
# Core factory – returns flat list of Crews for parallel dispatch
# ----------------------------------------------------------------------
def dispatch_crews(plan: BlogPlan) -> List[Crew]:
    """Creates one Crew per worker type per section (Researcher/Writer/Editor/Image).
    Returns flat list ready for asyncio.gather in graph.py.
    """
    crews: List[Crew] = []

    for section in plan.sections:
        # 1. Researcher Crew (only if research is required for this section)
        if plan.research_required and (section.search_query or section.description):
            researcher = Agent(
                role="Expert Researcher",
                goal=f"Conduct deep research for section: {section.title}",
                backstory="You are a meticulous fact-finder. Use search tools to gather accurate, up-to-date information.",
                llm=llm,
                tools=[search_tool],
                verbose=True,
            )
            research_task = Task(
                description=(
                    f"Research the section '{section.title}'.\n"
                    f"Query: {section.search_query or section.description}\n"
                    "Return a structured ResearchResult summary with sources."
                ),
                agent=researcher,
                expected_output="ResearchResult object (summary + sources)",
            )
            crews.append(Crew(agents=[researcher], tasks=[research_task], verbose=1))

        # 2. Writer Crew
        writer = Agent(
            role="Professional Blog Writer",
            goal=f"Draft engaging section: {section.title}",
            backstory="You are an SEO-conscious technical writer who produces clear, citation-rich Markdown.",
            llm=llm,
            verbose=True,
        )
        write_task = Task(
            description=(
                f"Write the '{section.title}' section.\n"
                f"Target word count: {section.word_count}\n"
                f"Description: {section.description}\n"
                "Use [citation] markers and insert [IMAGE_PLACEHOLDER_{section.id}] after the first paragraph."
            ),
            agent=writer,
            expected_output="SectionDraft (Markdown content with placeholders)",
        )
        crews.append(Crew(agents=[writer], tasks=[write_task], verbose=1))

        # 3. Editor Crew (refinement – runs after Writer in graph.py sequencing)
        editor = Agent(
            role="Senior Editor",
            goal=f"Polish and optimize section: {section.title}",
            backstory="You are a detail-oriented editor focused on clarity, flow, SEO, tone, and exact word count.",
            llm=llm,
            verbose=True,
        )
        edit_task = Task(
            description=f"Refine the draft for '{section.title}' – improve clarity, SEO, and enforce word count.",
            agent=editor,
            expected_output="Refined SectionDraft",
        )
        crews.append(Crew(agents=[editor], tasks=[edit_task], verbose=1))

        # 4. Image Agent Crew
        image_agent = Agent(
            role="Image Generation Specialist",
            goal=f"Generate visual for section: {section.title}",
            backstory="You create high-quality, relevant images using Stable Diffusion v1.4.",
            llm=llm,
            tools=[image_gen_tool],
            verbose=True,
        )
        image_task = Task(
            description=f"Generate image using prompt: {section.image_prompt}",
            agent=image_agent,
            expected_output="ImageResult (file_path, alt_text, etc.)",
        )
        crews.append(Crew(agents=[image_agent], tasks=[image_task], verbose=1))

    return crews


# ----------------------------------------------------------------------
# Main node function (called by graph.py)
# ----------------------------------------------------------------------
def planner_node(state: CrewState) -> Dict[str, Any]:
    """Entry point: returns {"plan": BlogPlan, "crews": list[Crew]}"""
    topic: str = state.get("topic", "").strip()
    if not topic:
        raise ValueError("planner_node: topic is required in CrewState")

    # 1. Optional RAG enrichment (2026 best practice)
    rag_context: str = _get_rag_context(topic)

    # 2. Prepare prompt
    formatted_prompt = PLANNER_PROMPT.format(
        topic=topic,
        rag_context=rag_context or "No prior relevant research found."
    )

    # 3. Structured output with retries (Mistral stability)
    structured_llm = llm.with_structured_output(BlogPlan)

    plan: BlogPlan | None = None
    for attempt in range(3):
        try:
            raw_output = structured_llm.invoke(formatted_prompt)
            # Handle both Pydantic model and dict return styles
            plan = BlogPlan.model_validate(raw_output) if isinstance(raw_output, dict) else raw_output

            # Self-validation checkpoint (per execution plan)
            if len(plan.sections) < 3:
                raise ValueError(f"Generated only {len(plan.sections)} sections (minimum 3 required)")
            for sec in plan.sections:
                if not sec.title or not sec.image_prompt or sec.word_count < 200:
                    raise ValueError("Section validation failed: missing title/image_prompt or insufficient word_count")
            break
        except Exception as e:
            if attempt == 2:
                print(f"[planner_node] Structured output failed after 3 attempts: {e}. Using fallback plan.")
                plan = _create_default_plan(topic)
            continue

    # 4. Dispatch parallel crews
    crews = dispatch_crews(plan)

    return {
        "plan": plan,
        "crews": crews,
    }