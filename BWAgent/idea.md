# 🤖 Autonomous AI Blog Generation System
## End-to-End Multi-Agent Architecture: From Topic to Published Article

> **Stack:** LangGraph · Ollama (Mistral) · SearxNG · Stable Diffusion v1.4 · Streamlit · ChromaDB  
> **Pattern:** Orchestrator–Worker with Router, Planner, Researcher, Writers, Image Agent, Editor & Reducer  
> **Constraint:** 100% Free & Open-Source · Fully Local · No API Keys Required

---

## Table of Contents

1. [Project Discovery](#1-project-discovery)
2. [Agentic System Design](#2-agentic-system-design)
3. [Agent Workflow & State Management](#3-agent-workflow--state-management)
4. [Technical Architecture & Frameworks](#4-technical-architecture--frameworks)
5. [Tooling & Infrastructure](#5-tooling--infrastructure)
6. [Ethical, Safety & Scaling Considerations](#6-ethical-safety--scaling-considerations)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Project Folder Structure](#8-project-folder-structure)
9. [Simplified Core LangGraph Pipeline Reference](#9-simplified-core-langgraph-pipeline-reference)

---

## 1. Project Discovery

### Core Problem

Writing a high-quality, research-backed blog post is time-consuming. It requires topic research, outline planning, web search, fact gathering, multi-section drafting, citation management, and image sourcing — a workflow that normally takes hours. This system collapses all of that into a single autonomous pipeline that operates end-to-end with no human intervention beyond the initial topic prompt.

### Target Users

This system is built for three primary audiences:

**Content Creators & Bloggers** who need a research-first drafting assistant that can write 2,000+ word articles with citations and images automatically embedded. **Developer / ML Portfolio Builders** who want a demonstrable, interview-ready agentic AI project that showcases orchestrator–worker patterns, parallel execution, RAG, and multi-modal generation. **Small Teams & Startups** who need to produce authoritative SEO content at scale without a dedicated content team.

### Real-World Use Cases

1. **Tech Blog Automation** — A developer blogs about AI trends; the system researches, writes, and illustrates a full post in under 5 minutes.
2. **Educational Content Generation** — A teacher gives a topic; the system produces a well-cited, structured explainer with illustrative images.
3. **Product/Company Newsletter** — Marketing feeds in a theme; multiple articles are generated in parallel for A/B distribution.
4. **Research Digests** — Feed in a subject (e.g., "Quantum Computing 2025"); the system synthesizes the latest web findings into a readable digest.
5. **Interview Portfolio Demo** — Demonstrates mastery of LangGraph state machines, parallel node execution, RAG, and diffusion model APIs in one cohesive project.

### Constraints, Assumptions & Success Metrics

**Constraints:**
- All models run locally via Ollama (Mistral 7B) and Hugging Face `diffusers`
- Search must work without paid API keys (SearxNG self-hosted or DuckDuckGo)
- Target hardware: consumer GPU with at least 8 GB VRAM (RTX 3060+) or CPU-only fallback

**Assumptions:**
- Internet access is available during research phase
- The operator has Docker installed for SearxNG
- Stable Diffusion v1.4 is acceptable image quality for the use case

**Success Metrics:**
- End-to-end blog generation in under 6 minutes on consumer hardware
- At least 5 images generated per post (1 feature + 4 section-level)
- All citations are real URLs extracted from SearxNG results
- Blog output is a single well-structured Markdown or HTML file
- Zero hallucinated citations (all links are live URLs from search results)

### Scope Boundaries

**In-Scope:** Topic intake → planning → parallel web research → parallel section writing → image generation → citation injection → final blog assembly → Streamlit UI preview

**Out-of-Scope:** Automatic publishing to WordPress/Ghost (can be added as Phase 4), human-in-the-loop editing UI, fine-tuning of Mistral, video or audio generation, multilingual support (initial version)

---

## 2. Agentic System Design

### Agent Roster

The system uses **8 specialized agents** organized in a two-tier hierarchy: a Supervisor layer (Router + Planner + Reducer) and a Worker layer (Researcher × N, Writer × N, Image Agent, Editor).

---

#### 🧠 Agent 1: Router Node
**Role:** Entry point and intent classifier.  
**Responsibilities:**
- Receives the raw user topic string
- Determines whether the topic requires live web research or can be handled from LLM knowledge alone
- Routes to Planner with a `research_required: bool` flag
- Handles malformed or empty inputs gracefully

**Decision Logic:**
```
IF topic contains time-sensitive keywords (latest, 2025, new, recent, current)
  OR topic is a factual technical subject with evolving information
→ research_required = True
ELSE
→ research_required = False (LLM knowledge-only path)
```

**Tools:** None (pure LLM reasoning node)

---

#### 📋 Agent 2: Planner Node
**Role:** Strategic decomposition of the topic.  
**Responsibilities:**
- Generates a structured blog outline (title, 4–6 section names, descriptions, and target word counts)
- Produces a list of focused search queries (one per section that needs research)
- Assigns image prompt strings for each section
- Outputs a `BlogPlan` structured object that drives all downstream agents
- Determines parallelization strategy: which sections can be written concurrently

**Output Schema (Pydantic):**
```python
class Section(BaseModel):
    id: str
    title: str
    description: str
    word_count: int
    search_query: Optional[str]
    image_prompt: str

class BlogPlan(BaseModel):
    blog_title: str
    feature_image_prompt: str
    sections: List[Section]
    research_required: bool
```

**Tools:** Ollama/Mistral with structured output enforced via `model.with_structured_output(BlogPlan)`

---

#### 🔍 Agent 3: Researcher Worker (Parallel × N)
**Role:** Web research specialist, one instance per section needing research.  
**Responsibilities:**
- Takes a single `search_query` from the plan
- Executes 2–3 targeted searches via SearxNG or DuckDuckGo
- Fetches and parses top result page content
- Extracts key facts, statistics, and quotes
- Returns a `ResearchResult` with summarized content + source URLs

**Tools:**
- `SearxSearchWrapper` (from `langchain_community.utilities`) pointed at local SearxNG Docker instance
- `DuckDuckGoSearchRun` as fallback
- `BeautifulSoup4` for lightweight HTML content parsing
- Web fetch via `httpx` for full-page content retrieval

**Parallelization:** LangGraph's `Send()` API dispatches N simultaneous researcher instances — one per section that has a `search_query`.

---

#### ✍️ Agent 4: Writer Worker (Parallel × N)
**Role:** Section content generation, one instance per blog section.  
**Responsibilities:**
- Receives its section plan + research results (if any)
- Drafts the full section body with proper headings, paragraph flow, and inline `[citation]` markers
- Writes to match the target word count from the plan
- Embeds an `[IMAGE_PLACEHOLDER_{section_id}]` token where the image should appear
- Returns a completed `SectionDraft` with content and raw citation references

**Tools:** Ollama Mistral via `langchain_community.llms.Ollama` with carefully structured section writing prompt

**Parallelization:** All writer workers fire simultaneously using LangGraph's `Send()` API. Each uses its own `WorkerState` (isolated from others), and all outputs are merged into `completed_sections` via the `Annotated[list, operator.add]` state key pattern.

---

#### 📝 Agent 4b: Editor Worker (Sequential or Parallel Refinement)
**Role:** Quality assurance and refinement specialist.  
**Responsibilities:**
- Takes completed `SectionDraft` (or full assembled draft) from Writer workers
- Improves clarity, flow, tone consistency, SEO-friendliness, and adherence to any style guide
- Ensures target word count is met, removes redundancy, and enhances engagement
- Checks citation placement and overall structure
- Returns refined `SectionDraft` or full blog content

**Tools:** Ollama Mistral with a dedicated editing prompt (see Section 9 for reference implementation).

**Integration:** Runs after Writer workers (can be parallel per section or as a single post-write step before Citation Manager). This adds the classic Researcher → Writer → Editor linear refinement pattern while preserving the full parallel orchestration.

---

#### 🎨 Agent 5: Image Generation Agent (Parallel × N)
**Role:** Visual content creator, one instance per planned image.  
**Responsibilities:**
- Receives the image prompt from the plan (feature image or section image)
- Calls Stable Diffusion v1.4 via Hugging Face `diffusers` pipeline
- Saves generated images to `outputs/images/` with deterministic filenames
- Returns file path and alt text for embedding in final blog

**Tools:**
```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
```
- Runs on GPU if available, CPU fallback with reduced steps
- Image size: 512×512 (feature), 512×384 (section images)
- Guidance scale: 7.5, inference steps: 30 (GPU) / 15 (CPU)

**Parallelization:** Image generation is dispatched in parallel with writer workers using `asyncio.gather()` wrapped in LangGraph task nodes.

---

#### 📎 Agent 6: Citation Manager Node
**Role:** Reference integrity enforcer.  
**Responsibilities:**
- Aggregates all `ResearchResult` objects from Researcher workers
- Builds a global citation registry: `{[N]: {url, title, snippet}}`
- Resolves `[citation]` markers in writer output to numbered references `[1]`, `[2]`, etc.
- Deduplicates URLs
- Appends a formatted "References" section to the blog

**Tools:** Pure Python string processing; no LLM required for this node

---

#### 🔗 Agent 7: Reducer / Assembler Node
**Role:** Final synthesis and document assembly.  
**Responsibilities:**
- Receives all completed `SectionDraft` objects, image paths, and resolved citations
- Orders sections according to the original `BlogPlan`
- Substitutes `[IMAGE_PLACEHOLDER_{id}]` tokens with actual Markdown image syntax `![alt](path/to/image.png)`
- Injects citations inline
- Writes the final blog as a `.md` file and optionally converts to `.html`
- Generates a metadata block (title, estimated read time, date, tags)

**Tools:** Python string templating, `markdown` library for HTML conversion

---

### Inter-Agent Communication Pattern

The system uses a **hierarchical supervisor pattern with event-driven parallel dispatch**:

- The Supervisor (Router → Planner) manages top-level control flow
- Workers (Researchers, Writers, Image Agents, Editor) communicate exclusively through **shared graph state** — they never call each other directly
- The `Send()` API in LangGraph enables the Planner to dynamically spawn N workers based on the number of sections at runtime
- All worker outputs flow back to the shared state via `Annotated[list, operator.add]` — a reducer function that safely merges concurrent writes
- The Reducer node reads the fully assembled state and produces the final artifact

---

## 3. Agent Workflow & State Management

### Full Agent Lifecycle

```
INITIALIZATION
└── User enters topic in Streamlit UI
└── GraphState initialized with topic string
└── LangGraph graph compiled and invoked

ROUTING (Router Node)
└── LLM classifies topic → sets research_required flag
└── Conditional edge → Planner

PLANNING (Planner Node)
└── LLM generates BlogPlan (structured output)
└── Plan stored in graph state
└── Parallel dispatch prepared

PARALLEL EXECUTION (Workers — fired simultaneously)
├── N × Researcher Workers
│   └── SearxNG search → page fetch → summarize → return ResearchResult
├── N × Writer Workers (can start with available research)
│   └── Wait for matching ResearchResult → draft section → return SectionDraft
├── N × Editor Workers (refinement after writers)
│   └── Improve tone, SEO, clarity → refined drafts
└── N+1 × Image Agent Workers
    └── SD pipeline call → save image → return ImageResult

AGGREGATION (Citation Manager)
└── Merge all ResearchResults
└── Build citation registry
└── Resolve inline citation markers

ASSEMBLY (Reducer Node)
└── Order sections
└── Inject images
└── Inject citations
└── Write final .md + .html files

TERMINATION
└── Output paths stored in graph state
└── Streamlit UI reads output and renders preview
└── Download links activated
```

### State Management Strategy

The entire graph operates on a single typed `GraphState` TypedDict, which is the only object passed between nodes (now extended with fields for flexible refinement pipelines):

```python
from typing import TypedDict, Annotated, List, Optional
import operator

class GraphState(TypedDict):
    # Input
    topic: str
    
    # Router output
    research_required: bool
    
    # Planner output
    plan: Optional[BlogPlan]
    
    # Worker outputs — operator.add merges parallel writes safely
    research_results: Annotated[List[ResearchResult], operator.add]
    completed_sections: Annotated[List[SectionDraft], operator.add]
    generated_images: Annotated[List[ImageResult], operator.add]
    
    # Added for Editor refinement (from standard LangGraph patterns)
    research_data: Annotated[List[str], operator.add]
    edits: Annotated[List[str], operator.add]
    
    # Reducer outputs
    citation_registry: dict
    final_markdown: str
    final_html: str
    output_path: str
```

**Short-Term Memory:** The `GraphState` dict acts as the working memory for a single blog generation run. All agents read from and write to this state.

**Long-Term Memory (Optional Phase 2):** ChromaDB stores previously generated blogs and their research summaries as vector embeddings. On new runs, the Planner queries ChromaDB to avoid re-researching recently covered topics, and Writer workers can draw on prior research context for faster generation.

**Caching:** Research results are cached to disk as JSON per search query hash. If the same query is re-run within 24 hours, the cached result is used, bypassing SearxNG calls.

### Asynchronous Orchestration

LangGraph's `Send()` API enables true parallel worker dispatch within the Python event loop (including Editor workers):

```python
def dispatch_workers(state: GraphState):
    """Dispatch all researcher, writer, editor, and image workers in parallel"""
    sends = []
    for section in state["plan"].sections:
        if section.search_query:
            sends.append(Send("researcher_worker", {"section": section}))
        sends.append(Send("writer_worker", {"section": section}))
        sends.append(Send("editor_worker", {"section": section}))
        sends.append(Send("image_worker", {"section": section}))
    return sends
```

Image generation is the most latency-sensitive step. It runs in a `ThreadPoolExecutor` (since the `diffusers` pipeline is not async-native) wrapped inside a LangGraph task node, keeping it off the main event loop.

### Error Handling, Retries & Fallbacks

Every worker node is wrapped in a try/except with a three-tier fallback strategy:

**Tier 1 — Retry:** On transient failures (network timeouts, Ollama response errors), workers retry up to 3 times with exponential backoff (1s, 2s, 4s).

**Tier 2 — Degraded Fallback:** If a Researcher worker fails after retries, the Writer/Editor worker for that section receives an empty `ResearchResult` and writes/refines the section using LLM knowledge alone, marking it with a `[RESEARCH_FAILED]` warning tag.

**Tier 3 — Skip & Continue:** If an Image worker fails, the `[IMAGE_PLACEHOLDER_{id}]` token is replaced with a placeholder text box, and the blog assembly continues without blocking.

LangGraph's built-in checkpointer (using SQLite via `SqliteSaver`) saves graph state after every node completion. If the pipeline crashes mid-run, it can be resumed from the last successful checkpoint rather than starting over.

---

## 4. Technical Architecture & Frameworks

### Stack Justification

**LangGraph** is chosen over CrewAI and AutoGen for this project for three reasons. First, its `Send()` API provides the most elegant native support for the orchestrator–worker scatter-gather pattern. Second, its explicit `TypedDict` state gives full transparency and debuggability over the shared state. Third, it has native SQLite checkpointing for crash recovery and LangSmith integration for observability — both critical for production-grade builds.

**Ollama + Mistral 7B** is the LLM backend. Mistral 7B v0.3 is the best-in-class 7B model for instruction following, structured output generation, and text quality — all essential for a content generation pipeline. It runs at 4-bit quantization (Q4_K_M) on modest consumer hardware and supports tool-calling via its native function format, which LangGraph can leverage for the Router's decision logic.

**SearxNG** (self-hosted via Docker) is the search layer. It is completely free, aggregates Google, DuckDuckGo, Bing, and Wikipedia simultaneously, strips tracking, and integrates natively with LangChain's `SearxSearchWrapper`. DuckDuckGo (`langchain_community.tools.DuckDuckGoSearchRun`) is the zero-setup fallback.

**Stable Diffusion v1.4** via Hugging Face `diffusers` is the image generation engine. SD v1.4 is the specified target, and it generates 512×512 images in ~3–10 seconds on an RTX 3060. For CPU-only environments, step count is reduced to 15 and resolution to 384×384.

**ChromaDB** is the optional long-term vector store. It is in-process (no separate server needed), persists to disk, and integrates directly with LangChain's retriever interface. It stores research summaries and past blog outlines for retrieval-augmented planning.

**Streamlit** provides the UI. It is the fastest path to a functional, shareable frontend with file preview, download buttons, and progress feedback — zero JavaScript required.

### High-Level Architecture Diagram

```mermaid
graph TD
    UI([🖥️ Streamlit UI\nTopic Input]) --> ROUTER

    subgraph SUPERVISOR_LAYER["🧠 Supervisor Layer"]
        ROUTER["🔀 Router Node\nIntent Classification\nresearch_required: bool"]
        PLANNER["📋 Planner Node\nBlogPlan Generation\nStructured Output via Mistral"]
        REDUCER["🔗 Reducer Node\nAssembly + Citation Injection"]
    end

    subgraph WORKER_LAYER["⚙️ Worker Layer (Parallel Execution via Send API)"]
        direction LR
        R1["🔍 Researcher\nWorker 1"]
        R2["🔍 Researcher\nWorker 2"]
        Rn["🔍 Researcher\nWorker N"]

        W1["✍️ Writer\nWorker 1"]
        W2["✍️ Writer\nWorker 2"]
        Wn["✍️ Writer\nWorker N"]

        E1["📝 Editor\nWorker 1"]
        E2["📝 Editor\nWorker 2"]
        En["📝 Editor\nWorker N"]

        IM1["🎨 Image\nWorker 1\nSD v1.4"]
        IM2["🎨 Image\nWorker 2\nSD v1.4"]
        IMn["🎨 Image\nWorker N+1\nSD v1.4"]
    end

    subgraph TOOLS["🛠️ Tools & Services"]
        SEARXNG["🌐 SearxNG\n(Docker, Port 8080)"]
        DDGO["🦆 DuckDuckGo\n(Fallback)"]
        OLLAMA["🦙 Ollama\nMistral 7B\n(Port 11434)"]
        SDIFF["🖼️ Hugging Face\nStable Diffusion v1.4"]
        CHROMA["🗄️ ChromaDB\nLong-Term Memory"]
    end

    subgraph STATE["📦 Shared Graph State (TypedDict)"]
        S1["topic · plan · research_required"]
        S2["research_results: List (merged via operator.add)"]
        S3["completed_sections: List (merged via operator.add)"]
        S4["generated_images: List (merged via operator.add)"]
        S5["citation_registry · final_markdown"]
    end

    ROUTER -->|Sets research_required| PLANNER
    PLANNER -->|"Send(researcher_worker, section)"| R1
    PLANNER -->|"Send(researcher_worker, section)"| R2
    PLANNER -->|"Send(researcher_worker, section)"| Rn
    PLANNER -->|"Send(writer_worker, section)"| W1
    PLANNER -->|"Send(writer_worker, section)"| W2
    PLANNER -->|"Send(writer_worker, section)"| Wn
    PLANNER -->|"Send(editor_worker, section)"| E1
    PLANNER -->|"Send(editor_worker, section)"| E2
    PLANNER -->|"Send(editor_worker, section)"| En
    PLANNER -->|"Send(image_worker, prompt)"| IM1
    PLANNER -->|"Send(image_worker, prompt)"| IM2
    PLANNER -->|"Send(image_worker, prompt)"| IMn

    R1 & R2 & Rn -->|ResearchResult| STATE
    W1 & W2 & Wn -->|SectionDraft| STATE
    E1 & E2 & En -->|RefinedDraft| STATE
    IM1 & IM2 & IMn -->|ImageResult| STATE

    STATE -->|All workers complete| CITEMGR["📎 Citation Manager\nURL Registry Builder"]
    CITEMGR --> REDUCER
    REDUCER --> OUTPUT["📄 Output\n.md + .html Blog Post\nWith Images + Citations"]
    OUTPUT --> UI

    R1 & R2 & Rn <-->|"search(query)"| SEARXNG
    SEARXNG -.->|Fallback| DDGO
    W1 & W2 & Wn & E1 & E2 & En & ROUTER & PLANNER & REDUCER <-->|"generate(prompt)"| OLLAMA
    IM1 & IM2 & IMn <-->|"pipe(prompt)"| SDIFF
    PLANNER <-->|"similarity_search(topic)"| CHROMA
```

### Component Interaction Sequence

(The original sequence diagram remains unchanged except Editor workers are inserted after Writer workers in the parallel block.)

---

## 5. Tooling & Infrastructure

(Original content unchanged)

---

## 6. Ethical, Safety & Scaling Considerations

(Original content unchanged)

---

## 7. Implementation Roadmap

(Original content unchanged — the Editor agent fits naturally into Phase 2 parallelization and Phase 4 refinement features)

---

## 8. Project Folder Structure

(Original content unchanged — add `prompts/editor_prompt.txt` to the `prompts/` folder for the new Editor node)

---

## 9. Simplified Core LangGraph Pipeline Reference

### 1. High-level architecture

Think of this as a **multi-agent blog pipeline** (complementary to the full parallel system above):

- **Orchestrator (LangGraph)**  
  - Main state machine that routes work between agents.
  - Accepts a topic and outputs a finished blog draft + metadata.

- **Agents (workers)**  
  - `Researcher`: fetches background info (web via SearxNG, RAG, docs).
  - `Writer`: drafts the blog post given research.
  - `Editor`: improves tone, structure, SEO, and checks length (added as Agent 4b above).

- **Infrastructure layer**
  - **LangChain**: tools, prompts, memory, chains.
  - **LangGraph**: orchestrates nodes (agents) with state.
  - **LangSmith**: logs, traces, evals for debugging and tuning (optional).
  - **Workers** (if you want scale): each agent can be a separate worker (e.g., via FastAPI endpoints).

Diagram-style flow:

```
User topic
   ↓
LangGraph Orchestrator
   ↓
→ Researcher agent → Writer agent → Editor agent → Final blog
```

***

### 2. Core components (for your code)

#### State schema (LangGraph)

Define a shared `State` to carry data between agents (already extended in the main `GraphState` above):

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import MessagesState, START, END

class AgentState(MessagesState):
    topic: str
    research_data: List[str]
    blog_draft: str
    edits: List[str]
```

This lets **Researcher**, **Writer**, and **Editor** read and update the same state.

#### Researcher node (adapted to local stack)

This node can:
- Use `SearxSearchWrapper` or `DuckDuckGoSearchRun` for web lookup.
- Optionally call a local RAG over your docs.

```python
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.chat_models import ChatOllama

search_tool = SearxSearchWrapper(searx_host="http://localhost:8080")  # or DuckDuckGoSearchRun fallback

llm = ChatOllama(model="mistral", temperature=0.3)

def researcher_node(state: AgentState):
    topic = state["topic"]
    search_query = f"Blog background: {topic}"
    results = search_tool.run(search_query)  # returns string or parsed results
    return {
        "research_data": [results],  # or parsed list
        "messages": state["messages"] + [("system", "Researcher: gathered background info")]
    }
```

#### Writer node (adapted)

Prompt-based blog writer that uses research:

```python
from langchain_core.prompts import ChatPromptTemplate

PROMPT = """
You are a professional blog writer. Write an engaging, SEO-friendly blog post
about the given topic. Use the provided research snippets.

Topic: {topic}
Research snippets:
{research}

Write in markdown. Include:
- A short intro
- A few section headings
- 3–5 short paragraphs
- A brief conclusion
"""

prompt = ChatPromptTemplate.from_template(PROMPT)

def writer_node(state: AgentState):
    chain = prompt | llm
    blog = chain.invoke({
        "topic": state["topic"],
        "research": "\n".join(state["research_data"]),
    }).content
    return {
        "blog_draft": blog,
        "messages": state["messages"] + [("system", "Writer: generated draft")]
    }
```

#### Editor node (adapted)

Takes the draft and refines it:

```python
EDIT_PROMPT = """
You are an editor. Given this blog draft, improve:
- Clarity and flow
- Wordiness
- SEO-friendliness (add 2–3 natural keywords)
- Keep it under 800 words

Do not change the structure too much.

Draft:
{draft}

Return only the improved markdown.
"""

edit_prompt = ChatPromptTemplate.from_template(EDIT_PROMPT)

def editor_node(state: AgentState):
    chain = edit_prompt | llm
    improved = chain.invoke({"draft": state["blog_draft"]}).content
    return {
        "blog_draft": improved,
        "edits": [improved],
        "messages": state["messages"] + [("system", "Editor: improved draft")]
    }
```

***

### 3. Orchestrator (LangGraph workflow)

Wire the agents into a graph (this can be used as a lightweight sequential prototype before scaling to full parallel Send() dispatch):

```python
from langgraph.graph import StateGraph

graph_builder = StateGraph(AgentState)

graph_builder.add_node("Researcher", researcher_node)
graph_builder.add_node("Writer", writer_node)
graph_builder.add_node("Editor", editor_node)

graph_builder.set_entry_point("Researcher")
graph_builder.add_edge("Researcher", "Writer")
graph_builder.add_edge("Writer", "Editor")
graph_builder.add_edge("Editor", END)

blog_app = graph_builder.compile()
```

Run it:

```python
result = blog_app.invoke(
    {
        "topic": "How AI agents are changing software development",
        "research_data": [],
        "blog_draft": "",
        "edits": [],
        "messages": [("user", "Write a blog on this topic")],
    }
)

print(result["blog_draft"])
```

***

### 4. Orchestrator vs worker-node pattern (advanced)

If you want **true workers** (separate processes, not just nodes) — scale the full system:

- **Orchestrator** (LangGraph)  
  - Only decides “who runs next” and manages state.
- **Worker nodes** (FastAPI services or LangChain workers)  
  - Each agent runs as an HTTP service:
    - `/research` → researcher agent
    - `/write` → writer agent
    - `/edit` → editor agent

With **Orkes Conductor** or similar (optional for production):

- LangGraph dispatches tasks to worker services.
- Workers return results; orchestrator persists state and decides next step.

Minimal worker sketch (FastAPI):

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ResearchReq(BaseModel):
    topic: str

class ResearchResp(BaseModel):
    research_data: list[str]

@app.post("/research", response_model=ResearchResp)
def research_task(req: ResearchReq):
    # call researcher_node logic with local SearxNG/Ollama
    return ResearchResp(research_data=["snippet1", "snippet2"])
```

Then your LangGraph orchestrator calls `http.post("/research")`, etc.

***

### 5. Demo you can run locally (single file)

Here’s a **minimal, runnable demo** you can copy-paste and extend in 24 hours (fully adapted to the local Ollama + SearxNG stack):

```python
from typing import TypedDict, List
from langchain_community.utilities import SearxSearchWrapper  # or DuckDuckGoSearchRun
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, MessagesState, START, END

# Set up tools and LLM (local only)
search_tool = SearxSearchWrapper(searx_host="http://localhost:8080")
llm = ChatOllama(model="mistral", temperature=0.3)

# State
class AgentState(MessagesState):
    topic: str
    research_data: List[str]
    blog_draft: List[str]

# Researcher node
def researcher_node(state: AgentState):
    topic = state["topic"]
    results = search_tool.run(f"Blog background: {topic}")
    research = [results]
    return {
        "research_data": research,
    }

# Writer node
BLOG_PROMPT = """
Write a clear, engaging blog post about:

Topic: {topic}

Research snippets:
{research}

Format as markdown with 3–4 short paragraphs.
"""

prompt = ChatPromptTemplate.from_template(BLOG_PROMPT)
writer_chain = prompt | llm

def writer_node(state: AgentState):
    blog = writer_chain.invoke({
        "topic": state["topic"],
        "research": "\n".join(state["research_data"]),
    }).content
    return {"blog_draft": [blog]}

# Build graph (add Editor node similarly for full refinement)
graph_builder = StateGraph(AgentState)
graph_builder.add_node("Researcher", researcher_node)
graph_builder.add_node("Writer", writer_node)

graph_builder.set_entry_point("Researcher")
graph_builder.add_edge("Researcher", "Writer")
graph_builder.add_edge("Writer", END)

blog_app = graph_builder.compile()

# Run demo
if __name__ == "__main__":
    topic = "How AI agents are changing software development"
    result = blog_app.invoke(
        {
            "topic": topic,
            "research_data": [],
            "blog_draft": [],
            "messages": [("user", "Write a blog on this topic")],
        }
    )
    print(result["blog_draft"][0])
```

- Install (local only):
  ```bash
  pip install langchain langgraph langchain-community
  # No API keys needed
  ```
- Run the script and it outputs a blog draft. This serves as a quick prototype that can be expanded into the full 8-agent parallel system.

***

### 6. How to make it “custom” and production-ready

Given your constraints and 24-hour window:

- **Customization layer**:
  - Replace the `BLOG_PROMPT` / `EDIT_PROMPT` with your own style guide.
  - Add style rules (e.g., “Use UK English”, “Include 2–3 bullet lists”).
  - Plug in your local RAG (via ChromaDB retriever) instead of web search.

- **LangSmith**:
  - Wrap the app with `langsmith` tracing for node-level visibility.

- **Local compute**:
  - Already fully local with `ChatOllama` and SearxNG.

***

### 7. Next-step suggestion (for your 24h limit)

Given your time box:

1. **Day 1 morning (3–4 hours)**  
   - Run the minimal demo above and tweak prompts.
2. **Day 1 afternoon (4–6 hours)**  
   - Add your own RAG (ChromaDB) in place of search.
3. **Day 1 evening (2–4 hours)**  
   - Add the Editor node and basic checks (word count, headings, etc.).
4. **If you want true workers**  
   - Wrap each agent in a FastAPI service and wire them via LangGraph.

*This reference pipeline complements the main advanced architecture without replacing any of its parallel, image-enabled, citation-rich capabilities.*