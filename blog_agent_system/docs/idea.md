Here is the complete, production-grade architecture specification for an **Agentic AI Blog Writing System**.

---

# Agentic AI Blog Writing System — Complete Architecture Specification

## (1) System Architecture Overview

### Architectural Blueprint

The system adopts a **Supervisor-Based Multi-Agent Architecture** with LangGraph as the core orchestration engine. It implements a **hierarchical control flow** where a central `BlogOrchestrator` (supervisor agent) coordinates specialized worker agents through a stateful directed graph.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │   REST API   │  │  CLI Tool    │  │  Web UI      │                       │
│  │   (FastAPI)  │  │  (Typer)     │  │  (Streamlit) │                       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                       │
└─────────┼─────────────────┼─────────────────┼───────────────────────────────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER (LangGraph)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    BlogOrchestrator (Supervisor)                     │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │    │
│  │  │ Research │→ │ Outline  │→ │  Draft   │→ │  Editor  │→ │ Publish│ │    │
│  │  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │  │ Agent  │ │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────────┘ │    │
│  │       ↓              ↓              ↓            ↓                   │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │  │              PARALLEL WORKERS (Conditional Branching)            │ │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │ │    │
│  │  │  │ SEO Agent   │  │ FactChecker │  │ Image Generation Agent  │ │ │    │
│  │  │  │ (Parallel)  │  │ (Parallel)  │  │ (Conditional)           │ │ │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │ │    │
│  │  └─────────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────────────┐
│                         AGENT LAYER                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Planner   │ │  Researcher │ │   Writer    │ │   Critic    │           │
│  │   Agent     │ │   Agent     │ │   Agent     │ │   Agent     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────────────┐
│                      TOOL & FUNCTION LAYER                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  Web     │ │   RAG    │ │  Code    │ │  File    │ │  Image   │          │
│  │  Search  │ │ Retrieve │ │ Executor │ │   I/O    │ │ Generate │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────────────┐
│                    LLM INTEGRATION LAYER                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   OpenAI    │ │  Anthropic  │ │   Ollama    │ │   Gemini    │           │
│  │   GPT-4o    │ │   Claude 3  │ │  Llama 3    │ │   Flash     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────────────┐
│                    MEMORY & STATE LAYER                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │   Short-Term │ │   Long-Term  │ │   Episodic   │ │   Shared     │       │
│  │   (Redis)    │ │  (ChromaDB)  │ │ (PostgreSQL) │ │   State      │       │
│  │              │ │              │ │              │ │ (LangGraph)  │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────────────┐
│                   DATA PERSISTENCE LAYER                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │  PostgreSQL  │ │   ChromaDB   │ │    S3/MinIO  │ │   Redis      │       │
│  │  (Metadata)  │ │  (Vectors)   │ │  (Artifacts) │ │  (Cache)     │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Orchestration Model: Multi-Agent with Supervisor

- **Pattern**: Hierarchical Supervisor-Workers (not pure peer-to-peer)
- **Rationale**: Blog writing requires strict sequential dependencies (research before outline, outline before draft) with selective parallelization (SEO and fact-checking can run simultaneously on a completed draft)
- **Control Flow**:
  - **Sequential**: Research → Outline → Draft → Edit
  - **Parallel**: SEO optimization + Fact-checking (fork-join pattern)
  - **Conditional Branching**: Quality gate evaluates draft score; if < 0.7, loops back to Writer with critique feedback
  - **Loop Termination**: Max 3 revision iterations or quality score ≥ 0.85

### Layer Connectivity

All layers communicate through a **centralized state object** (LangGraph `StateGraph`) that acts as the single source of truth. Agents do not communicate directly; they read from and write to the shared state, ensuring loose coupling and deterministic replay.

---

## (2) Project File/Folder Hierarchy

```
blog_agent_system/
├── README.md                          # Project overview, setup instructions, architecture diagram
├── pyproject.toml                     # Poetry dependencies, project metadata, tool configs
├── .env.example                       # Template for all environment variables
├── docker-compose.yml                 # Local infrastructure: PostgreSQL, Redis, ChromaDB, MinIO
├── Makefile                           # Common commands: test, lint, migrate, run
│
├── src/
│   └── blog_agent_system/
│       ├── __init__.py
│       │
│       ├── main.py                    # Application entry point; FastAPI app factory
│       ├── cli.py                     # Typer CLI for local development and debugging
│       │
│       ├── config/                    # ─── CONFIGURATION LAYER ───
│       │   ├── __init__.py
│       │   ├── settings.py            # Pydantic-Settings; env var validation & defaults
│       │   ├── llm_config.py          # Per-agent model selection, temperature, token budgets
│       │   ├── prompts/               # Jinja2 prompt templates (version-controlled text)
│       │   │   ├── research.j2
│       │   │   ├── outline.j2
│       │   │   ├── draft.j2
│       │   │   ├── edit.j2
│       │   │   ├── seo.j2
│       │   │   └── critique.j2
│       │   └── agents.yaml            # Agent role definitions, system prompts, tool assignments
│       │
│       ├── core/                      # ─── CORE ORCHESTRATION ───
│       │   ├── __init__.py
│       │   ├── state.py               # Pydantic state schema; typed graph state definition
│       │   ├── graph.py               # LangGraph construction; nodes, edges, conditional routing
│       │   ├── orchestrator.py        # Supervisor logic; delegates to agents, manages lifecycle
│       │   └── checkpoint.py          # LangGraph persistence adapter (PostgresSaver setup)
│       │
│       ├── agents/                    # ─── AGENT LAYER ───
│       │   ├── __init__.py
│       │   ├── base.py                # BaseAgent abstract class; shared init, LLM binding, execute()
│       │   ├── research_agent.py      # Gathers sources, summarizes findings, extracts quotes
│       │   ├── outline_agent.py       # Structures blog sections, headings, key points per section
│       │   ├── writer_agent.py        # Generates prose per section; maintains voice consistency
│       │   ├── editor_agent.py        # Grammar, style, clarity improvements; line-edits
│       │   ├── seo_agent.py           # Keyword optimization, meta descriptions, readability
│       │   ├── fact_checker_agent.py  # Verifies claims against sources; confidence scoring
│       │   └── image_agent.py         # Generates/retrieves cover images and section visuals
│       │
│       ├── tools/                     # ─── TOOL & FUNCTION LAYER ───
│       │   ├── __init__.py
│       │   ├── registry.py            # Tool registration decorator; discovery and metadata
│       │   ├── base.py                # BaseTool class; input schema, execution, error handling
│       │   ├── web_search.py          # Tavily/Brave Search integration; result ranking
│       │   ├── rag_retriever.py       # ChromaDB vector search; context assembly
│       │   ├── code_executor.py       # E2B/sandboxed Python execution for data viz generation
│       │   ├── file_io.py             # Read/write local artifacts; markdown, images, JSON
│       │   └── image_generator.py     # Stable Diffusion/Flux API for blog illustrations
│       │
│       ├── llm/                       # ─── LLM INTEGRATION LAYER ───
│       │   ├── __init__.py
│       │   ├── provider.py            # Abstract LLMProvider; unified interface
│       │   ├── openai_provider.py     # GPT-4o, GPT-4o-mini implementations
│       │   ├── anthropic_provider.py  # Claude 3.5 Sonnet/Haiku implementations
│       │   ├── ollama_provider.py     # Local model support for cost-sensitive operations
│       │   ├── factory.py             # Provider instantiation based on config/agent role
│       │   ├── token_manager.py       # Token counting, budget enforcement, truncation
│       │   └── structured_output.py   # JSON schema enforcement; Pydantic model parsing
│       │
│       ├── memory/                    # ─── MEMORY & STATE MANAGEMENT ───
│       │   ├── __init__.py
│       │   ├── short_term.py          # Redis-backed conversation buffer; sliding window
│       │   ├── long_term.py           # ChromaDB vector store; embedding-based retrieval
│       │   ├── episodic.py            # PostgreSQL conversation history; thread persistence
│       │   ├── shared_state.py        # LangGraph state accessors; thread-scoped read/write
│       │   └── eviction.py            # LRU eviction, TTL policies, memory compaction
│       │
│       ├── persistence/               # ─── DATA PERSISTENCE LAYER ───
│       │   ├── __init__.py
│       │   ├── database.py            # SQLAlchemy engine, session factory, connection pooling
│       │   ├── models/                # ORM models (Alembic-managed)
│       │   │   ├── __init__.py
│       │   │   ├── task.py            # Blog generation tasks: status, config, timestamps
│       │   │   ├── agent_run.py       # Per-agent execution logs: input, output, latency
│       │   │   ├── document.py        # Generated blog posts: markdown, metadata, versions
│       │   │   └── feedback.py        # Human-in-the-loop feedback for RLHF
│       │   ├── repositories/          # Repository pattern; data access abstraction
│       │   │   ├── __init__.py
│       │   │   ├── task_repo.py
│       │   │   └── document_repo.py
│       │   └── vector_store.py        # ChromaDB client wrapper; collection management
│       │
│       ├── models/                    # ─── PYDANTIC SCHEMAS ───
│       │   ├── __init__.py
│       │   ├── blog.py                # BlogPost, Section, Metadata schemas
│       │   ├── research.py            # Source, Claim, Evidence schemas
│       │   └── workflow.py            # TaskRequest, TaskStatus, QualityScore schemas
│       │
│       ├── api/                       # ─── API LAYER ───
│       │   ├── __init__.py
│       │   ├── router.py              # FastAPI route definitions
│       │   ├── dependencies.py        # DI container: DB sessions, Redis, vector store
│       │   └── schemas.py             # Request/response DTOs
│       │
│       └── utils/                     # ─── UTILITIES ───
│           ├── __init__.py
│           ├── logging.py             # Structured JSON logging with correlation IDs
│           ├── exceptions.py          # Custom exception hierarchy
│           └── validators.py          # Input sanitization, URL validation
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures: test DB, mock LLM, fake state
│   ├── unit/                          # Isolated agent/tool tests
│   │   ├── test_agents.py
│   │   ├── test_tools.py
│   │   └── test_llm_providers.py
│   ├── integration/                   # Multi-agent workflow tests
│   │   ├── test_graph_execution.py
│   │   └── test_memory_persistence.py
│   └── e2e/                           # Full blog generation scenarios
│       └── test_blog_workflow.py
│
├── alembic/                           # Database migration scripts
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│
├── docs/                              # Architecture Decision Records (ADRs)
│   ├── adr-001-orchestrator-choice.md
│   ├── adr-002-memory-strategy.md
│   └── adr-003-llm-abstraction.md
│
└── scripts/
    ├── seed_db.py                     # Initial data seeding
    └── health_check.py                # Dependency health verification
```

---

## (3) Agent Layer

### Agent Roles & Responsibilities

| Agent | Role | LLM | Core Responsibility | Tools |
|-------|------|-----|---------------------|-------|
| **ResearchAgent** | Information Gatherer | GPT-4o-mini (fast, cheap) | Searches web, extracts key facts, identifies authoritative sources, summarizes conflicting viewpoints | WebSearch, RAGRetriever |
| **OutlineAgent** | Structure Designer | Claude 3.5 Haiku | Transforms research into hierarchical outline with H2/H3 headings, estimated word counts per section, narrative flow logic | FileIO |
| **WriterAgent** | Content Creator | GPT-4o / Claude 3.5 Sonnet | Generates prose section-by-section; maintains consistent tone, incorporates sources, handles transitions | RAGRetriever, FileIO |
| **EditorAgent** | Quality Refiner | Claude 3.5 Sonnet | Line-edits for clarity, grammar, style guide adherence; suggests structural improvements | FileIO |
| **SEOAgent** | Search Optimizer | GPT-4o-mini | Keyword density analysis, meta description generation, header tag optimization, readability scoring (Flesch-Kincaid) | WebSearch (for keyword volume) |
| **FactCheckerAgent** | Veracity Validator | Claude 3.5 Haiku | Cross-references draft claims against ResearchAgent sources; assigns confidence scores; flags hallucinations | WebSearch, RAGRetriever |
| **ImageAgent** | Visual Designer | DALL-E 3 / Flux | Generates cover images and section illustrations based on content themes | ImageGenerator |

### Agent-to-Agent Communication Protocol

Agents do **not** communicate directly. All interaction is **state-mediated** through the LangGraph shared state object:

```python
# src/blog_agent_system/core/state.py
from typing import Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
import operator

class Section(BaseModel):
    heading: str
    content: str = ""
    word_count: int = 0
    sources: list[str] = []

class BlogState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    target_audience: str = "general"
    tone: str = "informative"
    research_findings: list[dict] = []
    outline: list[Section] = []
    draft: str = ""
    draft_sections: list[Section] = []
    seo_metadata: dict = {}
    fact_check_results: list[dict] = []
    quality_score: float = 0.0
    revision_count: int = 0
    max_revisions: int = 3
    final_blog: str = ""
    status: str = "pending"  # pending, researching, outlining, drafting, editing, complete
```

**Communication Pattern**:
1. ResearchAgent writes to `state.research_findings`
2. OutlineAgent reads `research_findings`, writes to `state.outline`
3. WriterAgent reads `outline`, writes to `state.draft_sections`
4. EditorAgent reads `draft_sections`, writes to `state.draft`
5. SEOAgent and FactCheckerAgent read `draft` in parallel, write to respective fields

### Task Delegation

The Orchestrator uses **conditional edges** to delegate:

```python
# src/blog_agent_system/core/graph.py (simplified)
from langgraph.graph import StateGraph, END
from blog_agent_system.agents import (
    ResearchAgent, OutlineAgent, WriterAgent, 
    EditorAgent, SEOAgent, FactCheckerAgent
)

def create_blog_graph():
    builder = StateGraph(BlogState)
    
    # Register nodes
    builder.add_node("research", ResearchAgent().execute)
    builder.add_node("outline", OutlineAgent().execute)
    builder.add_node("write", WriterAgent().execute)
    builder.add_node("edit", EditorAgent().execute)
    builder.add_node("seo", SEOAgent().execute)
    builder.add_node("fact_check", FactCheckerAgent().execute)
    builder.add_node("quality_gate", quality_gate_node)
    
    # Sequential edges
    builder.set_entry_point("research")
    builder.add_edge("research", "outline")
    builder.add_edge("outline", "write")
    builder.add_edge("write", "edit")
    
    # Parallel fork after edit
    builder.add_edge("edit", "seo")
    builder.add_edge("edit", "fact_check")
    
    # Join parallel branches
    builder.add_edge("seo", "quality_gate")
    builder.add_edge("fact_check", "quality_gate")
    
    # Conditional loopback
    builder.add_conditional_edges(
        "quality_gate",
        lambda state: "accept" if state.quality_score >= 0.85 or state.revision_count >= 3 else "revise",
        {"accept": END, "revise": "write"}
    )
    
    return builder.compile()
```

### Agent Instantiation & Configuration

```python
# src/blog_agent_system/agents/base.py
from abc import ABC, abstractmethod
from typing import Any
from blog_agent_system.llm.factory import LLMFactory
from blog_agent_system.config.llm_config import AgentLLMConfig
from blog_agent_system.memory.shared_state import StateAccessor

class BaseAgent(ABC):
    def __init__(self, role: str, config: AgentLLMConfig):
        self.role = role
        self.llm = LLMFactory.create(config)
        self.state = StateAccessor()
        self.tools = []
    
    def bind_tools(self, tools: list):
        self.tools = tools
        self.llm = self.llm.bind_tools(tools)
    
    @abstractmethod
    async def execute(self, state: BlogState) -> dict:
        """Execute agent logic and return state updates."""
        pass
    
    def get_system_prompt(self) -> str:
        return f"You are the {self.role}. Follow your specialized instructions precisely."
```

---

## (4) Orchestration & Workflow Engine

### Orchestrator Design: LangGraph

LangGraph is selected over CrewAI for this system because:
- **Stateful cycles** are first-class (critical for revision loops)
- **Fine-grained control** over state transitions and conditional branching
- **Built-in persistence** via checkpointers for fault tolerance
- **Deterministic replay** for debugging and audit trails

### Graph Node Definitions

| Node | Type | Input State | Output State | Idempotency |
|------|------|-------------|--------------|-------------|
| `research` | Agent Node | `topic`, `target_audience` | `research_findings`, `messages` | Yes (same topic → same sources) |
| `outline` | Agent Node | `research_findings` | `outline`, `messages` | Yes |
| `write` | Agent Node | `outline`, `revision_count` | `draft_sections`, `messages` | No (revisions mutate) |
| `edit` | Agent Node | `draft_sections` | `draft`, `messages` | No |
| `seo` | Parallel Agent | `draft` | `seo_metadata`, `messages` | Yes |
| `fact_check` | Parallel Agent | `draft`, `research_findings` | `fact_check_results`, `messages` | Yes |
| `quality_gate` | Control Node | `draft`, `seo_metadata`, `fact_check_results` | `quality_score`, `revision_count` | Deterministic |
| `image_gen` | Conditional Agent | `outline` | `cover_image_url`, `messages` | No |

### Edge Conditions

```python
# src/blog_agent_system/core/graph.py
from typing import Literal

def should_generate_images(state: BlogState) -> Literal["generate", "skip"]:
    """Conditional entry for image generation based on user preference."""
    return "generate" if state.get("include_images", False) else "skip"

def parallel_join_condition(state: BlogState) -> Literal["quality_gate", "wait"]:
    """Ensures both SEO and FactChecker complete before quality gate."""
    required_keys = ["seo_metadata", "fact_check_results"]
    if all(k in state and state[k] for k in required_keys):
        return "quality_gate"
    return "wait"

def revision_router(state: BlogState) -> Literal["accept", "revise"]:
    """Routes to END or back to writer based on quality score."""
    if state.quality_score >= 0.85:
        return "accept"
    if state.revision_count >= state.max_revisions:
        return "accept"  # Force accept after max iterations
    return "revise"
```

### State Schema (Detailed)

```python
# src/blog_agent_system/core/state.py
from typing import Annotated, Sequence, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
import operator

class Source(BaseModel):
    url: str
    title: str
    snippet: str
    credibility_score: float = Field(ge=0.0, le=1.0)

class SEOData(BaseModel):
    title_tag: str
    meta_description: str
    keywords: list[str]
    readability_score: float
    keyword_density: dict[str, float]

class FactCheckResult(BaseModel):
    claim: str
    verified: bool
    confidence: float
    source_ref: Optional[str]
    correction: Optional[str]

class BlogState(BaseModel):
    # Messaging
    messages: Annotated[Sequence[BaseMessage], operator.add] = []
    
    # Input parameters
    topic: str = Field(..., min_length=5, max_length=500)
    target_audience: str = "technical professionals"
    tone: str = "informative yet conversational"
    word_count_target: int = Field(default=1500, ge=500, le=5000)
    include_images: bool = True
    style_guide: str = "AP"  # AP, Chicago, MLA
    
    # Workflow state
    status: str = "pending"
    current_step: str = "init"
    
    # Agent outputs
    research_findings: list[Source] = []
    outline: list[Section] = []
    draft_sections: list[Section] = []
    draft: str = ""
    edited_draft: str = ""
    seo_metadata: Optional[SEOData] = None
    fact_check_results: list[FactCheckResult] = []
    cover_image_url: Optional[str] = None
    section_images: dict[str, str] = {}  # heading -> url
    
    # Quality & loop control
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    revision_count: int = Field(default=0, ge=0, le=5)
    max_revisions: int = 3
    revision_feedback: str = ""
    
    # Final output
    final_blog: str = ""
    export_format: str = "markdown"  # markdown, html, pdf
    
    class Config:
        arbitrary_types_allowed = True
```

### Loop Termination Criteria

1. **Quality Threshold**: `quality_score >= 0.85` (configurable)
2. **Iteration Cap**: `revision_count >= max_revisions` (default 3)
3. **Timeout**: Global workflow timeout of 10 minutes
4. **User Abort**: External signal via checkpoint thread ID
5. **Error Budget**: Max 2 consecutive tool failures per agent

---

## (5) Tool & Function Layer

### Tool Inventory

| Tool | Provider | Purpose | Input Schema | Output Schema |
|------|----------|---------|--------------|---------------|
| `web_search` | Tavily API | Real-time information retrieval | `query: str, max_results: int, search_depth: str` | `list[Source]` |
| `rag_retrieve` | ChromaDB | Semantic search over knowledge base | `query: str, top_k: int, filter: dict` | `list[Document]` |
| `code_execute` | E2B Sandbox | Generate data visualizations/charts | `code: str, language: str, timeout: int` | `output: str, artifacts: list` |
| `file_read` | Local FS | Read templates, style guides, existing content | `path: str, encoding: str` | `content: str` |
| `file_write` | Local FS | Persist drafts, final outputs | `path: str, content: str, format: str` | `success: bool, bytes_written: int` |
| `image_generate` | DALL-E 3 / Flux | Create blog illustrations | `prompt: str, size: str, style: str` | `url: str, revised_prompt: str` |
| `keyword_analyze` | SEMrush API (mock) | SEO keyword volume/difficulty | `keywords: list[str]` | `metrics: dict` |

### Tool Registration & Discovery

```python
# src/blog_agent_system/tools/registry.py
from typing import Callable, Dict
from blog_agent_system.tools.base import BaseTool

class ToolRegistry:
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, tool: BaseTool):
        cls._tools[tool.name] = tool
        return tool
    
    @classmethod
    def get(cls, name: str) -> BaseTool:
        if name not in cls._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return cls._tools[name]
    
    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls._tools.keys())

def tool(name: str, description: str, args_schema: type):
    """Decorator for tool registration."""
    def decorator(func: Callable):
        tool_instance = BaseTool(
            name=name,
            description=description,
            func=func,
            args_schema=args_schema
        )
        ToolRegistry.register(tool_instance)
        return func
    return decorator
```

### Tool Base Class

```python
# src/blog_agent_system/tools/base.py
from pydantic import BaseModel, ValidationError
from typing import Callable, Any
from blog_agent_system.utils.exceptions import ToolExecutionError

class BaseTool:
    def __init__(self, name: str, description: str, func: Callable, args_schema: type[BaseModel]):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema
    
    def validate_input(self, kwargs: dict) -> BaseModel:
        try:
            return self.args_schema(**kwargs)
        except ValidationError as e:
            raise ToolExecutionError(f"Invalid input for {self.name}: {e}")
    
    async def invoke(self, **kwargs) -> Any:
        validated = self.validate_input(kwargs)
        try:
            result = await self.func(**validated.model_dump())
            return {
                "tool": self.name,
                "status": "success",
                "result": result,
                "input": validated.model_dump()
            }
        except Exception as e:
            return {
                "tool": self.name,
                "status": "error",
                "error": str(e),
                "input": validated.model_dump()
            }
```

### Tool Invocation Example

```python
# src/blog_agent_system/tools/web_search.py
from pydantic import BaseModel, Field
from blog_agent_system.tools.registry import tool
import aiohttp

class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)
    search_depth: str = Field(default="basic", pattern="^(basic|advanced)$")
    include_domains: list[str] = []

@tool(
    name="web_search",
    description="Search the web for current information on a topic",
    args_schema=WebSearchInput
)
async def web_search(query: str, max_results: int, search_depth: str, include_domains: list) -> list:
    api_key = settings.TAVILY_API_KEY
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_domains": include_domains,
                "api_key": api_key
            }
        ) as resp:
            data = await resp.json()
            return [
                {
                    "url": r["url"],
                    "title": r["title"],
                    "snippet": r["content"],
                    "credibility_score": r.get("score", 0.5)
                }
                for r in data.get("results", [])
            ]
```

### Result Validation

All tool outputs are validated against Pydantic schemas before being written to state. The `ToolRegistry` wraps execution in try-catch blocks, returning structured error objects that agents can interpret for retry logic.

---

## (6) Memory & State Management

### Memory Taxonomy

| Type | Purpose | Backend | Retrieval Strategy | Eviction Policy |
|------|---------|---------|-------------------|-----------------|
| **Short-Term** | In-context window management; recent turns | Redis (List) | LIFO sliding window | TTL 1 hour; max 20 messages |
| **Long-Term** | Semantic retrieval of domain knowledge | ChromaDB (Vector) | Similarity search (cosine) | None (append-only) |
| **Episodic** | Conversation history across sessions | PostgreSQL | Time-range query; thread ID | Soft delete after 90 days |
| **Shared** | Inter-agent state synchronization | LangGraph Checkpoint (PostgreSQL) | Exact key lookup | Thread-scoped lifecycle |

### Short-Term Memory (In-Context)

```python
# src/blog_agent_system/memory/short_term.py
import redis
from langchain_core.messages import BaseMessage
from typing import Sequence

class ShortTermMemory:
    def __init__(self, redis_client: redis.Redis, max_messages: int = 20, ttl: int = 3600):
        self.redis = redis_client
        self.max_messages = max_messages
        self.ttl = ttl
    
    async def add_messages(self, thread_id: str, messages: Sequence[BaseMessage]):
        key = f"stm:{thread_id}"
        pipe = self.redis.pipeline()
        for msg in messages:
            pipe.lpush(key, msg.json())
        pipe.ltrim(key, 0, self.max_messages - 1)
        pipe.expire(key, self.ttl)
        await pipe.execute()
    
    async def get_messages(self, thread_id: str) -> list[BaseMessage]:
        key = f"stm:{thread_id}"
        raw = await self.redis.lrange(key, 0, -1)
        return [BaseMessage.parse_json(m) for m in raw]
```

### Long-Term Memory (Vector Store)

```python
# src/blog_agent_system/memory/long_term.py
import chromadb
from langchain_openai import OpenAIEmbeddings

class LongTermMemory:
    def __init__(self, collection_name: str = "blog_knowledge"):
        self.client = chromadb.HttpClient(host="chromadb", port=8000)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    async def store(self, documents: list[str], metadata: list[dict], ids: list[str]):
        embeddings = await self.embeddings.aembed_documents(documents)
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
    
    async def retrieve(self, query: str, top_k: int = 5, filter: dict = None) -> list[dict]:
        query_embedding = await self.embeddings.aembed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        return [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
```

### Episodic Memory (Conversation History)

```python
# src/blog_agent_system/memory/episodic.py
from sqlalchemy.orm import Session
from blog_agent_system.persistence.models import ConversationTurn
from datetime import datetime, timedelta

class EpisodicMemory:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def record_turn(self, thread_id: str, agent: str, role: str, content: str, tokens_used: int):
        turn = ConversationTurn(
            thread_id=thread_id,
            agent=agent,
            role=role,
            content=content,
            tokens_used=tokens_used,
            timestamp=datetime.utcnow()
        )
        self.db.add(turn)
        self.db.commit()
    
    def get_thread_history(self, thread_id: str, limit: int = 100) -> list[ConversationTurn]:
        return self.db.query(ConversationTurn).filter(
            ConversationTurn.thread_id == thread_id
        ).order_by(ConversationTurn.timestamp.desc()).limit(limit).all()
    
    def archive_old_threads(self, days: int = 90):
        cutoff = datetime.utcnow() - timedelta(days=days)
        self.db.query(ConversationTurn).filter(
            ConversationTurn.timestamp < cutoff
        ).update({"archived": True})
        self.db.commit()
```

### Shared State (LangGraph Checkpoint)

```python
# src/blog_agent_system/memory/shared_state.py
from langgraph.checkpoint.postgres import PostgresSaver
from blog_agent_system.core.state import BlogState

class StateAccessor:
    def __init__(self, checkpointer: PostgresSaver):
        self.checkpointer = checkpointer
    
    async def read(self, thread_id: str) -> BlogState:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await self.checkpointer.aget(config)
        return BlogState(**checkpoint["channel_values"]) if checkpoint else BlogState()
    
    async def write(self, thread_id: str, updates: dict):
        # LangGraph handles this automatically via node returns,
        # but this is used for external state injection
        config = {"configurable": {"thread_id": thread_id}}
        await self.checkpointer.aput(config, updates)
```

---

## (7) LLM Integration Layer

### Provider Abstraction

```python
# src/blog_agent_system/llm/provider.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
from pydantic import BaseModel

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list, tools: list = None, response_format: type[BaseModel] = None) -> str:
        pass
    
    @abstractmethod
    async def stream(self, messages: list) -> AsyncIterator[str]:
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass
```

### Provider Implementations

```python
# src/blog_agent_system/llm/openai_provider.py
from openai import AsyncOpenAI
from blog_agent_system.llm.provider import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 4096):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def generate(self, messages, tools=None, response_format=None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    def count_tokens(self, text: str) -> int:
        # Use tiktoken for accurate counting
        import tiktoken
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))
```

### Model Selection Per Agent Role

```python
# src/blog_agent_system/config/llm_config.py
from pydantic import BaseModel
from typing import Optional

class AgentLLMConfig(BaseModel):
    provider: str  # openai, anthropic, ollama
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    retry_attempts: int = 3

AGENT_LLM_CONFIGS = {
    "research": AgentLLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.3, max_tokens=2048),
    "outline": AgentLLMConfig(provider="anthropic", model="claude-3-haiku-20240307", temperature=0.5, max_tokens=2048),
    "writer": AgentLLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022", temperature=0.8, max_tokens=8192),
    "editor": AgentLLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022", temperature=0.4, max_tokens=8192),
    "seo": AgentLLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.2, max_tokens=1024),
    "fact_checker": AgentLLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.1, max_tokens=2048),
    "image": AgentLLMConfig(provider="openai", model="dall-e-3", temperature=1.0, max_tokens=1024),
}
```

### Prompt Templates (Jinja2)

```python
# src/blog_agent_system/config/prompts/draft.j2
You are an expert blog writer. Write section {{ section_number }} of a blog post.

TOPIC: {{ topic }}
SECTION HEADING: {{ heading }}
TARGET AUDIENCE: {{ audience }}
TONE: {{ tone }}
WORD COUNT: {{ word_count }}
STYLE GUIDE: {{ style_guide }}

RESEARCH CONTEXT:
{% for source in sources %}
- {{ source.title }}: {{ source.snippet }}
{% endfor %}

PREVIOUS SECTION SUMMARY:
{{ previous_section_summary }}

INSTRUCTIONS:
1. Write engaging, original prose
2. Incorporate 2-3 sources naturally
3. Use subheadings if appropriate
4. End with a transition to the next section
5. Follow {{ style_guide }} style conventions

OUTPUT ONLY THE SECTION CONTENT. No meta-commentary.
```

### Structured Output Enforcement

```python
# src/blog_agent_system/llm/structured_output.py
from pydantic import BaseModel, Field
from typing import Type

class OutlineOutput(BaseModel):
    title: str = Field(..., description="Compelling blog post title")
    sections: list[dict] = Field(..., description="List of sections with heading and key_points")
    estimated_read_time: int = Field(..., description="Estimated read time in minutes")

async def generate_structured(
    provider: LLMProvider,
    messages: list,
    output_schema: Type[BaseModel]
) -> BaseModel:
    # Add schema instruction to system prompt
    schema_json = output_schema.model_json_schema()
    messages[0]["content"] += f"\n\nYou must respond with valid JSON matching this schema: {schema_json}"
    
    raw = await provider.generate(messages, response_format=output_schema)
    return output_schema.model_validate_json(raw)
```

### Token Budget Management

```python
# src/blog_agent_system/llm/token_manager.py
class TokenBudgetManager:
    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens
        self.reserved_output = 4096
        self.available_input = max_tokens - self.reserved_output
    
    def truncate_context(self, messages: list, provider: LLMProvider) -> list:
        total = sum(provider.count_tokens(m["content"]) for m in messages)
        while total > self.available_input and len(messages) > 2:
            # Remove oldest non-system message
            for i, msg in enumerate(messages):
                if msg["role"] != "system":
                    total -= provider.count_tokens(msg["content"])
                    messages.pop(i)
                    break
        return messages
```

### Retry & Fallback Logic

```python
# src/blog_agent_system/llm/factory.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMFactory:
    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError))
    )
    def create(config: AgentLLMConfig):
        try:
            if config.provider == "openai":
                return OpenAIProvider(config.model, config.temperature, config.max_tokens)
            elif config.provider == "anthropic":
                return AnthropicProvider(config.model, config.temperature, config.max_tokens)
            elif config.provider == "ollama":
                return OllamaProvider(config.model, config.temperature, config.max_tokens)
        except Exception as e:
            # Fallback to Ollama local model if cloud providers fail
            return OllamaProvider("llama3.1:70b", 0.7, 4096)
```

---

## (8) Data Persistence Layer

### Database Schema Design (PostgreSQL)

```sql
-- tasks table: Orchestration-level tracking
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) NOT NULL UNIQUE,
    topic TEXT NOT NULL,
    target_audience VARCHAR(100),
    tone VARCHAR(50),
    word_count_target INTEGER DEFAULT 1500,
    status VARCHAR(50) DEFAULT 'pending',
    quality_score DECIMAL(3,2),
    revision_count INTEGER DEFAULT 0,
    final_document_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- agent_runs table: Per-agent execution audit
CREATE TABLE agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    agent_name VARCHAR(50) NOT NULL,
    input_state JSONB,
    output_state JSONB,
    tokens_input INTEGER,
    tokens_output INTEGER,
    latency_ms INTEGER,
    status VARCHAR(20) DEFAULT 'running',
    error TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- documents table: Versioned blog outputs
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    version INTEGER NOT NULL DEFAULT 1,
    title VARCHAR(500),
    content TEXT NOT NULL,
    format VARCHAR(20) DEFAULT 'markdown',
    seo_metadata JSONB,
    fact_check_summary JSONB,
    word_count INTEGER,
    reading_time_minutes INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- conversation_turns table: Episodic memory
CREATE TABLE conversation_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(50),
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    tokens_used INTEGER,
    tool_calls JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    archived BOOLEAN DEFAULT FALSE
);

-- feedback table: Human-in-the-loop for RLHF
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    comments TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_thread ON tasks(thread_id);
CREATE INDEX idx_agent_runs_task ON agent_runs(task_id);
CREATE INDEX idx_conversation_thread ON conversation_turns(thread_id, timestamp DESC);
CREATE INDEX idx_documents_task ON documents(task_id, version DESC);
```

### Vector Store Configuration (ChromaDB)

```python
# src/blog_agent_system/persistence/vector_store.py
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_server_host="chromadb",
            chroma_server_http_port=8000,
            anonymized_telemetry=False
        ))
        
        # Collections for different knowledge domains
        self.collections = {
            "blog_knowledge": self.client.get_or_create_collection("blog_knowledge"),
            "style_guides": self.client.get_or_create_collection("style_guides"),
            "previous_blogs": self.client.get_or_create_collection("previous_blogs")
        }
    
    def get_collection(self, name: str):
        return self.collections.get(name)
```

### ORM Models (SQLAlchemy)

```python
# src/blog_agent_system/persistence/models/task.py
from sqlalchemy import Column, String, Integer, DateTime, DECIMAL, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from blog_agent_system.persistence.database import Base
import uuid

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String(255), unique=True, nullable=False, index=True)
    topic = Column(Text, nullable=False)
    target_audience = Column(String(100))
    tone = Column(String(50))
    word_count_target = Column(Integer, default=1500)
    status = Column(String(50), default="pending", index=True)
    quality_score = Column(DECIMAL(3, 2))
    revision_count = Column(Integer, default=0)
    final_document_id = Column(UUID(as_uuid=True))
    created_at = Column(DateTime(timezone=True), default=func.now())
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
```

### Repository Pattern

```python
# src/blog_agent_system/persistence/repositories/task_repo.py
from sqlalchemy.orm import Session
from blog_agent_system.persistence.models import Task
from typing import Optional

class TaskRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, task: Task) -> Task:
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task
    
    def get_by_thread(self, thread_id: str) -> Optional[Task]:
        return self.db.query(Task).filter(Task.thread_id == thread_id).first()
    
    def update_status(self, task_id: uuid.UUID, status: str):
        task = self.db.query(Task).filter(Task.id == task_id).first()
        task.status = status
        if status == "complete":
            task.completed_at = func.now()
        self.db.commit()
```

---

## (9) End-to-End Execution Flow

### Scenario: User requests "Write a 2000-word technical blog post on quantum error correction for software engineers"

**Step 1: Request Ingestion**
- FastAPI receives POST `/api/v1/blog/generate` with topic, audience="software engineers", tone="technical but accessible", word_count=2000
- Request validated against `TaskRequest` Pydantic schema
- `TaskRepository.create()` inserts row into `tasks` table with `status="pending"`
- Thread ID `thread_abc123` generated and returned to client

**Step 2: Orchestrator Initialization**
- `BlogOrchestrator` loads graph from `create_blog_graph()`
- `PostgresSaver` initializes checkpoint for `thread_abc123`
- Initial state populated: `topic="quantum error correction"`, `word_count_target=2000`, `revision_count=0`

**Step 3: Research Agent Execution**
- Graph enters `research` node
- `ResearchAgent` instantiated with GPT-4o-mini config
- Agent reads `topic` and `target_audience` from state
- **Tool Call 1**: `web_search` invoked with query="quantum error correction for software engineers 2024", max_results=10
- Tavily API returns 10 sources; stored in `state.research_findings`
- **Tool Call 2**: `rag_retrieve` queries ChromaDB "blog_knowledge" collection for existing quantum computing content
- Returns 3 relevant previous blog sections
- Agent synthesizes findings into structured `Source` objects
- Node returns `{"research_findings": [...], "status": "researching"}`

**Step 4: Outline Agent Execution**
- Graph transitions `research` → `outline` (sequential edge)
- `OutlineAgent` (Claude 3 Haiku) reads `research_findings`
- **LLM Invocation**: System prompt loaded from `outline.j2`, injected with research context
- LLM generates structured outline:
  ```json
  {
    "title": "Quantum Error Correction: A Software Engineer's Guide to Fault-Tolerant Computing",
    "sections": [
      {"heading": "Why Classical Error Correction Falls Short", "key_points": [...], "word_count": 300},
      {"heading": "The Qubit: A Brief Primer", "key_points": [...], "word_count": 400},
      {"heading": "Surface Codes and Logical Qubits", "key_points": [...], "word_count": 600},
      {"heading": "Implementing QEC in Practice", "key_points": [...], "word_count": 500},
      {"heading": "The Road Ahead", "key_points": [...], "word_count": 200}
    ]
  }
  ```
- `structured_output.py` validates against `OutlineOutput` schema
- Node returns `{"outline": [...], "status": "outlining"}`

**Step 5: Writer Agent Execution**
- Graph transitions `outline` → `write`
- `WriterAgent` (Claude 3.5 Sonnet) reads `outline` and `research_findings`
- **Loop**: For each section in outline:
  - **LLM Invocation**: `draft.j2` template rendered with section context, sources, style guide
  - Generated prose appended to `state.draft_sections`
  - Token budget checked via `TokenBudgetManager`; context truncated if needed
- After all sections: `draft_sections` concatenated into `state.draft`
- Node returns `{"draft_sections": [...], "draft": "...", "status": "drafting"}`

**Step 6: Editor Agent Execution**
- Graph transitions `write` → `edit`
- `EditorAgent` (Claude 3.5 Sonnet) reads `draft_sections`
- **LLM Invocation**: Line-edits entire draft for clarity, grammar, AP style adherence
- Returns `edited_draft` with tracked changes summary
- Node returns `{"draft": "<edited_content>", "status": "editing"}`

**Step 7: Parallel Fork (SEO + Fact-Checking)**
- Graph splits from `edit` to both `seo` and `fact_check` simultaneously

**Step 7a: SEO Agent (Parallel)**
- `SEOAgent` (GPT-4o-mini) reads `draft`
- **Tool Call**: `keyword_analyze` on extracted keywords ["quantum error correction", "surface code", "logical qubit"]
- **LLM Invocation**: Generates `SEOData` with title tag, meta description, keyword density analysis
- Node returns `{"seo_metadata": {...}}`

**Step 7b: Fact Checker Agent (Parallel)**
- `FactCheckerAgent` (GPT-4o-mini) reads `draft` and `research_findings`
- **LLM Invocation**: Extracts factual claims from draft
- **Tool Calls**: For each unverified claim, `web_search` invoked for corroboration
- Results validated against original sources
- Returns `fact_check_results` with confidence scores
- One claim flagged: "Surface codes require 1000 physical qubits per logical qubit" → corrected to "thousands" with source reference
- Node returns `{"fact_check_results": [...]}`

**Step 8: Quality Gate (Join)**
- Graph waits for both parallel branches (LangGraph handles synchronization)
- `quality_gate` node executes:
  - Calculates composite score: 0.4 * readability + 0.3 * factual_accuracy + 0.3 * seo_compliance
  - Score: 0.78 (below 0.85 threshold)
  - `revision_count` incremented to 1
  - Generates `revision_feedback`: "Strengthen the transition between sections 2 and 3; clarify the threshold theorem explanation"
- Conditional edge routes to `revise` → back to `write`

**Step 9: Revision Loop**
- `WriterAgent` re-executes with `revision_feedback` and `revision_count=1`
- Regenerates affected sections with feedback incorporated
- `EditorAgent` re-edits
- SEO and FactChecker re-run (results may differ slightly)
- Quality Gate: Score 0.88 (above threshold)
- Conditional edge routes to `accept`

**Step 10: Image Generation (Conditional)**
- `should_generate_images` returns `"generate"` (user requested images)
- `ImageAgent` reads `outline` headings
- **Tool Calls**: `image_generate` invoked for cover image + 2 section illustrations
- DALL-E 3 generates images; URLs stored in `state.cover_image_url` and `state.section_images`

**Step 11: Final Assembly**
- Graph reaches `END`
- `final_blog` assembled from `draft` + `seo_metadata` + image markdown embeds
- `DocumentRepository.create()` inserts final version into `documents` table
- `TaskRepository.update_status()` sets `status="complete"`, `quality_score=0.88`
- Episodic memory records all turns to `conversation_turns`

**Step 12: Response Delivery**
- FastAPI returns JSON response with `task_id`, `thread_id`, `final_blog` markdown, `quality_score`, `reading_time`
- Client can poll `GET /api/v1/blog/{thread_id}/status` or receive webhook

---

## (10) Architectural Principles & Technology Reference

### Non-Negotiable Design Principles

| Principle | Rule | Rationale |
|-----------|------|-----------|
| **State Immutability** | Agents never mutate state in-place; they return delta updates | Enables deterministic replay, time-travel debugging, and fault recovery |
| **Tool Idempotency** | All tools must be safely retryable without side effects | Critical for resilience in LLM-driven loops |
| **Fail-Fast Validation** | Pydantic schemas at every boundary (API, state, tool I/O) | Prevents garbage-in-garbage-out in agent chains |
| **Observability by Design** | Every agent run, tool call, and LLM invocation is logged with correlation IDs | Essential for debugging non-deterministic LLM behavior |
| **Provider Agnosticism** | LLM provider swappable via configuration without code changes | Avoids vendor lock-in; enables cost optimization |
| **Memory Tiering** | Hot data in Redis, warm in PostgreSQL, cold in S3/MinIO | Cost-performance optimization for different access patterns |
| **Graceful Degradation** | If image generation fails, blog proceeds without images; if SEO fails, content still delivered | Partial success > total failure in production |
| **Human-in-the-Loop** | Quality gates and approval checkpoints at critical transitions | Safety for high-stakes content; enables RLHF data collection |

### Complete Technology Stack

| Component | Technology | Version Target | Role | Justification |
|-----------|-----------|----------------|------|---------------|
| **Language** | Python | 3.11+ | Primary runtime | Native async/await, rich ML ecosystem |
| **Orchestration** | LangGraph | ^0.2.0 | Workflow engine | First-class stateful cycles, checkpointing, deterministic execution |
| **LLM Framework** | LangChain | ^0.3.0 | LLM abstraction | Standardized interfaces for models, prompts, and output parsing |
| **LLM Providers** | OpenAI SDK | ^1.0.0 | GPT-4o/4o-mini | Best-in-class structured output, tool use |
| | Anthropic SDK | ^0.34.0 | Claude 3.5 Sonnet/Haiku | Superior long-context writing quality |
| | Ollama | ^0.3.0 | Local Llama 3.1 | Cost-free fallback, data privacy for sensitive drafts |
| **Web Search** | Tavily Python | ^0.5.0 | Research tool | Optimized for LLMs; returns cleaned content, not just links |
| **Vector Store** | ChromaDB | ^0.5.0 | Long-term memory | Embedded-friendly, metadata filtering, no external dependency in dev |
| **Database** | PostgreSQL | 16+ | Structured persistence | ACID compliance, JSONB for flexible state, proven at scale |
| **ORM** | SQLAlchemy | ^2.0.0 | Database abstraction | Async support, type-safe queries, Alembic integration |
| **Migrations** | Alembic | ^1.13.0 | Schema versioning | Industry standard for SQLAlchemy |
| **Cache / STM** | Redis | 7.2+ | Short-term memory, rate limiting | Sub-millisecond latency, native TTL, pub/sub for real-time |
| **Structured Output** | Pydantic | ^2.9.0 | Schema validation | Runtime validation, JSON schema generation, type safety |
| **Settings** | Pydantic-Settings | ^2.5.0 | Configuration management | Environment-based config with validation |
| **Prompts** | Jinja2 | ^3.1.0 | Template engine | Logic-capable templating, sandboxed execution |
| **API Layer** | FastAPI | ^0.115.0 | REST API | Async-native, automatic OpenAPI docs, dependency injection |
| **CLI** | Typer | ^0.12.0 | Command-line interface | Type-hint driven CLI, minimal boilerplate |
| **Sandbox** | E2B | ^1.0.0 | Code execution | Secure, ephemeral sandboxes for data visualization generation |
| **Image Gen** | DALL-E 3 (via OpenAI) | API v1 | Visual content | High-quality illustration generation |
| **Token Counting** | tiktoken | ^0.7.0 | OpenAI token counting | Exact token counts for budget management |
| **Retry Logic** | tenacity | ^9.0.0 | Resilience | Decorator-based retry with exponential backoff |
| **Logging** | structlog | ^24.4.0 | Structured logging | JSON logs, context binding, performance tracking |
| **Testing** | pytest | ^8.3.0 | Test framework | Async test support, fixtures, parametrize |
| **Containerization** | Docker | 25.0+ | Environment consistency | Reproducible dev/prod parity |
| **Object Storage** | MinIO | RELEASE.2024 | Artifact storage | S3-compatible, self-hosted image/document storage |
| **Process Management** | Supervisor / systemd | N/A | Production process control | Ensures uptime, auto-restart on failure |

### Version Lock Rationale

- **LangGraph 0.2+**: Required for `StateGraph` with Pydantic state schemas and `PostgresSaver` checkpointing
- **Python 3.11+**: Required for `typing.Self`, improved `asyncio.TaskGroup`, and 10-15% performance gains over 3.10
- **PostgreSQL 16+**: Required for `pgvector` extension compatibility (if migrating to pgvector later) and improved JSONB performance
- **Pydantic 2.x**: Required for `model_json_schema()` and `TypeAdapter` used in structured output enforcement

---

This architecture provides a **deterministic, observable, and resilient** foundation for production blog generation. The LangGraph orchestrator ensures that complex multi-agent workflows with loops are manageable and debuggable, while the layered design allows individual components (LLM providers, memory backends, tools) to be swapped without cascading changes.