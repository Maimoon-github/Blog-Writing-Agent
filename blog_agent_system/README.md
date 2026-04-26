# Blog Agent System

**Agentic AI Blog Writing System** — A multi-agent pipeline that automates the entire blog creation workflow using LangGraph orchestration, specialized AI agents, and a tiered memory architecture.

## Architecture

```
Research → Outline → Write → Edit → [SEO + Fact-Check] → Quality Gate → Publish
                                ↑                              │
                                └──── Revision (if < 0.85) ────┘
```

**7 Specialized Agents** orchestrated through LangGraph with state-mediated communication:

| Agent | Model | Purpose |
|-------|-------|---------|
| Research | GPT-4o-mini | Web search, source gathering |
| Outline | Claude Haiku 4.5 | Structural design |
| Writer | Claude Sonnet 4.6 | Prose generation |
| Editor | Claude Sonnet 4.6 | Grammar, clarity, style |
| SEO | GPT-4o-mini | Keyword optimization |
| Fact Checker | GPT-4o-mini | Claim verification |
| Image | DALL-E 3 | Cover art, illustrations |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API keys: OpenAI, Anthropic, Tavily

### Setup

```bash
# 1. Clone and enter
cd blog_agent_system

# 2. Start infrastructure
cp .env.example .env
# Edit .env with your API keys
docker compose up -d

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Check health
make health

# 5. Generate a blog
make cli-generate TOPIC="Quantum Error Correction for Software Engineers"
```

### API Server

```bash
make serve
# Open http://localhost:8080/docs for Swagger UI
```

```bash
curl -X POST http://localhost:8080/api/v1/blog/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "Introduction to Rust for Python Developers", "word_count_target": 2000}'
```

## Project Structure

```
src/blog_agent_system/
├── main.py              # FastAPI app factory
├── cli.py               # Typer CLI
├── config/              # Settings, LLM config, prompts
├── core/                # LangGraph state, graph, orchestrator
├── agents/              # 7 specialized agents
├── tools/               # Web search, RAG, file I/O
├── llm/                 # Provider abstraction (OpenAI, Anthropic, Ollama)
├── memory/              # Short-term (Redis), Long-term (ChromaDB), Episodic (PostgreSQL)
├── persistence/         # Database, ORM models, repositories
├── models/              # Pydantic schemas
├── api/                 # FastAPI routes
└── utils/               # Logging, exceptions, validators
```

## Development

```bash
make dev          # Install dev dependencies
make lint         # Run ruff linter
make format       # Auto-format code
make test         # Run all tests
make test-cov     # Tests with coverage
```

## Architecture Decisions

See [`docs/`](docs/) for Architecture Decision Records:

- [ADR-001: Orchestrator Choice — LangGraph](docs/adr-001-orchestrator-choice.md)
- [ADR-002: Memory Strategy — Tiered Architecture](docs/adr-002-memory-strategy.md)
- [ADR-003: LLM Abstraction — Provider-Agnostic Layer](docs/adr-003-llm-abstraction.md)

## License

MIT
