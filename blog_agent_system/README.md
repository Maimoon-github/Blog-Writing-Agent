# Blog Agent System

**Agentic AI Blog Writing System** — Fully local multi-agent pipeline using **Ollama** (Mistral + Gemma2) + LangGraph.

## Architecture
Research → Outline → Write → Edit → [SEO + Fact-Check] → Quality Gate → (optional Image)
↑
└──── Revision loop (max 3) ────┘

## Models

- Mistral 7B: Research, Outline, SEO, Fact-Check
- Gemma 2 9B: Writer, Editor
- Qwen 3-VL 8B: Image Generator

## Getting Started

**Prerequisites:**
- Python 3.11+
- Docker & Docker Compose

**Setup:**
```bash
# Clone the repository
git clone https://github.com/yourusername/blog-agent-system.git
cd blog-agent-system

# Install dependencies (Poetry recommended but pip also works)
pip install -r requirements.txt

# Start infrastructure
docker compose up -d

# Pull required models (optional — will auto-pull on first run)
docker compose exec ollama ollama pull mistral:7b
docker compose exec ollama ollama pull gemma2:9b
docker compose exec ollama ollama pull qwen3-vl:8b
```

## Usage

### CLI (generate a blog)
```bash
# Generate a blog post
python -m blog_agent_system.cli generate "Your blog topic here" --word-count 1500
```

### API Server
```bash
# Start API server
uvicorn blog_agent_system.main:app --reload --host [IP_ADDRESS] --port 8000
```

**Generate via API:**
```bash
curl -X POST "http://localhost:8000/api/v1/blog/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Introduction to LangGraph",
    "word_count_target": 1500
  }'
```



**All agents run 100% locally on Ollama** — no cloud API keys required.

## Quick Start

### 1. Start Infrastructure

```bash
cp .env.example .env
docker compose up -d --pull always
make pull-ollama          # Pull Mistral + Gemma2