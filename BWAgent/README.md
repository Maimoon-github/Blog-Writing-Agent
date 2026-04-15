# BWAgent

BWAgent is an autonomous blog generation system built for local open-source tooling.
It uses LangGraph orchestration, Ollama/Mistral for LLMs, SearxNG for search, Stable Diffusion for images,
and ChromaDB for research context persistence.

## Quick start

1. Copy `.env.example` to `.env`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Ollama locally and optionally run SearxNG on port 8080.
4. Launch the Streamlit UI:
   ```bash
   streamlit run ui/app.py
   ```

## Project structure

- `agents/` — Graph node implementations for routing, planning, research, writing, editing, images, citations, and assembly.
- `graph/` — LangGraph state definitions and workflow configuration.
- `prompts/` — prompt templates for each agent role.
- `tools/` — local search, web scraping, and Stable Diffusion helpers.
- `memory/` — cache and ChromaDB persistence wrappers.
- `ui/` — Streamlit application entrypoint.
- `tests/` — unit and integration test scaffolding.

## Output

Generated blogs, HTML, and images are written under `outputs/`.
