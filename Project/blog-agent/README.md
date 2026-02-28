# Blog Agent — Autonomous AI Blog Generation System

[![LangGraph](https://img.shields.io/badge/🤖%20LangGraph-Framework-blue)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/🦙%20Ollama-Local%20LLMs-green)](https://ollama.ai/)
[![Streamlit](https://img.shields.io/badge/📊%20Streamlit-UI-red)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/📇%20ChromaDB-Vector%20Store-yellow)](https://www.trychroma.com/)
[![Stable Diffusion](https://img.shields.io/badge/🎨%20Stable%20Diffusion-Image%20Gen-orange)](https://huggingface.co/CompVis/stable-diffusion-v1-4)
[![Python 3.11](https://img.shields.io/badge/🐍%20Python-3.11+-blue)](https://python.org/)

An autonomous multi‑agent system that researches, writes, and illustrates complete blog posts using LangGraph, local LLMs (Ollama), and Stable Diffusion. All agents run locally – no cloud API dependencies.

---

## Architecture Overview

```mermaid
flowchart TD
    User[User Input] --> Router[Router Agent]
    Router --> Planner[Planner Agent]
    Planner --> ResearchDispatch{Research Required?}
    ResearchDispatch -- Yes --> ResearchWorkers[Researcher Agent (per section)] 
    ResearchDispatch -- No --> WriterWorkers[Writer Agent (per section)]
    ResearchWorkers --> ResearchJoin{research_complete}
    ResearchJoin --> WriterWorkers
    WriterWorkers --> WriterJoin{writer_complete}
    WriterJoin --> ImageDispatch[Image Agent (feature + per section)]
    ImageDispatch --> ImageJoin{image_complete}
    ImageJoin --> Citation[Citation Manager]
    Citation --> Reducer[Reducer Agent]
    Reducer --> Output[(Markdown + HTML)]
    
    style Router fill:#f9f,stroke:#333
    style Planner fill:#bbf,stroke:#333
    style ResearchWorkers fill:#afa,stroke:#333
    style WriterWorkers fill:#ffa,stroke:#333
    style ImageDispatch fill:#faa,stroke:#333
    style Citation fill:#ccf,stroke:#333
    style Reducer fill:#cfc,stroke:#333
```

**Key:**  
- **Router**: Analyzes topic and determines if research is needed.  
- **Planner**: Generates a structured blog plan (sections, target word counts, image prompts).  
- **Researcher** (parallel per section): Searches the web, fetches pages, and summarises content.  
- **Writer** (parallel per section): Writes Markdown drafts, inserting citation markers.  
- **Image Agent** (parallel): Generates images using Stable Diffusion.  
- **Citation Manager**: Maps citation keys to source URLs.  
- **Reducer**: Assembles the final blog (Markdown + HTML) and saves files.

---

## Prerequisites

- **Docker** – for SearxNG (search engine)  
- **Ollama** – local LLM server (pull `mistral` or any supported model)  
- **Python 3.11+**  
- **8GB+ VRAM** (e.g., RTX 3060+) for GPU acceleration (CPU fallback works but is slow)  
- **10GB free disk** – for Stable Diffusion model (~4GB) and vector store data  

---

## Quick Start

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/blog-agent.git
   cd blog-agent
   ```

2. **Configure environment**  
   ```bash
   cp .env.example .env
   # Edit .env if needed (defaults work out‑of‑box)
   ```

3. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull the default LLM model**  
   ```bash
   ollama pull mistral
   ```

5. **Start the SearxNG search engine**  
   ```bash
   docker compose -f docker/docker-compose.yml up -d searxng
   ```

6. **Download Stable Diffusion model (first run only)**  
   ```bash
   python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')"
   ```

7. **Launch the Streamlit UI**  
   ```bash
   streamlit run app/ui.py
   ```

   Open `http://localhost:8501` in your browser.

---

## Environment Variables

| Variable                | Default                   | Description |
|-------------------------|---------------------------|-------------|
| `OLLAMA_BASE_URL`       | `http://localhost:11434`  | Ollama server URL |
| `OLLAMA_MODEL`          | `mistral`                 | LLM model name |
| `SEARXNG_BASE_URL`      | `http://localhost:8080`   | SearxNG instance URL |
| `SD_MODEL_ID`           | `CompVis/stable-diffusion-v1-4` | Hugging Face model ID for image generation |
| `SD_DEVICE`             | `cuda`                    | Device for diffusion (`cuda`, `cpu`, `mps`) |
| `SD_IMAGE_WIDTH`        | `512`                     | Generated image width |
| `SD_IMAGE_HEIGHT`       | `512`                     | Generated image height |
| `SD_INFERENCE_STEPS`    | `30`                      | Number of denoising steps |
| `CHROMA_PERSIST_DIR`    | `./outputs/chroma`        | Directory for ChromaDB vector store |
| `CACHE_DIR`             | `./outputs/cache`         | Directory for disk cache (search results) |
| `CACHE_TTL_SECONDS`     | `86400`                   | Cache TTL (24h) |
| `OUTPUT_DIR`            | `./outputs`               | Root output folder (blogs, images, logs) |

---

## Agent Roster

| Agent             | Role                                                                 | Tools / Dependencies |
|-------------------|----------------------------------------------------------------------|----------------------|
| **Router**        | Classifies topic as requiring research; performs safety check        | `ChatOllama`         |
| **Planner**       | Generates a structured blog plan (title, sections, prompts)         | `ChatOllama` (structured output) |
| **Researcher**    | Performs web search, fetches content, and writes research summary   | `search()`, `fetch_page_content()`, `cache`, `ChromaDB`, `ChatOllama` |
| **Writer**        | Writes a Markdown section draft from research & description         | `ChatOllama` |
| **Image Agent**   | Generates images using Stable Diffusion                              | `generate_image()` |
| **Citation Manager** | Maps citation markers (`[SOURCE_N]`) to source URLs               | `re`, state |
| **Reducer**       | Assembles final Markdown/HTML, saves files                           | `markdown` library, `pathlib` |

---

## Running Tests

- **Unit tests** (fast, mock‑based)  
  ```bash
  pytest tests/ -m "not slow"
  ```

- **Integration tests** (require Ollama and SearxNG running)  
  ```bash
  pytest tests/ -m slow
  ```

---

## Limitations

- **CPU‑only image generation** is very slow (20+ minutes per image). A GPU with ≥8GB VRAM is strongly recommended.
- **Mistral’s structured output** may rarely fail to produce valid JSON; an automatic fallback to regex parsing handles most cases.
- **SearxNG** can be rate‑limited by some engines; a fallback to DuckDuckGo is built in.
- The system is designed for English‑language content only.

---

## License

MIT License – see [LICENSE](LICENSE) file for details.