"""
Single source of truth for all configuration constants in the
Autonomous Blog Generation Agent (8-agent parallel pipeline).

Never hardcode values in agents/, tools/, graph.py, streamlit_app.py
or any other file. Import from here to keep the entire system tunable.

References:
- roadmap.html → Phase 2 Step 2 + config.py file card
- idea.md → full local stack (Ollama + SearxNG + SD v1.4 + ChromaDB)
- CrewAI v1.14.2 compatibility requirements
"""

# === LLM / Ollama Configuration ===
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "llama3.2:3b"  # NOT "mistral:7b"       # or "llama3.2:3b" if it fits your GPU
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 2048

# === Search Layer ===
SEARXNG_HOST: str = "http://localhost:8080"

# === Image Generation (Stable Diffusion v1.4 via diffusers) ===
SD_MODEL_ID: str = "CompVis/stable-diffusion-v1-4"
USE_GPU: bool = True                     # Set False for CPU-only mode

# Note: image_gen.py reads USE_GPU at import time and adjusts these
IMG_SIZE: tuple[int, int] = (512, 512)   # GPU default → (384, 384) on CPU
IMG_STEPS: int = 50                      # GPU default → 15 on CPU

# === Output Paths & Persistence (all gitignored) ===
OUTPUT_BLOGS: str = "outputs/blogs"
OUTPUT_IMAGES: str = "outputs/images"
CHROMA_DB_DIR: str = "chroma_db"

# === Caching ===
CACHE_DIR: str = ".cache"
CACHE_TTL_SEC: int = 86400               # 24 hours

# Explicit export list for clean imports
__all__ = [
    "OLLAMA_BASE_URL", "OLLAMA_MODEL", "TEMPERATURE", "MAX_TOKENS",
    "SEARXNG_HOST",
    "SD_MODEL_ID", "USE_GPU", "IMG_SIZE", "IMG_STEPS",
    "OUTPUT_BLOGS", "OUTPUT_IMAGES", "CHROMA_DB_DIR",
    "CACHE_DIR", "CACHE_TTL_SEC",
]

# Quick override notes:
# • CPU-only hardware → set USE_GPU = False
# • Different model     → change OLLAMA_MODEL (e.g. "ollama/mistral:7b-instruct")
# All agents/tools import via: from config import OLLAMA_MODEL, SEARXNG_HOST, ...