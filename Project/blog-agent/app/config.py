"""Centralized configuration for the blog agent.

Loads environment variables from a .env file and provides typed constants.
All output directories are created at import time.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file (if present).
# Does not override already-set environment variables.
load_dotenv()

# ---------------------------------------------------------------------------
# Basic Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

SEARXNG_BASE_URL: str = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")

# Stable Diffusion
SD_MODEL_ID: str = os.getenv("SD_MODEL_ID", "CompVis/stable-diffusion-v1-4")
SD_DEVICE: str = os.getenv("SD_DEVICE", "cuda")  # "cuda", "cpu", or "mps"
SD_IMAGE_WIDTH: int = int(os.getenv("SD_IMAGE_WIDTH", "512"))
SD_IMAGE_HEIGHT: int = int(os.getenv("SD_IMAGE_HEIGHT", "512"))
SD_INFERENCE_STEPS: int = int(os.getenv("SD_INFERENCE_STEPS", "30"))

# ChromaDB persistence
CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./outputs/chroma"))

# Cache settings
CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./outputs/cache"))
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 hours

# ---------------------------------------------------------------------------
# Output Directory Structure
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
IMAGES_DIR: Path = OUTPUT_DIR / "images"
BLOGS_DIR: Path = OUTPUT_DIR / "blogs"
LOGS_DIR: Path = OUTPUT_DIR / "logs"

# ---------------------------------------------------------------------------
# Ensure Required Directories Exist
# ---------------------------------------------------------------------------
for directory in [CHROMA_PERSIST_DIR, CACHE_DIR, IMAGES_DIR, BLOGS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)