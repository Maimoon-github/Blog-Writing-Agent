"""Configuration settings for BWAgent."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if available.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")

SD_MODEL_ID = os.getenv("SD_MODEL_ID", "CompVis/stable-diffusion-v1-4")
SD_DEVICE = os.getenv("SD_DEVICE", "cuda")
SD_IMAGE_WIDTH = int(os.getenv("SD_IMAGE_WIDTH", "512"))
SD_IMAGE_HEIGHT = int(os.getenv("SD_IMAGE_HEIGHT", "512"))
SD_INFERENCE_STEPS = int(os.getenv("SD_INFERENCE_STEPS", "30"))
SD_SAFETY_CHECKER = os.getenv("SD_SAFETY_CHECKER", "true").strip().lower() in ("1", "true", "yes")

CHROMA_PERSIST_DIR = BASE_DIR / Path(os.getenv("CHROMA_PERSIST_DIR", "./outputs/chroma"))
CACHE_DIR = BASE_DIR / Path(os.getenv("CACHE_DIR", "./outputs/cache"))

OUTPUT_DIR = BASE_DIR / Path(os.getenv("OUTPUT_DIR", "./outputs"))
IMAGES_DIR = OUTPUT_DIR / "images"
BLOGS_DIR = OUTPUT_DIR / "blogs"
LOGS_DIR = OUTPUT_DIR / "logs"

for directory in [CHROMA_PERSIST_DIR, CACHE_DIR, IMAGES_DIR, BLOGS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
