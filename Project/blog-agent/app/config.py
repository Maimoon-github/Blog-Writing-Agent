import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ollama
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

# SearXNG
SEARXNG_BASE_URL: str = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")

# Stable Diffusion
SD_MODEL_ID: str = os.getenv("SD_MODEL_ID", "CompVis/stable-diffusion-v1-4")
SD_DEVICE: str = os.getenv("SD_DEVICE", "cuda")
SD_IMAGE_WIDTH: int = int(os.getenv("SD_IMAGE_WIDTH", "512"))
SD_IMAGE_HEIGHT: int = int(os.getenv("SD_IMAGE_HEIGHT", "512"))
SD_INFERENCE_STEPS: int = int(os.getenv("SD_INFERENCE_STEPS", "30"))

# ChromaDB persistence directory
CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./memory/chroma_data"))

# Cache directory
CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./memory/cache"))
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

# Output directories
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
IMAGES_DIR: Path = OUTPUT_DIR / "images"
BLOGS_DIR: Path = OUTPUT_DIR / "blogs"
LOGS_DIR: Path = OUTPUT_DIR / "logs"

# Create all required directories
for directory in [
    OUTPUT_DIR,
    CHROMA_PERSIST_DIR,
    CACHE_DIR,
    IMAGES_DIR,
    BLOGS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)