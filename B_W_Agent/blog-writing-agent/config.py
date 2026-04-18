# config.py
OLLAMA_MODEL = "mistral:7b"          # or "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.3
MAX_TOKENS = 2048

# For image generation fallback
USE_GPU = True  # Set False if no CUDA