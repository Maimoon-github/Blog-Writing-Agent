# Setup

1. Install Python dependencies from `requirements.txt`.
2. Configure local services in `env.example` and `config/config.yaml`.
3. Run Ollama Mistral locally and verify `http://localhost:11434`.
4. Start SearxNG in Docker on port 8080 or use DuckDuckGo fallback.
5. Ensure `diffusers` and `transformers` can access Stable Diffusion v1.4.
6. Use `ui/app.py` as the main Streamlit entrypoint when ready.
