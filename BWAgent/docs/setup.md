# Setup

1. Copy `env.example` to `.env` and edit local service endpoints.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Ollama locally and verify `http://localhost:11434`.
4. Start SearxNG locally with Docker Compose:
   ```bash
   docker compose up -d
   ```
5. Confirm SearxNG is reachable at `http://localhost:8080`.
6. Make sure your machine can load the Stable Diffusion model, or update `SD_DEVICE` to `cpu`.
7. Run the Streamlit UI with:
   ```bash
   streamlit run ui/app.py
   ```
