# Ollama

- Local LLM backend for Mistral 7B.
- Expose Ollama API at `http://localhost:11434`.
- Use `langchain_community.llms.Ollama` for prompt generation and structured output.
- Keep model settings low-latency: `temperature=0.3` and local quantization if available.
