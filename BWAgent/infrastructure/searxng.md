# SearxNG

- Primary search layer for local free web research.
- Run SearxNG in Docker on port `8080`.
- Configure LangChain `SearxSearchWrapper` to use `http://localhost:8080`.
- Fallback: DuckDuckGo when SearxNG is unavailable.
- Use results for Researcher workers and citation extraction.
