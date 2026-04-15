# Memory

- Short-term memory: `GraphState` carries live run state between nodes.
- Long-term memory: optional ChromaDB vector store for past blog plans and research summaries.
- Cache research results by query hash to reduce repeated web calls.
- Persist checkpoints for restart/resume support.
