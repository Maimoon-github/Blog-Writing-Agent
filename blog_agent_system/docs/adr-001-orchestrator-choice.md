# ADR-001: Orchestrator Choice — LangGraph over CrewAI and AutoGen

**Status:** Accepted  
**Date:** 2026-04-26  
**Deciders:** Architecture Team  
**Affected:** `core/graph.py`, `core/orchestrator.py`, `core/state.py`

---

## Context

The blog writing system requires a workflow engine capable of coordinating multiple specialized AI agents through a complex, stateful pipeline. The pipeline includes:

- Strict sequential dependencies (research → outline → draft → edit)
- Parallel execution branches (SEO optimization and fact-checking running simultaneously)
- Conditional loopback (quality gate routing back to writer for revisions)
- Persistent state across interruptions (crash recovery, human-in-the-loop pauses)

We evaluated three primary candidates: **LangGraph**, **CrewAI**, and **AutoGen**.

---

## Decision

We will use **LangGraph** as the core orchestration engine.

---

## Consequences

### Positive

- **First-class stateful cycles:** LangGraph treats cycles as native graph constructs rather than workarounds. This is critical for our revision loop (draft → edit → quality gate → [revise → draft]).
- **Fine-grained control:** We define every node, edge, and conditional branch explicitly. This determinism is essential for debugging non-deterministic LLM outputs.
- **Built-in persistence:** `PostgresSaver` and `SqliteSaver` provide checkpointing out-of-the-box. An agent crash mid-draft does not lose progress; resumption starts from the last checkpoint.
- **Pydantic state schemas:** LangGraph 0.2+ supports typed state objects with validation, preventing agents from corrupting shared state with malformed outputs.
- **LangChain ecosystem compatibility:** Reuses our existing investment in LangChain tools, prompts, and model integrations without adapter layers.

### Negative

- **Steeper learning curve:** Developers must understand graph theory concepts (nodes, edges, conditional routing) versus CrewAI's higher-level "crew" and "task" metaphors.
- **More boilerplate:** Defining a graph requires more code than CrewAI's declarative YAML-style configurations.
- **Tighter coupling to LangChain:** While beneficial for our stack, migrating away from LangChain in the future would require rewriting the orchestration layer.

---

## Alternatives Considered

### CrewAI

**Why rejected:** CrewAI excels at simple, sequential task delegation with role-based agents. However, it lacks native support for:
- **Stateful cycles:** Revision loops require manual state management outside the framework.
- **Parallel branch synchronization:** Fork-join patterns are not first-class; achieving synchronized parallel execution requires fragile workarounds.
- **Checkpointing:** No built-in persistence mechanism for long-running workflows.

CrewAI was deemed suitable for simpler, one-shot agent workflows but insufficient for our iterative, quality-gated blog generation pipeline.

### AutoGen (Microsoft)

**Why rejected:** AutoGen's conversational agent pattern is powerful for open-ended, multi-turn dialogues. However:
- **Non-determinism:** Agent conversations are dynamic and difficult to reproduce, complicating debugging and testing.
- **State opacity:** The group chat manager abstracts state transitions, making it hard to enforce strict business logic (e.g., "always fact-check before publishing").
- **Overhead:** AutoGen's conversational routing introduces latency and token costs unsuitable for a structured content generation pipeline.

AutoGen was retained as a future option for interactive brainstorming modes but rejected for the core production pipeline.

### Pure Asyncio / Custom Code

**Why rejected:** Building a custom state machine in raw Python would provide maximum flexibility but would require us to re-implement:
- Checkpointing and crash recovery
- Parallel execution with join synchronization
- State serialization and schema validation
- Observability hooks

The engineering cost outweighed the benefits given LangGraph's alignment with our requirements.

---

## Implementation Notes

- The `BlogState` Pydantic model in `core/state.py` is the single source of truth for all inter-agent communication.
- `core/graph.py` defines the `StateGraph` with explicit conditional edges for the quality gate loopback.
- `PostgresSaver` is configured in `core/checkpoint.py` for production-grade persistence.
- All agent nodes are idempotent where possible; retrying a node after a crash must not duplicate side effects (e.g., tool calls use idempotency keys).

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [LangGraph vs. CrewAI: Stateful Workflows](https://blog.langchain.dev/langgraph-vs-crewai/)
