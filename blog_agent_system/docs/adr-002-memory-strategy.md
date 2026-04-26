# ADR-002: Memory Strategy — Tiered Memory with Redis, ChromaDB, and PostgreSQL

**Status:** Accepted  
**Date:** 2026-04-26  
**Deciders:** Architecture Team  
**Affected:** `memory/`, `persistence/`, `core/state.py`

---

## Context

Agentic systems require multiple memory types operating at different timescales and access patterns:

- **Short-term:** Recent conversation turns within a single session (hot, ephemeral)
- **Long-term:** Domain knowledge accumulated across sessions (warm, semantic)
- **Episodic:** Historical conversation logs for audit and replay (cold, chronological)
- **Shared:** Real-time state synchronization between agents in an active workflow (transactional)

A single storage backend cannot optimally serve all four access patterns. We needed a strategy that balances latency, cost, query flexibility, and durability.

---

## Decision

We will implement a **four-tier memory architecture**:

| Tier | Technology | Purpose | Access Pattern |
|------|-----------|---------|----------------|
| **Short-Term** | Redis (List + TTL) | In-context window buffer | LIFO sliding window, sub-ms |
| **Long-Term** | ChromaDB (Vector) | Semantic knowledge retrieval | Cosine similarity, embedding-based |
| **Episodic** | PostgreSQL (JSONB) | Conversation history & audit | Time-range queries, thread-scoped |
| **Shared** | PostgreSQL (LangGraph Checkpoint) | Inter-agent state sync | Exact key lookup, ACID |

---

## Consequences

### Positive

- **Cost-performance optimization:** Hot data lives in Redis (memory), warm vectors in ChromaDB (disk + memory-mapped), and cold records in PostgreSQL (disk). This avoids over-provisioning expensive RAM for rarely accessed historical data.
- **Specialized query capabilities:** Vector search for long-term memory, relational queries for episodic history, and simple key-value operations for short-term caching—each backend optimized for its query pattern.
- **Durability where it matters:** Episodic and shared state are ACID-compliant via PostgreSQL, ensuring no data loss for business-critical audit trails and workflow checkpoints.
- **Independent scaling:** Redis can be scaled vertically for throughput, ChromaDB horizontally for vector volume, and PostgreSQL via read replicas for historical queries.

### Negative

- **Operational complexity:** Four storage systems increase deployment and monitoring surface area.
- **Consistency challenges:** Data may exist in multiple tiers simultaneously (e.g., a fact stored in short-term Redis and long-term ChromaDB). We accept eventual consistency between tiers; the shared state (LangGraph checkpoint) remains strongly consistent.
- **Data migration overhead:** Moving data between tiers (e.g., promoting short-term insights to long-term knowledge) requires explicit ETL logic in `memory/eviction.py`.

---

## Alternatives Considered

### Single Backend: PostgreSQL Only

**Why rejected:** While PostgreSQL with `pgvector` could theoretically handle all tiers, it fails on latency and cost for short-term memory (Redis is 10-100x faster for simple list operations) and vector search at scale (ChromaDB offers better embedding-specific optimizations and metadata filtering).

### Single Backend: Redis Only

**Why rejected:** Redis lacks durable persistence guarantees and vector search capabilities required for long-term semantic memory. Using Redis Streams for episodic history would be prohibitively expensive at scale.

### Single Backend: ChromaDB Only

**Why rejected:** ChromaDB is not a general-purpose database. It lacks ACID transactions, time-range query performance, and the relational integrity required for workflow checkpoints and audit trails.

### Redis + PostgreSQL (without ChromaDB)

**Why rejected:** Using PostgreSQL `pgvector` for all vector needs was evaluated. ChromaDB was chosen over `pgvector` because:
- **Developer experience:** ChromaDB's Python-native API reduces boilerplate for embedding management.
- **Metadata filtering:** ChromaDB offers more flexible filtering during vector search (critical for multi-tenant blog knowledge separation).
- **Decoupling:** Vector workloads are isolated from transactional database load, preventing resource contention.

*Note:* We retain the option to migrate from ChromaDB to `pgvector` in the future if operational consolidation becomes a priority. The `VectorStore` abstraction in `persistence/vector_store.py` encapsulates this dependency.

---

## Implementation Notes

### Short-Term Memory (`memory/short_term.py`)
- Redis List with `LPUSH` / `LTRIM` for sliding window
- TTL of 1 hour; max 20 messages per thread
- Used primarily for maintaining conversation context within a single agent turn

### Long-Term Memory (`memory/long_term.py`)
- ChromaDB collection per domain (`blog_knowledge`, `style_guides`, `previous_blogs`)
- OpenAI `text-embedding-3-small` for embeddings (cost-efficient, high quality)
- Append-only; no eviction policy. Knowledge grows monotonically.

### Episodic Memory (`memory/episodic.py`)
- PostgreSQL `conversation_turns` table with JSONB for flexible metadata
- Indexed by `(thread_id, timestamp DESC)` for fast thread reconstruction
- Soft archive after 90 days; hard delete after 1 year per GDPR considerations

### Shared State (`memory/shared_state.py`)
- LangGraph `PostgresSaver` checkpoint mechanism
- Every node return triggers an atomic write to PostgreSQL
- Enables time-travel debugging and crash recovery

### Eviction & Compaction (`memory/eviction.py`)
- Short-term: Automatic via Redis TTL
- Episodic: Batch archive job runs nightly
- Long-term: Manual review process; no automatic eviction to prevent knowledge loss

---

## References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Redis Data Types: Lists](https://redis.io/docs/data-types/lists/)
- [PostgreSQL JSONB Documentation](https://www.postgresql.org/docs/current/datatype-json.html)
- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Embedding Models: OpenAI text-embedding-3](https://platform.openai.com/docs/guides/embeddings)
