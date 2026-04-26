"""
RAG retriever tool — ChromaDB vector search for context assembly.
"""

from pydantic import BaseModel, Field

from blog_agent_system.tools.registry import tool
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class RAGRetrieveInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: str = Field(default="blog_knowledge")


@tool(
    name="rag_retrieve",
    description="Retrieve relevant content from the knowledge base via semantic search",
    args_schema=RAGRetrieveInput,
)
async def rag_retrieve(query: str, top_k: int, collection: str) -> list:
    """Perform semantic search against ChromaDB."""
    try:
        from blog_agent_system.memory.long_term import LongTermMemory

        memory = LongTermMemory(collection_name=collection)
        results = await memory.retrieve(query=query, top_k=top_k)
        logger.info("rag_retrieve.complete", result_count=len(results))
        return results
    except Exception as e:
        logger.warning("rag_retrieve.error", error=str(e))
        return []
