"""
Long-term memory — ChromaDB vector store for semantic knowledge retrieval.
"""

import chromadb
from langchain_openai import OpenAIEmbeddings

from blog_agent_system.config.settings import settings
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class LongTermMemory:
    """Semantic knowledge retrieval using ChromaDB vector search."""

    def __init__(self, collection_name: str = "blog_knowledge"):
        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
        logger.info("long_term_memory.initialized", collection=collection_name)

    async def store(self, documents: list[str], metadata: list[dict], ids: list[str]) -> None:
        """Store documents with embeddings."""
        embeddings = await self.embeddings.aembed_documents(documents)
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids,
        )
        logger.info("long_term_memory.stored", count=len(documents))

    async def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[dict]:
        """Retrieve semantically similar documents."""
        query_embedding = await self.embeddings.aembed_query(query)
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]