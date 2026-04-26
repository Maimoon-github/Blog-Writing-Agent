"""
ChromaDB vector store client wrapper.
"""

import chromadb

from blog_agent_system.config.settings import settings
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB client wrapper with pre-configured collections."""

    def __init__(self):
        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

        self.collections = {
            "blog_knowledge": self.client.get_or_create_collection("blog_knowledge"),
            "style_guides": self.client.get_or_create_collection("style_guides"),
            "previous_blogs": self.client.get_or_create_collection("previous_blogs"),
        }

        logger.info("vector_store.initialized", collections=list(self.collections.keys()))

    def get_collection(self, name: str):
        """Return a collection by name."""
        return self.collections.get(name)