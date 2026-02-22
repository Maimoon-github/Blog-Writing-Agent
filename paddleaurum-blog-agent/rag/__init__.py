"""RAG knowledge base setup and ingestion utilities."""

from .embeddings import get_embedding_function
from .ingest import main as ingest

__all__ = [
    "get_embedding_function",
    "ingest",
]