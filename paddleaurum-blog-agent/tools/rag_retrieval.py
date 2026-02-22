"""ChromaDB similarity search wrapper for RAG retrieval."""

import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Default embedding model (all-MiniLM-L6-v2)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Persist directory from environment or default
DEFAULT_PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", "./chromadb_store")

# Global vector store cache to avoid re‑initializing on every call
_vector_store_cache = {}


def _get_vector_store(collection_name: str) -> Chroma:
    """Get or create a Chroma vector store for the given collection."""
    if collection_name in _vector_store_cache:
        return _vector_store_cache[collection_name]

    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=DEFAULT_PERSIST_DIR,
        embedding_function=embeddings,
    )
    _vector_store_cache[collection_name] = vectordb
    return vectordb


def rag_retrieve(
    query: str,
    collection_name: str,
    k: int = 5,
    filter: Optional[dict] = None,
) -> List[str]:
    """
    Retrieve top-k relevant document chunks from a ChromaDB collection.

    Args:
        query: The search query.
        collection_name: Name of the ChromaDB collection.
        k: Number of documents to return.
        filter: Optional metadata filter.

    Returns:
        List of document page contents (strings).
    """
    try:
        vectordb = _get_vector_store(collection_name)
        docs = vectordb.similarity_search(query, k=k, filter=filter)
        return [doc.page_content for doc in docs]
    except Exception as e:
        # Log error and return empty list – calling code should handle gracefully
        print(f"RAG retrieval error (collection={collection_name}): {e}")
        return []