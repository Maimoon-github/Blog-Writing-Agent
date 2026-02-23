"""tools/rag_retrieval.py

ChromaDB similarity search wrapper for the paddleaurum.com pipeline.

All five knowledge-base collections are named as module constants so agents
can reference them without hardcoding strings.  The vector store is initialised
once per (collection_name, persist_dir) pair and cached for the process lifetime.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import CHROMADB_PERSIST_DIR, EMBEDDING_MODEL

# ── Known collection names ────────────────────────────────────────────────────
# Import these in agents/tools rather than hardcoding strings.

COLLECTION_PICKLEBALL_RULES   = "pickleball_rules"       # USAPA 2024/2025 rulebook
COLLECTION_COACHING_MATERIALS = "coaching_materials"     # drills, strategies, technique guides
COLLECTION_SEO_GUIDELINES     = "seo_guidelines"         # SEO checklist, best practices
COLLECTION_PUBLISHED_ARTICLES = "published_articles"     # prevents topical duplication
COLLECTION_KEYWORD_HISTORY    = "keyword_history"        # used keywords + performance data

# ── Internal store cache ──────────────────────────────────────────────────────
# Key: (collection_name, persist_dir) — avoids re-initialising when the same
# collection is queried by multiple agents in one process.
_cache: Dict[Tuple[str, str], Chroma] = {}


def _get_store(collection_name: str, persist_dir: str) -> Chroma:
    """Return a cached Chroma instance for the given collection and directory."""
    key = (collection_name, persist_dir)
    if key not in _cache:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _cache[key] = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    return _cache[key]


def rag_retrieve(
    query: str,
    collection_name: str,
    k: int = 5,
    metadata_filter: Optional[dict] = None,
    persist_dir: str = CHROMADB_PERSIST_DIR,
) -> List[str]:
    """
    Return the top-k document chunks most similar to `query`.

    Parameters
    ----------
    query           : Natural-language search query.
    collection_name : ChromaDB collection to search (use module constants above).
    k               : Number of results to return.
    metadata_filter : Optional ChromaDB metadata filter dict.
    persist_dir     : Path to the ChromaDB persist directory; defaults to
                      CHROMADB_PERSIST_DIR from config/settings.py.

    Returns
    -------
    List[str]
        Page content strings in descending similarity order.
        Returns an empty list on any retrieval failure so callers can
        degrade gracefully without crashing the pipeline.
    """
    if not query or not query.strip():
        return []

    try:
        store = _get_store(collection_name, persist_dir)
        docs = store.similarity_search(query, k=k, filter=metadata_filter)
        return [doc.page_content for doc in docs]
    except Exception as exc:
        # Non-fatal: log and return empty so the calling node can continue.
        import logging
        logging.getLogger(__name__).warning(
            "RAG retrieval failed (collection=%s): %s", collection_name, exc
        )
        return []