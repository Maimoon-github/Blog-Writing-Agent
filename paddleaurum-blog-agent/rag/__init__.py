# rag/__init__.py
"""rag/ — Knowledge base setup and ChromaDB ingestion for paddleaurum.com.

Public interface
----------------
    from rag import get_embedding_function, ingest
    from rag import COLLECTION_PICKLEBALL_RULES, COLLECTION_COACHING_MATERIALS,
                 COLLECTION_SEO_GUIDELINES, COLLECTION_PUBLISHED_ARTICLES,
                 COLLECTION_KEYWORD_HISTORY

    ingest(persist_dir="./chromadb_store", reset=False)
"""

from rag.embeddings import get_embedding_function
from rag.ingest import main as ingest

# Expose collection name constants for retrieval-time usage
from rag.collections.pickleball_rules import COLLECTION_NAME as COLLECTION_PICKLEBALL_RULES
from rag.collections.coaching_materials import COLLECTION_NAME as COLLECTION_COACHING_MATERIALS
from rag.collections.seo_guidelines import COLLECTION_NAME as COLLECTION_SEO_GUIDELINES
from rag.collections.published_articles import COLLECTION_NAME as COLLECTION_PUBLISHED_ARTICLES
from rag.collections.keyword_history import COLLECTION_NAME as COLLECTION_KEYWORD_HISTORY

__all__ = [
    "get_embedding_function",
    "ingest",
    "COLLECTION_PICKLEBALL_RULES",
    "COLLECTION_COACHING_MATERIALS",
    "COLLECTION_SEO_GUIDELINES",
    "COLLECTION_PUBLISHED_ARTICLES",
    "COLLECTION_KEYWORD_HISTORY",
]