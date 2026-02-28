# import json
# from typing import List

# import chromadb
# import structlog

# from app.config import CHROMA_PERSIST_DIR


# class ChromaStore:
#     def __init__(
#         self,
#         persist_directory: str = str(CHROMA_PERSIST_DIR),
#         collection_name: str = "blog_research",
#     ):
#         self.client = chromadb.PersistentClient(path=persist_directory)
#         self.collection = self.client.get_or_create_collection(name=collection_name)
#         self.collection_name = collection_name
#         self.logger = structlog.get_logger(__name__)

#     def add_research(
#         self,
#         section_id: str,
#         query: str,
#         summary: str,
#         source_urls: List[str],
#     ) -> None:
#         """Add a research summary to the ChromaDB collection."""
#         self.collection.add(
#             documents=[summary],
#             metadatas=[
#                 {
#                     "section_id": section_id,
#                     "query": query,
#                     "source_urls_json": json.dumps(source_urls),
#                 }
#             ],
#             ids=[f"research_{section_id}"],
#         )
#         self.logger.info(
#             "research_added",
#             section_id=section_id,
#             query=query,
#             source_url_count=len(source_urls),
#         )

#     def search_similar(self, query: str, n_results: int = 3) -> List[dict]:
#         """Search for similar research summaries by query text."""
#         try:
#             results = self.collection.query(
#                 query_texts=[query],
#                 n_results=n_results,
#             )

#             parsed: List[dict] = []

#             documents = results.get("documents") or []
#             metadatas = results.get("metadatas") or []

#             # Both are lists-of-lists (one inner list per query text)
#             for doc, meta in zip(
#                 documents[0] if documents else [],
#                 metadatas[0] if metadatas else [],
#             ):
#                 parsed.append(
#                     {
#                         "summary": doc,
#                         "section_id": meta.get("section_id", ""),
#                         "query": meta.get("query", ""),
#                         "source_urls": json.loads(meta.get("source_urls_json", "[]")),
#                     }
#                 )

#             return parsed

#         except Exception as exc:  # noqa: BLE001
#             self.logger.warning("search_similar_failed", query=query, error=str(exc))
#             return []

#     def clear(self) -> None:
#         """Delete and recreate the collection, removing all stored research."""
#         self.client.delete_collection(name=self.collection_name)
#         self.collection = self.client.get_or_create_collection(name=self.collection_name)
#         self.logger.info("chroma_collection_cleared", collection=self.collection_name)


# # Shared singleton — imported by other modules
# chroma_store = ChromaStore()
































"""ChromaDB wrapper for storing and retrieving research summaries with metadata.

This module provides a thread-safe abstraction over ChromaDB's persistent client,
specifically designed for the blog agent's research pipeline. It handles collection
management, metadata serialization, error resilience, and comprehensive logging.

Typical usage:
    from memory.chroma_store import chroma_store

    # Add research
    chroma_store.add_research(
        section_id="introduction",
        query="attention mechanisms",
        summary="Attention allows models to focus on relevant parts...",
        source_urls=["https://arxiv.org/abs/1706.03762"]
    )

    # Search similar
    results = chroma_store.search_similar("transformer attention", n_results=5)
"""

import json
import threading
from typing import Any, Dict, List, Optional

import chromadb
import structlog
from chromadb.errors import ChromaError

from app.config import CHROMA_PERSIST_DIR

# Module logger
logger = structlog.get_logger(__name__)


class ChromaStore:
    """Thread-safe ChromaDB client for research data.

    This class encapsulates a ChromaDB collection for storing research summaries
    with metadata (section_id, query, source_urls). All public methods are
    protected by a reentrant lock to ensure thread safety within a single process.

    Attributes:
        persist_directory: Directory where ChromaDB persists data.
        collection_name: Name of the ChromaDB collection.
        client: ChromaDB PersistentClient instance.
        collection: ChromaDB Collection instance (None if initialization failed).
        logger: Structured logger bound with context.
        _lock: Reentrant lock for thread-safe operations.
    """

    def __init__(
        self,
        persist_directory: str = str(CHROMA_PERSIST_DIR),
        collection_name: str = "blog_research",
    ) -> None:
        """Initialize the ChromaStore.

        Args:
            persist_directory: Directory for ChromaDB persistence.
            collection_name: Name of the collection to use/create.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = logger.bind(
            persist_directory=persist_directory,
            collection=collection_name,
        )
        self._lock = threading.RLock()
        self._init_client()

    def _init_client(self) -> None:
        """Initialize ChromaDB client and collection.

        Sets self.collection to None if initialization fails. This allows the
        store to operate in a degraded mode where all operations log errors
        and return gracefully.
        """
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            self.logger.info("chroma.initialized")
        except Exception as e:
            self.logger.error(
                "chroma.init_failed",
                error=str(e),
                exc_info=True,
            )
            self.collection = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_collection(self) -> bool:
        """Check if collection is available and log warning if not.

        Returns:
            True if collection is available, False otherwise.
        """
        if self.collection is None:
            self.logger.warning("chroma.collection_unavailable")
            return False
        return True

    def _serialize_metadata(
        self,
        section_id: str,
        query: str,
        source_urls: List[str],
    ) -> Dict[str, str]:
        """Prepare metadata for ChromaDB storage.

        Args:
            section_id: Blog section identifier.
            query: Search query used to generate this research.
            source_urls: List of source URLs.

        Returns:
            Dictionary with string values suitable for ChromaDB metadata.
        """
        return {
            "section_id": section_id,
            "query": query,
            "source_urls_json": json.dumps(source_urls, ensure_ascii=False),
        }

    def _parse_result(
        self,
        document: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse a single search result from ChromaDB.

        Args:
            document: Document text (summary).
            metadata: Metadata dictionary from ChromaDB.

        Returns:
            Parsed result dictionary with summary, section_id, query, source_urls.
        """
        try:
            source_urls = json.loads(metadata.get("source_urls_json", "[]"))
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.warning(
                "chroma.metadata_parse_failed",
                error=str(e),
                metadata=metadata,
            )
            source_urls = []

        return {
            "summary": document,
            "section_id": metadata.get("section_id", ""),
            "query": metadata.get("query", ""),
            "source_urls": source_urls,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_research(
        self,
        section_id: str,
        query: str,
        summary: str,
        source_urls: List[str],
    ) -> None:
        """Add or update a research summary in the collection.

        Uses upsert to make the operation idempotent – if the same section_id
        already exists, it will be overwritten.

        Args:
            section_id: Blog section identifier (used as document ID).
            query: Search query that produced this research.
            summary: Research summary text.
            source_urls: List of source URLs.
        """
        # Input validation
        if not section_id or not section_id.strip():
            self.logger.warning("add_research.invalid_section_id")
            return
        if not summary or not summary.strip():
            self.logger.warning("add_research.empty_summary", section_id=section_id)
            return

        with self._lock:
            if not self._check_collection():
                return

            doc_id = f"research_{section_id}"
            metadata = self._serialize_metadata(section_id, query, source_urls)

            try:
                self.collection.upsert(
                    ids=[doc_id],
                    documents=[summary],
                    metadatas=[metadata],
                )
                self.logger.info(
                    "add_research.success",
                    section_id=section_id,
                    query=query,
                    source_url_count=len(source_urls),
                )
            except ChromaError as e:
                self.logger.error(
                    "add_research.chroma_error",
                    section_id=section_id,
                    error=str(e),
                    exc_info=True,
                )
            except Exception as e:
                self.logger.error(
                    "add_research.unexpected_error",
                    section_id=section_id,
                    error=str(e),
                    exc_info=True,
                )

    def search_similar(
        self,
        query: str,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search for research summaries similar to the query.

        Args:
            query: Search query text.
            n_results: Maximum number of results to return.

        Returns:
            List of result dictionaries, each containing:
            - summary: The research summary text
            - section_id: Associated blog section
            - query: Original search query that generated this summary
            - source_urls: List of source URLs
            Returns empty list on error or if collection unavailable.
        """
        # Input validation
        if not query or not query.strip():
            self.logger.debug("search_similar.empty_query")
            return []
        if n_results < 1:
            n_results = 1

        with self._lock:
            if not self._check_collection():
                return []

            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["documents", "metadatas"],
                )

                parsed_results = []
                # ChromaDB returns lists-of-lists; we have one query, so take first element
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]

                for doc, meta in zip(documents, metadatas):
                    parsed_results.append(self._parse_result(doc, meta))

                self.logger.debug(
                    "search_similar.success",
                    query=query,
                    n_results=len(parsed_results),
                )
                return parsed_results

            except ChromaError as e:
                self.logger.warning(
                    "search_similar.chroma_error",
                    query=query,
                    error=str(e),
                )
                return []
            except Exception as e:
                self.logger.error(
                    "search_similar.unexpected_error",
                    query=query,
                    error=str(e),
                    exc_info=True,
                )
                return []

    def clear(self) -> None:
        """Delete and recreate the collection, removing all stored research.

        Use with caution – this operation is irreversible.
        """
        with self._lock:
            if not self._check_collection():
                # Try to reinitialize if collection is None
                self._init_client()
                if self.collection is None:
                    return

            try:
                # Delete the existing collection
                self.client.delete_collection(self.collection_name)
                self.logger.info("clear.collection_deleted", collection=self.collection_name)

                # Recreate it
                self.collection = self.client.create_collection(self.collection_name)
                self.logger.info("clear.collection_recreated", collection=self.collection_name)

            except ChromaError as e:
                self.logger.error(
                    "clear.chroma_error",
                    error=str(e),
                    exc_info=True,
                )
                # Attempt to recover by reinitializing
                self._init_client()
            except Exception as e:
                self.logger.error(
                    "clear.unexpected_error",
                    error=str(e),
                    exc_info=True,
                )
                self._init_client()

    # ------------------------------------------------------------------
    # Async wrappers (for asyncio applications)
    # ------------------------------------------------------------------

    async def add_research_async(
        self,
        section_id: str,
        query: str,
        summary: str,
        source_urls: List[str],
    ) -> None:
        """Asynchronous version of add_research.

        Runs the synchronous method in a thread pool to avoid blocking
        the asyncio event loop.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.add_research,
            section_id,
            query,
            summary,
            source_urls,
        )

    async def search_similar_async(
        self,
        query: str,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Asynchronous version of search_similar."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.search_similar,
            query,
            n_results,
        )

    async def clear_async(self) -> None:
        """Asynchronous version of clear."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.clear)


# ----------------------------------------------------------------------
# Module-level singleton – import and use this throughout the application.
# ----------------------------------------------------------------------
chroma_store = ChromaStore()