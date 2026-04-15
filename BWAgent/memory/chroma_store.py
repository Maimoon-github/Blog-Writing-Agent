"""ChromaDB persistence wrapper for BWAgent."""

import json
import threading
from typing import Any, Dict, List

import chromadb
import structlog
from chromadb.errors import ChromaError

from config.settings import CHROMA_PERSIST_DIR

logger = structlog.get_logger(__name__)


class ChromaStore:
    def __init__(self, persist_directory: str = str(CHROMA_PERSIST_DIR), collection_name: str = "blog_research") -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = logger.bind(persist_directory=persist_directory, collection=collection_name)
        self._lock = threading.RLock()
        self._init_client()

    def _init_client(self) -> None:
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            self.logger.info("chroma.initialized")
        except Exception as exc:
            self.logger.error("chroma.init_failed", error=str(exc))
            self.collection = None

    def _check_collection(self) -> bool:
        if self.collection is None:
            self.logger.warning("chroma.unavailable")
            return False
        return True

    def _serialize_metadata(self, section_id: str, query: str, source_urls: List[str]) -> Dict[str, str]:
        return {
            "section_id": section_id,
            "query": query,
            "source_urls_json": json.dumps(source_urls, ensure_ascii=False),
        }

    def _parse_result(self, document: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        try:
            source_urls = json.loads(metadata.get("source_urls_json", "[]"))
        except Exception:
            source_urls = []
        return {
            "summary": document,
            "section_id": metadata.get("section_id", ""),
            "query": metadata.get("query", ""),
            "source_urls": source_urls,
        }

    def add_research(self, section_id: str, query: str, summary: str, source_urls: List[str]) -> None:
        if not section_id or not summary:
            self.logger.warning("chroma.invalid_research", section_id=section_id)
            return
        with self._lock:
            if not self._check_collection():
                return
            try:
                doc_id = f"research_{section_id}"
                metadata = self._serialize_metadata(section_id, query, source_urls)
                self.collection.upsert(ids=[doc_id], documents=[summary], metadatas=[metadata])
                self.logger.info("chroma.add_research", section_id=section_id, urls=len(source_urls))
            except ChromaError as exc:
                self.logger.error("chroma.upsert_failed", section_id=section_id, error=str(exc))
            except Exception as exc:
                self.logger.error("chroma.unexpected_error", section_id=section_id, error=str(exc))

    def search_similar(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        if not query or not self._check_collection():
            return []
        with self._lock:
            try:
                results = self.collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                return [self._parse_result(doc, meta) for doc, meta in zip(documents, metadatas)]
            except ChromaError as exc:
                self.logger.warning("chroma.query_failed", query=query, error=str(exc))
            except Exception as exc:
                self.logger.error("chroma.query_error", query=query, error=str(exc))
        return []

    async def add_research_async(self, section_id: str, query: str, summary: str, source_urls: List[str]) -> None:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.add_research, section_id, query, summary, source_urls)

    async def search_similar_async(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.search_similar, query, n_results)


chroma_store = ChromaStore()
