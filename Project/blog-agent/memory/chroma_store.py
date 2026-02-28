import json
from typing import List

import chromadb
import structlog

from app.config import CHROMA_PERSIST_DIR


class ChromaStore:
    def __init__(
        self,
        persist_directory: str = str(CHROMA_PERSIST_DIR),
        collection_name: str = "blog_research",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.collection_name = collection_name
        self.logger = structlog.get_logger(__name__)

    def add_research(
        self,
        section_id: str,
        query: str,
        summary: str,
        source_urls: List[str],
    ) -> None:
        """Add a research summary to the ChromaDB collection."""
        self.collection.add(
            documents=[summary],
            metadatas=[
                {
                    "section_id": section_id,
                    "query": query,
                    "source_urls_json": json.dumps(source_urls),
                }
            ],
            ids=[f"research_{section_id}"],
        )
        self.logger.info(
            "research_added",
            section_id=section_id,
            query=query,
            source_url_count=len(source_urls),
        )

    def search_similar(self, query: str, n_results: int = 3) -> List[dict]:
        """Search for similar research summaries by query text."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
            )

            parsed: List[dict] = []

            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []

            # Both are lists-of-lists (one inner list per query text)
            for doc, meta in zip(
                documents[0] if documents else [],
                metadatas[0] if metadatas else [],
            ):
                parsed.append(
                    {
                        "summary": doc,
                        "section_id": meta.get("section_id", ""),
                        "query": meta.get("query", ""),
                        "source_urls": json.loads(meta.get("source_urls_json", "[]")),
                    }
                )

            return parsed

        except Exception as exc:  # noqa: BLE001
            self.logger.warning("search_similar_failed", query=query, error=str(exc))
            return []

    def clear(self) -> None:
        """Delete and recreate the collection, removing all stored research."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.logger.info("chroma_collection_cleared", collection=self.collection_name)


# Shared singleton — imported by other modules
chroma_store = ChromaStore()