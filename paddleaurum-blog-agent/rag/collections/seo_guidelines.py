"""rag/collections/seo_guidelines.py

Ingestion for the seo_guidelines ChromaDB collection.

Source: SEO checklist markdown file containing on-page SEO rules, best
practices, and keyword strategy guidelines for paddleaurum.com.

Smaller chunk_size (500) and minimal overlap (50) are intentional: each SEO
rule is a discrete, self-contained check, and oversized chunks would cause
unrelated rules to pollute a single retrieval result.

Collection name must match COLLECTION_SEO_GUIDELINES in tools/rag_retrieval.py.
"""

import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

COLLECTION_NAME: str = "seo_guidelines"

_DEFAULT_SOURCE_PATH:   str = os.path.join("data", "seo_checklist.md")
_DEFAULT_CHUNK_SIZE:    int = 500
_DEFAULT_CHUNK_OVERLAP: int = 50


def load(
    persist_directory: str,
    embedding_function,
    source_path: str = _DEFAULT_SOURCE_PATH,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> Chroma:
    """
    Load the SEO checklist document into the seo_guidelines collection.

    Parameters
    ----------
    persist_directory : ChromaDB persist directory.
    embedding_function: Embedding model instance from rag/embeddings.py.
    source_path       : Path to the SEO checklist (.md or .txt).
    chunk_size        : Maximum characters per chunk.
    chunk_overlap     : Overlap between adjacent chunks.

    Returns
    -------
    Chroma
        Initialised vector store for this collection.

    Raises
    ------
    FileNotFoundError
        If source_path does not exist.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"SEO checklist not found: {source_path}")

    documents = TextLoader(source_path, autodetect_encoding=True).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )

    print(f"[{COLLECTION_NAME}] {len(chunks)} chunks ingested from {source_path}")
    return vectordb