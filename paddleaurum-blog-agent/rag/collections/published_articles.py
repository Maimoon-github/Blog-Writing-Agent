"""rag/collections/published_articles.py

Ingestion for the published_articles ChromaDB collection.

Source: directory of previously published paddleaurum.com articles exported
from WordPress as Markdown or text files.

This collection serves two purposes:
  1. The Coach Writer RAG-retrieves previously published content so it can
     cross-reference and internally link to existing articles.
  2. Before each pipeline run, the planner queries this collection to detect
     topical overlap and avoid publishing duplicate content.

The filename (without extension) is stored as `slug` metadata so queries can
filter by article identity.

Collection name must match COLLECTION_PUBLISHED_ARTICLES in tools/rag_retrieval.py.
"""

import os
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma

COLLECTION_NAME: str = "published_articles"

_DEFAULT_POSTS_DIR:     str = os.path.join("data", "published_posts")
_DEFAULT_CHUNK_SIZE:    int = 1000
_DEFAULT_CHUNK_OVERLAP: int = 200


def _inject_slug_metadata(documents: List[Document]) -> List[Document]:
    """Add the filename stem as 'slug' metadata for each document."""
    for doc in documents:
        source: str = doc.metadata.get("source", "")
        slug = os.path.splitext(os.path.basename(source))[0]
        doc.metadata["slug"] = slug
    return documents


def load(
    persist_directory: str,
    embedding_function,
    posts_dir: str = _DEFAULT_POSTS_DIR,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> Chroma:
    """
    Load all published article files (.md, .txt) into the published_articles collection.

    Parameters
    ----------
    persist_directory : ChromaDB persist directory.
    embedding_function: Embedding model instance from rag/embeddings.py.
    posts_dir         : Directory containing exported WordPress posts.
    chunk_size        : Maximum characters per chunk.
    chunk_overlap     : Overlap between adjacent chunks.

    Returns
    -------
    Chroma
        Initialised vector store for this collection.

    Raises
    ------
    FileNotFoundError
        If posts_dir does not exist.
    ValueError
        If no article files are found in posts_dir.
    """
    if not os.path.exists(posts_dir):
        raise FileNotFoundError(f"Published posts directory not found: {posts_dir}")

    documents: List[Document] = []
    for glob in ("**/*.md", "**/*.txt"):
        loader = DirectoryLoader(
            posts_dir,
            glob=glob,
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            silent_errors=True,
        )
        documents.extend(loader.load())

    if not documents:
        raise ValueError(f"No article files (.md, .txt) found in {posts_dir}")

    _inject_slug_metadata(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )

    print(f"[{COLLECTION_NAME}] {len(chunks)} chunks ingested from {posts_dir}")
    return vectordb