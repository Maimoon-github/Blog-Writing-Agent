"""rag/collections/coaching_materials.py

Ingestion for the coaching_materials ChromaDB collection.

Source: directory of coaching guides — drills, strategies, and technique files.
Supports .txt, .md, and .pdf files.
Collection name must match COLLECTION_COACHING_MATERIALS in tools/rag_retrieval.py.
"""

import os
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma

COLLECTION_NAME: str = "coaching_materials"

_DEFAULT_SOURCE_DIR:    str = os.path.join("data", "coaching_manuals")
_DEFAULT_CHUNK_SIZE:    int = 1000
_DEFAULT_CHUNK_OVERLAP: int = 200


def _load_text_files(source_dir: str) -> List[Document]:
    """Load all .txt and .md files using TextLoader."""
    docs: List[Document] = []
    for glob in ("**/*.txt", "**/*.md"):
        try:
            loader = DirectoryLoader(
                source_dir,
                glob=glob,
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True},
                silent_errors=True,
            )
            docs.extend(loader.load())
        except Exception:
            pass
    return docs


def _load_pdf_files(source_dir: str) -> List[Document]:
    """Load all .pdf files via PyPDFLoader (one loader per file)."""
    docs: List[Document] = []
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                try:
                    pdf_path = os.path.join(root, filename)
                    docs.extend(PyPDFLoader(pdf_path).load())
                except Exception as exc:
                    print(f"[{COLLECTION_NAME}] Could not load PDF {filename}: {exc}")
    return docs


def load(
    persist_directory: str,
    embedding_function,
    source_dir: str = _DEFAULT_SOURCE_DIR,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> Chroma:
    """
    Load all coaching materials (.txt, .md, .pdf) into the coaching_materials collection.

    Parameters
    ----------
    persist_directory : ChromaDB persist directory.
    embedding_function: Embedding model instance from rag/embeddings.py.
    source_dir        : Root directory containing coaching document files.
    chunk_size        : Maximum characters per chunk.
    chunk_overlap     : Overlap between adjacent chunks.

    Returns
    -------
    Chroma
        Initialised vector store for this collection.

    Raises
    ------
    FileNotFoundError
        If source_dir does not exist.
    ValueError
        If no documents are found in source_dir.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Coaching materials directory not found: {source_dir}")

    documents = _load_text_files(source_dir) + _load_pdf_files(source_dir)

    if not documents:
        raise ValueError(f"No documents found in {source_dir}")

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

    print(f"[{COLLECTION_NAME}] {len(chunks)} chunks ingested from {source_dir}")
    return vectordb