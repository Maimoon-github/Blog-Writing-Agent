"""rag/collections/pickleball_rules.py

Ingestion for the pickleball_rules ChromaDB collection.

Source: USAPA 2024/2025 official rulebook (PDF).
Collection name must match COLLECTION_PICKLEBALL_RULES in tools/rag_retrieval.py.
"""

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

COLLECTION_NAME: str = "pickleball_rules"

_DEFAULT_PDF_PATH:    str = os.path.join("data", "usapa_rulebook_2025.pdf")
_DEFAULT_CHUNK_SIZE:  int = 1000
_DEFAULT_CHUNK_OVERLAP: int = 200


def load(
    persist_directory: str,
    embedding_function,
    pdf_path: str = _DEFAULT_PDF_PATH,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> Chroma:
    """
    Load the USAPA rulebook PDF into the pickleball_rules ChromaDB collection.

    Each page is split into overlapping chunks so that rules spanning page
    boundaries remain retrievable as complete context.

    Parameters
    ----------
    persist_directory : ChromaDB persist directory.
    embedding_function: Embedding model instance from rag/embeddings.py.
    pdf_path          : Path to the USAPA rulebook PDF.
    chunk_size        : Maximum token count per chunk.
    chunk_overlap     : Overlap between adjacent chunks.

    Returns
    -------
    Chroma
        Initialised vector store for this collection.

    Raises
    ------
    FileNotFoundError
        If pdf_path does not exist.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"USAPA rulebook not found: {pdf_path}")

    documents = PyPDFLoader(pdf_path).load()

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

    print(f"[{COLLECTION_NAME}] {len(chunks)} chunks ingested from {pdf_path}")
    return vectordb