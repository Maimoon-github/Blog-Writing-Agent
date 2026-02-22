"""Configuration and ingestion for pickleball_rules ChromaDB collection."""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

COLLECTION_NAME = "pickleball_rules"
# Path to the USAPA rulebook PDF (adjust as needed)
DEFAULT_PDF_PATH = os.path.join("data", "usapa_rulebook_2025.pdf")


def load(
    persist_directory: str,
    embedding_function,
    pdf_path: str = DEFAULT_PDF_PATH,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Chroma:
    """
    Load the USAPA rulebook PDF, split into chunks, and add to ChromaDB collection.
    Returns the Chroma vector store object.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    # Create or update collection
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )
    vectordb.persist()
    print(f"Loaded {len(chunks)} chunks into '{COLLECTION_NAME}' collection.")
    return vectordb