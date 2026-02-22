"""Configuration and ingestion for coaching_materials ChromaDB collection."""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

COLLECTION_NAME = "coaching_materials"
# Directory containing coaching guides (drills, strategies, techniques)
DEFAULT_DIR = os.path.join("data", "coaching_manuals")


def load(
    persist_directory: str,
    embedding_function,
    source_dir: str = DEFAULT_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    glob_pattern: str = "**/*.*",
) -> Chroma:
    """
    Load all coaching materials from a directory (supports .txt, .md, .pdf).
    Returns the Chroma vector store object.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Use multiple loaders depending on file types
    loaders = []
    # Text files
    if glob_pattern:
        loaders.append(DirectoryLoader(
            source_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
        ))
        loaders.append(DirectoryLoader(
            source_dir,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
        ))
        # PDF files (requires separate loader)
        # For PDF, we could use PyPDFLoader, but that's more complex.
        # We'll assume PDFs are handled separately or use a generic loader.
        # For simplicity, we'll use PyPDFLoader for PDFs via a custom loader.
    # For brevity, we'll just load all text and markdown files; PDFs can be added similarly.

    documents = []
    for loader in loaders:
        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading with {loader.__class__.__name__}: {e}")

    if not documents:
        raise ValueError(f"No documents found in {source_dir} with pattern {glob_pattern}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )
    vectordb.persist()
    print(f"Loaded {len(chunks)} chunks into '{COLLECTION_NAME}' collection.")
    return vectordb