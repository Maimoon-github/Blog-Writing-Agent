"""Configuration and ingestion for seo_guidelines ChromaDB collection."""

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

COLLECTION_NAME = "seo_guidelines"
# Path to SEO checklist file
DEFAULT_CHECKLIST_PATH = os.path.join("data", "seo_checklist.md")


def load(
    persist_directory: str,
    embedding_function,
    checklist_path: str = DEFAULT_CHECKLIST_PATH,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    Load the SEO checklist markdown file and add to ChromaDB.
    Returns the Chroma vector store object.
    """
    if not os.path.exists(checklist_path):
        raise FileNotFoundError(f"SEO checklist not found: {checklist_path}")

    loader = TextLoader(checklist_path, autodetect_encoding=True)
    documents = loader.load()

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