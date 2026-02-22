"""Configuration and ingestion for published_articles ChromaDB collection."""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

COLLECTION_NAME = "published_articles"
# Directory containing exported WordPress articles (markdown or text)
DEFAULT_POSTS_DIR = os.path.join("data", "published_posts")


def load(
    persist_directory: str,
    embedding_function,
    posts_dir: str = DEFAULT_POSTS_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    glob_pattern: str = "**/*.md",
) -> Chroma:
    """
    Load all previously published articles (markdown files) and add to ChromaDB.
    Returns the Chroma vector store object.
    """
    if not os.path.exists(posts_dir):
        raise FileNotFoundError(f"Posts directory not found: {posts_dir}")

    loader = DirectoryLoader(
        posts_dir,
        glob=glob_pattern,
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    documents = loader.load()

    if not documents:
        raise ValueError(f"No markdown files found in {posts_dir}")

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