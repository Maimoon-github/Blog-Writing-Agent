"""Configuration and ingestion for keyword_history ChromaDB collection."""

import os
import csv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

COLLECTION_NAME = "keyword_history"
# CSV file containing keyword performance data
DEFAULT_CSV_PATH = os.path.join("data", "keyword_history.csv")
# Expected columns: keyword, volume, difficulty, cpc, etc.


def load(
    persist_directory: str,
    embedding_function,
    csv_path: str = DEFAULT_CSV_PATH,
) -> Chroma:
    """
    Load keyword history from a CSV file and create documents for each keyword.
    Each row becomes a document with keyword as page_content and other fields as metadata.
    Returns the Chroma vector store object.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Keyword history CSV not found: {csv_path}")

    documents = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use the keyword as the main text
            keyword = row.get("keyword", "").strip()
            if not keyword:
                continue
            # Convert all fields to strings for metadata
            metadata = {k: str(v) for k, v in row.items() if k != "keyword"}
            doc = Document(page_content=keyword, metadata=metadata)
            documents.append(doc)

    if not documents:
        raise ValueError("No keywords found in CSV")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )
    vectordb.persist()
    print(f"Loaded {len(documents)} keyword entries into '{COLLECTION_NAME}' collection.")
    return vectordb