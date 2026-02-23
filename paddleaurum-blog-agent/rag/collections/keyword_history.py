"""rag/collections/keyword_history.py

Ingestion for the keyword_history ChromaDB collection.

Source: CSV file containing previously used keywords and their performance data.

Expected CSV columns (all optional except 'keyword'):
    keyword    — the keyword phrase (used as Document.page_content)
    volume     — estimated monthly search volume
    difficulty — SEO difficulty score (0–100)
    cpc        — cost-per-click estimate
    used_at    — ISO date the keyword was last published
    article_slug — slug of the article that used this keyword

Each row becomes one Document.  No splitting is applied because each keyword
entry is already atomic — chunking would fragment the performance metadata.

The SEO Strategist queries this collection before finalising keyword selection
to avoid targeting keywords already covered by published articles.

Collection name must match COLLECTION_KEYWORD_HISTORY in tools/rag_retrieval.py.
"""

import csv
import os
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

COLLECTION_NAME: str = "keyword_history"

_DEFAULT_CSV_PATH: str = os.path.join("data", "keyword_history.csv")


def _csv_to_documents(csv_path: str) -> List[Document]:
    """Parse CSV rows into LangChain Documents.

    The keyword phrase becomes page_content; all other columns are stored as
    string metadata so they are available on retrieved documents.
    """
    documents: List[Document] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            keyword = row.get("keyword", "").strip()
            if not keyword:
                continue
            metadata = {k: str(v).strip() for k, v in row.items() if k != "keyword"}
            documents.append(Document(page_content=keyword, metadata=metadata))
    return documents


def load(
    persist_directory: str,
    embedding_function,
    csv_path: str = _DEFAULT_CSV_PATH,
) -> Chroma:
    """
    Load keyword history from a CSV file into the keyword_history collection.

    Parameters
    ----------
    persist_directory : ChromaDB persist directory.
    embedding_function: Embedding model instance from rag/embeddings.py.
    csv_path          : Path to the keyword history CSV file.

    Returns
    -------
    Chroma
        Initialised vector store for this collection.

    Raises
    ------
    FileNotFoundError
        If csv_path does not exist.
    ValueError
        If the CSV contains no valid keyword rows.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Keyword history CSV not found: {csv_path}")

    documents = _csv_to_documents(csv_path)

    if not documents:
        raise ValueError(f"No valid keyword rows found in {csv_path}")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )

    print(f"[{COLLECTION_NAME}] {len(documents)} keyword entries ingested from {csv_path}")
    return vectordb