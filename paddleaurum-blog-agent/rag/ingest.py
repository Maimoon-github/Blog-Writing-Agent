#!/usr/bin/env python
"""Main ingestion script: loads all knowledge base documents into ChromaDB."""

import os
import argparse
from rag.embeddings import get_embedding_function
from rag.collections import (
    pickleball_rules,
    coaching_materials,
    seo_guidelines,
    published_articles,
    keyword_history,
)

# Default ChromaDB persist directory (can be overridden via env)
DEFAULT_PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", "./chromadb_store")


def main(persist_dir: str, reset: bool = False):
    """
    Ingest all collections into ChromaDB.

    Args:
        persist_dir: Directory where ChromaDB data is stored.
        reset: If True, delete existing collections before loading.
    """
    if reset and os.path.exists(persist_dir):
        import shutil
        print(f"Resetting ChromaDB at {persist_dir}...")
        shutil.rmtree(persist_dir)

    embedding_fn = get_embedding_function()

    # Load each collection (order may not matter)
    collections = [
        ("pickleball_rules", pickleball_rules),
        ("coaching_materials", coaching_materials),
        ("seo_guidelines", seo_guidelines),
        ("published_articles", published_articles),
        ("keyword_history", keyword_history),
    ]

    for name, module in collections:
        print(f"\n--- Loading collection: {name} ---")
        try:
            module.load(
                persist_directory=persist_dir,
                embedding_function=embedding_fn,
            )
        except Exception as e:
            print(f"ERROR loading {name}: {e}")

    print("\nIngestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load knowledge base into ChromaDB.")
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=DEFAULT_PERSIST_DIR,
        help="ChromaDB persist directory",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing ChromaDB directory before loading",
    )
    args = parser.parse_args()
    main(args.persist_dir, args.reset)