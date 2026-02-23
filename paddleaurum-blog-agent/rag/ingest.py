#!/usr/bin/env python
"""rag/ingest.py

Main ingestion script: loads all five knowledge-base collections into ChromaDB.

Usage
-----
    python -m rag.ingest [--persist-dir PATH] [--reset] [--collection NAME]

CHROMADB_PERSIST_DIR is read from config/settings.py (overridable via env var).
"""

import argparse
import logging
import os
import shutil

from rag.embeddings import get_embedding_function
from rag.collections import (
    coaching_materials,
    keyword_history,
    pickleball_rules,
    published_articles,
    seo_guidelines,
)
from config.settings import CHROMADB_PERSIST_DIR

logger = logging.getLogger(__name__)

# Ordered so that foundational knowledge (rules, coaching) is ingested before
# derivative content (published articles, keyword history).
_COLLECTIONS = [
    ("pickleball_rules",    pickleball_rules),
    ("coaching_materials",  coaching_materials),
    ("seo_guidelines",      seo_guidelines),
    ("published_articles",  published_articles),
    ("keyword_history",     keyword_history),
]


def main(
    persist_dir: str = CHROMADB_PERSIST_DIR,
    reset: bool = False,
    collection: str | None = None,
) -> None:
    """
    Ingest all (or one) collection into ChromaDB.

    Parameters
    ----------
    persist_dir : ChromaDB persist directory.
    reset       : If True, delete the persist directory before loading.
    collection  : If set, ingest only this named collection and skip the rest.
    """
    if reset and os.path.exists(persist_dir):
        logger.info("Resetting ChromaDB at %s", persist_dir)
        shutil.rmtree(persist_dir)

    os.makedirs(persist_dir, exist_ok=True)
    embedding_fn = get_embedding_function()

    for name, module in _COLLECTIONS:
        if collection and name != collection:
            continue

        logger.info("Loading collection: %s", name)
        try:
            module.load(
                persist_directory=persist_dir,
                embedding_function=embedding_fn,
            )
            logger.info("Collection '%s' loaded successfully.", name)
        except FileNotFoundError as exc:
            logger.error("Skipping '%s' — source file not found: %s", name, exc)
        except Exception as exc:
            logger.exception("Failed to load collection '%s': %s", name, exc)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest knowledge base into ChromaDB.")
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=CHROMADB_PERSIST_DIR,
        help=f"ChromaDB persist directory (default: {CHROMADB_PERSIST_DIR})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing ChromaDB directory before loading.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        choices=[name for name, _ in _COLLECTIONS],
        help="Ingest a single named collection and skip the rest.",
    )
    args = parser.parse_args()
    main(persist_dir=args.persist_dir, reset=args.reset, collection=args.collection)