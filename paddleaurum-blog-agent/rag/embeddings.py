"""rag/embeddings.py

HuggingFace sentence-transformers embedding setup.

Imports EMBEDDING_MODEL from config/settings.py so the model name is defined
in exactly one place for both ingestion (rag/) and retrieval (tools/rag_retrieval.py).
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL

# Use GPU if available; fall back to CPU automatically.
_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")


def get_embedding_function(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Return a HuggingFaceEmbeddings instance for the given model.

    The model is downloaded and cached locally on first call.
    Embeddings are L2-normalised, which is required for cosine-similarity
    searches in ChromaDB.

    Parameters
    ----------
    model_name : Sentence-transformers model identifier.
                 Defaults to EMBEDDING_MODEL from config/settings.py.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": _DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )