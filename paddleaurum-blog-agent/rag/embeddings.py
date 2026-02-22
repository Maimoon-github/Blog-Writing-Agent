"""HuggingFace sentence-transformers embedding setup."""

import os
from langchain_huggingface import HuggingFaceEmbeddings

# Default embedding model (all-MiniLM-L6-v2)
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function(model_name: str = DEFAULT_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFaceEmbeddings instance for the given model.
    The model is downloaded and cached locally on first use.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # or "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},
    )