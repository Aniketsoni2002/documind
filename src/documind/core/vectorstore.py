"""Vector store management (ChromaDB + local HuggingFace embeddings).

Wraps ChromaDB so the rest of the app never touches embedding details. The
embedding model is loaded lazily and cached, so importing this module is cheap
and tests that don't need embeddings stay fast.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from documind.config import get_settings
from documind.utils.logging import get_logger

logger = get_logger("vectorstore")


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model once per process."""
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    """Return the persistent Chroma collection."""
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def add_documents(chunks: list[Document]) -> int:
    """Embed and store chunks. Returns the number of chunks added."""
    if not chunks:
        return 0
    store = get_vectorstore()
    store.add_documents(chunks)
    logger.info("Added %d chunk(s) to vector store", len(chunks))
    return len(chunks)


def clear_collection() -> None:
    """Delete all vectors (used by the 'reset' action and by tests)."""
    store = get_vectorstore()
    store.delete_collection()
    logger.info("Cleared vector store collection")
