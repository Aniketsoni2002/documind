"""Document loading and chunking.

Turns raw files (PDF, TXT, Markdown) into a list of LangChain ``Document``
chunks ready to be embedded. Kept deliberately dependency-light and pure so it
is trivial to unit-test without a running model or vector store.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from documind.config import get_settings
from documind.utils.logging import get_logger

logger = get_logger("loader")

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


class UnsupportedFileError(ValueError):
    """Raised when a file type we can't parse is submitted."""


def _loader_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8")
    raise UnsupportedFileError(
        f"Unsupported file type: {suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def load_documents(path: str | Path) -> list[Document]:
    """Load a single file into one Document per page/section."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    docs = _loader_for(path).load()
    # Loaders set "source" to the full path; normalise it to just the file name
    # so citations shown to users are human-readable and stable across machines.
    for doc in docs:
        doc.metadata["source"] = path.name
    logger.info("Loaded %d section(s) from %s", len(docs), path.name)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split loaded documents into overlapping chunks for retrieval."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunk(s)", len(chunks))
    return chunks


def load_and_split(path: str | Path) -> list[Document]:
    """Convenience: load a file and return retrieval-ready chunks."""
    return split_documents(load_documents(path))
