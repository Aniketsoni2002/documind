"""Tests for document loading and chunking (no models required)."""
from __future__ import annotations

import pytest

from documind.core.loader import (
    UnsupportedFileError,
    load_and_split,
    load_documents,
    split_documents,
)


def test_load_text_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Hello world. This is a test document.", encoding="utf-8")

    docs = load_documents(f)

    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content
    assert docs[0].metadata["source"] == "note.txt"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope.txt")


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(UnsupportedFileError):
        load_documents(f)


def test_splitting_produces_multiple_chunks(tmp_path, monkeypatch):
    # Force a tiny chunk size so we deterministically get several chunks.
    from documind import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("DOCUMIND_CHUNK_SIZE", "50")
    monkeypatch.setenv("DOCUMIND_CHUNK_OVERLAP", "10")

    f = tmp_path / "long.txt"
    f.write_text("word " * 200, encoding="utf-8")

    chunks = load_and_split(f)

    assert len(chunks) > 1
    assert all("start_index" in c.metadata for c in chunks)
    config.get_settings.cache_clear()


def test_split_documents_preserves_source(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nSome markdown content here.", encoding="utf-8")
    chunks = split_documents(load_documents(f))
    assert all(c.metadata["source"] == "doc.md" for c in chunks)
