"""API tests using FastAPI's TestClient with the RAG layer mocked out."""
from __future__ import annotations

import io

from fastapi.testclient import TestClient

from documind.api import main
from documind.core.rag import Answer


def _client() -> TestClient:
    return TestClient(main.app)


def test_health():
    resp = _client().get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "llm_model" in body


def test_ask_returns_answer(monkeypatch):
    monkeypatch.setattr(
        main,
        "answer_question",
        lambda q, top_k=None: Answer(text="42", sources=["book.pdf"]),
    )
    resp = _client().post("/ask", json={"question": "meaning of life?"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": "42", "sources": ["book.pdf"]}


def test_ask_rejects_empty_question():
    resp = _client().post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ingest_rejects_unsupported_type():
    files = {"file": ("bad.png", io.BytesIO(b"data"), "image/png")}
    resp = _client().post("/ingest", files=files)
    assert resp.status_code == 415


def test_ingest_indexes_supported_file(monkeypatch):
    monkeypatch.setattr(main, "ingest_file", lambda path: 7)
    files = {"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    resp = _client().post("/ingest", files=files)
    assert resp.status_code == 200
    assert resp.json() == {"filename": "notes.txt", "chunks_indexed": 7}
