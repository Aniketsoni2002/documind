"""Tests for the RAG chain using fakes (no Ollama / no downloads)."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from documind.core import rag


def _fake_llm() -> RunnableLambda:
    """A Runnable that stands in for ChatOllama in the ``prompt | llm`` chain.

    It receives the formatted prompt value and returns an AIMessage, exactly
    like a real chat model would — so the chain wiring is exercised for real.
    """

    def _respond(prompt_value):
        text = prompt_value.to_string()
        assert "Context:" in text  # the system prompt injected the context
        return AIMessage(content="Grounded answer based on the context.")

    return RunnableLambda(_respond)


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, _query):
        return self._docs


class _FakeStore:
    def __init__(self, docs):
        self._docs = docs

    def as_retriever(self, **_kwargs):
        return _FakeRetriever(self._docs)


def test_answer_uses_retrieved_sources(monkeypatch):
    docs = [
        Document(
            page_content="Refunds within 30 days.",
            metadata={"source": "policy.pdf"},
        ),
        Document(
            page_content="Contact support@x.com.",
            metadata={"source": "faq.md"},
        ),
    ]
    monkeypatch.setattr(rag, "get_vectorstore", lambda: _FakeStore(docs))
    monkeypatch.setattr(rag, "_get_llm", _fake_llm)

    answer = rag.answer_question("What is the refund policy?")

    assert "Grounded answer" in answer.text
    assert answer.sources == ["faq.md", "policy.pdf"]  # sorted + deduped


def test_empty_index_returns_helpful_message(monkeypatch):
    monkeypatch.setattr(rag, "get_vectorstore", lambda: _FakeStore([]))
    monkeypatch.setattr(rag, "_get_llm", _fake_llm)

    answer = rag.answer_question("anything?")

    assert "Upload a file first" in answer.text
    assert answer.sources == []


def test_context_formatting_includes_source_labels():
    docs = [Document(page_content="chunk text", metadata={"source": "a.pdf"})]
    formatted = rag._format_context(docs)
    assert "source: a.pdf" in formatted
    assert "chunk text" in formatted
