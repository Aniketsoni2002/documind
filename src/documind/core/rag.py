"""The RAG chain: retrieve relevant chunks, then answer with citations.

This is the heart of the project. It wires the retriever to a local Ollama LLM
with a prompt that (a) forces the model to answer *only* from the retrieved
context and (b) say "I don't know" rather than hallucinate — the behaviour a
reviewer actually wants to see in a RAG system.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from documind.config import get_settings
from documind.core.vectorstore import get_vectorstore
from documind.utils.logging import get_logger

logger = get_logger("rag")

SYSTEM_PROMPT = """You are DocuMind, a careful assistant that answers questions \
strictly from the provided context.

Rules:
- Use ONLY the information in the context below. Do not use outside knowledge.
- If the context does not contain the answer, say: "I don't have enough \
information in the provided documents to answer that."
- Be concise and cite the source file names you used.

Context:
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


@dataclass
class Answer:
    """A model answer plus the sources that grounded it."""

    text: str
    sources: list[str]


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source: {src})\n{doc.page_content}")
    return "\n\n".join(blocks)


def _get_llm() -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )


def answer_question(question: str, *, top_k: int | None = None) -> Answer:
    """Retrieve context for ``question`` and generate a grounded answer."""
    settings = get_settings()
    k = top_k or settings.top_k

    retriever = get_vectorstore().as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    if not docs:
        return Answer(
            text="No documents have been indexed yet. Upload a file first.",
            sources=[],
        )

    chain = _PROMPT | _get_llm()
    response = chain.invoke(
        {"context": _format_context(docs), "question": question}
    )

    sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
    logger.info("Answered question using %d chunk(s)", len(docs))
    return Answer(text=response.content, sources=sources)
