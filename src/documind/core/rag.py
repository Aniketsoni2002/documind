"""The RAG chain: retrieve relevant chunks, then answer with citations.

This is the heart of the project. It wires the retriever to a local Ollama LLM
with a prompt that (a) forces the model to answer *only* from the retrieved
context and (b) say "I don't know" rather than hallucinate — the behaviour a
reviewer actually wants to see in a RAG system.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

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


def _get_llm() -> BaseChatModel:
    """Build the chat model for the configured provider.

    Imports are local so that, e.g., a Groq-only cloud deploy never needs the
    Ollama client installed and vice versa.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise RuntimeError(
                "DOCUMIND_LLM_PROVIDER=groq but DOCUMIND_GROQ_API_KEY is not set."
            )
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    raise ValueError(
        f"Unknown DOCUMIND_LLM_PROVIDER={settings.llm_provider!r}. "
        "Use 'ollama' or 'groq'."
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
