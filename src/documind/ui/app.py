"""Streamlit UI for DocuMind — a friendly front end over the RAG pipeline.

Run with:  streamlit run src/documind/ui/app.py
It calls the core library directly (no API round-trip needed), so it works even
if you only want the UI.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from documind.config import get_settings
from documind.core.ingest import ingest_file
from documind.core.loader import SUPPORTED_SUFFIXES
from documind.core.rag import answer_question
from documind.core.vectorstore import clear_collection

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="centered")
settings = get_settings()

st.title("📄 DocuMind")
st.caption("Chat with your documents — 100% local, private, and free.")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF / TXT / Markdown",
        type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Index documents", type="primary"):
        for file in uploaded:
            dest = settings.upload_dir / Path(file.name).name
            dest.write_bytes(file.getbuffer())
            with st.spinner(f"Indexing {file.name}…"):
                n = ingest_file(dest)
            st.success(f"{file.name}: {n} chunks indexed")

    st.divider()
    if st.button("🗑️ Clear index"):
        clear_collection()
        st.info("Vector store cleared.")

    st.divider()
    st.caption(f"LLM: `{settings.llm_model}`")
    st.caption(f"Embeddings: `{settings.embedding_model.split('/')[-1]}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = answer_question(prompt)
        st.markdown(result.text)
        if result.sources:
            st.caption("Sources: " + ", ".join(f"`{s}`" for s in result.sources))
    st.session_state.messages.append(
        {"role": "assistant", "content": result.text}
    )
