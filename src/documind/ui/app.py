"""Streamlit UI for DocuMind — a friendly front end over the RAG pipeline.

Run with:  streamlit run src/documind/ui/app.py
It calls the core library directly (no API round-trip needed), so it works even
if you only want the UI.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

# Bridge Streamlit secrets → environment so pydantic-settings picks them up.
# Must run BEFORE the first get_settings() call. Any DOCUMIND_* key set in
# .streamlit/secrets.toml (or the Cloud "Secrets" box) is exported here.
try:
    for _key, _val in st.secrets.items():
        if _key.startswith("DOCUMIND_") and _key not in os.environ:
            os.environ[_key] = str(_val)
except Exception:  # no secrets file locally — that's fine
    pass

from documind.config import get_settings  # noqa: E402
from documind.core.ingest import ingest_file
from documind.core.loader import SUPPORTED_SUFFIXES, UnsupportedFileError
from documind.core.rag import answer_question
from documind.core.vectorstore import clear_collection, collection_count

st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
settings = get_settings()

# --- Styling ---------------------------------------------------------------
st.markdown(
    """
    <style>
      /* Tighten the main column and give the chat some breathing room. */
      .block-container { max-width: 900px; padding-top: 2.2rem; }

      /* Gradient headline. */
      .dm-title {
          font-size: 2.6rem; font-weight: 800; line-height: 1.1;
          background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 55%, #ec4899 100%);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          background-clip: text; margin-bottom: .2rem;
      }
      .dm-subtitle { color: #6b7280; font-size: 1.02rem; margin-bottom: 1.4rem; }

      /* Source "chips" under an answer. */
      .dm-source {
          display: inline-block; background: rgba(99,102,241,.12);
          color: #6366f1; border: 1px solid rgba(99,102,241,.25);
          padding: 2px 10px; border-radius: 999px; font-size: .78rem;
          margin: 2px 6px 2px 0; font-family: ui-monospace, monospace;
      }

      /* Example-prompt buttons on the empty state. */
      .stButton>button { border-radius: 10px; }

      /* Status pill in the sidebar. */
      .dm-pill {
          display: inline-block; padding: 3px 12px; border-radius: 999px;
          font-size: .8rem; font-weight: 600;
      }
      .dm-pill-ok  { background: rgba(16,185,129,.14); color: #059669; }
      .dm-pill-off { background: rgba(148,163,184,.18); color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _refresh_count() -> int:
    """Cheap wrapper so we only hit the store when we need a fresh number."""
    try:
        return collection_count()
    except Exception:  # pragma: no cover - defensive; store may be uninitialised
        return 0


# --- Header ----------------------------------------------------------------
st.markdown('<div class="dm-title">📄 DocuMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dm-subtitle">Chat with your documents — 100% local, '
    "private, and free.</div>",
    unsafe_allow_html=True,
)

# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("📁 Documents")

    indexed = _refresh_count()
    if indexed:
        st.markdown(
            f'<span class="dm-pill dm-pill-ok">● {indexed} chunks indexed</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="dm-pill dm-pill-off">● No documents yet</span>',
            unsafe_allow_html=True,
        )

    st.write("")
    uploaded = st.file_uploader(
        "Upload PDF / TXT / Markdown",
        type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
        accept_multiple_files=True,
        help="Files are processed locally and never leave your machine.",
    )

    if uploaded and st.button("⚡ Index documents", type="primary", width="stretch"):
        progress = st.progress(0.0, text="Starting…")
        total = len(uploaded)
        for i, file in enumerate(uploaded, start=1):
            progress.progress(
                (i - 1) / total, text=f"Indexing {file.name} ({i}/{total})…"
            )
            dest = settings.upload_dir / Path(file.name).name
            dest.write_bytes(file.getbuffer())
            try:
                n = ingest_file(dest)
                st.success(f"{file.name}: {n} chunks indexed")
            except UnsupportedFileError as exc:
                st.error(f"{file.name}: {exc}")
            except Exception as exc:  # pragma: no cover - surface any failure
                st.error(f"{file.name}: failed to index — {exc}")
        progress.progress(1.0, text="Done")
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear index", width="stretch"):
        clear_collection()
        st.session_state.messages = []
        st.toast("Vector store cleared.", icon="🧹")
        st.rerun()

    st.divider()
    st.caption("**Configuration**")
    if settings.llm_provider.lower() == "groq":
        st.caption(f"LLM · `{settings.groq_model}` _(Groq)_")
    else:
        st.caption(f"LLM · `{settings.llm_model}` _(Ollama)_")
    st.caption(f"Embeddings · `{settings.embedding_model.split('/')[-1]}`")
    st.caption(f"Top-K · `{settings.top_k}`")

# --- Chat state ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

EXAMPLES = [
    "What is this document about?",
    "Summarise the key points.",
    "What are the main takeaways?",
]


def _ask(question: str) -> None:
    """Run a question through the RAG pipeline and append both messages."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = answer_question(question)
        st.markdown(result.text)
        if result.sources:
            chips = "".join(
                f'<span class="dm-source">{s}</span>' for s in result.sources
            )
            st.markdown(chips, unsafe_allow_html=True)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.text,
            "sources": result.sources,
        }
    )


# --- Empty state -----------------------------------------------------------
if not st.session_state.messages:
    if _refresh_count() == 0:
        st.info(
            "👈 Upload a document in the sidebar and click **Index documents** "
            "to get started.",
            icon="📄",
        )
    else:
        st.caption("Try asking:")
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES, strict=True):
            if col.button(ex, key=f"ex-{ex}", width="stretch"):
                _ask(ex)
                st.rerun()

# --- Replay history --------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips = "".join(
                f'<span class="dm-source">{s}</span>' for s in msg["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)

# --- Input -----------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents…"):
    _ask(prompt)
    st.rerun()
