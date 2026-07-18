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
# A premium, token-driven design system injected over Streamlit's DOM. All
# colours are CSS custom properties so light/dark themes share one source of
# truth. Business logic below is untouched — this block only affects looks.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

      /* ===================== Design tokens ===================== */
      :root {
        --dm-accent:        #6366f1;
        --dm-accent-strong: #4f46e5;
        --dm-accent-soft:   rgba(99,102,241,.10);
        --dm-accent-ring:   rgba(99,102,241,.35);
        --dm-grad: linear-gradient(100deg, #6366f1 0%, #7c74f2 45%, #a855f7 100%);

        /* Cool neutrals, biased a touch toward the indigo accent. */
        --dm-bg:        #f7f8fb;
        --dm-surface:   #ffffff;
        --dm-surface-2: #f1f2f7;
        --dm-border:    #e6e8ef;
        --dm-text:      #1a1d29;
        --dm-text-soft: #5b6172;
        --dm-text-mute: #8a90a2;

        --dm-good:  #059669;  --dm-good-soft:  rgba(16,185,129,.12);
        --dm-warn:  #d97706;  --dm-warn-soft:  rgba(217,119,6,.12);
        --dm-err:   #dc2626;  --dm-err-soft:   rgba(220,38,38,.10);

        --dm-shadow-sm: 0 1px 2px rgba(24,27,40,.06), 0 1px 3px rgba(24,27,40,.05);
        --dm-shadow-md: 0 4px 14px rgba(24,27,40,.08), 0 2px 6px rgba(24,27,40,.05);
        --dm-shadow-lg: 0 12px 34px rgba(24,27,40,.12);

        --dm-r-sm: 10px; --dm-r-md: 14px; --dm-r-lg: 20px;
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --dm-bg:        #0c0e14;
          --dm-surface:   #14171f;
          --dm-surface-2: #1b1f2a;
          --dm-border:    #262b38;
          --dm-text:      #e9ebf1;
          --dm-text-soft: #a2a8bb;
          --dm-text-mute: #6b7186;
          --dm-accent-soft: rgba(99,102,241,.16);
          --dm-shadow-sm: 0 1px 2px rgba(0,0,0,.4);
          --dm-shadow-md: 0 6px 18px rgba(0,0,0,.45);
          --dm-shadow-lg: 0 16px 40px rgba(0,0,0,.55);
        }
      }

      /* ===================== Base ===================== */
      html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }
      .stApp { background: var(--dm-bg); color: var(--dm-text); }
      .block-container { max-width: 860px; padding-top: 2.4rem; padding-bottom: 6rem; }

      /* Streamlit's top toolbar/header — make it blend with the app bg
         instead of showing an opaque white/near-white bar. */
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"] { right: 8px; }
      code, kbd, .dm-source { font-family: 'JetBrains Mono', ui-monospace, monospace; }

      /* ===================== Header ===================== */
      .dm-brand { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
      .dm-logo {
        width: 46px; height: 46px; border-radius: 13px; flex: none;
        display: grid; place-items: center; font-size: 24px;
        background: var(--dm-grad); box-shadow: var(--dm-shadow-md);
      }
      .dm-title {
        font-size: 2.35rem; font-weight: 800; line-height: 1.05; letter-spacing: -.02em;
        background: var(--dm-grad); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; background-clip: text;
      }
      .dm-subtitle {
        color: var(--dm-text-soft); font-size: 1.02rem; font-weight: 450;
        margin: 2px 0 26px 0;
      }

      /* ===================== Sidebar ===================== */
      [data-testid="stSidebar"] {
        background: var(--dm-surface);
        border-right: 1px solid var(--dm-border);
      }
      [data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
      [data-testid="stSidebar"] h2 {
        font-size: 1.02rem !important; font-weight: 700; letter-spacing: -.01em;
      }
      [data-testid="stSidebar"] hr { margin: 1.1rem 0; border-color: var(--dm-border); }

      /* ===================== Buttons ===================== */
      .stButton > button {
        border-radius: var(--dm-r-sm); font-weight: 600; font-size: .9rem;
        border: 1px solid var(--dm-border); background: var(--dm-surface);
        color: var(--dm-text); box-shadow: var(--dm-shadow-sm);
        transition: transform .2s ease, box-shadow .2s ease,
                    background .2s ease, border-color .2s ease;
      }
      .stButton > button:hover {
        transform: translateY(-1px); box-shadow: var(--dm-shadow-md);
        border-color: var(--dm-accent-ring); color: var(--dm-text);
      }
      .stButton > button:active { transform: translateY(0); }
      .stButton > button:focus-visible {
        outline: none; box-shadow: 0 0 0 3px var(--dm-accent-ring);
      }
      /* Primary (type="primary") — gradient emphasis. */
      .stButton > button[kind="primary"] {
        background: var(--dm-grad); color: #fff; border: none;
        box-shadow: 0 4px 14px var(--dm-accent-ring);
      }
      .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px); box-shadow: 0 8px 22px var(--dm-accent-ring);
        color: #fff;
      }

      /* ===================== File uploader ===================== */
      [data-testid="stFileUploaderDropzone"] {
        background: var(--dm-surface-2);
        border: 1.5px dashed var(--dm-border); border-radius: var(--dm-r-md);
        transition: border-color .2s ease, background .2s ease, transform .2s ease;
      }
      [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--dm-accent); background: var(--dm-accent-soft);
      }

      /* ===================== Chat messages ===================== */
      [data-testid="stChatMessage"] {
        background: var(--dm-surface); border: 1px solid var(--dm-border);
        border-radius: var(--dm-r-md); padding: 2px 6px;
        box-shadow: var(--dm-shadow-sm); margin-bottom: 14px;
        animation: dm-rise .28s cubic-bezier(.22,.61,.36,1) both;
      }
      /* Assistant messages get an accent hairline on the left. */
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 3px solid var(--dm-accent);
      }
      @keyframes dm-rise {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
      }

      /* ===================== Chat input ===================== */
      /* The fixed bottom bar has a hardcoded white background in dark mode —
         align it with the app background so the input floats cleanly. */
      [data-testid="stBottom"] > div { background: var(--dm-bg); }
      [data-testid="stChatInput"] {
        border-radius: var(--dm-r-md); border: 1px solid var(--dm-border);
        background: var(--dm-surface); box-shadow: var(--dm-shadow-md);
        transition: border-color .2s ease, box-shadow .2s ease;
      }
      /* Streamlit paints an inner grey wrapper + textarea; align both to the
         surface token so the field matches the theme in light and dark. */
      [data-testid="stChatInput"] > div,
      [data-testid="stChatInput"] textarea {
        background: var(--dm-surface) !important; color: var(--dm-text) !important;
      }
      [data-testid="stChatInput"] textarea::placeholder { color: var(--dm-text-mute) !important; }
      [data-testid="stChatInput"]:focus-within {
        border-color: var(--dm-accent); box-shadow: 0 0 0 3px var(--dm-accent-ring);
      }

      /* ===================== Source chips ===================== */
      .dm-sources { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
      .dm-source {
        display: inline-flex; align-items: center; gap: 5px;
        background: var(--dm-accent-soft); color: var(--dm-accent);
        border: 1px solid var(--dm-accent-ring);
        padding: 3px 11px; border-radius: 999px; font-size: .76rem; font-weight: 500;
      }
      .dm-source::before { content: "📄"; font-size: .8em; }

      /* ===================== Status pill ===================== */
      .dm-pill {
        display: inline-flex; align-items: center; gap: 7px;
        padding: 5px 13px; border-radius: 999px; font-size: .8rem; font-weight: 600;
      }
      .dm-pill .dot { width: 7px; height: 7px; border-radius: 50%; }
      .dm-pill-ok  { background: var(--dm-good-soft); color: var(--dm-good); }
      .dm-pill-ok  .dot { background: var(--dm-good); box-shadow: 0 0 0 0 var(--dm-good);
                          animation: dm-pulse 2s infinite; }
      .dm-pill-off { background: var(--dm-surface-2); color: var(--dm-text-mute); }
      .dm-pill-off .dot { background: var(--dm-text-mute); }
      @keyframes dm-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(16,185,129,.4); }
        70%  { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
        100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
      }

      /* ===================== Empty-state hero ===================== */
      .dm-hero {
        text-align: center; padding: 40px 28px; border-radius: var(--dm-r-lg);
        background: var(--dm-surface); border: 1px solid var(--dm-border);
        box-shadow: var(--dm-shadow-md); animation: dm-rise .32s ease both;
      }
      .dm-hero .icon {
        width: 60px; height: 60px; margin: 0 auto 16px; border-radius: 16px;
        display: grid; place-items: center; font-size: 30px;
        background: var(--dm-accent-soft);
      }
      .dm-hero h3 { font-size: 1.15rem; font-weight: 700; margin: 0 0 6px; color: var(--dm-text); }
      .dm-hero p  { color: var(--dm-text-soft); font-size: .95rem; margin: 0; }

      .dm-eyebrow {
        font-size: .72rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; color: var(--dm-text-mute); margin: 4px 0 10px;
      }

      /* ===================== Alerts (soften) ===================== */
      [data-testid="stAlert"] { border-radius: var(--dm-r-md); border: none; box-shadow: var(--dm-shadow-sm); }

      /* ===================== Motion preference ===================== */
      @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation: none !important; transition: none !important; }
      }
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


def _render_sources(sources: list[str]) -> None:
    """Render citation chips inside a flex container."""
    if not sources:
        return
    chips = "".join(f'<span class="dm-source">{s}</span>' for s in sources)
    st.markdown(f'<div class="dm-sources">{chips}</div>', unsafe_allow_html=True)


# --- Header ----------------------------------------------------------------
st.markdown(
    '<div class="dm-brand">'
    '<div class="dm-logo">📄</div>'
    '<div class="dm-title">DocuMind</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="dm-subtitle">Chat with your documents — private, '
    "grounded answers with citations.</div>",
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
        _render_sources(result.sources)
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
        st.markdown(
            '<div class="dm-hero">'
            '<div class="icon">📁</div>'
            "<h3>Add a document to begin</h3>"
            "<p>Upload a PDF, TXT, or Markdown file in the sidebar and click "
            "<b>Index documents</b> — then ask anything about it.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="dm-eyebrow">Try asking</div>', unsafe_allow_html=True)
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES, strict=True):
            if col.button(ex, key=f"ex-{ex}", width="stretch"):
                _ask(ex)
                st.rerun()

# --- Replay history --------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _render_sources(msg.get("sources", []))

# --- Input -----------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents…"):
    _ask(prompt)
    st.rerun()
