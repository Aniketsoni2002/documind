# Changelog

All notable changes to DocuMind are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Swappable LLM provider** via `DOCUMIND_LLM_PROVIDER` (`ollama` | `groq`).
  The new `groq` provider uses the hosted Groq API (free tier, Llama 3.1) so the
  app can run on hosts that can't run Ollama.
- **Streamlit Community Cloud deploy**: root `streamlit_app.py` entrypoint,
  `.streamlit/config.toml` theme, `requirements-cloud.txt` (CPU-only torch), and
  a `secrets.toml.example` template. Streamlit secrets are bridged to config env
  vars automatically.
- One-command Docker Compose stack now includes the **Streamlit UI** (port 8501)
  alongside the API and Ollama, with the configured LLM model pulled
  automatically on first start.
- `collection_count()` helper on the vector store, powering an "N chunks
  indexed" status pill in the UI.

### Changed
- Polished the Streamlit UI: gradient header, sidebar index-status pill, upload
  progress bar with per-file error handling, example-prompt buttons on the empty
  state, source-citation chips that persist across reruns, and a wide layout.

## [1.0.0] — 2026-07-18

### Added
- Local RAG pipeline: document loading, chunking, HuggingFace embeddings, and
  persistent ChromaDB vector store.
- Grounded question answering via a local Ollama LLM, with source citations and
  an explicit "I don't know" fallback to avoid hallucination.
- Three interfaces: Streamlit chat UI, FastAPI REST API, and a CLI
  (`documind ingest / ask / reset`).
- Typed, environment-overridable configuration via `pydantic-settings`.
- Test suite (13 tests), `ruff` linting, and multi-version GitHub Actions CI.
- Docker and Docker Compose support for one-command local deployment.
