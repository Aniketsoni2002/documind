# Changelog

All notable changes to DocuMind are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
