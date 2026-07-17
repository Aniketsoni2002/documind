<div align="center">

# 📄 DocuMind

### Chat with your documents — 100% local, private, and free.

A production-grade **Retrieval-Augmented Generation (RAG)** application that lets you
ask natural-language questions about your PDFs, text and Markdown files and get
answers **grounded in your documents, with citations** — running entirely on your
own machine. No OpenAI key, no data leaving your laptop.

[![CI](https://github.com/Aniketsoni2002/documind/actions/workflows/ci.yml/badge.svg)](https://github.com/Aniketsoni2002/documind/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-13%20passing-brightgreen)

</div>

---

## ✨ Why this project

Most RAG demos are a single notebook that needs a paid API key and breaks the moment
you touch it. DocuMind is built the way you'd ship RAG in a real product:

- 🔒 **Fully local & private** — LLM via [Ollama](https://ollama.com), embeddings via
  HuggingFace `sentence-transformers`, vector store via ChromaDB. **Nothing is sent to a third party.**
- 🎯 **Grounded answers with citations** — the model is instructed to answer *only*
  from retrieved context and to say *"I don't know"* rather than hallucinate.
- 🧱 **Clean, layered architecture** — loaders, vector store, RAG chain, API and UI
  are cleanly separated and independently testable.
- 🖥️ **Three ways to use it** — a **Streamlit** chat UI, a **FastAPI** REST API, and a **CLI**.
- ✅ **Actually engineered** — typed config, structured logging, a **13-test suite**,
  **ruff** linting, a **multi-version CI pipeline**, and **Docker Compose** for one-command deploy.

---

## 🏗️ Architecture

```
                ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  Your docs ──▶ │   Loader &   │ ──▶ │  Embeddings  │ ──▶ │   ChromaDB   │
 (PDF/TXT/MD)   │   Chunker    │     │ (MiniLM,     │     │  (vectors)   │
                └──────────────┘     │  local)      │     └──────┬───────┘
                                     └──────────────┘            │
                                                        similarity search
                                                                 │
     "What is the refund policy?" ─────┐                         ▼
                                        │              ┌────────────────────┐
                                        └────────────▶ │   RAG Chain        │
                                                       │ (retrieve → prompt │
                                        ┌───────────── │  → Ollama LLM)     │
                                        │              └────────────────────┘
                                        ▼
                        "Refunds are allowed within 30 days.
                         Sources: policy.pdf"
```

**Layers** (`src/documind/`):

| Module | Responsibility |
|---|---|
| `core/loader.py` | Load PDF/TXT/MD → chunk with overlap, tag citations |
| `core/vectorstore.py` | HuggingFace embeddings + persistent ChromaDB collection |
| `core/rag.py` | The RAG chain: retrieve → grounded prompt → Ollama, return answer + sources |
| `core/ingest.py` | High-level "file → indexed" orchestration |
| `api/main.py` | FastAPI REST service (`/ask`, `/ingest`, `/health`) |
| `ui/app.py` | Streamlit chat interface |
| `cli.py` | `documind ingest / ask / reset` command line |
| `config.py` | Typed, env-overridable settings (pydantic-settings) |

---

## 🚀 Quickstart

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and a model pulled:

```bash
ollama pull llama3.2      # the local LLM DocuMind talks to
```

### 2. Install

```bash
git clone https://github.com/Aniketsoni2002/documind.git
cd documind
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Use it — pick your interface

**Streamlit chat UI** (easiest):
```bash
streamlit run src/documind/ui/app.py
```
Upload a document in the sidebar, click *Index*, then ask away.

**REST API**:
```bash
uvicorn documind.api.main:app --reload
# then, in another terminal:
curl -F "file=@mydoc.pdf" http://localhost:8000/ingest
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}'
```
Interactive API docs are auto-generated at **http://localhost:8000/docs**.

**CLI**:
```bash
documind ingest data/uploads/handbook.pdf
documind ask "What is the refund policy?"
```

---

## 🐳 Run everything with Docker

Ollama + the API, one command:

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.2   # first time only
```

---

## ⚙️ Configuration

Everything is configurable via environment variables (or a `.env` file — see
[`.env.example`](.env.example)). Sensible local defaults mean **it works out of the box**.

| Variable | Default | Description |
|---|---|---|
| `DOCUMIND_LLM_MODEL` | `llama3.2` | Ollama model name |
| `DOCUMIND_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `DOCUMIND_TOP_K` | `4` | Chunks retrieved per question |
| `DOCUMIND_CHUNK_SIZE` | `1000` | Characters per chunk |
| `DOCUMIND_CHUNK_OVERLAP` | `150` | Overlap between chunks |

---

## 🧪 Testing & quality

```bash
pytest                    # 13 tests, no Ollama/network required (fakes used)
ruff check src tests      # lint
```

The test suite covers loading/chunking, the RAG chain (with a fake LLM so it runs
offline in CI), and the FastAPI endpoints. CI runs on Python 3.10, 3.11 and 3.12.

---

## 🗺️ Roadmap

- [ ] Hybrid search (BM25 + dense) with reranking
- [ ] Conversational memory (multi-turn follow-ups)
- [ ] Streaming token responses in the UI
- [ ] Evaluation harness (RAGAS) for answer faithfulness

---

## 🛠️ Tech stack

**LangChain** · **Ollama** · **ChromaDB** · **HuggingFace sentence-transformers** ·
**FastAPI** · **Streamlit** · **Pydantic** · **pytest** · **ruff** · **Docker**

---

## 📄 License

MIT © Aniket Soni
