"""Central configuration for DocuMind.

All tunables live here and can be overridden with environment variables so the
app runs the same way locally, in Docker, and in CI. Nothing here requires a
paid API key — the defaults target a fully local stack (Ollama + HuggingFace).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings, populated from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_prefix="DOCUMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider selection -------------------------------------------
    # "ollama" runs a fully local model (default, private). "groq" uses the
    # hosted Groq API (free tier) so the app can run on hosts that can't run
    # Ollama, e.g. Streamlit Community Cloud.
    llm_provider: str = Field(default="ollama")  # "ollama" | "groq"
    llm_temperature: float = Field(default=0.1)

    # --- LLM (local via Ollama) -------------------------------------------
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="llama3.2")

    # --- LLM (hosted via Groq) --------------------------------------------
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.1-8b-instant")

    # --- Embeddings (local via HuggingFace sentence-transformers) ---------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # --- Vector store (local ChromaDB) ------------------------------------
    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    collection_name: str = Field(default="documind")

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=150)

    # --- Retrieval --------------------------------------------------------
    top_k: int = Field(default=4)

    # --- Storage ----------------------------------------------------------
    upload_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")

    def ensure_dirs(self) -> None:
        """Create the directories the app writes to, if they don't exist."""
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (single source of truth per process)."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
