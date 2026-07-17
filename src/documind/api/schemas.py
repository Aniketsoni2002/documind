"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question.")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Override retrieved chunk count."
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    embedding_model: str
