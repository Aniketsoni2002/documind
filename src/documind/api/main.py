"""FastAPI application exposing the RAG pipeline over HTTP.

Endpoints
---------
GET  /health          -> liveness + which models are configured
POST /ingest          -> upload a document (multipart) and index it
POST /ask             -> ask a question, get a grounded answer + sources
DELETE /documents     -> clear the index
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from documind.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestResponse,
)
from documind.config import get_settings
from documind.core.ingest import ingest_file
from documind.core.loader import SUPPORTED_SUFFIXES, UnsupportedFileError
from documind.core.rag import answer_question
from documind.core.vectorstore import clear_collection
from documind.utils.logging import get_logger

logger = get_logger("api")

app = FastAPI(
    title="DocuMind API",
    description="Local, private Retrieval-Augmented Generation over your documents.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    settings = get_settings()
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        allowed = sorted(SUPPORTED_SUFFIXES)
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type {suffix!r}. Allowed: {allowed}",
        )

    dest = settings.upload_dir / filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        chunks = ingest_file(dest)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    return IngestResponse(filename=filename, chunks_indexed=chunks)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    answer = answer_question(req.question, top_k=req.top_k)
    return AskResponse(answer=answer.text, sources=answer.sources)


@app.delete("/documents", status_code=204)
def reset() -> None:
    clear_collection()
