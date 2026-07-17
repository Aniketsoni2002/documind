# DocuMind API container. Ollama runs on the host (or a sibling container);
# point DOCUMIND_OLLAMA_BASE_URL at it.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

# Embedding model is downloaded on first run; mount a volume to cache it.
EXPOSE 8000

CMD ["uvicorn", "documind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
