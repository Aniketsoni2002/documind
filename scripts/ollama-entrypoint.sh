#!/bin/sh
# Start the Ollama server, then pull the configured model once it's up so the
# stack is usable straight after `docker compose up` — no manual pull needed.
set -e

MODEL="${DOCUMIND_LLM_MODEL:-llama3.2}"

# Start the server in the background.
/bin/ollama serve &
SERVER_PID=$!

echo "⏳ Waiting for Ollama to be ready…"
until ollama list >/dev/null 2>&1; do
    sleep 1
done

echo "⬇️  Pulling model: ${MODEL} (first run only; cached afterwards)"
ollama pull "${MODEL}" || echo "⚠️  Could not pull ${MODEL}; pull it manually later."

echo "✅ Ollama ready with ${MODEL}"

# Hand control back to the server process so the container stays alive.
wait "${SERVER_PID}"
