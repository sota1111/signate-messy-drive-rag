# Cloud Run image for the messy-drive RAG backend.
# Build context is the REPO ROOT (see scripts/deploy.sh) so we can COPY src/, config/,
# the prebuilt index, and the minimal glossary folder.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    INDEX_DIR=/app/index_store \
    CORPUS_DIR=/app/corpus

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Application code
COPY config /app/config
COPY src /app/src
COPY backend/app /app/app

# Prebuilt retrieval index + minimal glossary corpus (staged by scripts/deploy.sh)
COPY index_store /app/index_store
COPY backend/_corpus /app/corpus

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
