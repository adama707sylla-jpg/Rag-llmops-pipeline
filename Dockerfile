# ── Stage 1 : builder 
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2 : runtime 
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    OLLAMA_HOST=host-gateway \
    OLLAMA_PORT=11434 \
    LLM_MODEL=phi3:mini

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY api/        ./api/
COPY rag/        ./rag/
COPY ingestion/  ./ingestion/
COPY setup.py    .

RUN pip install -e . --no-deps

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
