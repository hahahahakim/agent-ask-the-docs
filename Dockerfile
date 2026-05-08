FROM python:3.12-slim

# Run as a non-root user
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
# Install dependencies and pre-bake the ONNX embedding model so the
# container never downloads it at runtime (offline-safe after build).
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()(['warmup'])"

# Copy application code (chown to appuser so it can write data/ at runtime)
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Send SIGTERM so uvicorn shuts down gracefully instead of being force-killed.
STOPSIGNAL SIGTERM

# IMPORTANT: --workers 1 is required.
# MemorySaver is an in-process Python dict. Multiple workers each get their
# own isolated memory space, breaking conversation continuity across requests.
# To scale horizontally, first migrate get_checkpointer() to SqliteSaver or
# a Redis-backed checkpointer in core/persistence.py.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "asyncio", \
     "--timeout-graceful-shutdown", "10"]
