# ── Base image ─────────────────────────────────────────────────────────────────
# Python 3.14 has no official Docker image yet; 3.12 is fully compatible.
FROM python:3.12-slim

WORKDIR /app

# ── System build tools (needed by some pip packages) ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (own layer so rebuilds are fast on code-only changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the Sentence-BERT model into the image ────────────────────────
# This avoids a runtime download on every fresh container start.
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy source code ───────────────────────────────────────────────────────────
# Docs/, Datasets/, and mlruns/ are intentionally excluded via .dockerignore
# and injected at runtime via docker-compose volume mounts.
COPY . .

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
