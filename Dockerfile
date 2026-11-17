# Build dependencies in an isolated stage; runtime contains only the app and its Python environment.
FROM python:3.12-slim AS builder

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv "$VIRTUAL_ENV"
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    HOME=/tmp \
    NEXUSIQ_DATA_DIR=/app/data/runtime

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 nexusiq \
    && useradd --system --uid 10001 --gid nexusiq --home-dir /app --no-create-home nexusiq \
    && mkdir -p /app/data/runtime \
    && chown -R nexusiq:nexusiq /app/data/runtime

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=nexusiq:nexusiq app/ ./app/
COPY --chown=nexusiq:nexusiq nexusiq/ ./nexusiq/
COPY --chown=nexusiq:nexusiq data/knowledge_base/ ./data/knowledge_base/

USER 10001:10001
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8501') + '/_stcore/health', timeout=3)"]
CMD ["sh", "-c", "exec streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
