# ── Stage 1: Build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ src/

# Install into a virtual env so we can copy it cleanly
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir .


# ── Stage 2: Production image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="hedgefund-team"
LABEL description="HedgeFund AI Options Trading Agent"

# Non-root user for production
RUN groupadd --gid 1000 hedgefund && \
    useradd --uid 1000 --gid hedgefund --shell /bin/bash --create-home hedgefund

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source and config
COPY src/ src/
COPY config/ config/

# Ensure venv is on PATH
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HEDGEFUND_ENV=production

# Switch to non-root
USER hedgefund

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "-m", "hedgefund"]
CMD ["dashboard"]
