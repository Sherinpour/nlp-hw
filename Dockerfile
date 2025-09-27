# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# Set non-root user
ARG USER=app
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # model/data caches inside container
    HF_HOME=/opt/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/.cache/huggingface \
    NLTK_DATA=/opt/.cache/nltk \
    STANZA_RESOURCES_DIR=/opt/.cache/stanza \
    TOKENIZERS_PARALLELISM=false \
    MPLCONFIGDIR=/opt/.cache/matplotlib

# System deps for lxml/sentencepiece/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates tini \
    libxml2-dev libxslt1-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create user and dirs
RUN groupadd -g ${GID} ${USER} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER} \
 && mkdir -p /workspace ${HF_HOME} ${NLTK_DATA} ${STANZA_RESOURCES_DIR} ${TRANSFORMERS_CACHE} /opt/bin \
 && chown -R ${USER}:${USER} /workspace /opt/.cache

WORKDIR /workspace

# Copy only manifests first (better layer caching)
COPY --chown=${USER}:${USER} requirements.txt ./

# Install Python deps (CPU torch)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir flask gunicorn

# Add project files
COPY --chown=${USER}:${USER} src/ ./src/
COPY --chown=${USER}:${USER} configs/ ./configs/
COPY --chown=${USER}:${USER} notebooks/ ./notebooks/
COPY --chown=${USER}:${USER} app.py ./
COPY --chown=${USER}:${USER} docker/entrypoint.sh /opt/bin/entrypoint.sh
COPY --chown=${USER}:${USER} docker/downloads.sh /opt/bin/downloads.sh
RUN chmod +x /opt/bin/*.sh

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/reports /workspace/artifacts

USER ${USER}

# Expose port for web application
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
