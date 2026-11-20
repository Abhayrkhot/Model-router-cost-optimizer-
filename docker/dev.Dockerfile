# Development image: Python 3.12 + uv.  The project's virtualenv lives in the bind-mounted
# repo (.venv), so dependency changes don't need an image rebuild.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir "uv==0.9.*"

ENV UV_LINK_MODE=copy \
    UV_CACHE_DIR=/cache/uv \
    HF_HOME=/cache/hf \
    FASTEMBED_CACHE_PATH=/cache/fastembed \
    PYTHONUNBUFFERED=1

WORKDIR /src
