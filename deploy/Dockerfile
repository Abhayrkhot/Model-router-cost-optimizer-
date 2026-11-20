# Gateway image: the package with the `gateway` extra only (no torch / scikit-learn / pandas),
# the ONNX embedding model baked in, running as a non-root user.
FROM python:3.12-slim AS build
RUN pip install --no-cache-dir "uv==0.9.*"
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 FASTEMBED_CACHE_PATH=/app/models
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra gateway --no-editable
RUN .venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

FROM python:3.12-slim
RUN useradd --uid 10001 --create-home app
COPY --from=build /app/.venv /app/.venv
COPY --from=build --chown=10001 /app/models /app/models
ENV PATH=/app/.venv/bin:$PATH \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=2s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=1)"
ENTRYPOINT ["modelrouter", "serve"]
CMD ["--config", "/etc/modelrouter/gateway.yaml"]
