ARG PYTHON_VERSION=3.12

# ---- build stage ---------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir --upgrade pip build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel --outdir /wheels

# ---- runtime stage -------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Non-root user
RUN groupadd -r thought && useradd -r -g thought -d /app -s /bin/bash thought
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    THOUGHT_DB_PATH=/data/thought.db

COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl 'thought-mcp[mcp,sqlite-vec]' \
    && rm -rf /wheels

RUN mkdir -p /data && chown -R thought:thought /data /app
USER thought

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sqlite3, os; sqlite3.connect(os.environ.get('THOUGHT_DB_PATH','/data/thought.db')).execute('SELECT 1')" || exit 1

ENTRYPOINT ["thought"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765"]
