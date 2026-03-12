# syntax=docker/dockerfile:1
# ── Stage 1: Base ─────────────────────────────────────
FROM python:3.10-slim AS base
WORKDIR /app
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-dev --no-interaction --no-ansi

# ── Stage 2: Train ────────────────────────────────────
FROM base AS train
COPY src/ ./src/
COPY configs/ ./configs/
COPY pipelines/ ./pipelines/
RUN poetry install --no-dev --no-interaction --no-ansi
ENTRYPOINT ["deeprl-recsys"]

# ── Stage 3: Serve (lightweight) ─────────────────────
FROM base AS serve
COPY src/ ./src/
COPY configs/serving.yaml ./configs/serving.yaml
RUN poetry install --no-dev --no-interaction --no-ansi
EXPOSE 8000
ENTRYPOINT ["deeprl-recsys", "serve"]
