# syntax=docker/dockerfile:1.7

# 1) DEPS STAGE: build the virtualenv from lockfile only
FROM python:3.14-slim@sha256:fa0acdcd760f0bf265bc2c1ee6120776c4d92a9c3a37289e17b9642ad2e5b83b AS deps
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Only copy lock/manifest so Docker cache stays hot unless deps change
COPY pyproject.toml uv.lock ./

# Build cached dependency venv strictly from lockfile
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy \
    uv sync --frozen --no-install-project --no-editable

# 2) FINAL STAGE: install the project into the prebuilt venv, then run
FROM python:3.14-slim@sha256:fa0acdcd760f0bf265bc2c1ee6120776c4d92a9c3a37289e17b9642ad2e5b83b AS final
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Bring in the ready-to-go dependency venv
COPY --from=deps /app/.venv /app/.venv

# Copy source
COPY . .

# Install the project itself without re-resolving dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    uv sync --frozen --no-editable

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

ENTRYPOINT ["engramcp"]
