# syntax=docker/dockerfile:1.7

# 1) DEPS STAGE: build the virtualenv from lockfile only
FROM python:3.13-slim@sha256:49b618b8afc2742b94fa8419d8f4d3b337f111a0527d417a1db97d4683cb71a6 AS deps
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.29@sha256:db9370c2b0b837c74f454bea914343da9f29232035aa7632a1b14dc03add9edb /uv /uvx /bin/

# Only copy lock/manifest so Docker cache stays hot unless deps change
COPY pyproject.toml uv.lock ./

# Build cached dependency venv strictly from lockfile
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy \
    uv sync --frozen --no-install-project --no-editable

# 2) FINAL STAGE: install the project into the prebuilt venv, then run
FROM python:3.13-slim@sha256:49b618b8afc2742b94fa8419d8f4d3b337f111a0527d417a1db97d4683cb71a6 AS final
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.29@sha256:db9370c2b0b837c74f454bea914343da9f29232035aa7632a1b14dc03add9edb /uv /uvx /bin/

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
