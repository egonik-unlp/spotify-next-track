# syntax=docker/dockerfile:1.7
#
# Docker image for the spotify-next-track lensing instance: one container that
# runs `lensing-server` (HTTP API + the built React UI) and spawns the Python
# predictor subprocesses. Postgres and Qdrant are
# separate services (see deploy/docker-compose.yml).
#
# Multi-stage: Rust workspace → Vite UI → slim Python runtime that carries the
# predictor venv. Build context is the REPO ROOT:
#   docker build -f deploy/Dockerfile -t ghcr.io/egonik-unlp/spotify-nexttrack .

# ── Stage 1: Rust workspace (server, pipeline, predictor binaries) ────────────
FROM rust:1-bookworm AS rust-build
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends \
      pkg-config libssl-dev cmake clang && rm -rf /var/lib/apt/lists/*
# Only the files cargo needs, so the layer cache survives UI/predictor edits.
# domain.toml is baked into lensing-core at compile time (include_str! in
# domain.rs), so it must be present at the workspace root for the build.
COPY Cargo.toml Cargo.lock domain.toml ./
COPY crates ./crates
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --workspace && \
    mkdir -p /out && \
    find target/release -maxdepth 1 -type f -executable -exec cp {} /out/ \;

# ── Stage 2: Vite + React UI → ui/dist (served by lensing-server) ─────────────
FROM node:22-bookworm AS ui-build
WORKDIR /ui
COPY ui/package.json ui/package-lock.json* ./
RUN npm install --no-fund --no-audit
COPY ui/ ./
RUN npm run build

# ── Stage 3: runtime — slim Python + the predictor venv ───────────
FROM python:3.14-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# libgomp1: OpenMP runtime for torch/xgboost/lightgbm. curl: container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 curl ca-certificates && rm -rf /var/lib/apt/lists/*

# predictors/.venv — mirrors `zig build py-setup` (PyTorch CPU wheel index).
COPY predictors/requirements.txt ./predictors/requirements.txt
RUN python3 -m venv predictors/.venv && \
    predictors/.venv/bin/pip install --no-cache-dir -r predictors/requirements.txt \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple

# App source + built artifacts. cwd is the repo root the server expects
# (registry.toml, domain.toml, models.toml, ui/dist, predictors/, data/).
COPY predictors/ ./predictors/
COPY registry.toml domain.toml models.toml ./
COPY scripts/ ./scripts/
COPY --from=rust-build /out/ ./target/release/
COPY --from=ui-build /ui/dist ./ui/dist
COPY deploy/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# data/ is a mounted volume at runtime (datasets, runs, model snapshots).
EXPOSE 8096
ENTRYPOINT ["./entrypoint.sh"]
