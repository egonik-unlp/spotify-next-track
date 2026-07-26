#!/bin/sh
# lensing-server container entrypoint.
#
# The server runs its own additive DB schema migrations on startup. `migrate-data`
# is the idempotent file→Postgres backfill (seeds model definitions from
# models.toml + the dataset/model index from data/); running it every boot is
# safe and heals a fresh Postgres from the baked/mounted files. Non-fatal so a
# restored instance (DB already populated) still starts if the backfill no-ops.
set -e

: "${LENSING_PORT:=8096}"
: "${LENSING_COLLECTION:=spotify_tracks}"
: "${LENSING_DATABASE_URL:?LENSING_DATABASE_URL is required}"
: "${LENSING_QDRANT_URL:?LENSING_QDRANT_URL is required}"

echo "[entrypoint] migrate-data (idempotent file→Postgres backfill)…"
target/release/lensing-server --database-url "$LENSING_DATABASE_URL" migrate-data || \
  echo "[entrypoint] migrate-data returned non-zero (continuing)"

echo "[entrypoint] starting lensing-server on :$LENSING_PORT (qdrant=$LENSING_QDRANT_URL, collection=$LENSING_COLLECTION)"
exec target/release/lensing-server \
  --port "$LENSING_PORT" \
  --qdrant-url "$LENSING_QDRANT_URL" \
  --collection "$LENSING_COLLECTION" \
  --database-url "$LENSING_DATABASE_URL"
