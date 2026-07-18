#!/usr/bin/env bash
# export-instance.sh — capture this running instance into a portable bundle.
#
# Produces a self-contained directory (default ./instance-bundle/) holding:
#   postgres.dump                 metadata DB (pg_dump -Fc, no owners/privs)
#   qdrant-<collection>.snapshot  one Qdrant snapshot per corpus collection
#   data-essentials.tar.zst       data/{models,datasets,best-models.json}
#   manifest.json                 filenames, sizes, sha256, collections, git rev
#
# Upload that directory to Cloudflare R2 (see the tail of this script), or pack
# it into a single tarball with --tarball, and a teammate restores the whole
# instance on a fresh machine with scripts/restore-instance.sh.
#
# NOTE: the historical data/runs/ tree (finished training-run artifact dirs) is
# intentionally NOT bundled — restored runs still appear in the UI from
# Postgres, but their per-run artifact dirs are absent. Set BUNDLE_RUNS=1 to
# include them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- args --------------------------------------------------------------------
# --tarball / -t : also pack the bundle dir into a single ./instance-bundle.tar
#                  (a Google-Drive-friendly one-file upload; same as TARBALL=1)
TARBALL="${TARBALL:-0}"
for arg in "$@"; do
  case "$arg" in
    --tarball|-t) TARBALL=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ---- config (override via env) ------------------------------------------------
OUT_DIR="${OUT_DIR:-$ROOT/instance-bundle}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6337}"
# Corpus + manual-track collections, space-separated. Mirrors the pipeline's
# AE chain: the base corpus (spotify_tracks) → content AE (spotify_tracks_content)
# → Song-AE (spotify_tracks_song_ae, the canonical dataset source), plus the
# cleaned variant and the manual-track collection. Collections absent on this
# instance are skipped, not fatal (below), so an instance without manual tracks
# or without the cleaned variant still exports cleanly.
QDRANT_COLLECTIONS="${QDRANT_COLLECTIONS:-spotify_tracks spotify_tracks_content spotify_tracks_song_ae spotify_tracks-clean-one manual-tracks}"
PG_SERVICE="${PG_SERVICE:-postgres}"           # docker-compose service name
# Instances that borrow a DB inside another instance's running postgres
# container (shared-container setup) set PG_CONTAINER to that container name;
# the dump then uses `docker exec <container>` instead of `docker compose exec`.
PG_CONTAINER="${PG_CONTAINER:-}"
PG_USER="${LENSING_DB_USER:-pg}"
PG_DB="${LENSING_DB_NAME:-lensing}"
BUNDLE_RUNS="${BUNDLE_RUNS:-0}"

log()  { printf '\033[1;36m[export]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[export]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[export] %s\033[0m\n' "$*" >&2; exit 1; }

command -v docker  >/dev/null || die "docker not found"
command -v curl    >/dev/null || die "curl not found"
command -v python3 >/dev/null || die "python3 not found"
command -v sha256sum >/dev/null || die "sha256sum not found"

if command -v zstd >/dev/null; then
  TAR_COMP=(--use-compress-program=zstd); DATA_ARCHIVE="data-essentials.tar.zst"
else
  warn "zstd not found — falling back to gzip"
  TAR_COMP=(--gzip); DATA_ARCHIVE="data-essentials.tar.gz"
fi

mkdir -p "$OUT_DIR"
log "bundle → $OUT_DIR"

# ---- 1. Postgres -------------------------------------------------------------
log "dumping Postgres ($PG_DB) from compose service '$PG_SERVICE'…"
if [ -n "$PG_CONTAINER" ]; then
  docker exec -i "$PG_CONTAINER" \
    pg_dump -U "$PG_USER" -d "$PG_DB" -Fc --no-owner --no-privileges \
    > "$OUT_DIR/postgres.dump" \
    || die "pg_dump failed via container '$PG_CONTAINER' (db '$PG_DB')"
else
  docker compose exec -T "$PG_SERVICE" \
    pg_dump -U "$PG_USER" -d "$PG_DB" -Fc --no-owner --no-privileges \
    > "$OUT_DIR/postgres.dump" \
    || die "pg_dump failed — is the database up? (zig build db-up)"
fi
log "  postgres.dump $(du -h "$OUT_DIR/postgres.dump" | cut -f1)"

# ---- 2. Qdrant snapshots -----------------------------------------------------
QDRANT_JSON_COLS=""
for col in $QDRANT_COLLECTIONS; do
  # Skip a collection that isn't present on this instance (e.g. manual-tracks
  # before any manual entry exists) instead of aborting the whole export. Only
  # the collections actually snapshotted are recorded in the manifest, so
  # restore stays consistent. A genuine connection failure still aborts.
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$QDRANT_URL/collections/$col") \
    || die "Qdrant unreachable at $QDRANT_URL"
  if [ "$code" = "404" ]; then
    warn "collection '$col' absent — skipping"
    continue
  fi
  [ "$code" = "200" ] || die "unexpected HTTP $code querying collection '$col' at $QDRANT_URL"
  log "snapshotting Qdrant collection '$col'…"
  snap=$(curl -fsS -X POST "$QDRANT_URL/collections/$col/snapshots" \
           | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["name"])') \
    || die "snapshot create failed for '$col' — is Qdrant at $QDRANT_URL?"
  out="$OUT_DIR/qdrant-$col.snapshot"
  curl -fsS "$QDRANT_URL/collections/$col/snapshots/$snap" -o "$out" \
    || die "snapshot download failed for '$col'"
  # tidy the server-side snapshot we just created
  curl -fsS -X DELETE "$QDRANT_URL/collections/$col/snapshots/$snap" >/dev/null || true
  log "  qdrant-$col.snapshot $(du -h "$out" | cut -f1)"
  QDRANT_JSON_COLS="$QDRANT_JSON_COLS\"$col\","
done

# ---- 3. data/ essentials -----------------------------------------------------
DATA_PATHS=()
[ -e data/best-models.json ] && DATA_PATHS+=(data/best-models.json)
for d in models datasets cli-runs seq; do [ -e "data/$d" ] && DATA_PATHS+=("data/$d"); done
if [ "$BUNDLE_RUNS" = "1" ]; then
  [ -e data/runs ] && DATA_PATHS+=(data/runs)
  warn "BUNDLE_RUNS=1 — including data/runs/ (large)"
fi
log "archiving ${DATA_PATHS[*]}…"
tar "${TAR_COMP[@]}" -cf "$OUT_DIR/$DATA_ARCHIVE" "${DATA_PATHS[@]}" \
  || die "tar failed"
log "  $DATA_ARCHIVE $(du -h "$OUT_DIR/$DATA_ARCHIVE" | cut -f1)"

# ---- 4. manifest -------------------------------------------------------------
log "writing manifest.json…"
GIT_REV=$(git rev-parse HEAD 2>/dev/null || echo unknown)
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
(
  cd "$OUT_DIR"
  files_json=""
  for f in postgres.dump "$DATA_ARCHIVE" qdrant-*.snapshot; do
    [ -e "$f" ] || continue
    sha=$(sha256sum "$f" | cut -d' ' -f1)
    sz=$(stat -c%s "$f")
    files_json="$files_json{\"name\":\"$f\",\"bytes\":$sz,\"sha256\":\"$sha\"},"
  done
  cat > manifest.json <<JSON
{
  "created_at": "$STAMP",
  "git_rev": "$GIT_REV",
  "bundle_scope": "$([ "$BUNDLE_RUNS" = 1 ] && echo full || echo essentials)",
  "postgres": { "db": "$PG_DB", "dump": "postgres.dump" },
  "qdrant": { "collections": [ ${QDRANT_JSON_COLS%,} ] },
  "data_archive": "$DATA_ARCHIVE",
  "files": [ ${files_json%,} ]
}
JSON
)
python3 -m json.tool "$OUT_DIR/manifest.json" >/dev/null || die "manifest.json is not valid JSON"

log "done. Bundle contents:"
ls -lh "$OUT_DIR"

# ---- 5. (optional) single tarball for Google Drive et al. --------------------
# The files inside are already compressed, so the outer wrapper is a plain tar
# (no double-compression) holding the bundle at top level (./manifest.json …).
if [ "$TARBALL" = "1" ]; then
  TARBALL_OUT="${TARBALL_OUT:-$ROOT/instance-bundle.tar}"
  log "packing single upload tarball → $TARBALL_OUT…"
  tar -C "$OUT_DIR" -cf "$TARBALL_OUT" . || die "tar failed"
  log "  $(basename "$TARBALL_OUT") $(du -h "$TARBALL_OUT" | cut -f1)"
  cat <<EOF

────────────────────────────────────────────────────────────────────────────
Single-file bundle ready: $TARBALL_OUT
Upload it to Google Drive (or any host), then restore on the other machine:

  scripts/restore-instance.sh /path/to/instance-bundle.tar
  (or: zig build restore-instance -- /path/to/instance-bundle.tar)

restore-instance.sh auto-detects a .tar/.tar.zst/.tar.gz file and unpacks it.
────────────────────────────────────────────────────────────────────────────
EOF
  exit 0
fi

cat <<EOF

────────────────────────────────────────────────────────────────────────────
Next: host the bundle on Cloudflare R2 so teammates can pull it.

  # one-time: create a private bucket
  wrangler r2 bucket create spotify-engagement-dumps

  # upload every file in the bundle
  for f in "$OUT_DIR"/*; do
    wrangler r2 object put "spotify-engagement-dumps/\$(basename "\$f")" --file "\$f"
  done

Then on the other machine, point restore-instance.sh at the bundle. Either:
  • a local copy:    scripts/restore-instance.sh /path/to/instance-bundle
  • or an HTTP base: DUMP_BASE_URL=https://<r2-public-or-presigned-base>/ \\
                       scripts/restore-instance.sh

(For a private bucket, generate presigned URLs or run \`wrangler r2 object get\`
into a local dir first, then pass that dir.)
────────────────────────────────────────────────────────────────────────────
EOF
