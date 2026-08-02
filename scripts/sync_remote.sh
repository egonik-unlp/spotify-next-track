#!/usr/bin/env bash
# Push this instance's code + frozen artifacts to the remote worker.
#
# The remote (`homeserver`, ~/lensing-worker) runs two jobs for us: the lensing
# distributed training worker, and — since the cold-start work — the MiniLM text
# encode that `predictors/seq_coldstart.py --remote` ships over ssh. Both need the
# predictor sources to match this checkout, because the remote runs THE SAME module
# file. A stale copy there is the failure mode this exists to prevent: it would
# encode with a drifted document format and place cold tracks slightly wrong, which
# nothing downstream could detect.
#
# ONE DIRECTION ONLY: local → remote. The remote is a compute node, not a source of
# truth; nothing it produces is copied back here (runs report to the hub over the
# API). --delete is deliberately NOT passed, so remote-only state (its venv, the
# worker log, its own data/) survives.
#
#   scripts/sync_remote.sh              # sync
#   scripts/sync_remote.sh --dry-run    # show what would change
#   scripts/sync_remote.sh --install-cron [HH:MM]   # daily, default 04:30
#
# Cron installs a line calling this script; output goes to the log below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${COLDSTART_REMOTE:-homeserver:~/lensing-worker}"
HOST="${REMOTE%%:*}"
RPATH="${REMOTE#*:}"
LOG="${SYNC_REMOTE_LOG:-$HOME/.cache/lensing-sync-remote.log}"

# What the remote actually needs. NOT the whole repo: data/ is tens of GB of run
# dirs and model snapshots the worker fetches from the hub on demand, and .git,
# node_modules and the venvs are either huge or platform-specific.
PATHS=(
  "predictors/"
  "pipeline/"
  "registry.toml"
  "domain.toml"
)
EXCLUDES=(
  --exclude ".venv" --exclude "__pycache__" --exclude "*.pyc"
  --exclude "pipeline/corpus/.venv" --exclude "pipeline/corpus/cache"
  --exclude "pipeline/corpus/account-data" --exclude "pipeline/corpus/out"
  --exclude "pipeline/corpus/master_df.csv"
)

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

install_cron() {
  local at="${1:-04:30}" hh mm line
  hh="${at%%:*}"; mm="${at##*:}"
  [[ "$hh" =~ ^[0-9]{1,2}$ && "$mm" =~ ^[0-9]{1,2}$ ]] || { echo "bad time: $at (want HH:MM)" >&2; exit 2; }
  line="${mm#0} ${hh#0} * * * $ROOT/scripts/sync_remote.sh >> $LOG 2>&1"
  # Idempotent: drop any previous entry for this script before adding.
  local current
  current="$(crontab -l 2>/dev/null | grep -vF "scripts/sync_remote.sh" || true)"
  printf '%s\n%s\n' "$current" "$line" | grep -v '^$' | crontab -
  echo "installed: $line"
  crontab -l | grep -F "sync_remote.sh"
}

DRY=()
case "${1:-}" in
  -h|--help) usage 0 ;;
  --install-cron) install_cron "${2:-04:30}"; exit 0 ;;
  --dry-run) DRY=(--dry-run --itemize-changes) ;;
  "") ;;
  *) echo "unknown argument: $1" >&2; usage 2 ;;
esac

echo "=== $(date -Is) sync $ROOT → $HOST:$RPATH"
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" true 2>/dev/null; then
  echo "remote $HOST unreachable — nothing done (this is not an error: the local"
  echo "path stays authoritative and seq_coldstart falls back to encoding here)"
  exit 0
fi

cd "$ROOT"
rsync -az --human-readable "${DRY[@]}" "${EXCLUDES[@]}" \
  --out-format='  %o %n' \
  "${PATHS[@]}" --relative \
  "$HOST:$RPATH/"

# The remote half of the cold-start encoder must be able to run at all; say so
# plainly rather than discovering it mid-journey.
if [[ ${#DRY[@]} -eq 0 ]]; then
  if ssh -o BatchMode=yes "$HOST" \
      "$RPATH/predictors/.venv/bin/python -c 'import sentence_transformers' 2>/dev/null"; then
    echo "  remote encode: ready"
  else
    echo "  remote encode: sentence-transformers MISSING on $HOST"
    echo "                 ($RPATH/predictors/.venv/bin/pip install sentence-transformers)"
  fi
fi
echo "=== done"
