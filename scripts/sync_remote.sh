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
#   scripts/sync_remote.sh --setup      # sync + install what the remote is missing
#   scripts/sync_remote.sh --check      # prove both places encode IDENTICALLY
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

# The point of --check: the two halves must be the SAME setup, not merely both
# present. A drifted model or document format over there would place cold-started
# tracks slightly wrong and nothing downstream could tell — so prove the remote
# returns bit-identical vectors for the same text before trusting it.
run_check() {
  echo "=== parity check: $HOST vs local"
  COLDSTART_REMOTE="$REMOTE" "$ROOT/predictors/.venv/bin/python" - <<'PY'
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.environ.get("SYNC_ROOT", "."), "predictors"))
import seq_coldstart as cs
docs = ["Parity probe. Artist: Nobody. Album: Nothing (single, 2020). Genre: test.",
        "Segunda prueba con acentos: canción. Artist: Alguien. Genre: electronica argentina."]
b = cs.Basis()
remote = cs.remote_encoder(os.environ["COLDSTART_REMOTE"], b.text_model, lambda m: print(f"  {m}"))
if remote is None:
    print("  no remote configured"); raise SystemExit(1)
R = np.asarray(remote(docs)); L = np.asarray(
    cs.text_encoder(b.text_model).encode(docs, normalize_embeddings=False, show_progress_bar=False))
worst = float(np.abs(R - L).max())
for i in range(len(docs)):
    c = float(R[i] @ L[i] / (np.linalg.norm(R[i]) * np.linalg.norm(L[i])))
    print(f"  doc{i}: cos {c:.6f}  max|delta| {np.abs(R[i]-L[i]).max():.3e}")
print("  PARITY OK (bit-identical)" if worst == 0.0 else
      f"  PARITY WARN: max|delta| {worst:.3e} — the two sides are not the same model")
raise SystemExit(0 if worst == 0.0 else 1)
PY
}

setup_remote() {
  echo "=== setup $HOST"
  ssh -o BatchMode=yes "$HOST" "
    set -e
    $RPATH/predictors/.venv/bin/python -c 'import sentence_transformers' 2>/dev/null \
      || $RPATH/predictors/.venv/bin/pip install -q sentence-transformers
    $RPATH/predictors/.venv/bin/python -c 'import sentence_transformers as s, torch; \
      print(\"  sentence-transformers\", s.__version__, \"· torch\", torch.__version__)'
  "
}

DRY=(); MODE=sync
case "${1:-}" in
  -h|--help) usage 0 ;;
  --install-cron) install_cron "${2:-04:30}"; exit 0 ;;
  --dry-run) DRY=(--dry-run --itemize-changes) ;;
  --setup) MODE=setup ;;
  --check) MODE=check ;;
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
    echo "                 (run: scripts/sync_remote.sh --setup)"
  fi
fi

[[ "$MODE" == "setup" ]] && setup_remote
if [[ "$MODE" == "setup" || "$MODE" == "check" ]]; then
  SYNC_ROOT="$ROOT" run_check
fi
echo "=== done"
