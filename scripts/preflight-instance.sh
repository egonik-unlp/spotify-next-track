#!/usr/bin/env bash
# preflight-instance.sh — fast, non-fatal check run before `zig build serve`.
#
# A populated instance keeps its binary artifacts under data/ (gitignored, NOT
# in the repo). If they're missing, this is almost certainly a fresh checkout
# on a new machine that still needs its Postgres + Qdrant + data/ bundle
# restored. We only WARN and point at restore-instance.sh — we never block the
# server or touch any store (set PREFLIGHT_STRICT=1 to make it fatal in CI).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -e data/models ] && [ -e data/best-models.json ]; then
  exit 0   # looks populated — say nothing, stay out of the way
fi

printf '\033[1;33m
┌─ instance not initialized ─────────────────────────────────────────────┐
│ data/ has no model artifacts — this looks like a fresh machine.         │
│ The corpus + metadata + artifacts live in a separate bundle, not git.   │
│                                                                          │
│   Restore them first:                                                    │
│     scripts/restore-instance.sh /path/to/instance-bundle                │
│   or:  DUMP_BASE_URL=https://<r2-base>/ scripts/restore-instance.sh      │
│                                                                          │
│ (Create a bundle from a working machine with scripts/export-instance.sh)│
└──────────────────────────────────────────────────────────────────────────┘\033[0m
' >&2

[ "${PREFLIGHT_STRICT:-0}" = "1" ] && exit 1
exit 0
