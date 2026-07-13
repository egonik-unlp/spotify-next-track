#!/usr/bin/env bash
# One-time: build the corpus-build venv. gensim (word2vec co-listening vectors)
# requires Python <3.13, so this venv is SEPARATE from predictors/.venv (3.14).
set -euo pipefail
cd "$(dirname "$0")"   # pipeline/corpus

PY312="$(command -v python3.12 || true)"
[ -n "$PY312" ] || PY312="$HOME/.local/share/mise/installs/python/3.12.13/bin/python3"
[ -x "$PY312" ] || { echo "need a Python 3.12 interpreter (gensim has no 3.13+ wheels)"; exit 1; }

"$PY312" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "corpus venv ready: $(.venv/bin/python --version)"
