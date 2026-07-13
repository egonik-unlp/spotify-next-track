#!/usr/bin/env bash
# Full, self-contained corpus refresh for this instance — raw Spotify export to a
# trainable Song-AE dataset. No external checkout required.
#
#   [1] corpus build : ingest -> enrich -> clean -> sessionize -> aggregates ->
#                      embed -> transitions -> ids -> upsert  (=> spotify_tracks;
#                      the binary `rotation` target is materialized by aggregates)
#   [2] AE chain     : embed_content -> fetch_spotify_meta ->
#                      fetch_audio_features --write -> build_song_ae --latent 64
#                      (=> spotify_tracks_content, spotify_tracks_song_ae)
#   [3] dataset      : POST the canonical Song-AE recipe to the lensing server
#
# Usage:  pipeline/refresh_corpus.sh [--skip-enrich] [--source DIR]
# Prereqs: pipeline/corpus/.venv (run pipeline/corpus/setup_venv.sh once),
#          predictors/.venv, a running lensing server (:8096), reachable Qdrant.
set -euo pipefail
cd "$(dirname "$0")/.."          # instance root
CORPUS_PY=pipeline/corpus/.venv/bin/python
AE_PY=predictors/.venv/bin/python
SERVER=${LENSING_SERVER:-http://localhost:8096}

[ -x "$CORPUS_PY" ] || { echo "missing $CORPUS_PY — run pipeline/corpus/setup_venv.sh first"; exit 1; }
[ -x "$AE_PY" ] || { echo "missing $AE_PY (predictors venv)"; exit 1; }

echo "===== [1/3] corpus build -> spotify_tracks ====="
"$CORPUS_PY" -m pipeline.corpus.run_all "$@"

echo "===== [2/3] AE chain -> spotify_tracks_content / _song_ae ====="
"$AE_PY" pipeline/embed_content.py
"$AE_PY" pipeline/fetch_spotify_meta.py
"$AE_PY" pipeline/fetch_audio_features.py --write   # --write is REQUIRED (else no af_*)
"$AE_PY" pipeline/build_song_ae.py --latent 64

echo "===== [3/3] build Song-AE dataset ====="
curl -s -X POST "$SERVER/api/datasets" -H 'Content-Type: application/json' \
  -d @pipeline/corpus/dataset_recipe.json
echo
echo "===== DONE. Dataset build is async (poll /api/builds/{id}); then retrain"
echo "       the good definitions on the new dataset_id via POST /api/runs. ====="
