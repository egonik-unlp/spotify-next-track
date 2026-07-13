"""Pathfinder configuration.

Endpoints and the service port default to standalone values but are overridable
via env vars so the lensing-server can spawn `pathfinder.service` as a sidecar
pointed at whatever Qdrant URL / port it is itself using (see
crates/lensing-server/src/pathfinder.rs).
"""
import os
from pathlib import Path

QDRANT_URL = os.environ.get("PATHFINDER_QDRANT_URL", "http://localhost:6335")
# Defaults for standalone runs; the lensing-server injects COLLECTION + API at
# sidecar spawn (it does NOT inject MODEL, so this default must be a real model).
QDRANT_COLLECTION = os.environ.get("PATHFINDER_COLLECTION", "spotify_tracks_song_ae")
LENSING_API = os.environ.get("PATHFINDER_LENSING_API", "http://localhost:8096")
MODEL_NAME = os.environ.get("PATHFINDER_MODEL", "rotation-fit")

# Port the API-only service binds (loopback). The Rust proxy forwards here.
SERVICE_PORT = int(os.environ.get("PATHFINDER_PORT", "8097"))

# Transition + context tables. Shipped alongside the package (copied from
# pipeline/out/transitions.json in the SpotifyData repo); regenerate there with
# `python -m pipeline.transitions` and re-copy to refresh.
TRANSITIONS_JSON = Path(__file__).resolve().parent / "transitions.json"

KNN_K = 20

# A* cost weights:
#   cost(u->v) = W_DIST * cos_dist(u,v)            stay near on the taste map
#              + W_FIT  * (1 - habitfit(v))         land on tracks you engage with
#              + W_DIV  * genre_jump_penalty        don't lurch between genres
#              + W_TRANS* (1 - transition(u->v))    follow how you actually listen (#7)
#              + W_CTX  * (1 - context_fit(v|ctx))  fit the time-of-day / shuffle mood (#8)
# The W_TRANS / W_CTX terms are min-max normalized across the candidate next
# hops at each expansion, so they express a *relative* preference among the
# reachable neighbors rather than an absolute scale.
W_DIST = 1.0
W_FIT = 0.5
W_DIV = 0.5
W_TRANS = 0.6
W_CTX = 0.4

# Transition back-off: track-level bigram evidence is trusted in proportion to
# n_u / (n_u + TRANSITION_BACKOFF_K); below that it defers to the (denser)
# artist- and genre-level conditional transition tendencies.
TRANSITION_BACKOFF_K = 8.0
TRANSITION_ARTIST_WEIGHT = 0.6   # artist vs genre split of the backed-off prior
TRANSITION_GENRE_WEIGHT = 0.4

# Diversity constraints
MAX_CONSECUTIVE_ARTIST = 2     # hard: no 3 same-artist tracks in a row
ARTIST_SHARE_DIVISOR = 4       # hard: <= ceil(length/4) tracks per artist
GENRE_JUMP_PENALTY = 0.15      # soft: charged when genre changes vs previous hop
