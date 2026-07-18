"""Shared paths and constants for the corpus-build pipeline.

Vendored into this instance (from the SpotifyData pathfinder repo) so the
instance is self-contained: a raw Spotify "Extended streaming history" export
can be turned into the `spotify_tracks` Qdrant corpus here, with no external
checkout. Paths default to in-repo locations and are env-overridable.
"""
import os
from pathlib import Path

# .../<instance>/pipeline/corpus/config.py
PKG_DIR = Path(__file__).resolve().parent              # pipeline/corpus
REPO_ROOT = PKG_DIR.parents[1]                          # instance root

# ---- Raw Spotify account export (the GDPR "Extended streaming history") ----
#
# A fresh export's "Extended streaming history" folder always contains the
# COMPLETE history (Spotify re-sends everything, not just the delta), so
# `ingest` rebuilds master_df.csv from scratch on every run. To inject newer
# data: drop the fresh export in place of `extended-2026/` (or point ingest at
# it with `--source`, or set LENSING_CORPUS_SOURCE) and re-run the pipeline.
#
# Both export filename shapes are accepted: the older `endsong_*.json` and the
# newer `Streaming_History_Audio_*.json` (same rich schema, just renamed). The
# lean "Account data" StreamingHistory*.json is NOT a play source — it lacks the
# track URI and reason_start/reason_end fields the model needs.
EXTENDED_DIR = Path(
    os.environ.get(
        "LENSING_CORPUS_SOURCE",
        REPO_ROOT / "extended-2026" / "Spotify Extended Streaming History",
    )
)
ENDSONG_GLOBS = ["endsong*.json", "Streaming_History_Audio_*.json"]
# Back-compat alias (older code referenced a single glob).
ENDSONG_GLOB = ENDSONG_GLOBS[0]

OUT_DIR = PKG_DIR / "out"
MASTER_CSV = PKG_DIR / "master_df.csv"

# Spotify Web API metadata caches (large; gitignored — regenerable via `enrich`).
CACHE_DIR = PKG_DIR / "cache"
ARTIST_JSON = CACHE_DIR / "data.json"
TRACK_DATA_JSON = CACHE_DIR / "track_data_block_1.json"

# Deliberate-curation signals (small; committed). Joined onto per-track
# aggregates by track URI.
ACCOUNT_DIR = PKG_DIR / "account-data"
LIBRARY_JSON = ACCOUNT_DIR / "YourLibrary.json"
PLAYLISTS_JSON = ACCOUNT_DIR / "Playlist1.json"
ONREPEAT_JSON = ACCOUNT_DIR / "OnRepeatContents.json"

PLAYS_PARQUET = OUT_DIR / "plays.parquet"
SESSIONS_PARQUET = OUT_DIR / "sessions.parquet"
TRACK_FEATURES_PARQUET = OUT_DIR / "track_features.parquet"
TRACK_VECTORS_NPY = OUT_DIR / "track_vectors.npy"
TRACK_VECTOR_URIS_JSON = OUT_DIR / "track_vector_uris.json"
ID_MAP_JSON = OUT_DIR / "id_map.json"
TRANSITIONS_JSON = OUT_DIR / "transitions.json"

# Ensure the gitignored working dirs exist (fresh clones won't have them).
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cleaning / sessionization
SKIP_THRESHOLD_MS = 30_000      # Spotify's own "stream" threshold
SESSION_GAP_MINUTES = 30

# ---- Transition model (next-track + context; consumed by the pathfinder) ----
#
# A transition u->v is weighted by how the *destination* play ended: a track
# you let finish (or replayed) is a satisfying place to land; one you skipped
# forward off is not. reason_end values absent here default to NEUTRAL_SAT.
REASON_END_SATISFACTION = {
    "trackdone": 1.0,   # played to the end — the strongest positive
    "backbtn": 0.9,     # pressed back — engaged (replay / re-listen)
    "clickrow": 0.8,    # deliberately clicked into it
    "endplay": 0.6,     # ended the session on it — neutral-positive
    "remote": 0.6,      # cast/remote handoff — neutral
    "logout": 0.5,      # session boundary, not a judgement
    "fwdbtn": 0.25,     # skipped forward off it — negative
    "trackerror": 0.0,  # never really heard it
}
NEUTRAL_SAT = 0.5

# Laplace smoothing for the artist/genre conditional transition tables.
TRANSITION_SMOOTHING = 0.5
# A context-genre lift is only trusted once the genre has at least this many
# plays in that context bucket; below it the lift is shrunk toward 1.0 (no
# signal). Keeps thin buckets from inventing strong preferences.
CONTEXT_MIN_SUPPORT = 20
# Track-level time-of-day lift is only emitted for tracks with at least this
# many total qualifying plays (otherwise the per-bucket counts are noise).
CONTEXT_TRACK_MIN_PLAYS = 8

# Time-of-day buckets (hour of the play timestamp). Kept in sync with
# pathfinder/transitions.py:tod_bucket — identical boundaries on both sides.
TOD_BUCKETS = [
    ("night", range(0, 6)),       # 00:00–05:59
    ("morning", range(6, 12)),    # 06:00–11:59
    ("afternoon", range(12, 18)), # 12:00–17:59
    ("evening", range(18, 24)),   # 18:00–23:59
]

# Word2Vec (co-listening embeddings)
EMBEDDING_DIM = 200
W2V_PARAMS = dict(
    vector_size=EMBEDDING_DIM,
    sg=1,                # skip-gram: better for small corpora / rare items
    window=10,
    negative=10,
    ns_exponent=0.75,
    epochs=30,
    min_count=5,
    sample=1e-3,
    seed=42,
    workers=4,
)
# min_count=5 left only 1,866 learned tracks (graph too thin for pathfinding);
# 3 is the planned fallback — noisier vectors, but a connected taste manifold.
W2V_PARAMS.update(
    min_count=3,
)

# Qdrant — env-overridable; defaults match the instance's shared corpus.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6337")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "spotify_tracks")
UPSERT_BATCH_SIZE = 256

# ---- Spotify Web API (metadata enrichment; consumed by `enrich`) ----
#
# Client-credentials flow — only needs an app's client id + secret (no user
# login), enough for the public /v1/tracks and /v1/search endpoints. Set them
# in the instance .env (SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET) or the
# environment. Enrichment is incremental: it only calls the API for track URIs /
# artists not already cached, so re-running with no new data costs zero requests
# and needs no credentials.
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SPOTIFY_TRACKS_BATCH = 50       # /v1/tracks accepts up to 50 ids per call
SPOTIFY_REQUEST_TIMEOUT = 30    # seconds


def _load_dotenv() -> None:
    """Minimal .env loader (no python-dotenv dependency); never overrides
    variables already present in the real environment."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def spotify_credentials() -> tuple[str | None, str | None]:
    _load_dotenv()
    return os.environ.get("SPOTIFY_CLIENT_ID"), os.environ.get("SPOTIFY_CLIENT_SECRET")
