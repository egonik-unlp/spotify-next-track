"""Incrementally enrich master_df tracks/artists with Spotify Web API metadata.

Replaces the old apicalls notebook. Writes the two cache files the aggregates
stage joins on:

  * track_data_block_1.json — {block_index: {"tracks": [<track object>, ...]}},
    one block per /v1/tracks?ids= call (≤50 ids). Carries duration, popularity,
    explicit, release date, album_type, artist_count.
  * data.json — {artist_query_name: <full /v1/search?type=artist response>}.
    Carries genres, artist popularity, followers.

INCREMENTAL: only track URIs / artist names not already present in the caches
are fetched, then appended. Re-running with no new data makes zero API calls
and needs no credentials — so it is safe to leave in `run_all`. New blocks are
appended with fresh keys, so aggregates.py (which reads block["tracks"] across
all blocks) keeps working unchanged.

Credentials: client-credentials flow (no user login). Set SPOTIFY_CLIENT_ID and
SPOTIFY_CLIENT_SECRET in a .env at the repo root or the environment. Get them
from https://developer.spotify.com/dashboard (create an app).

Usage: python -m pipeline.enrich [--dry-run]
"""
import argparse
import base64
import json
import time

import pandas as pd
import requests

from .config import (
    ARTIST_JSON,
    MASTER_CSV,
    SPOTIFY_API_BASE,
    SPOTIFY_REQUEST_TIMEOUT,
    SPOTIFY_TOKEN_URL,
    SPOTIFY_TRACKS_BATCH,
    TRACK_DATA_JSON,
    spotify_credentials,
)


# --------------------------------------------------------------------------- #
# cache I/O
# --------------------------------------------------------------------------- #
def _load_json(path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _cached_track_ids(blocks: dict) -> set[str]:
    ids = set()
    for block in blocks.values():
        for t in block.get("tracks", []) or []:
            if t and t.get("id"):
                ids.add(t["id"])
    return ids


def _next_block_key(blocks: dict) -> int:
    keys = [int(k) for k in blocks.keys() if str(k).isdigit()]
    return max(keys) + 1 if keys else 0


# --------------------------------------------------------------------------- #
# Spotify Web API (client-credentials)
# --------------------------------------------------------------------------- #
class Spotify:
    def __init__(self, client_id: str, client_secret: str):
        self._auth = base64.b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()
        self._token = None
        self._token_expiry = 0.0

    def _ensure_token(self, now: float) -> None:
        if self._token and now < self._token_expiry - 30:
            return
        resp = requests.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {self._auth}"},
            timeout=SPOTIFY_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        self._token_expiry = now + body.get("expires_in", 3600)

    def get(self, path: str, params: dict, now: float) -> dict:
        """GET with token refresh and 429 Retry-After backoff. `now` is passed
        in (rather than read here) to keep the module unit-testable / clock-free
        at import time; callers thread time.monotonic() through."""
        for attempt in range(6):
            self._ensure_token(now + attempt)
            try:
                resp = requests.get(
                    f"{SPOTIFY_API_BASE}{path}",
                    params=params,
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=SPOTIFY_REQUEST_TIMEOUT,
                )
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt  # transient connection drop/timeout — backoff
                print(f"enrich: connection error on {path} ({e}), retrying in {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "2")) + 1
                print(f"enrich: rate-limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 401:
                self._token = None  # force refresh on next attempt
                continue
            if resp.status_code in (500, 502, 503, 504):
                wait = 2 ** attempt  # transient gateway/5xx — exp. backoff
                print(f"enrich: {resp.status_code} on {path}, retrying in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"enrich: gave up on {path} after retries")


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def run(dry_run: bool = False) -> None:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"{MASTER_CSV.name} not found — run `python -m pipeline.ingest` first"
        )
    master = pd.read_csv(MASTER_CSV, usecols=["spotify_track_uri", "master_metadata_album_artist_name"])

    # Distinct track ids (strip the spotify:track: prefix) and artist names.
    uris = master["spotify_track_uri"].dropna().unique()
    want_track_ids = {u.split(":")[-1] for u in uris if isinstance(u, str) and ":" in u}
    want_artists = {
        a for a in master["master_metadata_album_artist_name"].dropna().unique()
        if isinstance(a, str) and a.strip()
    }

    track_blocks = _load_json(TRACK_DATA_JSON)
    artist_cache = _load_json(ARTIST_JSON)

    new_track_ids = sorted(want_track_ids - _cached_track_ids(track_blocks))
    new_artists = sorted(want_artists - set(artist_cache.keys()))

    print(
        f"enrich: {len(want_track_ids)} distinct tracks "
        f"({len(new_track_ids)} new), "
        f"{len(want_artists)} distinct artists ({len(new_artists)} new)"
    )

    if not new_track_ids and not new_artists:
        print("enrich: caches already current — no API calls needed")
        return

    if dry_run:
        print("enrich: --dry-run, not calling the API")
        return

    client_id, client_secret = spotify_credentials()
    if not (client_id and client_secret):
        raise RuntimeError(
            "enrich: new tracks/artists to fetch but SPOTIFY_CLIENT_ID / "
            "SPOTIFY_CLIENT_SECRET are unset. Add them to .env (see "
            ".env.example), or run `python -m pipeline.run_all --skip-enrich` "
            "to reuse the existing caches."
        )

    spotify = Spotify(client_id, client_secret)
    start = time.monotonic()

    # ---- tracks: batched /v1/tracks?ids= ----
    block_key = _next_block_key(track_blocks)
    for i in range(0, len(new_track_ids), SPOTIFY_TRACKS_BATCH):
        batch = new_track_ids[i : i + SPOTIFY_TRACKS_BATCH]
        body = spotify.get("/tracks", {"ids": ",".join(batch)}, time.monotonic() - start)
        track_blocks[str(block_key)] = {"tracks": body.get("tracks", [])}
        block_key += 1
        if (i // SPOTIFY_TRACKS_BATCH) % 10 == 0:
            print(f"enrich: tracks {min(i + SPOTIFY_TRACKS_BATCH, len(new_track_ids))}/{len(new_track_ids)}")
    if new_track_ids:
        with open(TRACK_DATA_JSON, "w", encoding="utf-8") as f:
            json.dump(track_blocks, f, ensure_ascii=False)
        print(f"enrich: wrote {len(new_track_ids)} new tracks to {TRACK_DATA_JSON.name}")

    # ---- artists: one /v1/search per artist name (keyed by query name) ----
    for n, artist in enumerate(new_artists):
        artist_cache[artist] = spotify.get(
            "/search", {"q": artist, "type": "artist", "limit": 5}, time.monotonic() - start
        )
        if n % 50 == 0:
            print(f"enrich: artists {n + 1}/{len(new_artists)}")
            with open(ARTIST_JSON, "w", encoding="utf-8") as f:  # checkpoint
                json.dump(artist_cache, f, ensure_ascii=False)
    if new_artists:
        with open(ARTIST_JSON, "w", encoding="utf-8") as f:
            json.dump(artist_cache, f, ensure_ascii=False)
        print(f"enrich: wrote {len(new_artists)} new artists to {ARTIST_JSON.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many tracks/artists would be fetched, without calling the API",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
