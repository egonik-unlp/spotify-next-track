#!/usr/bin/env python3
"""Enrich the corpus with extra INTRINSIC song info from Spotify's metadata
endpoints (NOT the deprecated audio-features — /tracks and /artists still work
with client-credentials auth). Adds song attributes we don't already hold, to
feed a richer song embedding:

  per track  (GET /v1/tracks?ids=, batch 50):
    sp_duration_ms, sp_explicit, sp_track_number, sp_n_markets, sp_release_year
  per artist (GET /v1/artists?ids=, batch 50):
    sp_genres  -- the artist's MULTI-genre tag list (e.g. Slipknot ->
    [nu metal, metal, alternative metal, rap metal, heavy metal]), a far richer
    style signal than the single genre_primary. Unioned over a track's artists.

Credentials come from .env (SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET).
Writes into spotify_tracks_content payload.metadata (key-merge, non-destructive).

Run: predictors/.venv/bin/python pipeline/fetch_spotify_meta.py [--limit N]
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6337"
DST = "spotify_tracks_content"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"


def load_env(path=".env"):
    env = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); env[k] = v
    return env


class Spotify:
    def __init__(self, cid, csec):
        self.cid, self.csec = cid, csec
        self.tok = None
        self._auth()

    def _auth(self):
        body = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self.cid, "client_secret": self.csec,
        }).encode()
        with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=body), timeout=30) as r:
            self.tok = json.load(r)["access_token"]

    def get(self, path, ids):
        qs = urllib.parse.urlencode({"ids": ",".join(ids)})
        for attempt in range(5):
            req = urllib.request.Request(f"{API}/{path}?{qs}",
                                         headers={"Authorization": f"Bearer {self.tok}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    self._auth(); continue
                if e.code == 429:
                    time.sleep(int(e.headers.get("Retry-After", "2")) + 1); continue
                raise
        return {}


def year_of(date: str):
    return int(date[:4]) if date and date[:4].isdigit() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    env = load_env()
    sp = Spotify(env["SPOTIFY_CLIENT_ID"], env["SPOTIFY_CLIENT_SECRET"])
    client = QdrantClient(url=QDRANT_URL, timeout=180)

    # point_id -> spotify track id
    pid_sid, offset = [], None
    while True:
        pts, offset = client.scroll(DST, limit=2048, offset=offset,
                                    with_payload=["metadata.track_uri"], with_vectors=False)
        for p in pts:
            uri = ((p.payload or {}).get("metadata", {}) or {}).get("track_uri", "")
            if uri:
                pid_sid.append((p.id, uri.split(":")[-1]))
        if offset is None or (args.limit and len(pid_sid) >= args.limit):
            break
    if args.limit:
        pid_sid = pid_sid[: args.limit]
    sid_pid = {s: p for p, s in pid_sid}
    sids = [s for _, s in pid_sid]
    print(f"enriching {len(sids)} tracks")

    # 1) tracks -> per-track fields + artist ids
    track_meta = {}              # sid -> dict
    sid_artists = {}             # sid -> [artist_id]
    for i in range(0, len(sids), 50):
        chunk = sids[i:i + 50]
        for t in (sp.get("tracks", chunk).get("tracks") or []):
            if not t:
                continue
            sid = t["id"]
            track_meta[sid] = {
                "sp_duration_ms": t.get("duration_ms"),
                "sp_explicit": 1 if t.get("explicit") else 0,
                "sp_track_number": t.get("track_number"),
                "sp_n_markets": len(t.get("available_markets", []) or []),
                "sp_release_year": year_of((t.get("album") or {}).get("release_date", "")),
            }
            sid_artists[sid] = [a["id"] for a in (t.get("artists") or []) if a.get("id")]
        time.sleep(0.1)
        print(f"  tracks {min(i + 50, len(sids))}/{len(sids)}", flush=True)

    # 2) artists -> genres
    all_artist_ids = sorted({aid for ids in sid_artists.values() for aid in ids})
    print(f"  fetching genres for {len(all_artist_ids)} distinct artists")
    artist_genres = {}
    for i in range(0, len(all_artist_ids), 50):
        chunk = all_artist_ids[i:i + 50]
        for a in (sp.get("artists", chunk).get("artists") or []):
            if a:
                artist_genres[a["id"]] = a.get("genres", []) or []
        time.sleep(0.1)
        print(f"  artists {min(i + 50, len(all_artist_ids))}/{len(all_artist_ids)}", flush=True)

    # 3) merge + write
    wrote = 0
    items = list(track_meta.items())
    for i in range(0, len(items), 256):
        for sid, meta in items[i:i + 256]:
            genres = sorted({g for aid in sid_artists.get(sid, []) for g in artist_genres.get(aid, [])})
            payload = {k: v for k, v in meta.items() if v is not None}
            payload["sp_genres"] = genres
            payload["sp_genre_count"] = len(genres)
            client.set_payload(DST, payload=payload, points=[sid_pid[sid]], key="metadata")
            wrote += 1
        print(f"  wrote {min(i + 256, len(items))}/{len(items)}", flush=True)

    n = len(sids)
    print("\n==== ENRICHMENT COVERAGE ====")
    print(f"  tracks queried:      {n}")
    print(f"  track meta resolved: {len(track_meta)} ({len(track_meta)/n:.1%})")
    withg = sum(1 for sid in track_meta if any(artist_genres.get(a) for a in sid_artists.get(sid, [])))
    print(f"  with >=1 sp_genre:   {withg} ({withg/n:.1%})")
    print(f"  distinct artists:    {len(all_artist_ids)}; distinct genres seen: "
          f"{len({g for gs in artist_genres.values() for g in gs})}")


if __name__ == "__main__":
    main()
