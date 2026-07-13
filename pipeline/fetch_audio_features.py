#!/usr/bin/env python3
"""Fetch acoustic audio features for the corpus from ReccoBeats (a free,
non-Spotify drop-in for Spotify's deprecated audio-features endpoint), keyed by
the Spotify track ids we already hold in metadata.track_uri.

Flow:  Spotify ids --batch--> /v1/track  (Spotify id -> ReccoBeats uuid + isrc)
                    --batch--> /v1/audio-features  (uuid -> 11 features)

The 11 features (danceability, energy, valence, tempo, acousticness,
instrumentalness, loudness, speechiness, liveness, key, mode) describe what the
track SOUNDS like — orthogonal to the artist/genre identity the one-hots
already capture. We add them as structured NUMERIC fields (not an embedding;
the content-text embedding experiment showed dense vectors lose to categoricals
here).

Usage:
  fetch_audio_features.py --limit 200            # coverage probe, no writes
  fetch_audio_features.py --write                # full run, writes to payload
Writes go to the DST collection's payload (default spotify_tracks_content —
NOT the shared spotify_tracks corpus the sibling project reads).
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6335"
API = "https://api.reccobeats.com/v1"
FEATURES = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "instrumentalness", "loudness", "speechiness", "liveness", "key", "mode",
]
BATCH = 40          # ids per ReccoBeats call
PAUSE = 0.3         # polite throttle between calls (s)


def sid_from_href(href: str) -> str:
    # https://open.spotify.com/track/<id>
    return href.rstrip("/").split("/")[-1] if href else ""


# ReccoBeats' WAF 403s the default Python-urllib User-Agent; send a normal one.
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) lensing-pipeline/1.0",
}


def get_json(path: str, ids: list[str]) -> dict:
    qs = urllib.parse.urlencode({"ids": ",".join(ids)})
    req = urllib.request.Request(f"{API}/{path}?{qs}", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="spotify_tracks_content")
    ap.add_argument("--limit", type=int, default=0, help="0 = all points")
    ap.add_argument("--write", action="store_true", help="write features to payload")
    args = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL, timeout=120)

    # point_id -> spotify_id  (read from the SOURCE corpus payloads)
    pid_to_sid, sid_to_pid = {}, {}
    offset = None
    while True:
        pts, offset = client.scroll(
            args.collection, limit=2048, offset=offset,
            with_payload=["metadata.track_uri"], with_vectors=False,
        )
        for p in pts:
            uri = ((p.payload or {}).get("metadata", {}) or {}).get("track_uri", "")
            sid = uri.split(":")[-1] if uri else ""
            if sid:
                pid_to_sid[p.id] = sid
                sid_to_pid[sid] = p.id
        if offset is None or (args.limit and len(pid_to_sid) >= args.limit):
            break

    sids = list(sid_to_pid)[: args.limit] if args.limit else list(sid_to_pid)
    print(f"querying ReccoBeats for {len(sids)} tracks (collection {args.collection!r})")

    # 1) resolve Spotify ids -> ReccoBeats uuids
    sid_to_uuid, uuid_to_sid, isrc_hits = {}, {}, 0
    for i in range(0, len(sids), BATCH):
        chunk = sids[i : i + BATCH]
        try:
            content = get_json("track", chunk).get("content", [])
        except Exception as e:  # noqa: BLE001
            print(f"  resolve batch {i} error: {e}")
            content = []
        for t in content:
            sid = sid_from_href(t.get("href", ""))
            if sid and t.get("id"):
                sid_to_uuid[sid] = t["id"]
                uuid_to_sid[t["id"]] = sid
                if t.get("isrc"):
                    isrc_hits += 1
        time.sleep(PAUSE)
        print(f"  resolved {len(sid_to_uuid)}/{min(i + BATCH, len(sids))}", flush=True)

    # 2) uuids -> audio features
    feats_by_pid = {}
    uuids = list(uuid_to_sid)
    for i in range(0, len(uuids), BATCH):
        chunk = uuids[i : i + BATCH]
        try:
            content = get_json("audio-features", chunk).get("content", [])
        except Exception as e:  # noqa: BLE001
            print(f"  features batch {i} error: {e}")
            content = []
        for f in content:
            sid = sid_from_href(f.get("href", "")) or uuid_to_sid.get(f.get("id"), "")
            pid = sid_to_pid.get(sid)
            if pid is None:
                continue
            vals = {k: f[k] for k in FEATURES if f.get(k) is not None}
            if len(vals) >= 8:  # require a mostly-complete row
                feats_by_pid[pid] = vals
        time.sleep(PAUSE)
        print(f"  features {len(feats_by_pid)}/{len(uuids)}", flush=True)

    n = len(sids)
    print("\n==== COVERAGE ====")
    print(f"  queried:           {n}")
    print(f"  resolved to RB id: {len(sid_to_uuid)}  ({len(sid_to_uuid)/n:.1%})")
    print(f"  with ISRC:         {isrc_hits}")
    print(f"  with features:     {len(feats_by_pid)}  ({len(feats_by_pid)/n:.1%})")

    if args.write and feats_by_pid:
        print(f"\nwriting {len(FEATURES)} audio features to {len(feats_by_pid)} payloads ...")
        items = list(feats_by_pid.items())
        for i in range(0, len(items), 256):
            for pid, vals in items[i : i + 256]:
                # key="metadata" MERGES these keys into the existing metadata
                # object (does NOT overwrite the other metadata fields).
                client.set_payload(
                    args.collection,
                    payload={f"af_{k}": v for k, v in vals.items()},
                    points=[pid],
                    key="metadata",
                )
            print(f"  wrote {min(i + 256, len(items))}/{len(items)}", flush=True)
        print("done writing payloads.")
    elif args.write:
        print("nothing to write.")


if __name__ == "__main__":
    main()
