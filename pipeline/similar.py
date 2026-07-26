#!/usr/bin/env python3
"""'Songs like this' over the musical-distance collection `spotify_tracks_content_metric`.

Resolves a Spotify track (URL / URI / bare id / --name substring) to its stored
vector and returns nearest neighbours under the chosen tightness<->discovery dial.
Distance shown is the proper metric d = sqrt(2 - 2*cos) in [0, 2].

Run:
  predictors/.venv/bin/python pipeline/similar.py https://open.spotify.com/track/<id>
  predictors/.venv/bin/python pipeline/similar.py --name "du hast" --dial sonic -k 10
"""
import argparse
import hashlib
import math

from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6337"
DST = "spotify_tracks_content_metric"


def uri_to_id(uri: str) -> int:
    return int.from_bytes(hashlib.sha256(uri.encode()).digest()[:8], "little")


def bare_id(tok: str) -> str:
    tok = tok.split("?", 1)[0].rstrip("/")
    return tok.rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def label(payload: dict) -> str:
    m = (payload or {}).get("metadata", {}) or {}
    return f"{m.get('track_name','?')} — {m.get('artist','?')} [{m.get('genre_primary','?')}]"


def resolve_by_name(client, needle: str, dst: str):
    """Case-insensitive substring match on track_name; pick the most-played hit."""
    needle = needle.lower()
    best = None
    off = None
    while True:
        pts, off = client.scroll(dst, limit=4096, offset=off, with_payload=True, with_vectors=False)
        for p in pts:
            m = (p.payload or {}).get("metadata", {}) or {}
            nm = (m.get("track_name") or "").lower()
            if needle in nm:
                plays = m.get("play_count") or 0
                if best is None or plays > best[2]:
                    best = (p.id, p.payload, plays)
        if off is None:
            break
    return (best[0], best[1]) if best else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track", nargs="?", help="Spotify URL, URI, or bare track id")
    ap.add_argument("--name", help="resolve by track-name substring instead")
    ap.add_argument("--dial", default="balanced", choices=["balanced", "tight", "sonic"])
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--dst", default=DST)
    args = ap.parse_args()
    client = QdrantClient(url=QDRANT_URL, timeout=120)

    if args.name:
        pid, payload = resolve_by_name(client, args.name, args.dst)
        if pid is None:
            raise SystemExit(f"no track matching name ~ {args.name!r}")
    elif args.track:
        pid = uri_to_id(f"spotify:track:{bare_id(args.track)}")
        got = client.retrieve(args.dst, ids=[pid], with_payload=True, with_vectors=False)
        if not got:
            raise SystemExit(f"track id {pid} not in {args.dst} (cold / not in corpus)")
        payload = got[0].payload
    else:
        raise SystemExit("give a track (URL/URI/id) or --name")

    got = client.retrieve(args.dst, ids=[pid], with_payload=False, with_vectors=[args.dial])
    vec = got[0].vector[args.dial]

    res = client.query_points(args.dst, query=vec, using=args.dial,
                              limit=args.k + 1, with_payload=True).points
    print(f"\nseed [{args.dial}]: {label(payload)}\n")
    shown = 0
    for r in res:
        if r.id == pid:
            continue
        d = math.sqrt(max(0.0, 2.0 - 2.0 * r.score))
        print(f"  d={d:.3f}  {label(r.payload)}")
        shown += 1
        if shown >= args.k:
            break


if __name__ == "__main__":
    main()
