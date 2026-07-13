#!/usr/bin/env python3
"""Build a playlist PATH between two Spotify songs.

Give two Spotify track URLs/ids; this resolves each to a node in the AE
song-embedding graph (directly if the track is in your library, else by
AE-encoding it and SNAPPING to its nearest library track), then runs the
Pathfinder A* search between them — a journey biased toward taste-fit
(rotation-fit W_FIT) over the song-character manifold.

Run: predictors/.venv/bin/python pipeline/path_from_spotify.py <url1> <url2> [--length 12]
(uses predictors/.venv for the AE encode; shells the pathfinder.cli in its own venv)
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_spotify_url as P  # reuse fetch + feature build + Encoder

QDRANT_URL = "http://localhost:6335"
COL = "spotify_tracks_song_ae"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def encode(url, tok, pre, enc, st_model):
    sid = re.sub(r".*[/:]", "", url.split("?")[0])
    tr = P.sp_get(tok, f"tracks/{sid}")
    aids = [a["id"] for a in tr.get("artists", []) if a.get("id")]
    arts = P.sp_get(tok, "artists?ids=" + ",".join(aids[:50])).get("artists", []) if aids else []
    sg = sorted({g for a in arts for g in (a.get("genres") or [])})
    af = P.recco_features(sid)
    meta = {
        "track_name": tr.get("name"), "album": (tr.get("album") or {}).get("name"),
        "artist": (tr.get("artists") or [{}])[0].get("name"),
        "artist_count": len(tr.get("artists", []) or []),
        "track_popularity": tr.get("popularity"),
        "artist_popularity": arts[0].get("popularity") if arts else None,
        "artist_followers": (arts[0].get("followers") or {}).get("total") if arts else None,
        "album_type": (tr.get("album") or {}).get("album_type"),
        "release_year": int(((tr.get("album") or {}).get("release_date") or "0")[:4] or 0) or None,
        "sp_duration_ms": tr.get("duration_ms"), "sp_explicit": 1 if tr.get("explicit") else 0,
        "sp_track_number": tr.get("track_number"),
        "sp_n_markets": len(tr.get("available_markets", []) or []),
        "sp_genres": sg, "sp_genre_count": len(sg), "genre_primary": sg[0] if sg else None,
        **{f"af_{k}": af.get(k) for k in ["danceability", "energy", "valence", "tempo",
            "acousticness", "instrumentalness", "loudness", "speechiness", "liveness", "key", "mode"]},
    }
    emb = st_model.encode([P.content_doc(meta)], normalize_embeddings=True)[0]
    row = P.build_input_row(meta, emb, pre)
    with torch.no_grad():
        lat = enc(torch.from_numpy(row).unsqueeze(0)).numpy()[0]
    return sid, meta, lat / (np.linalg.norm(lat) + 1e-9)


def resolve_node(client, sid, lat):
    """Return (track_uri, label). Use the exact library track if present, else
    snap to the nearest library track in AE space."""
    hit, _ = client.scroll(COL, limit=1, with_payload=["metadata.track_uri", "metadata.track_name", "metadata.artist"],
                           scroll_filter=Filter(must=[FieldCondition(
                               key="metadata.track_uri", match=MatchValue(value=f"spotify:track:{sid}"))]))
    if hit:
        m = hit[0].payload["metadata"]
        return m["track_uri"], f"in library: {m.get('artist')} — {m.get('track_name')}"
    nn = client.query_points(COL, query=lat.tolist(), limit=1,
                             with_payload=["metadata.track_uri", "metadata.track_name", "metadata.artist"]).points[0]
    m = nn.payload["metadata"]
    return m["track_uri"], f"snapped (not in library) -> {m.get('artist')} — {m.get('track_name')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start_url"); ap.add_argument("end_url")
    ap.add_argument("--length", type=int, default=12)
    # Transitions ON by default now (the shipped table covers 82% of the corpus);
    # pass --no-transitions to fall back to distance + taste-fit + diversity only.
    ap.add_argument("--no-transitions", action="store_true")
    ap.add_argument("--context", default="now",
                    help="time-of-day bias: now|any|night|morning|afternoon|evening")
    ap.add_argument("--shuffle", default="any", help="any|shuffle|linear")
    args = ap.parse_args()

    import json
    pre = json.load(open(os.path.join(ROOT, "pipeline/artifacts/song_ae_preprocess.json")))
    enc = P.Encoder(pre["din"], pre["hidden"], pre["latent"])
    sd = torch.load(os.path.join(ROOT, "pipeline/artifacts/song_ae.pt"), map_location="cpu")
    enc.load_state_dict({k: v for k, v in sd.items() if k.startswith("enc.")}); enc.eval()
    st_model = SentenceTransformer(pre["text_model"])
    tok = P.sp_token(*[P.env()[k] for k in ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET")])
    client = QdrantClient(url=QDRANT_URL, timeout=120)

    endpoints = []
    for label, url in [("START", args.start_url), ("END", args.end_url)]:
        sid, meta, lat = encode(url, tok, pre, enc, st_model)
        uri, how = resolve_node(client, sid, lat)
        print(f"{label}: {meta['artist']} — {meta['track_name']}  [{how}]")
        endpoints.append(uri)

    print("\nrunning pathfinder...\n")
    env = {**os.environ, "PATHFINDER_LENSING_API": "http://localhost:8096",
           "PATHFINDER_COLLECTION": COL, "PATHFINDER_MODEL": "rotation-fit"}
    cmd = [os.path.join(ROOT, "pathfinder/.venv/bin/python"), "-m", "pathfinder.cli",
           "--start", endpoints[0], "--end", endpoints[1], "--length", str(args.length),
           "--context", args.context, "--shuffle", args.shuffle]
    if args.no_transitions:
        cmd.append("--no-transitions")
    subprocess.run(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
