#!/usr/bin/env python3
"""Build a learned, information-rich SONG embedding via an autoencoder.

Goal: pack a lot of intrinsic song information into one dense vector. We
assemble a multimodal feature row per track and train an autoencoder to
compress it into a dense latent that RECONSTRUCTS every modality; the latent
becomes the stored vector of a new collection. Prediction is a separate, later
concern --- this is pure representation.

Feature blocks (each weighted EQUALLY in the reconstruction loss, so the 384
text dims don't drown out the 11 acoustic ones):
  text         384  sentence embedding of title/artist/album/genre (reused
                    from spotify_tracks_content's stored vector)
  numeric       ~10 popularity/followers/year/artist_count + Spotify
                    duration/markets/track_no/explicit/genre_count (z-scored)
  acoustic      22  11 ReccoBeats audio features + missing flags (z-scored)
  categorical   ~N  MULTI-genre multi-hot (Spotify artist genres, top-N) +
                    genre_primary bit + album_type one-hot

Source: spotify_tracks_content (text emb as vector; af_* + sp_* in payload).
Dest:   spotify_tracks_song_ae  (same ids + payloads, vector = AE latent).

Run: predictors/.venv/bin/python pipeline/build_song_ae.py [--latent 64] [--epochs 200]
"""
import argparse
import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_URL = "http://localhost:6335"
SRC = "spotify_tracks_content"
DST = "spotify_tracks_song_ae"

# numeric fields (original + Spotify-enriched); log1p the wide-range ones
NUMERICS = ["artist_popularity", "track_popularity", "artist_followers",
            "release_year", "artist_count",
            "sp_duration_ms", "sp_n_markets", "sp_track_number",
            "sp_explicit", "sp_genre_count"]
LOG_NUMERICS = {"artist_followers", "sp_duration_ms"}
ACOUSTICS = ["af_danceability", "af_energy", "af_valence", "af_tempo",
             "af_acousticness", "af_instrumentalness", "af_loudness",
             "af_speechiness", "af_liveness", "af_key", "af_mode"]
GENRE_TOPN = 120


def zscore(col):
    mu, sd = np.nanmean(col), np.nanstd(col)
    return (np.nan_to_num(col, nan=mu) - mu) / (sd if sd > 1e-9 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    client = QdrantClient(url=QDRANT_URL, timeout=180)

    ids, text, payloads = [], [], []
    num_raw = {k: [] for k in NUMERICS}
    ac_raw = {k: [] for k in ACOUSTICS}
    multigenre, albumtypes = [], []
    offset = None
    while True:
        pts, offset = client.scroll(SRC, limit=2048, offset=offset,
                                    with_payload=True, with_vectors=True)
        for p in pts:
            m = (p.payload or {}).get("metadata", {}) or {}
            ids.append(p.id); payloads.append(p.payload); text.append(p.vector)
            for k in NUMERICS:
                v = m.get(k)
                num_raw[k].append(float(v) if isinstance(v, (int, float)) else math.nan)
            for k in ACOUSTICS:
                v = m.get(k)
                ac_raw[k].append(float(v) if isinstance(v, (int, float)) else math.nan)
            gs = list(m.get("sp_genres") or [])
            gp = m.get("genre_primary")
            if gp:
                gs.append(gp)
            multigenre.append(gs)
            albumtypes.append(m.get("album_type") or "__other__")
        if offset is None:
            break
    n = len(ids)
    print(f"loaded {n} tracks; text dim {len(text[0])}")
    text = np.asarray(text, dtype=np.float32)

    # numeric block (record z-score stats for one-shot inference parity)
    cols, num_stats = [], {}
    for k in NUMERICS:
        arr = np.asarray(num_raw[k], dtype=np.float64)
        if k in LOG_NUMERICS:
            arr = np.log1p(np.where(arr >= 0, arr, np.nan))
        mu, sd = float(np.nanmean(arr)), float(np.nanstd(arr))
        sd = sd if sd > 1e-9 else 1.0
        num_stats[k] = {"mean": mu, "std": sd, "log": k in LOG_NUMERICS}
        cols.append((np.nan_to_num(arr, nan=mu) - mu) / sd)
    numeric = np.stack(cols, 1).astype(np.float32)

    # acoustic block (+ missing flags)
    ac_cols, ac_flag, ac_stats = [], [], {}
    for k in ACOUSTICS:
        arr = np.asarray(ac_raw[k], dtype=np.float64)
        mu, sd = float(np.nanmean(arr)), float(np.nanstd(arr))
        sd = sd if sd > 1e-9 else 1.0
        ac_stats[k] = {"mean": mu, "std": sd}
        ac_cols.append((np.nan_to_num(arr, nan=mu) - mu) / sd)
        ac_flag.append((~np.isfinite(arr)).astype(np.float32))
    acoustic = np.concatenate([np.stack(ac_cols, 1), np.stack(ac_flag, 1)], 1).astype(np.float32)

    # categorical block: multi-genre multi-hot (top-N) + album_type one-hot
    gcount = Counter(g for gs in multigenre for g in gs)
    top = [g for g, _ in gcount.most_common(GENRE_TOPN)]
    gidx = {g: i for i, g in enumerate(top)}
    gmat = np.zeros((n, len(top)), np.float32)
    for i, gs in enumerate(multigenre):
        for g in gs:
            if g in gidx:
                gmat[i, gidx[g]] = 1.0
    at_vals = sorted(set(albumtypes)); amap = {a: i for i, a in enumerate(at_vals)}
    amat = np.zeros((n, len(at_vals)), np.float32)
    for i, a in enumerate(albumtypes):
        amat[i, amap[a]] = 1.0
    categorical = np.concatenate([gmat, amat], 1)

    blocks = [("text", text), ("numeric", numeric), ("acoustic", acoustic), ("categorical", categorical)]
    X = np.concatenate([b for _, b in blocks], 1).astype(np.float32)
    spans, s = [], 0
    for nm, b in blocks:
        spans.append((nm, s, s + b.shape[1])); s += b.shape[1]
    print(f"input dim {X.shape[1]} :: " + ", ".join(f"{nm}[{a}:{c}]" for nm, a, c in spans))
    print(f"  multi-genre vocab: {len(top)} (of {len(gcount)} distinct), album_type: {len(at_vals)}")

    Xt = torch.from_numpy(X); din, L = X.shape[1], args.latent

    class AE(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(din, args.hidden), nn.ReLU(), nn.Linear(args.hidden, L))
            s.dec = nn.Sequential(nn.Linear(L, args.hidden), nn.ReLU(), nn.Linear(args.hidden, din))

        def forward(s, x):
            z = s.enc(x); return s.dec(z), z

    model = AE(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def block_loss(recon, target):
        return sum(((recon[:, a:c] - target[:, a:c]) ** 2).mean() for _, a, c in spans)

    bs = 256; idx = np.arange(n)
    for ep in range(args.epochs):
        model.train(); np.random.shuffle(idx); tot = 0.0
        for i in range(0, n, bs):
            b = torch.from_numpy(idx[i:i + bs]); xb = Xt[b]
            recon, _ = model(xb); loss = block_loss(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  block-loss {tot / n:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        recon, Z = model(Xt); recon = recon.numpy(); Z = Z.numpy()
    print("\nper-block reconstruction R^2 (how well the latent preserves each modality):")
    for nm, a, c in spans:
        mse = ((recon[:, a:c] - X[:, a:c]) ** 2).mean(); var = X[:, a:c].var() + 1e-9
        print(f"  {nm:12} dims {c-a:4d}  R^2 {1 - mse / var:.3f}")

    # save artifacts so a brand-new song can be encoded identically (one-shot)
    from pathlib import Path
    import json as _json
    art = Path("pipeline/artifacts"); art.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), art / "song_ae.pt")
    _json.dump({
        "din": int(din), "hidden": int(args.hidden), "latent": int(L),
        "numerics": NUMERICS, "log_numerics": sorted(LOG_NUMERICS), "num_stats": num_stats,
        "acoustics": ACOUSTICS, "ac_stats": ac_stats,
        "genre_vocab": top, "album_types": at_vals,
        "spans": [[nm, int(a), int(c)] for nm, a, c in spans],
        "text_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    }, open(art / "song_ae_preprocess.json", "w"), indent=1)
    print(f"saved AE artifacts to {art}/ (song_ae.pt + song_ae_preprocess.json)")

    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    client.recreate_collection(DST, vectors_config=VectorParams(size=L, distance=Distance.COSINE))
    for i in range(0, n, 256):
        client.upsert(DST, points=[PointStruct(id=ids[j], vector=Zn[j].tolist(), payload=payloads[j])
                                   for j in range(i, min(i + 256, n))])
        print(f"  upserted {min(i + 256, n)}/{n}", flush=True)
    print(f"done: {DST!r} has {client.get_collection(DST).points_count} points, dim {L}")


if __name__ == "__main__":
    main()
