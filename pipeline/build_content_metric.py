#!/usr/bin/env python3
"""Build the MUSICAL-DISTANCE metric collection `spotify_tracks_content_metric`.

A symmetric "how alike do these sound/feel" distance over INTRINSIC content —
the sonic counterpart to the behavioral co-listening space in `spotify_tracks`.
Empirically (see experiments/ + memory): pure ReccoBeats acoustics are near-noise
for musical similarity, and the whitened Song-AE latent is the workhorse; text
name/tags and a genre prototype add real coherence. So the metric is a
block-weighted, partial-whitened, L2-normalized fusion; Euclidean on the result
is a PROPER metric (== sqrt(2-2cos), so Qdrant Cosine ranks identically).

Blocks (all intrinsic, leak-safe — NO playback/behavioral fields):
  ae     64-d Song-AE latent (spotify_tracks_song_ae)      -- ZCA-whitened
  txt    384-d name/album/genre text embedding (content)   -- raw
  meta   5 numerics (popularity/followers/year/artists)     -- z-scored + whitened
  genre  per-genre prototype = mean whitened-AE of genre    -- raw

Three named vectors give a tightness<->discovery DIAL (same collection):
  balanced (DEFAULT) -- coherent but keeps cross-artist variety
  tight              -- maximizes genre/artist purity (leans same-artist)
  sonic              -- AE-only; most cross-artist "sounds alike" discovery

Source: spotify_tracks_song_ae (AE vector) + spotify_tracks_content (text vector,
        af_*/sp_* payload). Dest: spotify_tracks_content_metric (3 named vectors,
        same ids + payloads). Safe to run while lensing-server is up — it only
        creates a NEW collection and never touches existing ones.

Run: predictors/.venv/bin/python pipeline/build_content_metric.py
     [--dst spotify_tracks_content_metric] [--profiles balanced,tight,sonic]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_URL = "http://localhost:6337"
AE_SRC = "spotify_tracks_song_ae"
CONTENT_SRC = "spotify_tracks_content"
META_FIELDS = ["artist_popularity", "track_popularity", "artist_followers",
               "release_year", "artist_count"]

# profile := {block: (weight, whiten_alpha)}; alpha 0=raw, 1=full ZCA.
PROFILES = {
    "balanced": {"ae": (1.0, 1.0), "txt": (0.4, 0.0), "meta": (0.3, 1.0), "genre": (0.3, 0.0)},
    "tight":    {"ae": (1.0, 1.0), "txt": (0.7, 0.0), "meta": (0.3, 1.0), "genre": (0.7, 0.0)},
    "sonic":    {"ae": (1.0, 1.0)},
}
RNG = np.random.default_rng(42)


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X / n


def zscore(X):
    mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd; Z[np.isnan(Z)] = 0.0
    return Z, mu, sd


def partial_whiten(X, alpha):
    """W(alpha) = V diag(lam^(-alpha/2)) V^T on centered X. alpha 0=raw, 1=full ZCA.
    Eigenvalues floored at 1e-3*max so tiny/noise directions aren't over-inflated."""
    mu = X.mean(0); Xc = X - mu
    if alpha <= 0:
        return Xc, mu, np.eye(X.shape[1])
    cov = np.cov(Xc, rowvar=False)
    val, vec = np.linalg.eigh(cov)
    val = np.clip(val, 1e-3 * val.max(), None)
    W = vec @ np.diag(val ** (-alpha / 2)) @ vec.T
    return Xc @ W, mu, W


def scale_block(X, n=6000):
    """Divide by median pairwise distance so blocks combine on a comparable scale."""
    ii = RNG.integers(0, len(X), n); jj = RNG.integers(0, len(X), n)
    d = np.linalg.norm(X[ii] - X[jj], axis=1)
    med = float(np.median(d[d > 0]) or 1.0)
    return X / med, med


def assemble(blocks, cfg):
    """Concatenate sqrt(w)*scaled(whiten(block)); return (L2-normalized matrix, spec)."""
    parts, spec = [], {}
    for name, (w, alpha) in cfg.items():
        if w <= 0:
            continue
        Xw, mu, W = partial_whiten(blocks[name], alpha)
        Xs, med = scale_block(Xw)
        parts.append(np.sqrt(w) * Xs)
        spec[name] = {"weight": w, "alpha": alpha, "median": med, "dim": int(blocks[name].shape[1])}
    V = l2(np.concatenate(parts, 1).astype(np.float32))
    return V, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="spotify_tracks_content_metric")
    ap.add_argument("--profiles", default="balanced,tight,sonic",
                    help="comma list of profiles to store as named vectors (first = default)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for p in profiles:
        assert p in PROFILES, f"unknown profile {p!r} (have {list(PROFILES)})"

    client = QdrantClient(url=QDRANT_URL, timeout=180)

    # ---- read AE latents ----------------------------------------------------
    ae_by_id = {}
    off = None
    while True:
        pts, off = client.scroll(AE_SRC, limit=2048, offset=off,
                                 with_payload=False, with_vectors=True)
        for p in pts:
            ae_by_id[p.id] = p.vector
        if off is None:
            break
    print(f"AE latents: {len(ae_by_id)}")

    # ---- read content (text vector + payload) in a fixed id order -----------
    ids, text, payloads, meta_raw, genres = [], [], [], {k: [] for k in META_FIELDS}, []
    off = None
    while True:
        pts, off = client.scroll(CONTENT_SRC, limit=2048, offset=off,
                                 with_payload=True, with_vectors=True)
        for p in pts:
            if p.id not in ae_by_id:
                continue
            m = (p.payload or {}).get("metadata", {}) or {}
            ids.append(p.id); text.append(p.vector); payloads.append(p.payload)
            genres.append(m.get("genre_primary"))
            for k in META_FIELDS:
                v = m.get(k)
                meta_raw[k].append(float(v) if isinstance(v, (int, float)) else np.nan)
        if off is None:
            break
    n = len(ids)
    print(f"joined tracks: {n}; text dim {len(text[0])}")

    ae = np.asarray([ae_by_id[i] for i in ids], dtype=np.float64)
    txt = np.asarray(text, dtype=np.float64)
    meta_cols = []
    for k in META_FIELDS:
        arr = np.asarray(meta_raw[k], dtype=np.float64)
        if k == "artist_followers":
            arr = np.log1p(np.clip(arr, 0, None))
        meta_cols.append(arr)
    meta_z, meta_mu, meta_sd = zscore(np.column_stack(meta_cols))

    # genre prototype = per-genre mean of the fully-whitened AE latent
    ae_white = partial_whiten(ae, 1.0)[0]
    proto_by_genre = {}
    genres_arr = np.array(genres, dtype=object)
    for g in set(genres):
        mask = genres_arr == g
        proto_by_genre[g] = ae_white[mask].mean(0) if mask.any() else ae_white.mean(0)
    genre_b = np.asarray([proto_by_genre[g] for g in genres])

    blocks = {"ae": ae, "txt": txt, "meta": meta_z, "genre": genre_b}

    # ---- assemble each requested profile ------------------------------------
    vectors, specs, dims = {}, {}, {}
    for prof in profiles:
        V, spec = assemble(blocks, PROFILES[prof])
        vectors[prof] = V; specs[prof] = spec; dims[prof] = V.shape[1]
        print(f"profile {prof:9s} dim {V.shape[1]:4d}  blocks={PROFILES[prof]}")

    # ---- (re)create collection with named vectors + upsert ------------------
    vconf = {prof: VectorParams(size=dims[prof], distance=Distance.COSINE) for prof in profiles}
    client.recreate_collection(args.dst, vectors_config=vconf)
    for i in range(0, n, args.batch):
        j1 = min(i + args.batch, n)
        client.upsert(args.dst, points=[
            PointStruct(id=ids[j],
                        vector={prof: vectors[prof][j].tolist() for prof in profiles},
                        payload=payloads[j])
            for j in range(i, j1)])
        print(f"  upserted {j1}/{n}", flush=True)
    cnt = client.get_collection(args.dst).points_count
    print(f"done: {args.dst!r} has {cnt} points; named vectors {profiles} (default={profiles[0]})")

    # ---- reproducibility manifest -------------------------------------------
    art = Path("pipeline/artifacts"); art.mkdir(parents=True, exist_ok=True)
    json.dump({
        "collection": args.dst, "n": n, "default_profile": profiles[0],
        "profiles": {p: PROFILES[p] for p in profiles},
        "specs": specs, "meta_fields": META_FIELDS,
        "recipe": "per block: center by stored mean, W(alpha) whiten, /median, *sqrt(weight); "
                  "concat; L2-normalize. genre block = per-genre mean of full-whitened AE. "
                  "Euclidean on stored vectors is a proper metric == sqrt(2-2cos).",
        "sources": {"ae": AE_SRC, "text_and_payload": CONTENT_SRC},
    }, open(art / "content_metric.json", "w"), indent=1)
    np.savez(art / "content_metric.npz",
             ae_whiten=partial_whiten(ae, 1.0)[2], ae_mean=ae.mean(0),
             meta_mean=meta_mu, meta_std=meta_sd,
             genre_vocab=np.array([g if g is not None else "" for g in proto_by_genre], dtype=object),
             genre_protos=np.array([proto_by_genre[g] for g in proto_by_genre], dtype=np.float32))
    print(f"saved manifest -> {art}/content_metric.json (+ .npz)")


if __name__ == "__main__":
    main()
