#!/usr/bin/env python3
"""Build a linear (PCA) SONG embedding — the linear baseline for the AE.

Companion to `build_song_ae.py`: assembles the IDENTICAL leak-safe multimodal
feature matrix (same intrinsic blocks, same z-score/multi-hot construction) and
compresses it with PCA instead of a nonlinear autoencoder. The compressed,
L2-normalized rows become the stored vectors of a new collection so a sequence
artifact can be built over them and compared apples-to-apples against the AE
spaces under a frozen GRU readout.

Feature blocks (exactly as in build_song_ae.py — NO behavioral/target fields):
  text 384 (content emb) + numeric (~10, z-scored) + acoustic 22 (z + missing
  flags) + categorical (top-120 multi-genre multi-hot + album_type one-hot).

BLOCK-COMPOSITION knobs (all backward-compatible; the DEFAULTS reproduce the
original full-matrix output byte-for-byte):
  --blocks         subset/order of blocks to include (default all four)
  --block-weights  per-block scalar multipliers, e.g. text=2,acoustic=0.25
  --standardize    per-column z-score (diagonal whitening) each selected block
                   before weighting + PCA (de-loads high-variance num/acoustic)

Source: spotify_tracks_content (text emb as vector; af_* + sp_* in payload).
Dest:   <--dst>  (same ids + payloads, vector = PCA projection, L2-normalized).

Run: predictors/.venv/bin/python pipeline/build_song_pca.py --latent 64 --dst spotify_tracks_song_pca64
"""
import argparse
import math
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_URL = "http://localhost:6337"
SRC = "spotify_tracks_content"

# feature spec — MUST stay identical to build_song_ae.py (leak-safe intrinsic set)
NUMERICS = ["artist_popularity", "track_popularity", "artist_followers",
            "release_year", "artist_count",
            "sp_duration_ms", "sp_n_markets", "sp_track_number",
            "sp_explicit", "sp_genre_count"]
LOG_NUMERICS = {"artist_followers", "sp_duration_ms"}
ACOUSTICS = ["af_danceability", "af_energy", "af_valence", "af_tempo",
             "af_acousticness", "af_instrumentalness", "af_loudness",
             "af_speechiness", "af_liveness", "af_key", "af_mode"]
GENRE_TOPN = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--dst", required=True,
                    help="destination Qdrant collection (e.g. spotify_tracks_song_pca64)")
    ap.add_argument("--artifact-prefix", default=None,
                    help="basename for saved artifacts (default song_pca<latent>)")
    ap.add_argument("--blocks", default="text,numeric,acoustic,categorical",
                    help="comma list of feature blocks to include, in order "
                         "(default all four = unchanged full matrix)")
    ap.add_argument("--block-weights", default="",
                    help="comma list block=weight (e.g. text=2,acoustic=0.25); "
                         "unlisted blocks default to 1.0")
    ap.add_argument("--standardize", action="store_true",
                    help="per-column z-score (diagonal whitening) each selected "
                         "block before weighting + PCA (default off = unchanged)")
    args = ap.parse_args()
    prefix = args.artifact_prefix or f"song_pca{args.latent}"
    np.random.seed(42)
    client = QdrantClient(url=QDRANT_URL, timeout=180)

    # ---- feature assembly (verbatim parity with build_song_ae.py) ------------
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

    ac_cols, ac_flag, ac_stats = [], [], {}
    for k in ACOUSTICS:
        arr = np.asarray(ac_raw[k], dtype=np.float64)
        mu, sd = float(np.nanmean(arr)), float(np.nanstd(arr))
        sd = sd if sd > 1e-9 else 1.0
        ac_stats[k] = {"mean": mu, "std": sd}
        ac_cols.append((np.nan_to_num(arr, nan=mu) - mu) / sd)
        ac_flag.append((~np.isfinite(arr)).astype(np.float32))
    acoustic = np.concatenate([np.stack(ac_cols, 1), np.stack(ac_flag, 1)], 1).astype(np.float32)

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

    # ---- block selection / standardization / weighting -----------------------
    # DEFAULTS (all four blocks, no standardize, unit weights) reproduce the
    # original full-matrix output byte-for-byte.
    all_blocks = {"text": text, "numeric": numeric,
                  "acoustic": acoustic, "categorical": categorical}
    sel = [b.strip() for b in args.blocks.split(",") if b.strip()]
    for b in sel:
        assert b in all_blocks, f"unknown block {b!r} (have {list(all_blocks)})"
    weights = {}
    for tok in (t.strip() for t in args.block_weights.split(",") if t.strip()):
        k, _, v = tok.partition("=")
        weights[k.strip()] = float(v)

    blocks = []
    for nm in sel:
        B = all_blocks[nm].astype(np.float32)
        if args.standardize:
            mu = B.mean(0); sd = B.std(0); sd[sd < 1e-9] = 1.0
            B = ((B - mu) / sd).astype(np.float32)
        w = weights.get(nm, 1.0)
        if w != 1.0:
            B = (B * np.float32(w)).astype(np.float32)
        blocks.append((nm, B))

    X = np.concatenate([b for _, b in blocks], 1).astype(np.float32)
    spans, s = [], 0
    for nm, b in blocks:
        spans.append((nm, s, s + b.shape[1])); s += b.shape[1]
    print(f"input dim {X.shape[1]} :: " + ", ".join(f"{nm}[{a}:{c}]" for nm, a, c in spans))
    print(f"  blocks={sel} standardize={args.standardize} weights={weights}")
    print(f"  multi-genre vocab: {len(top)} (of {len(gcount)} distinct), album_type: {len(at_vals)}")

    # ---- PCA (the only difference from the AE builder) -----------------------
    L = args.latent
    pca = PCA(n_components=L, random_state=42)
    Z = pca.fit_transform(X).astype(np.float32)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA latent {L}: cumulative EVR {evr.sum():.4f}  "
          f"(component 1 {evr[0]:.4f}, last {evr[-1]:.5f})")

    # save artifacts for one-shot inference parity (feature spec + PCA basis)
    from pathlib import Path
    import json as _json
    art = Path("pipeline/artifacts"); art.mkdir(parents=True, exist_ok=True)
    np.savez(art / f"{prefix}.npz",
             components=pca.components_.astype(np.float32),
             mean=pca.mean_.astype(np.float32),
             explained_variance_ratio=evr.astype(np.float32))
    _json.dump({
        "din": int(X.shape[1]), "latent": int(L), "method": "pca",
        "numerics": NUMERICS, "log_numerics": sorted(LOG_NUMERICS), "num_stats": num_stats,
        "acoustics": ACOUSTICS, "ac_stats": ac_stats,
        "genre_vocab": top, "album_types": at_vals,
        "spans": [[nm, int(a), int(c)] for nm, a, c in spans],
        "blocks": sel,
        "block_weights": {nm: weights.get(nm, 1.0) for nm in sel},
        "standardize": bool(args.standardize),
        "cumulative_evr": float(evr.sum()),
        "text_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    }, open(art / f"{prefix}_preprocess.json", "w"), indent=1)
    print(f"saved PCA artifacts to {art}/ ({prefix}.npz + {prefix}_preprocess.json)")

    # L2-normalize rows (parity with the AE) and upsert (same ids + payloads)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    client.recreate_collection(args.dst, vectors_config=VectorParams(size=L, distance=Distance.COSINE))
    for i in range(0, n, 256):
        client.upsert(args.dst, points=[PointStruct(id=ids[j], vector=Zn[j].tolist(), payload=payloads[j])
                                        for j in range(i, min(i + 256, n))])
        print(f"  upserted {min(i + 256, n)}/{n}", flush=True)
    print(f"done: {args.dst!r} has {client.get_collection(args.dst).points_count} points, dim {L}")


if __name__ == "__main__":
    main()
