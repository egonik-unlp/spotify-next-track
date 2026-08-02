#!/usr/bin/env python3
"""Recover the content-metric block statistics a COLD track needs.

`build_content_metric.py` fuses four blocks into the musical-distance vector, and
`partial_whiten` CENTERS every block — even at alpha 0, where it is otherwise a
no-op. But `content_metric.npz` only persists what the AE and meta blocks needed
for their own construction (`ae_whiten`, `ae_mean`, `meta_mean`, `meta_std`,
`genre_protos`). Three centers and one whitening matrix are therefore missing, and
without them a track outside the corpus cannot be placed in the same fused space:

    txt      mean of the 384-d text block            (alpha 0 → center only)
    meta     mean + ZCA of the z-scored 5-d block    (alpha 1)
    genre    mean of the per-row genre prototype     (alpha 0 → center only)

This derives them from the corpus and caches them. It is strictly READ-ONLY: one
scroll over `spotify_tracks_content`, no collection is created or written. Notably
it does NOT re-run build_content_metric.py, which would recreate the collection the
lab reads its mood space from.

  predictors/.venv/bin/python pipeline/coldstart_metric_stats.py [--force]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "pipeline/artifacts"
OUT = ART / "content_metric_coldstart.npz"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6337")
CONTENT_SRC = "spotify_tracks_content"


def partial_whiten_fit(X: np.ndarray, alpha: float):
    """The fit half of build_content_metric.partial_whiten — same eigenvalue floor,
    same convention — returning (mu, W) so a single row can be transformed later."""
    mu = X.mean(0)
    if alpha <= 0:
        return mu, np.eye(X.shape[1])
    cov = np.cov(X - mu, rowvar=False)
    val, vec = np.linalg.eigh(cov)
    val = np.clip(val, 1e-3 * val.max(), None)
    return mu, vec @ np.diag(val ** (-alpha / 2)) @ vec.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-derive even if cached")
    ap.add_argument("--url", default=QDRANT_URL)
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT.relative_to(ROOT)} exists — use --force to re-derive")
        return 0

    cm = json.loads((ART / "content_metric.json").read_text())
    npz = np.load(ART / "content_metric.npz", allow_pickle=True)
    meta_fields = cm["meta_fields"]
    profile = cm["default_profile"]
    genre_vocab = [str(g) for g in npz["genre_vocab"]]
    genre_protos = np.asarray(npz["genre_protos"], dtype=np.float64)
    proto_by_genre = dict(zip(genre_vocab, genre_protos))
    meta_mean = np.asarray(npz["meta_mean"], dtype=np.float64)
    meta_std = np.asarray(npz["meta_std"], dtype=np.float64)

    client = QdrantClient(url=args.url, timeout=300)
    text_rows, meta_rows, genres = [], [], []
    offset = None
    while True:
        pts, offset = client.scroll(CONTENT_SRC, limit=2048, offset=offset,
                                    with_payload=True, with_vectors=True)
        for p in pts:
            m = (p.payload or {}).get("metadata", {}) or {}
            text_rows.append(p.vector)
            genres.append(m.get("genre_primary"))
            row = []
            for k in meta_fields:
                v = m.get(k)
                row.append(float(v) if isinstance(v, (int, float)) else np.nan)
            meta_rows.append(row)
        if offset is None:
            break
    n = len(text_rows)
    print(f"scrolled {n} tracks from {CONTENT_SRC}")

    txt = np.asarray(text_rows, dtype=np.float64)
    txt_mu, _ = partial_whiten_fit(txt, 0.0)

    # z-score with the STORED stats (log1p on followers, exactly as the builder),
    # then fit the ZCA the builder fitted on that z-scored block.
    meta = np.asarray(meta_rows, dtype=np.float64)
    fi = meta_fields.index("artist_followers")
    meta[:, fi] = np.log1p(np.clip(meta[:, fi], 0, None))
    z = (meta - meta_mean) / np.where(meta_std == 0, 1.0, meta_std)
    z[~np.isfinite(z)] = 0.0
    meta_zmu, meta_w = partial_whiten_fit(z, 1.0)

    fallback = genre_protos.mean(0)
    genre_block = np.asarray([proto_by_genre.get(str(g), fallback) for g in genres])
    genre_mu, _ = partial_whiten_fit(genre_block, 0.0)

    np.savez(OUT, txt_mean=txt_mu, meta_zmean=meta_zmu, meta_whiten=meta_w,
             genre_mean=genre_mu, genre_fallback=fallback,
             n=np.asarray(n), profile=np.asarray(profile))
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  txt_mean {txt_mu.shape} · meta_zmean {meta_zmu.shape} · "
          f"meta_whiten {meta_w.shape} · genre_mean {genre_mu.shape}")
    print(f"  profile {profile} · specs {list(cm['specs'][profile])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
