#!/usr/bin/env python3
"""Reduce a NAMED vector of an existing Qdrant collection to a PCA-<latent>
single-vector collection — the dim-matched item space for a sequence artifact.

Used to turn the musical-distance metric's `balanced` (517-d) named vector into
`spotify_tracks_song_pca192_balanced` (192-d), so the validated block-weighted /
partial-whitened metric geometry can be compared apples-to-apples (dim 192) with
the other item spaces under the champion sequence split. Copies ids + payloads
verbatim so the sequence vocab / items.json stay byte-identical.

Run: predictors/.venv/bin/python pipeline/reduce_named_vector.py \
       --src spotify_tracks_content_metric --vector-name balanced \
       --latent 192 --dst spotify_tracks_song_pca192_balanced
"""
import argparse

import numpy as np
from sklearn.decomposition import PCA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_URL = "http://localhost:6337"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source collection")
    ap.add_argument("--vector-name", required=True,
                    help="named vector on the source to reduce")
    ap.add_argument("--latent", type=int, default=192)
    ap.add_argument("--dst", required=True, help="destination collection")
    ap.add_argument("--artifact-prefix", default=None,
                    help="basename for saved artifacts (default <dst>)")
    args = ap.parse_args()
    prefix = args.artifact_prefix or args.dst
    np.random.seed(42)
    client = QdrantClient(url=QDRANT_URL, timeout=180)

    # ---- read the named vector + payloads in a fixed id order ----------------
    ids, vecs, payloads = [], [], []
    offset = None
    while True:
        pts, offset = client.scroll(args.src, limit=2048, offset=offset,
                                    with_payload=True,
                                    with_vectors=[args.vector_name])
        for p in pts:
            v = p.vector[args.vector_name]
            if v is None:
                continue
            ids.append(p.id); vecs.append(v); payloads.append(p.payload)
        if offset is None:
            break
    X = np.asarray(vecs, dtype=np.float32)
    n = len(ids)
    print(f"loaded {n} points; source vector {args.vector_name!r} dim {X.shape[1]}")

    # ---- PCA reduce ----------------------------------------------------------
    L = args.latent
    pca = PCA(n_components=L, random_state=42)
    Z = pca.fit_transform(X).astype(np.float32)
    evr = pca.explained_variance_ratio_
    print(f"PCA latent {L}: cumulative EVR {evr.sum():.4f} "
          f"(component 1 {evr[0]:.4f}, last {evr[-1]:.5f})")

    from pathlib import Path
    import json as _json
    art = Path("pipeline/artifacts"); art.mkdir(parents=True, exist_ok=True)
    np.savez(art / f"{prefix}.npz",
             components=pca.components_.astype(np.float32),
             mean=pca.mean_.astype(np.float32),
             explained_variance_ratio=evr.astype(np.float32))
    _json.dump({
        "din": int(X.shape[1]), "latent": int(L), "method": "pca-reduce",
        "source_collection": args.src, "source_vector": args.vector_name,
        "cumulative_evr": float(evr.sum()),
    }, open(art / f"{prefix}_preprocess.json", "w"), indent=1)
    print(f"saved artifacts to {art}/ ({prefix}.npz + {prefix}_preprocess.json)")

    # ---- L2-normalize + upsert (same ids + payloads) -------------------------
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    client.recreate_collection(args.dst, vectors_config=VectorParams(size=L, distance=Distance.COSINE))
    for i in range(0, n, 256):
        client.upsert(args.dst, points=[PointStruct(id=ids[j], vector=Zn[j].tolist(), payload=payloads[j])
                                        for j in range(i, min(i + 256, n))])
        print(f"  upserted {min(i + 256, n)}/{n}", flush=True)
    print(f"done: {args.dst!r} has {client.get_collection(args.dst).points_count} points, dim {L}")


if __name__ == "__main__":
    main()
