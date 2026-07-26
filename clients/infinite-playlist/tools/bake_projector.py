#!/usr/bin/env python3
"""Bake the cold-start projector for the CF-Worker backend (OFFLINE build tool —
the deployed backend is pure TS+WASM; this only produces the baked artifact).

Mirrors ../app tools/build_snapshot.py train_projector: a 2-layer MLP mapping
    [bge-m3 text emb (1024) ⊕ z(numerics) ⊕ z(acoustics) ⊕ missing_flags ⊕
     genre_multihot ⊕ album_onehot]
→ the 192-d song_pca192 latent (L2-normalized). Text is embedded with local
BAAI/bge-m3 (weights match Workers AI @cf/baai/bge-m3, which the Worker uses at
runtime). Feature order MUST match worker-core/src/project.rs.

Output: clients/infinite-playlist/public/model/projector.bin  (magic PFP1)

Run: predictors/.venv/bin/python clients/infinite-playlist/tools/bake_projector.py
"""
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parent
REPO = CLIENT.parent.parent
sys.path.insert(0, str(REPO))
from pipeline.embed_content import content_doc  # noqa: E402

ART = REPO / "pipeline" / "artifacts"
OUT = CLIENT / "public" / "model" / "projector.bin"
QDRANT = "http://localhost:6337"
COLL = "spotify_tracks_song_pca192"
TEXT_MODEL_CF = "@cf/baai/bge-m3"
TEXT_DIM = 1024
HIDDEN = 256


def main():
    pp = json.load(open(ART / "song_pca192_preprocess.json"))
    NUMERICS = pp["numerics"]; LOG = set(pp["log_numerics"]); NST = pp["num_stats"]
    ACOUSTICS = pp["acoustics"]; AST = pp["ac_stats"]
    GENRE = pp["genre_vocab"]; gidx = {g: i for i, g in enumerate(GENRE)}
    ALBUM = pp["album_types"]; amap = {a: i for i, a in enumerate(ALBUM)}

    # ---- 1) pull the whole corpus: metadata + target latent -----------------
    from qdrant_client import QdrantClient
    qc = QdrantClient(url=QDRANT, timeout=180)
    metas, Y = [], []
    off = None
    while True:
        pts, off = qc.scroll(COLL, limit=2048, offset=off,
                             with_payload=True, with_vectors=True)
        for p in pts:
            m = (p.payload or {}).get("metadata", {}) or {}
            v = p.vector
            if isinstance(v, dict):
                v = next(iter(v.values()))
            if not m or v is None:
                continue
            metas.append(m); Y.append(v)
        if off is None:
            break
    Y = np.asarray(Y, dtype=np.float32)
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    # bge-m3 on CPU is ~0.35s/doc; a diverse sample trains the MLP fine.
    cap = int(os.environ.get("PROJECTOR_CAP", "5000"))
    if cap and len(metas) > cap:
        sel = np.random.default_rng(0).choice(len(metas), cap, replace=False)
        metas = [metas[i] for i in sel]; Y = Y[sel]
        print(f"[corpus] sampled {cap} of {len(sel)} for training", flush=True)
    print(f"[corpus] {len(metas)} tracks · target dim {Y.shape[1]}", flush=True)

    # ---- 2) bge-m3 text embeddings (local; match @cf/baai/bge-m3) -----------
    import torch as _t
    _t.set_num_threads(max(1, os.cpu_count() or 4))
    from sentence_transformers import SentenceTransformer
    print("[bge-m3] loading local BAAI/bge-m3 ...", flush=True)
    st = SentenceTransformer("BAAI/bge-m3")
    st.max_seq_length = 128           # content_docs are short sentences; cap compute
    docs = [content_doc(m) for m in metas]
    text = st.encode(docs, normalize_embeddings=True, batch_size=64,
                     show_progress_bar=True).astype(np.float32)
    assert text.shape[1] == TEXT_DIM, text.shape

    # ---- 3) assemble features (order must match project.rs) -----------------
    def znum(k, v):
        v = float(v) if isinstance(v, (int, float)) else math.nan
        if k in LOG:
            v = math.log1p(v) if (v == v and v >= 0) else math.nan
        mu, sd = NST[k]["mean"], NST[k]["std"]
        v = mu if not (v == v) else v
        return (v - mu) / sd

    n = len(metas)
    NN = len(NUMERICS); NA = len(ACOUSTICS)
    in_dim = TEXT_DIM + NN + 2 * NA + len(GENRE) + len(ALBUM)
    X = np.zeros((n, in_dim), dtype=np.float32)
    X[:, :TEXT_DIM] = text
    for i, m in enumerate(metas):
        o = TEXT_DIM
        for k in NUMERICS:
            X[i, o] = znum(k, m.get(k)); o += 1
        for k in ACOUSTICS:  # z-values
            v = m.get(k); v = float(v) if isinstance(v, (int, float)) else math.nan
            mu, sd = AST[k]["mean"], AST[k]["std"]
            X[i, o] = ((mu if not (v == v) else v) - mu) / sd; o += 1
        for k in ACOUSTICS:  # missing flags
            v = m.get(k)
            X[i, o] = 0.0 if isinstance(v, (int, float)) else 1.0; o += 1
        gs = list(m.get("sp_genres") or [])
        gp = m.get("genre_primary")
        if gp:
            gs.append(gp)
        for g in gs:
            if g in gidx:
                X[i, o + gidx[g]] = 1.0
        o += len(GENRE)
        at = m.get("album_type") or "__other__"
        if at in amap:
            X[i, o + amap[at]] = 1.0

    # ---- 4) train the MLP ---------------------------------------------------
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)
    net = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    idx = np.arange(n)
    print(f"[train] in_dim={in_dim} (text {TEXT_DIM}) hidden={HIDDEN} out={Y.shape[1]}", flush=True)
    for ep in range(60):
        net.train(); np.random.shuffle(idx); tot = 0.0
        for b in range(0, n, 512):
            bi = idx[b:b + 512]
            xb = Xt[bi]; yb = Yt[bi]
            pred = net(xb)
            pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-9)
            loss = (1.0 - (pred * yb).sum(1)).mean()      # cosine loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(bi)
        if ep % 10 == 9 or ep == 0:
            print(f"  epoch {ep+1:>3}  cosine-loss {tot/n:.4f}", flush=True)

    # ---- 5) fidelity: mean cosine + self-recall@1 on a sample --------------
    net.eval()
    with torch.no_grad():
        P = net(Xt); P = P / (P.norm(dim=1, keepdim=True) + 1e-9)
        cos = float((P * Yt).sum(1).mean())
        s = min(2000, n); rng = np.random.default_rng(0)
        samp = rng.choice(n, s, replace=False)
        Yn = Yt.numpy()
        hit = 0
        for j in samp:
            sims = Yn @ P[j].numpy()
            if int(sims.argmax()) == j:
                hit += 1
    print(f"[fidelity] mean cos(projected, true) = {cos:.3f} · "
          f"self-recall@1 ({s} sample) = {hit/s:.3f}", flush=True)

    # ---- 6) write PFP1 ------------------------------------------------------
    w1 = net[0].weight.detach().numpy().astype("<f4")   # (hidden, in_dim)
    b1 = net[0].bias.detach().numpy().astype("<f4")
    w2 = net[2].weight.detach().numpy().astype("<f4")   # (out, hidden)
    b2 = net[2].bias.detach().numpy().astype("<f4")
    cfg = {
        "text_model": TEXT_MODEL_CF,
        "numerics": NUMERICS,
        "num_stats": {k: {"mean": NST[k]["mean"], "std": NST[k]["std"], "log": (k in LOG)} for k in NUMERICS},
        "acoustics": ACOUSTICS,
        "ac_stats": {k: {"mean": AST[k]["mean"], "std": AST[k]["std"]} for k in ACOUSTICS},
        "genre_vocab": GENRE,
        "album_types": ALBUM,
    }
    cfg_bytes = json.dumps(cfg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with open(OUT, "wb") as f:
        f.write(b"PFP1")
        f.write(struct.pack("<IIIII", 1, in_dim, HIDDEN, Y.shape[1], TEXT_DIM))
        f.write(w1.tobytes()); f.write(b1.tobytes())
        f.write(w2.tobytes()); f.write(b2.tobytes())
        f.write(struct.pack("<I", len(cfg_bytes))); f.write(cfg_bytes)
    print(f"[done] wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
