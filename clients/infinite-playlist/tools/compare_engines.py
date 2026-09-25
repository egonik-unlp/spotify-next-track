#!/usr/bin/env python3
"""Headless SIDE-BY-SIDE of every baked engine, over the browser's exact path.

This is the pre-publish smoke test AND the thing that makes the listening test worth
doing: it reads the same `manifest.json` / `catalog.json` / `latents.i16` the browser
reads, runs the same ONNX graphs, and reproduces `nearest()` from `app.js` term for term
(cosine + anchor*seed-centroid - stride*previous, hard per-artist cap, library-only
output). So the journeys printed here are the journeys you will hear.

It also prints the two walk-surface quantities the engines' labels make claims about —
`vibe` (mean cosine of the generated tracks to the seed centroid) and the median step
cosine, whose distance from the user's real 0.261 is `stride_err`. Those are the claims
your ears are being asked to adjudicate, so seeing them next to the tracklist is the
point: if `bank_v1` prints a visibly lower vibe AND sounds fine, the metric is not
measuring the experience.

  predictors/.venv/bin/python clients/infinite-playlist/tools/compare_engines.py
  predictors/.venv/bin/python clients/infinite-playlist/tools/compare_engines.py --seed-query "bonobo" --steps 15
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

DIR = Path(__file__).resolve().parent.parent / "public" / "model"
REAL_MEDIAN_STEP = 0.261        # the user's real listening median (tools/walk_eval.py)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-query", default="",
                    help="substring match on artist/name to pick seeds; default = a fixed "
                         "deterministic library slice so runs are comparable")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cap", type=int, default=1, help="max tracks per artist (1 = Explore)")
    args = ap.parse_args()

    man = json.loads((DIR / "manifest.json").read_text())
    cat = json.loads((DIR / "catalog.json").read_text())
    n, dim, scale = man["n"], man["dim"], man["scale"]
    raw = (np.frombuffer((DIR / "latents.i16").read_bytes(), dtype="<i2")
           .astype(np.float32) * scale).reshape(n, dim)
    norm = np.linalg.norm(raw, axis=1); norm[norm == 0] = 1.0
    il = np.array([1 if c.get("il", 1) else 0 for c in cat], dtype=bool)

    # ---- engines, exactly as app.js resolves them (defaults overridden by manifest) ----
    engines = {
        "gru":     dict(file="gru.onnx",     anchor=0.4, stride=0.0, label="GRU — original"),
        "dual":    dict(file="dualgru.onnx", anchor=0.8, stride=0.5, label="Dual-tower (DEPLOYED)"),
        "bank_e1": dict(file="banke1.onnx",  anchor=0.8, stride=0.5, label="Dual-tower, regularized"),
        "bank_v1": dict(file="bankv1.onnx",  anchor=0.8, stride=0.5, label="Bank ×3 + centroid"),
    }
    for key, blk in (("dual", "dualgru"), ("bank_e1", "bank_e1"), ("bank_v1", "bank_v1")):
        b = man.get(blk)
        if not b:
            continue
        for f in ("anchor", "stride", "file"):
            if b.get(f) is not None:
                engines[key][f] = b[f]
        if b.get("label"):
            engines[key]["label"] = b["label"]
    engines = {k: v for k, v in engines.items() if (DIR / v["file"]).exists()}

    # ---- seeds: deterministic so every engine starts from the identical prefix -------
    lib = np.flatnonzero(il)
    if args.seed_query:
        q = args.seed_query.lower()
        hits = [r for r in lib
                if q in (cat[r].get("artist") or "").lower()
                or q in (cat[r].get("name") or "").lower()]
        if not hits:
            raise SystemExit(f"no library track matches {args.seed_query!r}")
        seeds = hits[: args.seeds]
    else:
        seeds = [int(r) for r in lib[:: max(1, len(lib) // 977)][: args.seeds]]

    print("SEEDS")
    for r in seeds:
        print(f"  {cat[r]['artist']} — {cat[r]['name']}  [{cat[r].get('genre')}]")

    # anchor = normalized mean of L2-normalized seed latents (centroidOf; dominantCore is
    # the identity for <= 10 seeds, which is this case)
    c = np.zeros(dim, dtype=np.float32)
    for r in seeds:
        c += raw[r] / norm[r]
    anchor_vec = c / (np.linalg.norm(c) or 1.0)

    def nearest(pred, used, counts, cap, anchor_l, stride, prev):
        ok = il.copy()
        ok[list(used)] = False
        if cap > 0:
            for r in np.flatnonzero(ok):
                if counts.get(cat[r].get("artist"), 0) >= cap:
                    ok[r] = False
        if not ok.any():
            return nearest(pred, used, counts, 0, anchor_l, stride, prev) if cap > 0 else -1
        idx = np.flatnonzero(ok)
        sc = (raw[idx] @ pred) / norm[idx]
        if anchor_l:
            sc = sc + anchor_l * (raw[idx] @ anchor_vec) / norm[idx]
        if stride and prev >= 0:
            sc = sc - stride * (raw[idx] @ raw[prev]) / (norm[idx] * norm[prev])
        return int(idx[int(np.argmax(sc))])

    results = {}
    for key, eng in engines.items():
        sess = ort.InferenceSession(str(DIR / eng["file"]), providers=["CPUExecutionProvider"])
        seq = list(seeds)
        used = set(seq)
        counts = {}
        for r in seq:
            counts[cat[r].get("artist")] = counts.get(cat[r].get("artist"), 0) + 1
        picks = []
        for _ in range(args.steps):
            x = raw[seq][None, :, :].astype(np.float32)
            pred = sess.run(["next"], {"prefix": x})[0][0]
            r = nearest(pred, used, counts, args.cap, eng["anchor"], eng["stride"],
                        seq[-1] if seq else -1)
            if r < 0:
                break
            picks.append(r); seq.append(r); used.add(r)
            counts[cat[r].get("artist")] = counts.get(cat[r].get("artist"), 0) + 1

        u = raw[picks] / norm[picks][:, None]
        vibe = float((u @ anchor_vec).mean())
        full = np.array(list(seeds) + picks)
        uf = raw[full] / norm[full][:, None]
        steps = np.einsum("ij,ij->i", uf[:-1], uf[1:])
        med = float(np.median(steps))
        results[key] = dict(picks=picks, vibe=vibe, med=med,
                            stride_err=abs(med - REAL_MEDIAN_STEP),
                            genres=len({cat[r].get("genre") for r in picks}),
                            eng=eng)

    for key, res in results.items():
        eng = res["eng"]
        print(f"\n=== {key}  a{eng['anchor']}/s{eng['stride']}  — {eng['label']}")
        print(f"    vibe {res['vibe']:.3f}   median step {res['med']:.3f} "
              f"(stride_err {res['stride_err']:.3f} vs real {REAL_MEDIAN_STEP})   "
              f"distinct genres {res['genres']}")
        for i, r in enumerate(res["picks"], 1):
            print(f"    {i:2d}. {cat[r]['artist'][:28]:28s} {cat[r]['name'][:34]:34s} "
                  f"[{(cat[r].get('genre') or '')[:18]}]")

    print("\nOVERLAP with the deployed engine (shared tracks / steps, order-insensitive)")
    base = set(results["dual"]["picks"]) if "dual" in results else set()
    for key, res in results.items():
        if key == "dual":
            continue
        shared = len(base & set(res["picks"]))
        print(f"  {key:9s} {shared:2d}/{len(res['picks']):2d} shared with dual")
    print("\nIf two engines share nearly every track, the A/B cannot tell you anything —")
    print("that is the check this script exists to make BEFORE you spend time listening.")


if __name__ == "__main__":
    main()
