#!/usr/bin/env python3
"""Matched-sample HEAD-TO-HEAD of the walk-surface finalists.

The earlier sweeps each found a cell inside the target window (real-listening stride
AND a held vibe), but on DIFFERENT samples — the GRU/champion frontier at 20 sessions x
20 steps, the dual-tower arms at 12 x 15. A vibe gap of 0.588 vs 0.624 is well inside
what that mismatch alone could produce, so nothing was actually established.

This runs every arm on the IDENTICAL sessions and seeds, and reports PAIRED bootstrap
differences against the shipped configuration — the same instrument the rest of this
project uses. Per-session metrics are chosen so they can be paired:

  vibe       mean cosine of the generated tracks to the seed-core centroid   (higher better)
  stride_err |median step cosine - the user's real median|                   (LOWER better)
  drift      end-quarter minus start-quarter vibe                           (nearer 0 better)
  genres     distinct genres in the journey                                  (higher better)
  ground     graded relevance to what the user ACTUALLY played next          (higher better)

`stride_err` replaces the distributional W1 used earlier: W1 is computed over a pooled
sample and cannot be paired per session, which is exactly what made the previous
comparisons unfalsifiable.

  predictors/.venv/bin/python tools/walk_headtohead.py [--sessions 40] [--steps 20]
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import walk_eval as WE                                          # noqa: E402
sys.path.insert(0, str(ROOT / "predictors"))
import seq_common, seq_blend, seq_dualgru                       # noqa: E402

DUAL_LL = "best-seq-dualgru-20260725-143502-18b4a"     # latent/latent, fusion_layers=0
DUAL_CM = "best-seq-dualgru-20260725-143502-3e5d4"     # latent/cummean, fusion_layers=0


def paired(base, arm, n=2000, seed=1337):
    a = np.asarray(base, dtype=np.float64)
    b = np.asarray(arm, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    d = (b - a)[m]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bt = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="/tmp/walk_headtohead.json")
    args = ap.parse_args()

    art = seq_common.load_artifact(WE.DS)
    n = art.n_items
    lat = art.item_latents.astype(np.float64)
    unit = lat / np.linalg.norm(lat, axis=1, keepdims=True).clip(1e-12)
    artist_ids = seq_common._relevance_ids(art, "artist")
    genre_ids = seq_common._relevance_ids(art, "genre")
    music, mmask = seq_common._load_music_vectors(art)

    real = []
    for s in art.train_sessions:
        sq = np.asarray(art.session(int(s)), dtype=np.int64)
        if sq.size >= 2:
            real.append(WE.step_cosines(sq, unit))
    real_med = float(np.median(np.concatenate(real)))
    print(f"real median step {real_med:.3f}")

    # ---- score functions, built ONCE ----
    import onnxruntime as ort
    gsess = ort.InferenceSession(str(WE.GRU_ONNX), providers=["CPUExecutionProvider"])

    def gru_score(seq):
        x = lat[seq].astype(np.float32)[None, :, :]
        return unit @ gsess.run(["next"], {"prefix": x})[0][0]

    def dual_score(name):
        d = ROOT / "data/models" / name
        hp = seq_dualgru.load_hp(str(d / "hyperparams.json"))
        a = seq_dualgru.load_artifact(d)
        m = seq_dualgru.load_model(d, a, hp)
        fn = seq_dualgru.build_score_fn(m, a.item_latents)
        return lambda seq: np.asarray(fn(np.asarray(seq, dtype=np.int64)), dtype=np.float64)

    hp = seq_blend.load_hp(str(WE.CHAMP_DIR / "hyperparams.json"))
    cart = seq_blend.load_artifact(WE.CHAMP_DIR)
    cart = seq_blend._projected_artifact(cart, np.load(WE.CHAMP_DIR / "projection.npz")["W"])
    cmodel = seq_blend.load_model(WE.CHAMP_DIR, cart, hp)
    cfn = seq_blend.build_blend_score_fn(cart, cmodel, hp)

    def champ_score(seq):
        return np.asarray(cfn(np.asarray(seq, dtype=np.int64)), dtype=np.float64)

    # (label, score_fn, anchor, stride, z_terms)
    ARMS = [
        ("GRU shipped      a0.4 s0.0", gru_score, 0.4, 0.0, False),
        ("GRU tuned        a0.4 s0.3", gru_score, 0.4, 0.3, False),
        ("dual l/l    f0   a0.8 s0.4", dual_score(DUAL_LL), 0.8, 0.4, False),
        ("dual l/cummean f0 a0.8 s0.5", dual_score(DUAL_CM), 0.8, 0.5, False),
        ("champion         a1.0 s0.5", champ_score, 1.0, 0.5, True),
    ]
    print(f"{len(ARMS)} arms, identical sessions\n")

    rng = np.random.default_rng(1337)
    cands = [int(s) for s in art.test_sessions
             if np.asarray(art.session(int(s))).size >= WE.SEED_LEN + 5]
    sessions = [cands[int(i)] for i in rng.permutation(len(cands))[: args.sessions]]

    def zc(v, mask):
        m = v[mask]
        return (v - m.mean()) / (m.std() + 1e-9)

    per = {lbl: collections.defaultdict(list) for lbl, *_ in ARMS}
    for si, sid in enumerate(sessions):
        sq = np.asarray(art.session(int(sid)), dtype=np.int64)
        seed, truth = sq[:WE.SEED_LEN], sq[WE.SEED_LEN:]
        anch = unit[seed].mean(axis=0); anch /= (np.linalg.norm(anch) or 1.0)
        anch_sim = unit @ anch
        tm = truth[mmask[truth]]
        for lbl, score, a, s, zt in ARMS:
            seq = list(seed); used = set(seq)
            counts = collections.Counter(int(artist_ids[r]) for r in seq)
            picks = []
            for _ in range(args.steps):
                base = score(seq).copy()
                ok = np.ones(n, dtype=bool); ok[list(used)] = False
                for aid, c in counts.items():
                    if c >= 1 and aid >= 0:
                        ok[artist_ids == aid] = False
                if not ok.any():
                    ok = np.ones(n, dtype=bool); ok[list(used)] = False
                if not ok.any():
                    break
                prev = unit @ unit[seq[-1]]
                sc = (base + (a * zc(anch_sim, ok) if a else 0) - (s * zc(prev, ok) if s else 0)
                      if zt else base + a * anch_sim - s * prev)
                r = int(np.argmax(np.where(ok, sc, -np.inf)))
                picks.append(r); seq.append(r); used.add(r)
                counts[int(artist_ids[r])] += 1
            if len(picks) < 2:
                continue
            full = np.concatenate([seed, np.asarray(picks, dtype=np.int64)])
            v = unit[picks] @ anch
            h = max(1, len(picks) // 4)
            p = per[lbl]
            p["vibe"].append(float(v.mean()))
            p["stride_err"].append(abs(float(np.median(WE.step_cosines(full, unit))) - real_med))
            p["drift"].append(float(v[-h:].mean() - v[:h].mean()))
            p["genres"].append(float(len({int(genre_ids[i]) for i in picks})))
            pm = [i for i in picks if mmask[i]]
            p["ground"].append(float(np.mean(np.max(music[pm] @ music[tm].T, axis=1)))
                               if (tm.size and pm) else np.nan)
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(sessions)}")

    FIELDS = [("vibe", "↑"), ("stride_err", "↓"), ("drift", "~0"), ("genres", "↑"), ("ground", "↑")]
    print(f"\nMEANS over {len(sessions)} identical sessions")
    print(f"{'arm':29s}" + "".join(f"{f+g:>13s}" for f, g in FIELDS))
    print("-" * 94)
    means = {}
    for lbl, *_ in ARMS:
        p = per[lbl]
        means[lbl] = {f: float(np.nanmean(p[f])) for f, _ in FIELDS}
        print(f"{lbl:29s}" + "".join(f"{means[lbl][f]:13.3f}" for f, _ in FIELDS))

    base_lbl = ARMS[0][0]
    print(f"\nPAIRED Δ vs '{base_lbl.strip()}' (2000 resamples, rng 1337) — CI excluding 0 = real")
    out = {"real_median": real_med, "sessions": len(sessions), "steps": args.steps,
           "means": means, "paired": {}}
    for lbl, *_ in ARMS[1:]:
        print(f"  {lbl.strip()}")
        out["paired"][lbl] = {}
        for f, _ in FIELDS:
            d, lo, hi = paired(per[base_lbl][f], per[lbl][f])
            flag = "CI>0" if lo > 0 else ("CI<0" if hi < 0 else "straddles")
            out["paired"][lbl][f] = dict(delta=d, lo=lo, hi=hi, flag=flag)
            print(f"      Δ{f:11s} {d:+8.4f} [{lo:+.4f}, {hi:+.4f}]  {flag}")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
