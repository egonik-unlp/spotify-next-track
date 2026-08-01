#!/usr/bin/env python3
"""Why does the GROUNDING leg fail to separate any arm?

Grounding is the only leg that stops a pleasant-but-unrelated journey from winning —
the same role `music_rel` plays in the current crown, added in 2026-07-25 after the
3-factor version ranked `seq-mood` first at recall@10 0.024. In the walk head-to-head
it STRADDLES ZERO for all four arms, so it currently carries no weight. A crown whose
grounding leg cannot discriminate would repeat that failure, which is why this has to be
understood before the primary metric is changed.

Current definition, per session: mean over generated tracks of the MAX cosine, in the
sonic musical-distance space, to any track the user actually played after the seed.

Four candidate explanations, all testable here:
  A. SHORT TRUTH — the real continuation is (session length − seed), often just a few
     tracks, so the target set is tiny and noisy.
  B. SATURATION — max-cosine over any non-trivial target set is near 1 for almost
     anything, so the statistic has no headroom.
  C. VARIANCE — per-session spread dwarfs between-arm spread, so 40 sessions cannot
     resolve a real difference.
  D. WRONG AGGREGATION — max hides the distribution; mean / top-k / a hit-rate at a
     threshold may separate where max does not.

Reports the diagnostics for each, then re-scores the same journeys under alternative
aggregations to see whether any of them discriminates.

  predictors/.venv/bin/python tools/walk_ground_diag.py [--sessions 40] [--steps 20]
"""
from __future__ import annotations
import argparse, collections, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import walk_eval as WE                                     # noqa: E402
sys.path.insert(0, str(ROOT / "predictors"))
import seq_common, seq_dualgru                             # noqa: E402

DUAL_CM = "best-seq-dualgru-20260725-143502-3e5d4"
DUAL_LL = "best-seq-dualgru-20260725-143502-18b4a"


def paired(a, b, n=2000, seed=1337):
    a, b = np.asarray(a, float), np.asarray(b, float)
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
    args = ap.parse_args()

    art = seq_common.load_artifact(WE.DS)
    n = art.n_items
    lat = art.item_latents.astype(np.float64)
    unit = lat / np.linalg.norm(lat, axis=1, keepdims=True).clip(1e-12)
    aid = seq_common._relevance_ids(art, "artist")
    music, mmask = seq_common._load_music_vectors(art)
    if music is None:
        sys.exit("sonic vectors unavailable — grounding cannot be diagnosed")

    import onnxruntime as ort
    gsess = ort.InferenceSession(str(WE.GRU_ONNX), providers=["CPUExecutionProvider"])

    def dual(name):
        d = ROOT / "data/models" / name
        hp = seq_dualgru.load_hp(str(d / "hyperparams.json"))
        a = seq_dualgru.load_artifact(d)
        m = seq_dualgru.load_model(d, a, hp)
        fn = seq_dualgru.build_score_fn(m, a.item_latents)
        return lambda seq: np.asarray(fn(np.asarray(seq, dtype=np.int64)), float)

    ARMS = [
        ("GRU shipped   a0.4 s0.0", lambda seq: unit @ gsess.run(
            ["next"], {"prefix": lat[seq].astype(np.float32)[None, :, :]})[0][0], 0.4, 0.0),
        ("dual l/cummean a0.8 s0.5", dual(DUAL_CM), 0.8, 0.5),
        ("dual l/l       a0.8 s0.4", dual(DUAL_LL), 0.8, 0.4),
    ]

    rng = np.random.default_rng(1337)
    cands = [int(s) for s in art.test_sessions
             if np.asarray(art.session(int(s))).size >= WE.SEED_LEN + 5]
    sessions = [cands[int(i)] for i in rng.permutation(len(cands))[: args.sessions]]

    # ---- generate once per arm, keep the journeys so every variant re-scores the same walks
    journeys = {lbl: {} for lbl, *_ in ARMS}
    truths, truth_len = {}, {}
    for sid in sessions:
        sq = np.asarray(art.session(int(sid)), dtype=np.int64)
        seed, truth = sq[:WE.SEED_LEN], sq[WE.SEED_LEN:]
        truths[sid] = truth[mmask[truth]]
        truth_len[sid] = int(truths[sid].size)
        anch = unit[seed].mean(axis=0); anch /= (np.linalg.norm(anch) or 1.0)
        asim = unit @ anch
        for lbl, score, a, s in ARMS:
            seq, used = list(seed), set(seed)
            counts = collections.Counter(int(aid[r]) for r in seq)
            picks = []
            for _ in range(args.steps):
                sc = score(seq) + a * asim - s * (unit @ unit[seq[-1]])
                ok = np.ones(n, dtype=bool); ok[list(used)] = False
                for x, c in counts.items():
                    if c >= 1 and x >= 0:
                        ok[aid == x] = False
                if not ok.any():
                    ok = np.ones(n, dtype=bool); ok[list(used)] = False
                r = int(np.argmax(np.where(ok, sc, -np.inf)))
                picks.append(r); seq.append(r); used.add(r); counts[int(aid[r])] += 1
            journeys[lbl][sid] = [i for i in picks if mmask[i]]

    tl = np.array([truth_len[s] for s in sessions], float)
    print(f"A. TRUTH LENGTH — tracks the user actually played after the seed")
    print(f"   median {np.median(tl):.0f} · mean {tl.mean():.1f} · min {tl.min():.0f} "
          f"· max {tl.max():.0f} · share with <=3: {(tl <= 3).mean():.0%}")

    base = ARMS[0][0]
    variants = {
        "max (current)":  lambda S: float(np.mean(np.max(S, axis=1))),
        "mean":           lambda S: float(np.mean(S)),
        "top3 mean":      lambda S: float(np.mean(np.sort(S, axis=1)[:, -3:].mean(axis=1)))
                                    if S.shape[1] >= 3 else float(np.mean(np.max(S, axis=1))),
        "hit@0.7":        lambda S: float(np.mean(np.max(S, axis=1) >= 0.7)),
        "hit@0.85":       lambda S: float(np.mean(np.max(S, axis=1) >= 0.85)),
    }
    per = {v: {lbl: [] for lbl, *_ in ARMS} for v in variants}
    for sid in sessions:
        tm = truths[sid]
        if tm.size == 0:
            continue
        for lbl, *_ in ARMS:
            pm = journeys[lbl][sid]
            if not pm:
                continue
            S = music[pm] @ music[tm].T          # (generated, truth)
            for v, f in variants.items():
                per[v][lbl].append(f(S))

    print(f"\nB. SATURATION — distribution of the per-track max cosine (all arms pooled)")
    allmax = []
    for sid in sessions:
        tm = truths[sid]
        if tm.size == 0: continue
        for lbl, *_ in ARMS:
            pm = journeys[lbl][sid]
            if pm:
                allmax.append(np.max(music[pm] @ music[tm].T, axis=1))
    allmax = np.concatenate(allmax)
    print(f"   p10 {np.percentile(allmax,10):.3f} · median {np.median(allmax):.3f} "
          f"· p90 {np.percentile(allmax,90):.3f} · share >0.9: {(allmax>0.9).mean():.0%}")

    print(f"\nC. VARIANCE — between-arm spread vs per-session spread (current definition)")
    cur = per["max (current)"]
    means = {lbl: float(np.mean(cur[lbl])) for lbl in cur}
    between = max(means.values()) - min(means.values())
    within = float(np.mean([np.std(cur[lbl]) for lbl in cur]))
    print(f"   between-arm range {between:.4f} · mean per-session sd {within:.4f} "
          f"· ratio {between/within:.2f}")
    print(f"   → 40 sessions resolve about {1.96*within/np.sqrt(len(sessions)):.4f}; "
          f"the arms differ by {between:.4f}")

    print(f"\nD. AGGREGATION — paired Δ vs '{base.strip()}', does any variant separate?")
    for v in variants:
        print(f"   {v}")
        for lbl, *_ in ARMS[1:]:
            d, lo, hi = paired(per[v][base], per[v][lbl])
            flag = "CI>0" if lo > 0 else ("CI<0" if hi < 0 else "straddles")
            star = "  <<<" if flag != "straddles" else ""
            print(f"      {lbl:26s} {d:+8.4f} [{lo:+.4f}, {hi:+.4f}] {flag}{star}")


if __name__ == "__main__":
    main()
