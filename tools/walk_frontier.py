#!/usr/bin/env python3
"""Is the CONJUNCTION reachable — real-listening stride AND a held vibe?

`tools/walk_eval.py` measured that the two shipped engines fail in opposite
directions: the anchored GRU holds the vibe (0.609) but walks 2.5x tighter than the
user really does (median step 0.661 vs a real 0.261), while the champion matches the
real stride (0.317) but does not hold a vibe (0.187). Neither does both.

This sweeps two ORTHOGONAL, DETERMINISTIC controls to find out whether a cell exists
that does:

  * ANCHOR  — pull toward the seed-core centroid. Holds the geist.
      score += a * cos(candidate, seed_centroid)
  * STRIDE  — push away from the track just played. Enlarges each step WITHOUT
      letting the walk leave the seed's neighbourhood, because the anchor still
      applies. This is the lever, not sampling: sampling would loosen the stride too
      but would break the permalink's determinism guarantee (same recipe => same
      journey), which is now a shipped promise.
      score -= s * cos(candidate, previous_track)

Both engines get both controls. The GRU's base score is a raw cosine, so its terms are
raw cosines too — a=0.4, s=0 reproduces the SHIPPED config exactly. The champion's base
is a z-blend of three legs, so its terms are z-normalized over the candidates (a raw
cosine added to a z-blend would be scale-mismatched); its anchor is applied in the RAW
PCA-192 space, which walk_eval could not do and which is the whole point of arm 2.

  predictors/.venv/bin/python tools/walk_frontier.py [--sessions 20] [--steps 20]
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import walk_eval as WE                                     # noqa: E402
sys.path.insert(0, str(ROOT / "predictors"))
import seq_common, seq_blend                               # noqa: E402

# The target window: stride inside the real band's middle, vibe at least the shipped
# engine's 0.609 minus a little slack.
TARGET_STEP = (0.20, 0.38)
TARGET_VIBE = 0.50


def zc(v, mask):
    m = v[mask]
    return (v - m.mean()) / (m.std() + 1e-9)


def make_walk(kind, latents, unit_pca, artist_ids, n_items):
    """Returns walk(seed, steps, cap, a, s) -> picks, for 'gru' or 'champion'."""
    if kind == "gru":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(WE.GRU_ONNX), providers=["CPUExecutionProvider"])

        def raw_score(seq):
            x = latents[seq].astype(np.float32)[None, :, :]
            pred = sess.run(["next"], {"prefix": x})[0][0]
            return unit_pca @ pred
        zterms = False
    else:
        hp = seq_blend.load_hp(str(WE.CHAMP_DIR / "hyperparams.json"))
        art = seq_blend.load_artifact(WE.CHAMP_DIR)
        W = np.load(WE.CHAMP_DIR / "projection.npz")["W"]
        art = seq_blend._projected_artifact(art, W)
        model = seq_blend.load_model(WE.CHAMP_DIR, art, hp)
        fn = seq_blend.build_blend_score_fn(art, model, hp)

        def raw_score(seq):
            return np.asarray(fn(np.asarray(seq, dtype=np.int64)), dtype=np.float64)
        zterms = True

    def walk(seed, steps, cap, a, s):
        seq = list(seed)
        used = set(seq)
        counts = collections.Counter(int(artist_ids[r]) for r in seq)
        anchor = unit_pca[seed].mean(axis=0)
        anchor /= (np.linalg.norm(anchor) or 1.0)
        anch_sim = unit_pca @ anchor                       # RAW space, both engines
        picks = []
        for _ in range(steps):
            base = raw_score(seq).copy()
            ok = np.ones(n_items, dtype=bool)
            ok[list(used)] = False
            if cap:
                for art_id, c in counts.items():
                    if c >= cap and art_id >= 0:
                        ok[artist_ids == art_id] = False
                if not ok.any():
                    ok = np.ones(n_items, dtype=bool)
                    ok[list(used)] = False
            if not ok.any():
                break
            prev_sim = unit_pca @ unit_pca[seq[-1]]
            if zterms:
                sc = base + (a * zc(anch_sim, ok) if a else 0.0) \
                          - (s * zc(prev_sim, ok) if s else 0.0)
            else:
                sc = base + a * anch_sim - s * prev_sim
            sc = np.where(ok, sc, -np.inf)
            r = int(np.argmax(sc))
            picks.append(r); seq.append(r); used.add(r)
            counts[int(artist_ids[r])] += 1
        return picks
    return walk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="/tmp/walk_frontier.json")
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
    real = np.concatenate(real)
    band = (float(np.percentile(real, 10)), float(np.percentile(real, 90)))
    print(f"real median step {np.median(real):.3f} · band [{band[0]:.3f}, {band[1]:.3f}]")

    rng = np.random.default_rng(1337)
    cands = [int(s) for s in art.test_sessions
             if np.asarray(art.session(int(s))).size >= WE.SEED_LEN + 5]
    sessions = [cands[int(i)] for i in rng.permutation(len(cands))[: args.sessions]]
    print(f"{len(sessions)} held-out sessions x {args.steps} steps\n")

    GRIDS = {
        "GRU":      (make_walk("gru", lat, unit, artist_ids, n),
                     [0.0, 0.2, 0.4, 0.8], [0.0, 0.3, 0.6, 1.0]),
        "Champion": (make_walk("champion", lat, unit, artist_ids, n),
                     [0.0, 1.0, 2.0, 4.0], [0.0, 0.5]),
    }
    report = {}
    for kind, (walk, ANCH, STRIDE) in GRIDS.items():
        print(f"=== {kind}  (cap 1; a=anchor, s=stride)")
        print(f"    {'a':>4s}{'s':>5s}{'W1↓':>8s}{'medstep':>9s}{'vibe↑':>7s}"
              f"{'drift':>8s}{'genres':>7s}{'ground↑':>9s}   hit?")
        for a in ANCH:
            for s in STRIDE:
                steps_all, vibes, drifts, genres, grounds = [], [], [], [], []
                for sess_id in sessions:
                    sq = np.asarray(art.session(int(sess_id)), dtype=np.int64)
                    seed, truth = sq[:WE.SEED_LEN], sq[WE.SEED_LEN:]
                    picks = walk(seed, args.steps, 1, a, s)
                    if len(picks) < 2:
                        continue
                    anch = unit[seed].mean(axis=0); anch /= (np.linalg.norm(anch) or 1.0)
                    full = np.concatenate([seed, np.asarray(picks, dtype=np.int64)])
                    steps_all.append(WE.step_cosines(full, unit))
                    v = unit[picks] @ anch
                    vibes.append(float(v.mean()))
                    h = max(1, len(picks) // 4)
                    drifts.append(float(v[-h:].mean() - v[:h].mean()))
                    genres.append(len({int(genre_ids[i]) for i in picks}))
                    tm = truth[mmask[truth]]; pm = [i for i in picks if mmask[i]]
                    if tm.size and pm:
                        grounds.append(float(np.mean(np.max(music[pm] @ music[tm].T, axis=1))))
                sc = np.concatenate(steps_all)
                d = dict(w1=WE.wasserstein1(sc, real), med_step=float(np.median(sc)),
                         vibe=float(np.mean(vibes)), drift=float(np.mean(drifts)),
                         genres=float(np.mean(genres)),
                         ground=float(np.mean(grounds)) if grounds else float("nan"))
                hit = (TARGET_STEP[0] <= d["med_step"] <= TARGET_STEP[1]
                       and d["vibe"] >= TARGET_VIBE)
                report[f"{kind} a={a} s={s}"] = d
                print(f"    {a:4.1f}{s:5.1f}{d['w1']:8.3f}{d['med_step']:9.3f}"
                      f"{d['vibe']:7.3f}{d['drift']:+8.3f}{d['genres']:7.1f}"
                      f"{d['ground']:9.3f}   {'<<< YES' if hit else ''}")
        print()
    Path(args.out).write_text(json.dumps(
        {"real_median": float(np.median(real)), "band": band,
         "target_step": TARGET_STEP, "target_vibe": TARGET_VIBE,
         "sessions": len(sessions), "steps": args.steps, "cells": report}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
