#!/usr/bin/env python3
"""Curiosity: how does the DUAL-TOWER family score on the walk metrics?

Why it is worth asking. Two structural facts make dualgru interesting on the walk
surface in a way the one-shot leaderboard could never show:

  1. It retrieves in the RAW item-latent space (`build_score_fn(model,
     art.item_latents)`) — no learned projection. So a raw-space mood anchor is NATIVE
     to its geometry, unlike the champion, whose InfoNCE projection strips exactly the
     sound-similarity structure an anchor needs.
  2. One promoted arm uses `view_b = cummean` — that tower sees the CAUSAL MEAN of the
     prefix. That is structurally a "hold the running average" signal, i.e. a mood
     anchor baked into the architecture rather than bolted on at retrieval time.

The record dismissed this family: every `fusion_layers=1` arm lost to a single h256 GRU
on recall (CI<0), the best `fusion_layers=0` arm merely tied, and the `cummean/cummean`
arm was flagged as DEGENERATE for topping the old 3-factor holisticness at recall 0.066.
But every one of those judgements was made on one-shot next-track accuracy. If the
objective is a sequence that holds a geist, "degenerate" may have been the wrong word.

  predictors/.venv/bin/python tools/walk_dualgru.py [--sessions 12] [--steps 15]
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import walk_eval as WE                                    # noqa: E402
sys.path.insert(0, str(ROOT / "predictors"))
import seq_common, seq_dualgru                            # noqa: E402

ARMS = {
    "dual latent/latent f0": "best-seq-dualgru-20260725-143502-18b4a",
    "dual latent/CUMMEAN f0": "best-seq-dualgru-20260725-143502-3e5d4",
    "dual latent/latent f1": "best-seq-dualgru-20260726-161801-0df6f",
}
CELLS = [(0.8, 0.40), (0.8, 0.45)]   # s=0.3 gave med step 0.42, s=0.6 gave 0.10 — the
                                     # 0.20-0.38 window is between them, so bracket it.


def engine(model_name, unit):
    d = ROOT / "data/models" / model_name
    hp = seq_dualgru.load_hp(str(d / "hyperparams.json"))
    art = seq_dualgru.load_artifact(d)
    model = seq_dualgru.load_model(d, art, hp)
    fn = seq_dualgru.build_score_fn(model, art.item_latents)
    return fn, hp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--out", default="/tmp/walk_dualgru.json")
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
    print(f"real median step {np.median(real):.3f}\n")

    rng = np.random.default_rng(1337)
    cands = [int(s) for s in art.test_sessions
             if np.asarray(art.session(int(s))).size >= WE.SEED_LEN + 5]
    sessions = [cands[int(i)] for i in rng.permutation(len(cands))[: args.sessions]]

    report = {}
    print(f"{'arm':26s}{'a':>4s}{'s':>5s}{'W1↓':>8s}{'medstep':>9s}{'vibe↑':>7s}"
          f"{'drift':>8s}{'genres':>7s}{'ground↑':>9s}")
    print("-" * 83)
    for label, mname in ARMS.items():
        try:
            fn, hp = engine(mname, unit)
        except Exception as e:                              # noqa: BLE001
            print(f"{label:26s} SKIP ({type(e).__name__}: {e})")
            continue
        for a, s in CELLS:
            steps_all, vibes, drifts, genres, grounds = [], [], [], [], []
            for sid in sessions:
                sq = np.asarray(art.session(int(sid)), dtype=np.int64)
                seed, truth = sq[:WE.SEED_LEN], sq[WE.SEED_LEN:]
                anch = unit[seed].mean(axis=0); anch /= (np.linalg.norm(anch) or 1.0)
                anch_sim = unit @ anch
                seq = list(seed); used = set(seq)
                counts = collections.Counter(int(artist_ids[r]) for r in seq)
                picks = []
                for _ in range(args.steps):
                    base = np.asarray(fn(np.asarray(seq, dtype=np.int64)), dtype=np.float64)
                    ok = np.ones(n, dtype=bool); ok[list(used)] = False
                    for aid, c in counts.items():
                        if c >= 1 and aid >= 0:
                            ok[artist_ids == aid] = False
                    if not ok.any():
                        ok = np.ones(n, dtype=bool); ok[list(used)] = False
                    if not ok.any():
                        break
                    sc = base + a * anch_sim - s * (unit @ unit[seq[-1]])
                    r = int(np.argmax(np.where(ok, sc, -np.inf)))
                    picks.append(r); seq.append(r); used.add(r)
                    counts[int(artist_ids[r])] += 1
                if len(picks) < 2:
                    continue
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
            sc_all = np.concatenate(steps_all)
            d = dict(w1=WE.wasserstein1(sc_all, real), med_step=float(np.median(sc_all)),
                     vibe=float(np.mean(vibes)), drift=float(np.mean(drifts)),
                     genres=float(np.mean(genres)),
                     ground=float(np.mean(grounds)) if grounds else float("nan"))
            report[f"{label} a={a} s={s}"] = d
            hit = 0.20 <= d["med_step"] <= 0.38 and d["vibe"] >= 0.50
            print(f"{label:26s}{a:4.1f}{s:5.1f}{d['w1']:8.3f}{d['med_step']:9.3f}"
                  f"{d['vibe']:7.3f}{d['drift']:+8.3f}{d['genres']:7.1f}{d['ground']:9.3f}"
                  f"{'   <<< YES' if hit else ''}")
        print()
    Path(args.out).write_text(json.dumps(
        {"real_median": float(np.median(real)), "sessions": len(sessions),
         "steps": args.steps, "cells": report}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
