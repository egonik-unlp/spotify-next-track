#!/usr/bin/env python3
"""WALK-LEVEL evaluation — scores what this project actually ships.

The objective (domain.toml [metrics], restated by the user 2026-07-31): models are
CREATORS OF SEQUENCES that carry a mood / playlist-geist, enabling exploration of
genres that is COHESIVE. Holding a mood does not mean one genre — different genres are
welcome if they complement each other and preserve the vibe.

`holisticness@10` does not measure that. It is computed on a ONE-SHOT top-10, and its
`mood_coh` is anchored to the PREFIX centroid, which on a walk grows — so a journey
that drifts by a thousand small legal steps scores well at every step while ending in
the opposite region of the space. Drift is invisible by construction.

So this scores GENERATED JOURNEYS from held-out real sessions, on four axes:

  1. TRANSITION MATCH (primary) — do the walk's step-to-step distances look like this
     user's REAL listening? Per playlist-lab's finding: measure TRANSITIONS, NOT
     AVERAGES, calibrated against the user's own sessions. Averaging distance-to-a-
     centroid is the wrong shape: it penalises a legitimate genre move and rewards
     standing still. Reported as 1-Wasserstein distance to the real consecutive-pair
     cosine distribution (lower = more like real listening), plus two legible reads:
     the median step cosine (real: ~0.262) and the share of steps inside the real
     p10-p90 band.
  2. VIBE DRIFT — seed-anchored cosine (mean, and end-minus-start). Transitions alone
     can drift a long way via many individually-legal steps; this catches that.
  3. VARIETY — distinct artists / genres, counted as CREDIT, not cost.
  4. GROUNDING — graded relevance to what the user ACTUALLY played next in that
     session, so a pleasant unrelated drift cannot win (the lesson the crown learned
     when music_rel was added).

Measured in TWO spaces because they answer different questions: the PCA-192 item space
(where the walks and the recorded calibration live) and the sonic musical-distance
space (a purer "sounds alike" read).

Read-only: loads promoted models + the shipped gru.onnx, trains nothing, writes only
its own JSON report.

  predictors/.venv/bin/python tools/walk_eval.py [--sessions 40] [--steps 20]
"""
from __future__ import annotations
import argparse, collections, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "predictors"))
os.chdir(ROOT)

import seq_common                      # noqa: E402
import seq_blend                       # noqa: E402

DS = ROOT / "data/datasets/seq-20260715-131139"
CHAMP_DIR = ROOT / "data/models/best-seq-blend-20260728-005349-507bb"   # gated champion
GRU_ONNX = ROOT / "clients/infinite-playlist/public/model/gru.onnx"     # the SHIPPED engine
ANCHOR_L = 0.4          # the app's own anchor weight
SEED_LEN = 5


# --------------------------------------------------------------------------- #
# engines                                                                     #
# --------------------------------------------------------------------------- #
def gru_engine(latents):
    """The deployed browser engine: onnx GRU over raw PCA-192 + optional seed anchor."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(GRU_ONNX), providers=["CPUExecutionProvider"])
    unit = latents / np.linalg.norm(latents, axis=1, keepdims=True).clip(1e-12)

    def step(seq, anchor, lam):
        x = latents[seq].astype(np.float32)[None, :, :]
        pred = sess.run(["next"], {"prefix": x})[0][0]
        s = unit @ pred
        if anchor is not None and lam:
            s = s + lam * (unit @ anchor)
        return s
    return step


def champion_engine():
    """The promoted blend champion (R'+M+C' + markov_gate), full-vocab scores."""
    hp = seq_blend.load_hp(str(CHAMP_DIR / "hyperparams.json"))
    art = seq_blend.load_artifact(CHAMP_DIR)
    if hp.get("projection"):
        W = np.load(CHAMP_DIR / "projection.npz")["W"]
        art = seq_blend._projected_artifact(art, W)
    model = seq_blend.load_model(CHAMP_DIR, art, hp)
    fn = seq_blend.build_blend_score_fn(art, model, hp)
    # Its geometry is the PROJECTED space; anchoring there does not hold raw-space mood
    # (measured 2026-07-31), so the anchor is exposed but expected to be weak.
    P = art.item_latents.astype(np.float64)
    P = P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-12)

    def step(seq, anchor, lam):
        s = np.asarray(fn(np.asarray(seq, dtype=np.int64)), dtype=np.float64)
        if anchor is not None and lam:
            a = P[anchor] if isinstance(anchor, (list, np.ndarray)) and np.ndim(anchor) == 1 else None
            if a is None:
                s = s + lam * (P @ anchor)
        return s
    return step, P


def walk(step_fn, seed, steps, cap, anchor, lam, artist_ids, n_items):
    seq = list(seed)
    used = set(seq)
    counts = collections.Counter(int(artist_ids[r]) for r in seq)
    picks = []
    for _ in range(steps):
        s = np.asarray(step_fn(seq, anchor, lam), dtype=np.float64).copy()
        ok = np.ones(n_items, dtype=bool)
        ok[list(used)] = False
        if cap:
            for a, c in counts.items():
                if c >= cap and a >= 0:
                    ok[artist_ids == a] = False
            if not ok.any():                      # cap exhausted: fall back
                ok = np.ones(n_items, dtype=bool)
                ok[list(used)] = False
        if not ok.any():
            break
        s = np.where(ok, s, -np.inf)
        r = int(np.argmax(s))
        picks.append(r); seq.append(r); used.add(r)
        counts[int(artist_ids[r])] += 1
    return picks


# --------------------------------------------------------------------------- #
# metrics                                                                     #
# --------------------------------------------------------------------------- #
def wasserstein1(a, b):
    """1-Wasserstein between two 1-D samples (no scipy dependency)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    q = np.linspace(0, 1, 512)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def step_cosines(idx, unit):
    if len(idx) < 2:
        return np.zeros(0)
    A = unit[idx[:-1]]
    B = unit[idx[1:]]
    return np.sum(A * B, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="/tmp/walk_eval.json")
    args = ap.parse_args()

    art = seq_common.load_artifact(DS)
    n = art.n_items
    lat = art.item_latents.astype(np.float64)
    unit_pca = lat / np.linalg.norm(lat, axis=1, keepdims=True).clip(1e-12)
    artist_ids = seq_common._relevance_ids(art, "artist")
    genre_ids = seq_common._relevance_ids(art, "genre")
    music, mmask = seq_common._load_music_vectors(art)
    have_sonic = music is not None
    print(f"items {n} · sonic vectors {'yes' if have_sonic else 'NO'}")

    # ---- the REAL band: this user's own consecutive transitions (train only) ----
    real_pca, real_son = [], []
    for s in art.train_sessions:
        sq = np.asarray(art.session(int(s)), dtype=np.int64)
        if sq.size < 2:
            continue
        real_pca.append(step_cosines(sq, unit_pca))
        if have_sonic:
            m = mmask[sq]
            if m.sum() >= 2:
                real_son.append(step_cosines(sq[m], music))
    real_pca = np.concatenate(real_pca)
    real_son = np.concatenate(real_son) if real_son else np.zeros(0)
    band = (float(np.percentile(real_pca, 10)), float(np.percentile(real_pca, 90)))
    print(f"real train transitions: n={real_pca.size} median {np.median(real_pca):.3f} "
          f"p10-p90 [{band[0]:.3f}, {band[1]:.3f}]")

    # ---- held-out sessions long enough to seed AND to have a real continuation ----
    rng = np.random.default_rng(1337)
    cands = [int(s) for s in art.test_sessions
             if np.asarray(art.session(int(s))).size >= SEED_LEN + 5]
    pick = rng.permutation(len(cands))[: args.sessions]
    sessions = [cands[int(i)] for i in pick]
    print(f"scoring {len(sessions)} held-out sessions, {args.steps} steps each\n")

    gru_step = gru_engine(lat)
    champ_step, _P = champion_engine()

    ARMS = [
        ("GRU +anchor cap1  (SHIPPED)", gru_step, ANCHOR_L, 1),
        ("GRU +anchor cap3", gru_step, ANCHOR_L, 3),
        ("GRU  no-anchor cap1", gru_step, 0.0, 1),
        ("Champion cap1", champ_step, 0.0, 1),
        ("Champion cap3", champ_step, 0.0, 3),
    ]

    rows = {name: collections.defaultdict(list) for name, _, _, _ in ARMS}
    for si, s in enumerate(sessions):
        sq = np.asarray(art.session(int(s)), dtype=np.int64)
        seed, truth = sq[:SEED_LEN], sq[SEED_LEN:]
        a_pca = unit_pca[seed].mean(axis=0)
        a_pca /= (np.linalg.norm(a_pca) or 1.0)
        for name, step_fn, lam, cap in ARMS:
            picks = walk(step_fn, seed, args.steps, cap, a_pca, lam, artist_ids, n)
            if len(picks) < 2:
                continue
            full = np.concatenate([seed, np.asarray(picks, dtype=np.int64)])
            r = rows[name]
            r["step_pca"].append(step_cosines(full, unit_pca))
            r["vibe"].append(float(np.mean(unit_pca[picks] @ a_pca)))
            half = max(1, len(picks) // 4)
            r["drift"].append(float(np.mean(unit_pca[picks[-half:]] @ a_pca)
                                    - np.mean(unit_pca[picks[:half]] @ a_pca)))
            r["artists"].append(len({int(artist_ids[i]) for i in picks}))
            r["genres"].append(len({int(genre_ids[i]) for i in picks}))
            if have_sonic:
                tm = truth[mmask[truth]]
                pm = [i for i in picks if mmask[i]]
                if tm.size and pm:
                    r["ground"].append(float(np.mean(np.max(music[pm] @ music[tm].T, axis=1))))
                r["step_son"].append(step_cosines(np.asarray([i for i in full if mmask[i]]), music))
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(sessions)} sessions")

    print(f"\n{'arm':30s}{'W1↓':>7s}{'med step':>9s}{'in-band':>9s}{'vibe↑':>7s}"
          f"{'drift':>8s}{'artists':>8s}{'genres':>7s}{'ground↑':>8s}")
    print("-" * 93)
    report = {}
    for name, _, _, _ in ARMS:
        r = rows[name]
        if not r["step_pca"]:
            continue
        sc = np.concatenate(r["step_pca"])
        w1 = wasserstein1(sc, real_pca)
        inband = float(np.mean((sc >= band[0]) & (sc <= band[1])))
        g = float(np.mean(r["ground"])) if r["ground"] else float("nan")
        report[name] = dict(w1=w1, med_step=float(np.median(sc)), in_band=inband,
                            vibe=float(np.mean(r["vibe"])), drift=float(np.mean(r["drift"])),
                            artists=float(np.mean(r["artists"])), genres=float(np.mean(r["genres"])),
                            ground=g, n_sessions=len(r["step_pca"]))
        d = report[name]
        print(f"{name:30s}{d['w1']:7.3f}{d['med_step']:9.3f}{d['in_band']:9.1%}"
              f"{d['vibe']:7.3f}{d['drift']:+8.3f}{d['artists']:8.1f}{d['genres']:7.1f}{d['ground']:8.3f}")
    print(f"\nreference — the user's REAL sessions: med step {np.median(real_pca):.3f}, "
          f"in-band 80.0% by construction, W1 0.000")
    Path(args.out).write_text(json.dumps(
        {"band": band, "real_median": float(np.median(real_pca)),
         "n_real_transitions": int(real_pca.size), "steps": args.steps,
         "seed_len": SEED_LEN, "arms": report}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
