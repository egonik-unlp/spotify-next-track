#!/usr/bin/env python3
"""Part B: evaluate NEW leak-free base learners (content-kNN, 2nd-order Markov,
train-only item2vec) on the next-track retrieval task, scored on BOTH the exact
metrics and the graded-relevance metrics (artist@k / genre@k / artist_mrr), and
test whether any of them — standalone or blended into the champion R+M — beats
the champion (R+M z-blend, exact@10 0.1726).

Reuses, unmodified:
  * seq_common.load_artifact / eval_from_scores  (byte-identical ranking rules
    + the new graded metrics),
  * seq_baselines.markov_scorer                  (M, the bar),
  * seq_nexttrack.SeqNextLatent + the trained champion GRU checkpoint (R leg,
    no retrain),
  * seq_models.content_knn_scorer / markov2_scorer / item2vec_scorer (the new
    legs).

z-blend is byte-identical to seq_blend.train's blend_score_fn generalized to N
legs with per-leg weights: z-score each leg's full-vocab scores OVER THE
CANDIDATES (prefix excluded), sum weighted.

Bootstrap: 2000 resamples over test sessions for recall@10 CI; paired per-session
Δ vs the champion (the decision statistic). Run under the shared predictor venv."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seq_common import eval_from_scores, load_artifact, _relevance_ids  # noqa: E402
from seq_baselines import markov_scorer  # noqa: E402
from seq_nexttrack import SeqNextLatent  # noqa: E402
from seq_models import content_knn_scorer, markov2_scorer, item2vec_scorer  # noqa: E402

INST = ("/home/gonik/Documents/git/snappler/lensing-workspace/"
        "lensing-instances/spotify-next-track")
DATASET = Path(INST) / "data/seq/seq-20260713-014226"
CKPT_DIR = Path("/tmp/claude-1000/-home-gonik-Documents-git-snappler-lensing-"
                "workspace-lensing-instances-spotify-predict-engagement/"
                "29afb52f-e7e5-480b-be6c-cdd959b28caf/scratchpad/topo/t3_width256")
OUT = Path("/tmp/claude-1000/-home-gonik-Documents-git-snappler-lensing-"
           "workspace-lensing-instances-spotify-predict-engagement/"
           "29afb52f-e7e5-480b-be6c-cdd959b28caf/scratchpad")  # scratch json
K = 10
EPS = 1e-9


def build_gru_score_fn(art):
    hp = json.loads((CKPT_DIR / "hyperparams.json").read_text())
    D = art.latent_dim
    model = SeqNextLatent(D, hp["hidden"], hp["arch"], hp["num_layers"],
                          hp["dropout"], bidirectional=hp["bidirectional"],
                          residual=hp["residual"])
    model.load_state_dict(torch.load(CKPT_DIR / "model.pt", map_location="cpu"))
    model.eval()
    latents = art.item_latents
    item_norm = torch.nn.functional.normalize(torch.from_numpy(latents), dim=1)

    def raw(prefix: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(latents[prefix][None, :, :])
        with torch.no_grad():
            pred = torch.nn.functional.normalize(model.predict_next(x), dim=1)
            return (pred @ item_norm.t()).squeeze(0).numpy()

    cache: dict[bytes, np.ndarray] = {}

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        key = prefix.tobytes()
        v = cache.get(key)
        if v is None:
            v = raw(prefix).astype(np.float64)
            cache[key] = v
        return v
    return score_fn


def zblend(legs_weights, n_items):
    """legs_weights: list of (score_fn, weight). Returns a blended score_fn that
    z-normalizes each leg over the candidates (prefix excluded) then sums."""
    def score_fn(prefix: np.ndarray) -> np.ndarray:
        cand = np.ones(n_items, dtype=bool)
        cand[prefix] = False
        out = np.zeros(n_items, dtype=np.float64)
        for fn, w in legs_weights:
            v = np.asarray(fn(prefix), dtype=np.float64)
            z = (v - v[cand].mean()) / (v[cand].std() + EPS)
            out += w * z
        return out
    return score_fn


def per_session_hits(predictions, artist_ids, genre_ids, k=K):
    """Reconstruct per-session exact/artist/genre @k hit booleans from the
    predictions list (aligned across models: eval iterates test sessions in a
    fixed order)."""
    exact, art_h, gen_h = [], [], []
    for p in predictions:
        truth = int(p["actual"])
        topk = [int(i) for i in p["top_k_ids"]][:k]
        exact.append(1.0 if truth in topk else 0.0)
        a_true, g_true = int(artist_ids[truth]), int(genre_ids[truth])
        art_h.append(1.0 if a_true >= 0 and any(artist_ids[i] == a_true for i in topk) else 0.0)
        gen_h.append(1.0 if g_true >= 0 and any(genre_ids[i] == g_true for i in topk) else 0.0)
    return np.array(exact), np.array(art_h), np.array(gen_h)


def main():
    art = load_artifact(DATASET)
    n_items = art.n_items
    artist_ids = _relevance_ids(art, "artist")
    genre_ids = _relevance_ids(art, "genre")
    print(f"dataset: {n_items} items, {int(art.test_sessions.size)} test sessions",
          flush=True)

    t0 = time.time()
    gru = build_gru_score_fn(art)
    markov = markov_scorer(art)
    print("built GRU (cached) + Markov", flush=True)
    cknn_max = content_knn_scorer(art, agg="max")
    cknn_mean = content_knn_scorer(art, agg="mean")
    mk2 = markov2_scorer(art, backoff=True)
    print("built content-kNN(max/mean) + markov2(backoff)", flush=True)
    i2v = item2vec_scorer(art, dim=64, window=5, epochs=5, neg=5, seed=1337)
    print(f"trained item2vec ({time.time()-t0:.0f}s elapsed)", flush=True)

    # (name, score_fn) — standalone legs then blends. Champion first so it's the
    # paired-Δ reference.
    champion = zblend([(gru, 0.5), (markov, 0.5)], n_items)
    models = [
        ("R+M z-blend (CHAMPION)", champion),
        ("M  first-order Markov (bar)", markov),
        ("R  GRU-infonce h256", gru),
        ("C-kNN(max)  content", cknn_max),
        ("C-kNN(mean) content", cknn_mean),
        ("M2 2nd-order Markov (backoff)", mk2),
        ("I  item2vec (train-only)", i2v),
        # blends adding each new leg to the champion (equal-weight thirds)
        ("R+M+C  (+content)", zblend([(gru, 1/3), (markov, 1/3), (cknn_max, 1/3)], n_items)),
        ("R+M+I  (+item2vec)", zblend([(gru, 1/3), (markov, 1/3), (i2v, 1/3)], n_items)),
        ("R+M2   (markov2 for markov)", zblend([(gru, 0.5), (mk2, 0.5)], n_items)),
        ("R+M+M2 (+markov2)", zblend([(gru, 1/3), (markov, 1/3), (mk2, 1/3)], n_items)),
        ("M+C    (markov+content)", zblend([(markov, 0.5), (cknn_max, 0.5)], n_items)),
        ("R+C    (gru+content)", zblend([(gru, 0.5), (cknn_max, 0.5)], n_items)),
        ("R+M+C+I (all four)", zblend([(gru, 0.25), (markov, 0.25), (cknn_max, 0.25), (i2v, 0.25)], n_items)),
    ]

    results = []
    champ_exact = None
    for name, fn in models:
        m, preds = eval_from_scores(art, fn, k=K)
        ex, ah, gh = per_session_hits(preds, artist_ids, genre_ids)
        if champ_exact is None:
            champ_exact = ex  # champion is first
        results.append({"name": name, "metrics": m, "exact": ex, "artist": ah, "genre": gh})
        print(f"  {name:32s} R@10 {m['recall_at_k']:.4f}  "
              f"A@10 {m['artist_recall_at_k']:.4f}  G@10 {m['genre_recall_at_k']:.4f}  "
              f"MRR {m['mrr']:.4f}  aMRR {m['artist_mrr']:.4f}", flush=True)

    # Bootstrap recall@10 CI (2000 resamples) + paired Δ vs champion.
    rng = np.random.default_rng(1337)
    n = len(champ_exact)
    B = 2000
    idx = rng.integers(0, n, size=(B, n))
    for r in results:
        boot = r["exact"][idx].mean(axis=1)
        r["r10_ci"] = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        d = r["exact"] - champ_exact
        bd = d[idx].mean(axis=1)
        r["dvs_champ"] = float(d.mean())
        r["dvs_ci"] = (float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)))

    print("\n=== TABLE (exact + graded + CI + paired-Δ vs champion) ===", flush=True)
    hdr = (f"{'model':34s} {'R@10':>7s} {'[95% CI]':>16s} {'A@10':>6s} "
           f"{'G@10':>6s} {'MRR':>6s} {'aMRR':>6s} {'Δvs★':>8s} {'Δ CI':>18s}")
    print(hdr, flush=True)
    for r in results:
        m = r["metrics"]
        print(f"{r['name']:34s} {m['recall_at_k']:.4f} "
              f"[{r['r10_ci'][0]:.3f},{r['r10_ci'][1]:.3f}] "
              f"{m['artist_recall_at_k']:.4f} {m['genre_recall_at_k']:.4f} "
              f"{m['mrr']:.4f} {m['artist_mrr']:.4f} "
              f"{r['dvs_champ']:+.4f} [{r['dvs_ci'][0]:+.3f},{r['dvs_ci'][1]:+.3f}]",
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    dump = [{"name": r["name"], "metrics": r["metrics"], "r10_ci": r["r10_ci"],
             "dvs_champ": r["dvs_champ"], "dvs_ci": r["dvs_ci"]} for r in results]
    (OUT / "partb_results.json").write_text(json.dumps(dump, indent=2))
    print(f"\nwrote {OUT/'partb_results.json'}", flush=True)


if __name__ == "__main__":
    main()
