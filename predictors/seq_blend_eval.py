#!/usr/bin/env python3
"""Blend experiment: GRU-infonce candidate scores x first-order Markov bigram.

Question: does blending the GRU-infonce next-latent cosine scores with the
train-only first-order Markov (bigram + artist/genre back-off) affinity beat the
Markov bar on the next-track leave-last-out retrieval task?

Everything model-agnostic is REUSED from the existing predictors (no edits to
them):
  * seq_common.load_artifact / eval_from_scores / rank_metrics  -- identical
    prefix-exclusion (next-distinct) + metrics as every other run,
  * seq_baselines.markov_scorer                                  -- the exact
    train-only Markov per-item full-vocab score_fn (THE bar),
  * seq_nexttrack.SeqNextLatent                                  -- the GRU
    class; we load the trained checkpoint and reproduce its predict_next ->
    cosine-to-all-items score vector.

Two blend families, each fed through eval_from_scores so the ranking rules are
byte-identical to the standalone runs:
  1. weighted z-normalized: z-score each full-vocab score over the CANDIDATES
     (items not in the prefix), blend = a*z(gru) + (1-a)*z(markov), a in
     {0,.25,.5,.75,1}. z is a monotonic per-vector transform, so a=0 reproduces
     pure Markov and a=1 pure GRU exactly (endpoint sanity check).
  2. RRF: reciprocal rank fusion over candidate-only ranks,
     blend = 1/(k0+rank_gru) + 1/(k0+rank_markov), k0=60 (scale-free).

Run under the shared predictor venv (torch 2.12.0+cpu)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Import the existing, unmodified predictor modules (this file lives next to
# them in predictors/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seq_common import eval_from_scores, load_artifact, rank_metrics  # noqa: E402
from seq_baselines import markov_scorer  # noqa: E402
from seq_nexttrack import SeqNextLatent  # noqa: E402


DATASET = Path("/home/gonik/Documents/git/snappler/lensing-workspace/"
               "lensing-instances/spotify-next-track/data/seq/seq-20260713-014226")
CKPT_DIR = Path("/tmp/claude-1000/-home-gonik-Documents-git-snappler-lensing-"
                "workspace-lensing-instances-spotify-predict-engagement/"
                "29afb52f-e7e5-480b-be6c-cdd959b28caf/scratchpad/topo/t3_width256")
OUT_BASE = Path("/tmp/claude-1000/-home-gonik-Documents-git-snappler-lensing-"
                "workspace-lensing-instances-spotify-predict-engagement/"
                "29afb52f-e7e5-480b-be6c-cdd959b28caf/scratchpad/blend")

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
RRF_K0 = 60.0
EPS = 1e-9


def build_gru_score_fn(art, ckpt_dir: Path):
    """Load the trained GRU-infonce checkpoint and return a score_fn(prefix) ->
    full-vocab cosine vector, exactly as seq_nexttrack.train does at eval time."""
    hp = json.loads((ckpt_dir / "hyperparams.json").read_text())
    D = art.latent_dim
    model = SeqNextLatent(D, hp["hidden"], hp["arch"], hp["num_layers"],
                          hp["dropout"], bidirectional=hp["bidirectional"],
                          residual=hp["residual"])
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    latents = art.item_latents
    item_norm = torch.nn.functional.normalize(torch.from_numpy(latents), dim=1)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(latents[prefix][None, :, :])  # (1, T, D)
        with torch.no_grad():
            pred = model.predict_next(x)
            pred = torch.nn.functional.normalize(pred, dim=1)
            scores = (pred @ item_norm.t()).squeeze(0)
        return scores.numpy()
    return score_fn, hp


def candidate_ranks(scores: np.ndarray, prefix: np.ndarray, n_items: int) -> np.ndarray:
    """1-based descending rank of every item among the CANDIDATES (prefix items
    pushed to the bottom via -inf). Prefix items get large ranks; they are
    excluded downstream anyway."""
    s = scores.astype(np.float64).copy()
    s[prefix] = -np.inf
    order = np.argsort(-s, kind="stable")
    ranks = np.empty(n_items, dtype=np.float64)
    ranks[order] = np.arange(1, n_items + 1, dtype=np.float64)
    return ranks


def main() -> None:
    art = load_artifact(DATASET)
    n_items = art.n_items
    print(f"dataset {art.manifest['dataset_id']}: {n_items} items, "
          f"{int(art.test_sessions.size)} test sessions", flush=True)

    gru_fn, hp = build_gru_score_fn(art, CKPT_DIR)
    print(f"loaded GRU ckpt: {hp['arch']} h{hp['hidden']} x{hp['num_layers']} "
          f"loss={hp['loss']}", flush=True)
    markov_fn = markov_scorer(art)  # train-only, built exactly like seq-markov
    print("built train-only Markov scorer", flush=True)

    # Precompute both full-vocab score vectors ONCE per test session (the
    # expensive GRU forward + Markov scoring), cached by prefix bytes so every
    # blend config reuses them.
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    n_done = 0
    for s in art.test_sessions:
        seq = art.session(int(s))
        if seq.shape[0] < 2:
            continue
        prefix = seq[:-1].astype(np.int64)
        key = prefix.tobytes()
        if key not in cache:
            cache[key] = (gru_fn(prefix).astype(np.float64),
                          markov_fn(prefix).astype(np.float64))
        n_done += 1
    print(f"precomputed score vectors for {n_done} sessions "
          f"({len(cache)} distinct prefixes)", flush=True)

    def zblend_fn(alpha: float):
        def fn(prefix: np.ndarray) -> np.ndarray:
            gru, mk = cache[prefix.tobytes()]
            cand = np.ones(n_items, dtype=bool)
            cand[prefix] = False
            zg = (gru - gru[cand].mean()) / (gru[cand].std() + EPS)
            zm = (mk - mk[cand].mean()) / (mk[cand].std() + EPS)
            return alpha * zg + (1.0 - alpha) * zm
        return fn

    def rrf_fn(prefix: np.ndarray) -> np.ndarray:
        gru, mk = cache[prefix.tobytes()]
        rg = candidate_ranks(gru, prefix, n_items)
        rm = candidate_ranks(mk, prefix, n_items)
        return 1.0 / (RRF_K0 + rg) + 1.0 / (RRF_K0 + rm)

    # Run every config through the shared eval.
    results: list[tuple[str, dict, list[dict]]] = []
    for a in ALPHAS:
        m, preds = eval_from_scores(art, zblend_fn(a), k=10)
        label = f"z a={a:.2f}"
        results.append((label, m, preds))
        print(f"{label}: R@10={m['recall_at_10']:.4f} R@20={m['recall_at_20']:.4f} "
              f"MRR={m['mrr']:.4f} hit@10={m['hit_rate']:.4f}", flush=True)

    m_rrf, preds_rrf = eval_from_scores(art, rrf_fn, k=10)
    results.append(("RRF k0=60", m_rrf, preds_rrf))
    print(f"RRF k0=60: R@10={m_rrf['recall_at_10']:.4f} "
          f"R@20={m_rrf['recall_at_20']:.4f} MRR={m_rrf['mrr']:.4f} "
          f"hit@10={m_rrf['hit_rate']:.4f}", flush=True)

    # ---- table ----
    print("\n" + "=" * 78)
    print(f"{'config':<16}{'R@10':>10}{'R@20':>10}{'MRR':>10}{'hit@10':>10}"
          f"{'n_test':>10}")
    print("-" * 78)
    bar = results[0][1]  # a=0 == Markov bar
    for label, m, _ in results:
        print(f"{label:<16}{m['recall_at_10']:>10.4f}{m['recall_at_20']:>10.4f}"
              f"{m['mrr']:>10.4f}{m['hit_rate']:>10.4f}{m['n_test']:>10}")
    print("=" * 78)

    # ---- endpoint sanity ----
    r10_a0 = results[0][1]["recall_at_10"]
    r10_a1 = results[4][1]["recall_at_10"]
    print(f"\nendpoint check: a=0 R@10={r10_a0:.4f} (expect ~0.107 Markov), "
          f"a=1 R@10={r10_a1:.4f} (expect ~0.096 GRU)")

    # ---- best config (by R@10, tie-break MRR) ----
    best = max(results, key=lambda r: (r[1]["recall_at_10"], r[1]["mrr"]))
    print(f"best config: {best[0]}  R@10={best[1]['recall_at_10']:.4f} "
          f"MRR={best[1]['mrr']:.4f}")
    beats = best[1]["recall_at_10"] > 0.107 and best[1]["mrr"] > 0.069
    print(f"beats bar (R@10>0.107 AND MRR>0.069)? {beats}")

    # ---- predictions.json for best blend + a=0 control ----
    (OUT_BASE / "best").mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "alpha0").mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "best" / "predictions.json").write_text(json.dumps(best[2]))
    (OUT_BASE / "best" / "metrics.json").write_text(
        json.dumps({**best[1], "config": best[0]}))
    (OUT_BASE / "alpha0" / "predictions.json").write_text(json.dumps(results[0][2]))
    (OUT_BASE / "alpha0" / "metrics.json").write_text(
        json.dumps({**results[0][1], "config": "z a=0.00 (pure Markov bar)"}))
    print(f"\nwrote best -> {OUT_BASE/'best'}  and  a=0 control -> {OUT_BASE/'alpha0'}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
