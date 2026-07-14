#!/usr/bin/env python3
"""Next-track SEQUENCE blend predictor: GRU-infonce x train-only Markov.

The productionized winner of the blend experiment: at retrieval time each test
session's two full-vocab score vectors — the GRU-infonce next-latent cosine
scores and the train-only first-order Markov (track bigram + artist/genre
back-off) affinity — are z-normalized OVER THE CANDIDATES (items not already in
the prefix) and combined as

    blend = alpha * z(gru) + (1 - alpha) * z(markov)          (alpha default 0.5)

then fed through the shared leave-last-out eval. z is a monotonic per-vector
transform, so alpha=0 reproduces pure Markov and alpha=1 pure GRU exactly. The
z-norm + prefix-exclusion is byte-identical to the experiment script
(seq_blend_eval.py) that proved the win (Recall@10 0.107 -> 0.173,
MRR 0.069 -> 0.096, paired-delta CI entirely > 0).

Everything model-specific is REUSED, not reimplemented:
  * seq_nexttrack.fit          -- the EXACT GRU training procedure (seeding,
                                  val split, objective, early stopping) so the
                                  trained weights match a standalone seq-nexttrack
                                  run bit-for-bit,
  * seq_nexttrack.SeqNextLatent -- the recurrent model (predict_next -> next
                                  latent -> cosine over the vocab),
  * seq_baselines.markov_scorer -- the train-only Markov per-item full-vocab
                                  score_fn (THE bar),
  * seq_common.eval_from_scores / rank_metrics / write_outputs / make_fixture
                                  -- identical ranking rules, metrics shape, and
                                  output files as every other sequence run.

One train invocation does all of it: fit the GRU, build the Markov scorer, run
the z-norm blend eval, write metrics.json + predictions.json (SAME shapes as
seq-nexttrack) plus model.pt + hyperparams.json for reproducibility.

CLI (mirrors seq_nexttrack; the registry passes a `--hyperparams` FILE, an
inline JSON string is also accepted):

  seq_blend.py train --dataset <dir> --output <run_dir> --hyperparams <json|path>
  seq_blend.py fixture --output <dir>          # write a synthetic artifact

Run under the shared predictor venv (torch 2.12.0+cpu / numpy)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from seq_common import (
    emit,
    eval_from_scores,
    load_artifact,
    make_fixture,
    write_outputs,
)
from seq_baselines import markov_scorer
from seq_nexttrack import SEED, SeqNextLatent, fit

# Defaults ARE the winning blend config (this predictor productionizes it):
# hidden 256, gru, infonce, 40 epochs, patience 6, alpha 0.5. The GRU knobs not
# listed as tunables (tau/dropout/num_layers/val_fraction) sit at the same
# values seq-nexttrack used for the win so a default run reproduces it. `seed`
# defaults to seq_nexttrack's module SEED so training is deterministic and
# matches a standalone seq-nexttrack GRU exactly.
DEFAULTS = {
    "hidden": 256,          # recurrent hidden width (winning width)
    "arch": "gru",          # "gru" | "lstm"
    "loss": "infonce",      # "cosine" | "infonce" (infonce won)
    "epochs": 40,
    "patience": 6,          # early-stop patience (epochs; 0 = off)
    "lr": 1e-3,
    "seed": SEED,           # 1337 — deterministic; matches seq-nexttrack
    "alpha": 0.5,           # blend weight on the GRU leg (1-alpha on Markov)
    "batch_size": 128,
    "k": 10,                # Recall@k reported in metrics.recall_at_k
    # GRU knobs kept fixed at the winning config (not exposed in the registry,
    # but overridable via inline hyperparams for experiments).
    "tau": 0.07,            # InfoNCE temperature
    "dropout": 0.1,
    "num_layers": 1,
    "bidirectional": False,
    "residual": False,
    "val_fraction": 0.15,   # held-out slice of TRAIN sessions for early stop
}

EPS = 1e-9  # matches seq_blend_eval.py's z-norm epsilon exactly


def load_hp(spec: str) -> dict:
    """Accept a path to a JSON file (registry convention) or an inline JSON
    string. Merges over the winning-config DEFAULTS and validates."""
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    user = json.loads(raw)
    hp = {**DEFAULTS, **user}
    # Backward-compat with the old `layers` key (same as seq_nexttrack).
    if "num_layers" not in user and "layers" in user:
        hp["num_layers"] = int(user["layers"])
    hp["num_layers"] = int(hp["num_layers"])
    hp["bidirectional"] = bool(hp["bidirectional"])
    hp["residual"] = bool(hp["residual"])
    hp["seed"] = int(hp["seed"])
    hp["alpha"] = float(hp["alpha"])
    assert hp["arch"] in ("gru", "lstm"), "arch must be gru|lstm"
    assert hp["loss"] in ("cosine", "infonce"), "loss must be cosine|infonce"
    assert 0.0 <= hp["alpha"] <= 1.0, "alpha must be in [0, 1]"
    assert hp["num_layers"] >= 1, "num_layers must be >= 1"
    assert hp["epochs"] >= 1 and hp["batch_size"] >= 1
    return hp


def gru_score_fn(model: SeqNextLatent, latents: np.ndarray):
    """Full-vocab cosine score_fn(prefix) for a trained GRU — identical to
    seq_nexttrack's eval-time scoring and seq_blend_eval's build_gru_score_fn."""
    item_norm = torch.nn.functional.normalize(torch.from_numpy(latents), dim=1)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(latents[prefix][None, :, :])  # (1, T, D)
        with torch.no_grad():
            pred = model.predict_next(x)                    # (1, D) prefix -> next
            pred = torch.nn.functional.normalize(pred, dim=1)
            scores = (pred @ item_norm.t()).squeeze(0)      # (n_items,)
        return scores.numpy()
    return score_fn


def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    art = load_artifact(dataset)
    latents = art.item_latents
    n_items = art.n_items
    alpha = hp["alpha"]

    # (1) Train the GRU with the EXACT seq-nexttrack procedure + seed.
    model = fit(art, hp, seed=hp["seed"])
    # (2) Build the train-only Markov scorer (THE bar).
    markov_fn = markov_scorer(art)
    emit({"kind": "log", "msg": "built train-only Markov scorer"})
    gru_fn = gru_score_fn(model, latents)

    # (3) z-normalized blend over CANDIDATES (items not in the prefix), fed
    # through the shared eval so ranking rules match every other run exactly.
    # This is byte-identical to seq_blend_eval.py's zblend_fn(alpha).
    def blend_score_fn(prefix: np.ndarray) -> np.ndarray:
        gru = gru_fn(prefix).astype(np.float64)
        mk = markov_fn(prefix).astype(np.float64)
        cand = np.ones(n_items, dtype=bool)
        cand[prefix] = False
        zg = (gru - gru[cand].mean()) / (gru[cand].std() + EPS)
        zm = (mk - mk[cand].mean()) / (mk[cand].std() + EPS)
        return alpha * zg + (1.0 - alpha) * zm

    emit({"kind": "log", "msg":
          f"blending z(gru) x z(markov) at alpha={alpha:.2f} over "
          f"{int(art.test_sessions.size)} test sessions"})
    metrics, predictions = eval_from_scores(art, blend_score_fn, k=hp["k"])
    metrics["alpha"] = alpha  # record the blend weight (extra key, like baselines)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(run_dir, metrics, predictions)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "hyperparams.json").write_text(json.dumps(hp))

    emit({"kind": "log", "msg":
          f"blend alpha={alpha:.2f}  Recall@{hp['k']} {metrics['recall_at_k']:.3f}  "
          f"Recall@20 {metrics['recall_at_20']:.3f}  "
          f"MRR {metrics['mrr']:.3f}  hit@10 {metrics['hit_rate']:.3f}  "
          f"(n_test {metrics['n_test']})"})
    emit({"kind": "done"})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument("--output", required=True, type=Path)
    tr.add_argument("--hyperparams", required=True)  # file path or inline JSON
    fx = sub.add_parser("fixture", help="write a synthetic SEQUENCE artifact")
    fx.add_argument("--output", required=True, type=Path)
    fx.add_argument("--n-items", type=int, default=20)
    fx.add_argument("--n-sessions", type=int, default=40)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    if args.cmd == "train":
        train(args.dataset, args.output, args.hyperparams)
    else:
        make_fixture(args.output, n_items=args.n_items, n_sessions=args.n_sessions)


if __name__ == "__main__":
    main()
