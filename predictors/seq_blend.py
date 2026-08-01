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

from torch import nn

from seq_common import (
    SeqArtifact,
    emit,
    eval_from_scores,
    load_artifact,
    make_fixture,
    predict_ranking,
    write_outputs,
)
from seq_baselines import build_train_transitions, markov_scorer
from seq_models import content_knn_scorer
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
    "alpha": 0.5,           # blend weight on the GRU leg (1-alpha on Markov);
                            # applies ONLY to the 2-leg blend (content=False)
    "content": False,       # add the content-kNN third leg (R+M+C champion).
                            # When True the blend is the EQUAL-THIRDS z-average
                            # z(gru)+z(markov)+z(content) over the candidates
                            # (alpha is ignored) — the best-on-record recipe.
    "content_agg": "max",   # content-kNN prefix aggregation: "max" | "mean"
    # EVIDENCE GATE on the Markov leg (2026-07-27). `_zcand` is scale-invariant,
    # so it rescales a pure artist/genre back-off (n_u=0, 55.6% of canonical test
    # queries) back to a full one-third weight. When True the leg ABSTAINS there
    # and the divisor renormalizes. Default OFF => existing definitions are
    # bit-identical. Offline screen on the champion: 0.21174 -> 0.23201.
    "markov_gate": False,
    # LEARNED CONTENT PROJECTION (the R'+M+C' best-on-record recipe, 2026-07-18).
    # When True, a linear metric map of the item-latent space is fit from TRAIN
    # consecutive pairs (leak-free) toward next-track adjacency; the GRU and the
    # content-kNN leg then retrieve in that PROJECTED space (Markov is item-ID
    # based, unchanged) — i.e. the 3-leg content recipe on a supervised geometry.
    # projection=True implies content=True (the 3-leg blend). The projection.npz
    # is written to the run dir and baked into the promoted model for serving.
    "projection": False,
    "projection_rank": 0,             # 0 => full rank (= latent_dim)
    "projection_objective": "infonce",  # "infonce" (in-batch) | "bpr" (pairwise)
    "projection_tau": 0.07,           # InfoNCE temperature for the projection fit
    "projection_epochs": 40,
    "projection_lr": 1e-3,
    "batch_size": 128,
    "k": 10,                # Recall@k reported in metrics.recall_at_k
    # HARD per-artist cap on the top-k (2026-07-30). None/0 = off (bit-identical).
    # Unlike MMR (soft, score-space, needs sonic vectors) a cap cannot be
    # outscored, which is what a steep same-artist relevance gradient requires.
    "artist_cap": None,
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
    hp["content"] = bool(hp["content"])
    hp["markov_gate"] = bool(hp.get("markov_gate", False))
    hp["projection"] = bool(hp["projection"])
    hp["projection_rank"] = int(hp["projection_rank"])
    hp["projection_epochs"] = int(hp["projection_epochs"])
    hp["projection_tau"] = float(hp["projection_tau"])
    hp["projection_lr"] = float(hp["projection_lr"])
    # A learned projection retrieves the GRU + content legs in the projected
    # space — i.e. the 3-leg R'+M+C' recipe; force content on.
    if hp["projection"]:
        hp["content"] = True
    assert hp["arch"] in ("gru", "lstm"), "arch must be gru|lstm"
    assert hp["loss"] in ("cosine", "infonce"), "loss must be cosine|infonce"
    assert 0.0 <= hp["alpha"] <= 1.0, "alpha must be in [0, 1]"
    assert hp["content_agg"] in ("max", "mean"), "content_agg must be max|mean"
    assert hp["projection_objective"] in ("infonce", "bpr"), \
        "projection_objective must be infonce|bpr"
    assert hp["projection_rank"] >= 0, "projection_rank must be >= 0 (0 = full rank)"
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


def _zcand(v: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Z-normalize a score vector over the CANDIDATE items only (matches
    seq_blend_eval.py's z-norm exactly)."""
    return (v - v[cand].mean()) / (v[cand].std() + EPS)


def build_blend_score_fn(art, model: SeqNextLatent, hp: dict):
    """Assemble the full-vocab z-blend score_fn(prefix) from a (trained or
    reloaded) GRU + the train-only Markov leg + the optional content-kNN leg.

    SHARED by train and predict, so a promoted model ranks byte-identically to
    how it was evaluated:
      2-leg: alpha*z(gru)+(1-alpha)*z(markov)  (byte-identical to seq_blend_eval)
      3-leg (content=true): the EQUAL-THIRDS z-average of the three legs — the
      best-on-record R+M+C recipe; alpha is ignored.

    `markov_gate=true` ABSTAINS the Markov leg on queries where it has no
    track-level evidence. `markov_scorer` mixes its bigram row with an
    artist/genre back-off by `trust = n_u/(n_u+8)` (`seq_baselines.py:198`), so
    at n_u=0 the leg is PURE back-off — but `_zcand` is scale-invariant and
    rescales that evidence-free vector back to unit variance, letting it speak
    at a full one-third weight. On the canonical split that is 795/1431 (55.6%)
    of queries. Gating drops M there and renormalizes the divisor; queries with
    n_u>0 are bit-identical to the ungated blend. Default OFF, so every existing
    model definition reproduces exactly."""
    n_items = art.n_items
    alpha = hp["alpha"]
    content = hp["content"]
    gate = hp.get("markov_gate", False)
    gru_fn = gru_score_fn(model, art.item_latents)
    markov_fn = markov_scorer(art)                       # train-only (THE bar)
    emit({"kind": "log", "msg": "built train-only Markov scorer"})
    # Track-level bigram support n_u, for the evidence gate only. Same train-only
    # transitions markov_scorer builds internally — leak-free by construction.
    track_bigram = build_train_transitions(art)["track_bigram"] if gate else None
    if gate:
        emit({"kind": "log", "msg":
              "markov_gate ON — Markov leg abstains where the last prefix item "
              "has zero train bigram support"})
    content_fn = content_knn_scorer(art, agg=hp["content_agg"]) if content else None
    if content:
        emit({"kind": "log", "msg":
              f"built content-kNN leg (agg={hp['content_agg']}) on the item latents"})

    def blend_score_fn(prefix: np.ndarray) -> np.ndarray:
        cand = np.ones(n_items, dtype=bool)
        cand[prefix] = False
        zg = _zcand(gru_fn(prefix).astype(np.float64), cand)
        # w_m = 0 abstains the Markov leg; the divisor renormalizes so the
        # surviving legs keep unit total weight.
        w_m = 1.0
        if gate and sum(track_bigram.get(int(prefix[-1]), {}).values()) == 0:
            w_m = 0.0
        if content_fn is None:
            if w_m == 0.0:
                return zg
            return alpha * zg + (1.0 - alpha) * _zcand(
                markov_fn(prefix).astype(np.float64), cand)
        zm = _zcand(markov_fn(prefix).astype(np.float64), cand)
        zc = _zcand(content_fn(prefix).astype(np.float64), cand)
        return (zg + w_m * zm + zc) / (2.0 + w_m)

    return blend_score_fn


def load_model(model_dir: Path, art, hp: dict) -> SeqNextLatent:
    """Reconstruct the trained GRU from a promoted model dir's model.pt (the
    same weights train saved) at the artifact's latent dim + the hyperparams."""
    model = SeqNextLatent(
        latent_dim=art.latent_dim, hidden=hp["hidden"], arch=hp["arch"],
        num_layers=hp["num_layers"], dropout=hp["dropout"],
        bidirectional=hp["bidirectional"], residual=hp["residual"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Learned content projection (R'+M+C')                                        #
# --------------------------------------------------------------------------- #
def _train_pairs(art) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive within-TRAIN-session (u, v) item pairs, self-loops dropped
    (Markov convention). Leak-free — train sessions only."""
    us, vs = [], []
    for s in art.train_sessions:
        seq = art.session(int(s)).astype(np.int64)
        for a, b in zip(seq[:-1], seq[1:]):
            if int(a) != int(b):
                us.append(int(a)); vs.append(int(b))
    return np.asarray(us, dtype=np.int64), np.asarray(vs, dtype=np.int64)


def fit_projection(art, hp: dict) -> np.ndarray:
    """Learn a linear metric map W (rank x D) of the item-latent space toward
    next-track adjacency, from TRAIN consecutive pairs. Objective: InfoNCE
    (in-batch) or BPR (pairwise). Deterministic given hp['seed']. Leak-free
    (train pairs + intrinsic latents only). Returns W as float32 (rank, D)."""
    seed = hp["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Z = torch.from_numpy(art.item_latents.astype(np.float32))
    D = Z.shape[1]
    rank = hp["projection_rank"] or D
    tau = hp["projection_tau"]
    obj = hp["projection_objective"]
    us, vs = _train_pairs(art)
    n = len(us)
    W = nn.Linear(D, rank, bias=False)
    (nn.init.eye_ if rank == D else nn.init.orthogonal_)(W.weight)
    opt = torch.optim.Adam(W.parameters(), lr=hp["projection_lr"])
    us_t, vs_t = torch.from_numpy(us), torch.from_numpy(vs)
    n_items = art.n_items
    batch = 512
    for _ in range(hp["projection_epochs"]):
        perm = torch.from_numpy(rng.permutation(n))
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            a = torch.nn.functional.normalize(W(Z[us_t[idx]]), dim=1)
            p = torch.nn.functional.normalize(W(Z[vs_t[idx]]), dim=1)
            if obj == "infonce":
                logits = (a @ p.t()) / tau
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.arange(a.shape[0]))
            else:  # BPR: pairwise, one sampled negative per pair
                neg = torch.from_numpy(rng.integers(0, n_items, size=idx.shape[0]))
                zn = torch.nn.functional.normalize(W(Z[neg]), dim=1)
                loss = -torch.nn.functional.logsigmoid(
                    (a * p).sum(1) - (a * zn).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return W.weight.detach().numpy().astype(np.float32)  # (rank, D)


def _projected_artifact(art, W: np.ndarray):
    """A SeqArtifact whose item_latents are projected into the W-space and
    row-normalized (cosine = dot). Sessions/items/split are shared — so Markov
    (item-ID based) is unchanged while the GRU + content-kNN retrieve in the
    projected space."""
    Zp = art.item_latents.astype(np.float32) @ W.T          # (n_items, rank)
    Zp = Zp / (np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-12)
    manifest = dict(art.manifest)
    manifest["latent_dim"] = int(W.shape[0])
    return SeqArtifact(
        manifest=manifest, sessions=art.sessions, offsets=art.offsets,
        item_latents=Zp.astype(np.float32), train_sessions=art.train_sessions,
        test_sessions=art.test_sessions, items=art.items)


def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    art = load_artifact(dataset)
    run_dir.mkdir(parents=True, exist_ok=True)
    alpha = hp["alpha"]
    content = hp["content"]

    # (0) Optional learned content projection: fit W from train pairs, save it
    # for serving, and swap the artifact into the projected space so the GRU +
    # content-kNN retrieve there (Markov is unchanged) — the R'+M+C' recipe.
    if hp["projection"]:
        emit({"kind": "log", "msg":
              f"fitting content projection (obj={hp['projection_objective']}, "
              f"rank={hp['projection_rank'] or art.latent_dim}, "
              f"{hp['projection_epochs']}ep) from train consecutive pairs"})
        W = fit_projection(art, hp)
        np.savez(run_dir / "projection.npz", W=W)
        art = _projected_artifact(art, W)
        emit({"kind": "log", "msg":
              f"projected item latents -> dim {art.latent_dim} (R'+M+C')"})

    # (1) Train the GRU with the EXACT seq-nexttrack procedure + seed, then
    # (2/3) assemble the shared z-blend score_fn (Markov + optional content).
    model = fit(art, hp, seed=hp["seed"])
    blend_score_fn = build_blend_score_fn(art, model, hp)

    proj = hp["projection"]
    legs_desc = (
        "z(gru') x z(markov) x z(content') equal-thirds [projected]" if proj else
        "z(gru) x z(markov) x z(content) equal-thirds" if content else
        f"z(gru) x z(markov) at alpha={alpha:.2f}")
    emit({"kind": "log", "msg":
          f"blending {legs_desc} over "
          f"{int(art.test_sessions.size)} test sessions"})
    metrics, predictions = eval_from_scores(
        art, blend_score_fn, k=hp["k"],
        mmr_lambda=hp.get("mmr_lambda"), mmr_pool=int(hp.get("mmr_pool", 200)),
        artist_cap=hp.get("artist_cap"))
    metrics["alpha"] = alpha  # record the blend weight (extra key, like baselines)
    metrics["legs"] = "R'+M+C'" if proj else ("R+M+C" if content else "R+M")

    write_outputs(run_dir, metrics, predictions)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "hyperparams.json").write_text(json.dumps(hp))

    blend_tag = ("R'+M+C' thirds [proj]" if proj else
                 "R+M+C thirds" if content else f"alpha={alpha:.2f}")
    emit({"kind": "log", "msg":
          f"blend {blend_tag}  Recall@{hp['k']} {metrics['recall_at_k']:.3f}  "
          f"Recall@20 {metrics['recall_at_20']:.3f}  "
          f"MRR {metrics['mrr']:.3f}  hit@10 {metrics['hit_rate']:.3f}  "
          f"(n_test {metrics['n_test']})"})
    emit({"kind": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Serving-time ranking: reload the promoted blend (GRU weights + the
    train-only Markov/content legs rebuilt from the baked sequence artifact),
    rank the vocab for the caller's session prefix, and write the lensing
    predictions.json shape ([{row_id, predicted, top_k_ids}]).

    The promoted model dir carries BOTH the run's model.pt/hyperparams and a
    snapshot of the training sequence artifact (item_latents/items/sessions),
    so `load_artifact(model_dir)` reconstructs the exact vocab + train-only
    statistics the model was evaluated on — ranking is byte-identical to eval."""
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(model_dir)
    if hp["projection"]:
        # Rebuild the projected space from the baked projection.npz so the GRU
        # (trained in-projection) + content-kNN rank byte-identically to eval.
        W = np.load(model_dir / "projection.npz")["W"]
        art = _projected_artifact(art, W)
    model = load_model(model_dir, art, hp)
    blend_score_fn = build_blend_score_fn(art, model, hp)
    predict_ranking(art, blend_score_fn, input_dir, output, k=hp["k"],
                    mmr_lambda=hp.get("mmr_lambda"),
                    mmr_pool=int(hp.get("mmr_pool", 200)),
                    artist_cap=hp.get("artist_cap"))
    emit({"kind": "done"})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument("--output", required=True, type=Path)
    tr.add_argument("--hyperparams", required=True)  # file path or inline JSON
    pr = sub.add_parser("predict", help="rank the next track for a session prefix")
    pr.add_argument("--model", required=True, type=Path)
    pr.add_argument("--input", required=True, type=Path)  # dir with prefix.json
    pr.add_argument("--output", required=True, type=Path)
    fx = sub.add_parser("fixture", help="write a synthetic SEQUENCE artifact")
    fx.add_argument("--output", required=True, type=Path)
    fx.add_argument("--n-items", type=int, default=20)
    fx.add_argument("--n-sessions", type=int, default=40)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    if args.cmd == "train":
        train(args.dataset, args.output, args.hyperparams)
    elif args.cmd == "predict":
        predict(args.model, args.input, args.output)
    else:
        make_fixture(args.output, n_items=args.n_items, n_sessions=args.n_sessions)


if __name__ == "__main__":
    main()
