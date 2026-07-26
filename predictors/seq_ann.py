#!/usr/bin/env python3
"""Next-track SEQUENCE predictor — the GRU's NON-recurrent feed-forward twin.

This is the "A leg" of the next-track ensemble. Where seq_nexttrack.py runs a
GRU/LSTM over the session-prefix latents, this model has NO recurrence at all:
it pools the prefix into a fixed-width bag-of-prefix representation and feeds it
through a plain MLP to predict the next track's 64-dim latent. Retrieval then
ranks the whole vocab by cosine to that prediction — the exact same interface
(`score_fn(prefix) -> full-vocab cosine vector`) the blend / stacker harness
already consumes for the recurrent R leg.

Representation (per prefix step): concat of
  * the MEAN-POOL of the prefix latents seen so far, and
  * the LAST item's latent
=> a 2*latent_dim vector (128-d for the 64-d latents). An MLP with 2 hidden
layers (configurable width, ReLU, dropout) maps it to a 64-d next-latent.

Objective / training — IDENTICAL to the GRU's unidirectional path, reused
verbatim so this really is the GRU's feed-forward twin:
  * per-step teacher forcing over the prefix — for a session [a,b,c,d] the
    supervised steps are b|[a], c|[a,b], d|[a,b,c]; at step t the pooled
    prefix (bag of [seq[0..t]] + last item seq[t]) predicts seq[t+1]'s latent,
  * the SAME masked InfoNCE (or cosine) loss (seq_nexttrack.step_loss),
  * the SAME padded batching (seq_nexttrack.make_batches),
  * the SAME held-out-TRAIN-slice early stopping.
The pooling is causal: at step t only latents [0..t] enter the mean, so there
is no leakage of the target (bag-of-prefix, no recurrence).

CLI (mirrors seq_nexttrack; the registry passes a `--hyperparams` FILE, an
inline JSON string is also accepted):

  seq_ann.py train --dataset <dir> --output <run_dir> --hyperparams <json|path>
  seq_ann.py fixture --output <dir>          # write a synthetic artifact

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
    emit,
    eval_from_scores,
    load_artifact,
    make_fixture,
    predict_ranking,
    training_pairs,
    write_outputs,
)
# Reuse the GRU's EXACT batching + loss so the objective is byte-for-byte the
# same (this is the recurrent model's feed-forward twin, not a reimplementation).
from seq_nexttrack import SEED, make_batches, step_loss

DEFAULTS = {
    "hidden": 128,          # MLP hidden width (both hidden layers)
    "dropout": 0.1,
    "loss": "infonce",      # "cosine" | "infonce"
    "tau": 0.07,            # InfoNCE temperature
    "epochs": 30,
    "lr": 1e-3,
    "batch_size": 128,
    "patience": 5,          # early-stop patience (epochs; 0 = off)
    "val_fraction": 0.15,   # held-out slice of TRAIN sessions
    "seed": SEED,           # 1337 — deterministic
    "k": 10,                # Recall@k reported in metrics.recall_at_k
}


def load_hp(spec: str) -> dict:
    """Accept a path to a JSON file (registry convention) or an inline JSON
    string. Merges over DEFAULTS and validates."""
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    user = json.loads(raw)
    hp = {**DEFAULTS, **user}
    hp["seed"] = int(hp["seed"])
    assert hp["loss"] in ("cosine", "infonce"), "loss must be cosine|infonce"
    assert hp["epochs"] >= 1 and hp["batch_size"] >= 1
    return hp


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class AnnNextLatent(nn.Module):
    """Bag-of-prefix MLP: pooled prefix (mean-pool ++ last item) -> next latent.

    The pooling is CAUSAL and cumulative so the module can be trained with the
    GRU's per-step teacher forcing: for input latents x = (B, T, D) it emits a
    per-step representation at every t built only from x[:, 0..t]. There is no
    recurrence — each step is an independent MLP over a pooled window.

    forward()      -> (B, T, D) per-step next-latent preds (teacher forcing).
    predict_next() -> (B, D) final next-latent for a prefix (eval / scoring)."""

    def __init__(self, latent_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = hidden
        # Input = concat(mean-pool, last-item) = 2 * latent_dim.
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )

    @staticmethod
    def _causal_meanpool(x: torch.Tensor) -> torch.Tensor:
        """Cumulative mean of x (B, T, D) along T: position t = mean(x[:, 0..t]).

        Padded steps (t >= length) sit at the tail and contribute zeros to the
        running sum, so every REAL step's cumulative mean is exact; padded steps
        are dropped by the loss mask / gathered-at-length eval, never used."""
        csum = torch.cumsum(x, dim=1)                       # (B, T, D)
        counts = torch.arange(1, x.size(1) + 1, device=x.device,
                              dtype=x.dtype).view(1, -1, 1)  # (1, T, 1)
        return csum / counts

    def _pool_steps(self, x: torch.Tensor) -> torch.Tensor:
        """Per-step pooled representation (B, T, 2D): causal mean ++ last item.
        At step t the 'last item' is x[:, t] (the most recent prefix latent)."""
        mean = self._causal_meanpool(x)                     # (B, T, D)
        return torch.cat([mean, x], dim=-1)                 # (B, T, 2D)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Per-step next-latent predictions (B, T, D) for teacher forcing."""
        pooled = self._pool_steps(x)                        # (B, T, 2D)
        B, T, _ = pooled.shape
        out = self.net(pooled.reshape(B * T, -1))           # (B*T, D)
        return out.reshape(B, T, self.latent_dim)

    def predict_next(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Final predicted next-latent (B, D) from the last valid prefix step."""
        pooled = self._pool_steps(x)                        # (B, T, 2D)
        if lengths is not None:
            idx = (lengths.to(pooled.device) - 1).clamp(min=0)
            rep = pooled[torch.arange(pooled.size(0), device=pooled.device), idx]
        else:
            rep = pooled[:, -1, :]                           # full, unpadded prefix
        return self.net(rep)                                 # (B, D)


# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #
def fit(art, hp: dict, seed: int = SEED):
    """Train the feed-forward next-latent model on the artifact's TRAIN sessions
    and return the best (early-stopped) model in eval mode.

    Pure training — no disk writes, no retrieval eval — so the ensemble harness
    (blend / stacker) can reuse the EXACT procedure. Mirrors seq_nexttrack.fit
    (same seeding, val split, per-step objective, early stopping); the only
    difference is the network (bag-of-prefix MLP instead of a recurrent cell)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    D = art.latent_dim
    latents = art.item_latents

    all_train = training_pairs(art, art.train_sessions)
    assert all_train, "no usable train sessions (need length >= 2)"

    perm = rng.permutation(len(all_train))
    n_val = max(1, int(len(all_train) * hp["val_fraction"]))
    val_ids = set(perm[:n_val].tolist())
    tr_seqs = [s for i, s in enumerate(all_train) if i not in val_ids]
    va_seqs = [s for i, s in enumerate(all_train) if i in val_ids]

    emit({"kind": "log", "msg":
          f"dataset {art.manifest['dataset_id']}: {art.n_sessions} sessions, "
          f"{art.n_items} items, dim {D}; train pairs {len(tr_seqs)} / "
          f"val {len(va_seqs)} / test sessions {int(art.test_sessions.size)}"})
    emit({"kind": "log", "msg":
          f"ANN (bag-of-prefix MLP) hidden {hp['hidden']} x2, loss {hp['loss']}, "
          f"dropout {hp['dropout']}, lr {hp['lr']}, batch {hp['batch_size']}, "
          f"{hp['epochs']} epochs; objective per-step teacher forcing"})

    model = AnnNextLatent(D, hp["hidden"], hp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    def batch_loss(x, y, mask, lengths):
        pred = model(x, lengths)                            # (B, T, D)
        return step_loss(pred, y, mask, hp["loss"], hp["tau"]), int(mask.sum())

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, y, mask, lengths in make_batches(
                    va_seqs, latents, hp["batch_size"], rng, shuffle=False):
                loss, w = batch_loss(x, y, mask, lengths)
                tot += float(loss) * w
                n += w
        return tot / max(n, 1)

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad = 0
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y, mask, lengths in make_batches(
                tr_seqs, latents, hp["batch_size"], rng, shuffle=True):
            opt.zero_grad()
            loss, w = batch_loss(x, y, mask, lengths)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * w
            n += w
        tr_loss = tot / max(n, 1)
        vl = val_loss()
        emit({"kind": "epoch", "epoch": epoch, "total_epochs": hp["epochs"],
              "loss": tr_loss, "val_loss": vl})

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if hp["patience"] and bad >= hp["patience"]:
                emit({"kind": "log", "msg": f"early stop at epoch {epoch} "
                      f"(best val {best_val:.4f})"})
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def ann_score_fn(model: AnnNextLatent, latents: np.ndarray):
    """Full-vocab cosine score_fn(prefix) for a trained ANN — the SAME interface
    the ensemble harness uses for the GRU (seq_blend.gru_score_fn). This is the
    "A leg" full-vocab score vector."""
    item_norm = torch.nn.functional.normalize(torch.from_numpy(latents), dim=1)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(latents[prefix][None, :, :])  # (1, T, D)
        with torch.no_grad():
            pred = model.predict_next(x)                    # (1, D) prefix -> next
            pred = torch.nn.functional.normalize(pred, dim=1)
            scores = (pred @ item_norm.t()).squeeze(0)      # (n_items,)
        return scores.numpy()
    return score_fn


def load_model(model_dir: Path, art, hp: dict) -> "AnnNextLatent":
    """Reconstruct the trained pooled-MLP from a promoted dir's model.pt."""
    model = AnnNextLatent(art.latent_dim, hp["hidden"], hp["dropout"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Serving-time ranking: reload the ANN and rank the vocab for the caller's
    session prefix (see seq_common.predict_ranking)."""
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(model_dir)
    model = load_model(model_dir, art, hp)
    predict_ranking(art, ann_score_fn(model, art.item_latents), input_dir, output, k=hp["k"])
    emit({"kind": "done"})


def _capture(model: "AnnNextLatent"):
    """Per-model-SAE activation tap: the two post-ReLU hidden layers of the
    bag-of-prefix MLP. Returns (layer_names, step_acts_fn) for seq_model_sae."""
    net = model.net  # [Lin1, ReLU, Drop, Lin2, ReLU, Drop, Lin3]

    def step_acts_fn(seq_latents: np.ndarray):
        x = torch.from_numpy(seq_latents[None, :, :].astype(np.float32))  # (1,L,D)
        with torch.no_grad():
            pooled = model._pool_steps(x)[0]           # (L, 2D)
            h1 = torch.relu(net[0](pooled))            # (L, hidden)
            h2 = torch.relu(net[3](h1))                # (L, hidden); Drop=identity (eval)
        # Steps 0..L-2 have a next item; the last step has none.
        return [h1[:-1].numpy(), h2[:-1].numpy()]

    return ["hidden_1", "hidden_2"], step_acts_fn


def model_sae(model_dir: Path, dataset: Path, output: Path, *, layers, n_atoms,
              l1, epochs, cache_dir, label_atoms, dataset_sae, topk=0) -> None:
    """Per-model SAE: dictionary-learn the ANN's hidden activations and relate
    the atoms to next-item concepts (see seq_model_sae). ``topk>0`` uses a top-k
    SAE (l0==topk) instead of L1-induced sparsity."""
    import seq_model_sae
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(dataset)
    model = load_model(model_dir, art, hp)
    seq_model_sae.run(
        model_dir, dataset, output,
        capture=lambda _art: _capture(model),
        meta={"predictor": "seq-ann", "hidden": [hp["hidden"], hp["hidden"]]},
        layers=layers, n_atoms=n_atoms, l1=l1, epochs=epochs,
        cache_dir=cache_dir, label_atoms=label_atoms, dropped=True, topk=topk)


def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    art = load_artifact(dataset)
    latents = art.item_latents

    model = fit(art, hp, seed=hp["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)

    score_fn = ann_score_fn(model, latents)
    metrics, predictions = eval_from_scores(art, score_fn, k=hp["k"])
    write_outputs(run_dir, metrics, predictions)

    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "hyperparams.json").write_text(json.dumps(hp))

    emit({"kind": "log", "msg":
          f"Recall@{hp['k']} {metrics['recall_at_k']:.3f}  "
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
    pr = sub.add_parser("predict", help="rank the next track for a session prefix")
    pr.add_argument("--model", required=True, type=Path)
    pr.add_argument("--input", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    fx = sub.add_parser("fixture", help="write a synthetic SEQUENCE artifact")
    fx.add_argument("--output", required=True, type=Path)
    fx.add_argument("--n-items", type=int, default=20)
    fx.add_argument("--n-sessions", type=int, default=40)
    ms = sub.add_parser("model-sae", help="per-model SAE over the ANN's hidden activations")
    ms.add_argument("--model", required=True, type=Path)
    ms.add_argument("--dataset", required=True, type=Path)
    ms.add_argument("--output", required=True, type=Path)
    ms.add_argument("--layers", default="")  # "1,2"; empty = every hidden layer
    ms.add_argument("--n-atoms", type=int, default=0)
    ms.add_argument("--l1", type=float, default=0.0015)
    ms.add_argument("--epochs", type=int, default=40)
    ms.add_argument("--topk", type=int, default=0,
                    help="top-k SAE: hard-cap active atoms per row to K (l0==K); 0 = L1-only")
    ms.add_argument("--cache-dir", type=Path, default=None)
    ms.add_argument("--label-atoms", action="store_true")
    ms.add_argument("--dataset-sae", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    if args.cmd == "train":
        train(args.dataset, args.output, args.hyperparams)
    elif args.cmd == "predict":
        predict(args.model, args.input, args.output)
    elif args.cmd == "model-sae":
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
        model_sae(args.model, args.dataset, args.output, layers=layers,
                  n_atoms=args.n_atoms, l1=args.l1, epochs=args.epochs,
                  cache_dir=args.cache_dir, label_atoms=args.label_atoms,
                  dataset_sae=args.dataset_sae, topk=args.topk)
    else:
        make_fixture(args.output, n_items=args.n_items, n_sessions=args.n_sessions)


if __name__ == "__main__":
    main()
