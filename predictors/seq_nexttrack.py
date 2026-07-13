#!/usr/bin/env python3
"""Next-track SEQUENCE predictor (PyTorch, CPU).

A recurrent model (GRU or LSTM) consumes the frozen per-item latents of a
session prefix and predicts the *next* track's 64-dim latent; retrieval ranks
the whole vocab by cosine similarity to that prediction. This is the learned
counterpart to the popularity / recency / first-order-Markov baselines in
seq_baselines.py, evaluated on the exact same leave-last-out protocol.

Training (teacher forcing): for a session [a,b,c,d] the supervised next-step
targets are  b|[a], c|[a,b], d|[a,b,c]  — i.e. at every timestep the model
predicts the latent of the following item. Loss is either

  * `cosine`  : 1 - cos(pred, true_next_latent)   (default), or
  * `infonce` : in-batch InfoNCE with every other step's target latent (and the
                positive) as negatives, temperature `tau`.

Early stopping watches the same loss on a held-out slice of the TRAIN sessions.

CLI (mirrors the other predictors' arg shape; the registry passes a
`--hyperparams` FILE, but an inline JSON string is also accepted):

  seq_nexttrack.py train --dataset <dir> --output <run_dir> --hyperparams <json|path>
  seq_nexttrack.py fixture --output <dir>          # write a synthetic artifact

Run under the shared predictor venv (torch 2.12.0+cpu / numpy / sklearn)."""

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
    training_pairs,
    write_outputs,
)

DEFAULTS = {
    "hidden": 128,          # GRU/LSTM hidden width (128 / 256 typical)
    "arch": "gru",          # "gru" | "lstm"
    "layers": 1,            # recurrent layers
    "dropout": 0.1,
    "loss": "cosine",       # "cosine" | "infonce"
    "tau": 0.07,            # InfoNCE temperature
    "epochs": 30,
    "lr": 1e-3,
    "batch_size": 128,
    "patience": 5,          # early-stop patience (epochs; 0 = off)
    "val_fraction": 0.15,   # held-out slice of TRAIN sessions
    "k": 10,                # Recall@k reported in metrics.recall_at_k
}
SEED = 1337


def load_hp(spec: str) -> dict:
    """Accept either a path to a JSON file (registry convention) or an inline
    JSON string (the task's literal `--hyperparams '<json>'`)."""
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    hp = {**DEFAULTS, **json.loads(raw)}
    assert hp["arch"] in ("gru", "lstm"), "arch must be gru|lstm"
    assert hp["loss"] in ("cosine", "infonce"), "loss must be cosine|infonce"
    assert hp["epochs"] >= 1 and hp["batch_size"] >= 1
    return hp


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class SeqNextLatent(nn.Module):
    """Recurrent encoder over prefix latents -> predicted next latent.

    Input  : (batch, T, latent_dim) padded prefix latents.
    Output : (batch, T, latent_dim) per-step predicted next latent (train) or
             (batch, latent_dim) final-step prediction (eval)."""

    def __init__(self, latent_dim: int, hidden: int, arch: str,
                 layers: int, dropout: float):
        super().__init__()
        rnn_cls = nn.GRU if arch == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=latent_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        # x: (B, T, D). Pack to skip padding when lengths are given.
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.rnn(x)
        return self.head(self.drop(out))  # (B, T, D)


# --------------------------------------------------------------------------- #
# Batching                                                                    #
# --------------------------------------------------------------------------- #
def make_batches(seqs: list[np.ndarray], item_latents: np.ndarray,
                 batch_size: int, rng: np.random.Generator, shuffle: bool):
    """Yield padded teacher-forcing batches.

    For a session of item indices [i0, i1, ..., iL-1] the model input is the
    latents of [i0 .. iL-2] and the targets are the latents of [i1 .. iL-1]
    (predict the next item's latent at every step). Returns, per batch:
      x        : (B, Tmax, D) float32 input latents
      y        : (B, Tmax, D) float32 target latents
      mask     : (B, Tmax)    bool, True where a real step exists
      lengths  : (B,)         int input lengths (= L-1)"""
    idx = np.arange(len(seqs))
    if shuffle:
        rng.shuffle(idx)
    D = item_latents.shape[1]
    for start in range(0, len(idx), batch_size):
        batch_ids = idx[start:start + batch_size]
        batch = [seqs[i] for i in batch_ids]
        lengths = np.array([len(s) - 1 for s in batch], dtype=np.int64)
        Tmax = int(lengths.max())
        B = len(batch)
        x = np.zeros((B, Tmax, D), dtype=np.float32)
        y = np.zeros((B, Tmax, D), dtype=np.float32)
        mask = np.zeros((B, Tmax), dtype=bool)
        for r, seq in enumerate(batch):
            L = len(seq) - 1
            x[r, :L] = item_latents[seq[:-1]]
            y[r, :L] = item_latents[seq[1:]]
            mask[r, :L] = True
        yield (torch.from_numpy(x), torch.from_numpy(y),
               torch.from_numpy(mask), torch.from_numpy(lengths))


def step_loss(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
              loss_kind: str, tau: float) -> torch.Tensor:
    """Masked loss over all valid teacher-forcing steps."""
    m = mask.reshape(-1)                       # (B*T,)
    p = pred.reshape(-1, pred.shape[-1])[m]    # (N, D)
    t = y.reshape(-1, y.shape[-1])[m]          # (N, D)
    if p.shape[0] == 0:
        return pred.sum() * 0.0
    p = torch.nn.functional.normalize(p, dim=1)
    t = torch.nn.functional.normalize(t, dim=1)
    if loss_kind == "cosine":
        return (1.0 - (p * t).sum(dim=1)).mean()
    # InfoNCE: each row's positive is its own target; every step's target in
    # the batch (including the positive) serves as the candidate bank.
    logits = (p @ t.t()) / tau                 # (N, N)
    labels = torch.arange(p.shape[0], device=p.device)
    return torch.nn.functional.cross_entropy(logits, labels)


# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #
def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    art = load_artifact(dataset)
    D = art.latent_dim
    latents = art.item_latents

    all_train = training_pairs(art, art.train_sessions)
    assert all_train, "no usable train sessions (need length >= 2)"

    # Held-out slice of TRAIN sessions for early stopping.
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
          f"{hp['arch']} hidden {hp['hidden']} x{hp['layers']}, "
          f"loss {hp['loss']}, dropout {hp['dropout']}, lr {hp['lr']}, "
          f"batch {hp['batch_size']}, {hp['epochs']} epochs"})

    model = SeqNextLatent(D, hp["hidden"], hp["arch"], hp["layers"], hp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    run_dir.mkdir(parents=True, exist_ok=True)

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, y, mask, lengths in make_batches(
                    va_seqs, latents, hp["batch_size"], rng, shuffle=False):
                pred = model(x, lengths)
                nsteps = int(mask.sum())
                tot += float(step_loss(pred, y, mask, hp["loss"], hp["tau"])) * nsteps
                n += nsteps
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
            pred = model(x, lengths)
            loss = step_loss(pred, y, mask, hp["loss"], hp["tau"])
            loss.backward()
            opt.step()
            nsteps = int(mask.sum())
            tot += float(loss.detach()) * nsteps
            n += nsteps
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

    # Retrieval eval on TEST sessions. Precompute normalized item latents once;
    # the model's predicted next-latent for a prefix -> cosine over the vocab.
    item_norm = torch.nn.functional.normalize(torch.from_numpy(latents), dim=1)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(latents[prefix][None, :, :])  # (1, T, D)
        with torch.no_grad():
            pred = model(x)[:, -1, :]                       # final-step prediction
            pred = torch.nn.functional.normalize(pred, dim=1)
            scores = (pred @ item_norm.t()).squeeze(0)      # (n_items,)
        return scores.numpy()

    metrics, predictions = eval_from_scores(art, score_fn, k=hp["k"])
    write_outputs(run_dir, metrics, predictions)

    # Save weights + hyperparams so the run is reproducible.
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
