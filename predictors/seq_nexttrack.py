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

Network TOPOLOGY hyperparameters (for the topology experiment; all default to
the original single-layer unidirectional network so existing configs are
unchanged):

  * `num_layers`   : stacked GRU/LSTM (torch applies `dropout` between layers
                     when num_layers>1).
  * `bidirectional`: bidirectional encoder. See the LEAKAGE caveat below.
  * `residual`     : inter-layer residual connections for a manual stack of
                     single-layer cells (only meaningful with num_layers>1).

LEAKAGE caveat — per-step teacher forcing (predict item t+1 from the prefix
up to t) is only leak-free with a *unidirectional* RNN: a bidirectional RNN's
per-step output at t already sees items t+1.. from the backward pass, so it
would trivially copy the target. Therefore the training OBJECTIVE depends on
`bidirectional`:

  * `bidirectional=false` (default): the existing per-step teacher forcing over
    every next-step pair, head Linear(hidden -> 64).
  * `bidirectional=true`: a leak-free LAST-ITEM objective — encode the FULL
    prefix (all-but-last) bidirectionally, take the concat of both directions'
    final states (dim 2*hidden), head Linear(2*hidden -> 64), and predict ONLY
    the held-out next (last) item's latent. No intermediate steps are forced.
    This mirrors how eval already works (prefix -> next), so it is consistent.

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
    predict_ranking,
    training_pairs,
    write_outputs,
)

DEFAULTS = {
    "hidden": 128,          # GRU/LSTM hidden width (128 / 256 typical)
    "arch": "gru",          # "gru" | "lstm"
    "num_layers": 1,        # stacked recurrent layers (topology)
    "bidirectional": False, # bidirectional encoder -> last-item objective (topology)
    "residual": False,      # inter-layer residuals in a manual stack (topology)
    # Optional per-step MLP in front of the recurrence ("MLP -> GRU"). 0 = none,
    # which is byte-identical to the historical network (see the bit-identity
    # contract in SeqNextLatent.__init__).
    "pre_hidden": 0,
    "pre_layers": 1,
    "dropout": 0.1,
    "loss": "cosine",       # "cosine" | "infonce"
    "tau": 0.07,            # InfoNCE temperature
    "epochs": 30,
    "lr": 1e-3,
    "batch_size": 128,
    "patience": 5,          # early-stop patience (epochs; 0 = off)
    "val_fraction": 0.15,   # held-out slice of TRAIN sessions
    "k": 10,                # Recall@k reported in metrics.recall_at_k
    # Phase-2 anti-eager knobs (default OFF = byte-identical to before):
    "eager_beta": 0.0,      # weight of the anti-eager regularizer (0 = off)
    "eager_margin": 0.0,    # cos(pred, current) above this is penalized
    "mmr_lambda": None,     # MMR re-rank λ at eval (None/1.0 = off; <1 diversifies)
    "mmr_pool": 200,        # candidate pool the MMR re-rank operates over
}
SEED = 1337


def load_hp(spec: str) -> dict:
    """Accept either a path to a JSON file (registry convention) or an inline
    JSON string (the task's literal `--hyperparams '<json>'`)."""
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    user = json.loads(raw)
    hp = {**DEFAULTS, **user}
    # Backward-compat: the network depth used to be the `layers` key. Honor it
    # when a caller set `layers` but not the new canonical `num_layers`.
    if "num_layers" not in user and "layers" in user:
        hp["num_layers"] = int(user["layers"])
    hp["num_layers"] = int(hp["num_layers"])
    hp["pre_hidden"] = int(hp.get("pre_hidden", 0))
    hp["pre_layers"] = int(hp.get("pre_layers", 1))
    assert hp["pre_hidden"] >= 0, "pre_hidden must be >= 0"
    assert hp["pre_layers"] >= 1, "pre_layers must be >= 1"
    hp["bidirectional"] = bool(hp["bidirectional"])
    hp["residual"] = bool(hp["residual"])
    assert hp["arch"] in ("gru", "lstm"), "arch must be gru|lstm"
    assert hp["loss"] in ("cosine", "infonce"), "loss must be cosine|infonce"
    assert hp["num_layers"] >= 1, "num_layers must be >= 1"
    assert hp["epochs"] >= 1 and hp["batch_size"] >= 1
    return hp


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class SeqNextLatent(nn.Module):
    """Recurrent encoder over prefix latents -> predicted next latent.

    Topology is controlled by (num_layers, bidirectional, residual):

    * Unidirectional (bidirectional=False) — the default. Produces a per-step
      output (batch, T, hidden) that the head maps to (batch, T, latent_dim);
      trained with per-step teacher forcing. Depth is `num_layers`:
        - residual=False -> torch's native stacked RNN (inter-layer `dropout`
          when num_layers>1);
        - residual=True & num_layers>1 -> a MANUAL stack of `num_layers`
          single-layer cells with an inter-layer residual `out = layer(x) + x`
          between equal-width layers (torch's native multi-layer RNN has no
          inter-layer residual). The first layer maps latent_dim -> hidden; its
          residual is SKIPPED unless latent_dim == hidden (dim mismatch). With
          num_layers==1 residual is a no-op (falls back to the native path).

    * Bidirectional (bidirectional=True) — a native stacked bidirectional RNN.
      To stay LEAK-FREE (see the module docstring) it exposes only a LAST-ITEM
      prediction: the concat of both directions' final hidden states (dim
      2*hidden) mapped by the head to latent_dim. There is no per-step output,
      so it is trained on the held-out last item alone. `residual` does not
      apply to this path.

    forward()      -> (batch, T, latent_dim) per-step preds (unidirectional).
    predict_next() -> (batch, latent_dim) final next-latent (both modes; used
                      for eval, and for training in bidirectional mode)."""

    def __init__(self, latent_dim: int, hidden: int, arch: str,
                 num_layers: int, dropout: float,
                 bidirectional: bool = False, residual: bool = False,
                 pre_hidden: int = 0, pre_layers: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.arch = arch
        self.num_layers = num_layers
        self.bidirectional = bool(bidirectional)
        self.residual = bool(residual)
        # Bidirectional => leak-free last-item objective (no per-step forcing).
        self.last_item_only = self.bidirectional
        # A manual residual stack is only built when it is meaningful.
        self.manual_stack = (self.residual and num_layers > 1
                             and not self.bidirectional)

        rnn_cls = nn.GRU if arch == "gru" else nn.LSTM
        self.drop = nn.Dropout(dropout)

        # PRE-ENCODER (optional): a per-step nonlinear re-embedding in front of the
        # recurrence, making this an "MLP -> GRU" tower. A GRU's own input
        # transform is linear (W_i·x per gate), so this is a genuine expressivity
        # change — and, note, only because of the ReLU: a purely linear
        # pre-encoder at pre_hidden >= latent_dim would be exactly as expressive
        # as the bare GRU (W_i·(Vx) = (W_iV)x, no rank constraint).
        #
        # BIT-IDENTITY CONTRACT: at pre_hidden == 0 this constructs NO module at
        # all, so no parameters are drawn from the seeded RNG stream and the init
        # order of rnn/head is untouched. That keeps the family's on-record
        # baseline (recall@10 0.12299091544 for h256 infonce) reproducible to the
        # last digit — this predictor is the reference every campaign gates on, so
        # perturbing its default path would silently move the whole record.
        self.pre = None
        if pre_hidden and pre_hidden > 0:
            mods: list[nn.Module] = []
            d_in = latent_dim
            for _ in range(max(1, int(pre_layers))):
                mods += [nn.Linear(d_in, pre_hidden), nn.ReLU(), nn.Dropout(dropout)]
                d_in = pre_hidden
            self.pre = nn.Sequential(*mods)
        self.pre_hidden = int(pre_hidden or 0)
        in_size = self.pre_hidden if self.pre is not None else latent_dim

        if self.bidirectional:
            self.rnn = rnn_cls(
                input_size=in_size, hidden_size=hidden,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True)
            self.head = nn.Linear(2 * hidden, latent_dim)
        elif self.manual_stack:
            self.cells = nn.ModuleList([
                rnn_cls(input_size=(in_size if li == 0 else hidden),
                        hidden_size=hidden, num_layers=1, batch_first=True)
                for li in range(num_layers)
            ])
            self.head = nn.Linear(hidden, latent_dim)
        else:
            self.rnn = rnn_cls(
                input_size=in_size, hidden_size=hidden,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0)
            self.head = nn.Linear(hidden, latent_dim)

    def _pre_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pre-encoder per TIMESTEP (never across time — a pre-MLP that
        mixed timesteps would leak the teacher-forcing target)."""
        if self.pre is None:
            return x
        B, T, _ = x.shape
        return self.pre(x.reshape(B * T, -1)).reshape(B, T, -1)

    @staticmethod
    def _run(cell, x, lengths):
        """Run one RNN cell, packing to skip padding when lengths are given."""
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, state = cell(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, state = cell(x)
        return out, state

    def _encode_seq(self, x: torch.Tensor, lengths: torch.Tensor | None):
        """Per-step encoder output (B, T, hidden) for the unidirectional paths."""
        x = self._pre_encode(x)
        if self.manual_stack:
            h = x
            for li, cell in enumerate(self.cells):
                inp = h
                out, _ = self._run(cell, inp, lengths)
                # Inter-layer residual between equal-width layers; the first
                # layer changes width (latent_dim -> hidden) so skip unless the
                # dims already match.
                if li == 0:
                    if self.latent_dim == self.hidden:
                        out = out + inp
                else:
                    out = out + inp
                # Dropout between layers (not after the last), as torch does.
                if li < self.num_layers - 1:
                    out = self.drop(out)
                h = out
            return h
        out, _ = self._run(self.rnn, x, lengths)
        return out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        # Per-step next-latent prediction (unidirectional teacher forcing).
        assert not self.bidirectional, (
            "bidirectional model has no per-step output; use predict_next()")
        out = self._encode_seq(x, lengths)          # (B, T, hidden)
        return self.head(self.drop(out))            # (B, T, D)

    def predict_next(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Final predicted next-latent (B, latent_dim) for a prefix batch."""
        if self.bidirectional:
            _, state = self._run(self.rnn, self._pre_encode(x), lengths)
            h_n = state[0] if self.arch == "lstm" else state
            # h_n: (num_layers*2, B, hidden). Last layer's forward = -2, its
            # backward = -1; concat both directions -> (B, 2*hidden).
            rep = torch.cat([h_n[-2], h_n[-1]], dim=1)
            return self.head(self.drop(rep))        # (B, D)
        out = self._encode_seq(x, lengths)          # (B, T, hidden)
        if lengths is not None:
            idx = (lengths.to(out.device) - 1).clamp(min=0)
            last = out[torch.arange(out.size(0), device=out.device), idx]
        else:
            last = out[:, -1, :]
        return self.head(self.drop(last))           # (B, D)


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
              loss_kind: str, tau: float,
              x: torch.Tensor | None = None,
              eager_beta: float = 0.0, eager_margin: float = 0.0) -> torch.Tensor:
    """Masked loss over all valid teacher-forcing steps.

    Optional ANTI-EAGER regularizer (Phase 2): when `eager_beta > 0` and the
    input latents `x` are supplied, add `beta * mean(relu(cos(pred, x_current) -
    margin))`. This penalizes the model for predicting a near-copy of the
    CURRENT track's latent — the album-eager failure mode (next-album-track /
    same-artist) — while leaving the cosine/InfoNCE match to the true next
    latent intact. beta=0 (default) is byte-identical to the original loss."""
    m = mask.reshape(-1)                       # (B*T,)
    p = pred.reshape(-1, pred.shape[-1])[m]    # (N, D)
    t = y.reshape(-1, y.shape[-1])[m]          # (N, D)
    if p.shape[0] == 0:
        return pred.sum() * 0.0
    p = torch.nn.functional.normalize(p, dim=1)
    t = torch.nn.functional.normalize(t, dim=1)
    if loss_kind == "cosine":
        base = (1.0 - (p * t).sum(dim=1)).mean()
    else:
        # InfoNCE: each row's positive is its own target; every step's target in
        # the batch (including the positive) serves as the candidate bank.
        logits = (p @ t.t()) / tau             # (N, N)
        labels = torch.arange(p.shape[0], device=p.device)
        base = torch.nn.functional.cross_entropy(logits, labels)
    if eager_beta and x is not None:
        xc = torch.nn.functional.normalize(x.reshape(-1, x.shape[-1])[m], dim=1)
        eager = torch.nn.functional.relu((p * xc).sum(dim=1) - eager_margin).mean()
        base = base + eager_beta * eager
    return base


def last_item_loss(pred: torch.Tensor, y: torch.Tensor,
                   loss_kind: str, tau: float,
                   x: torch.Tensor | None = None,
                   eager_beta: float = 0.0, eager_margin: float = 0.0) -> torch.Tensor:
    """Leak-free last-item loss for the bidirectional path: one (pred, target)
    pair per session. Reuses the exact cosine / in-batch-InfoNCE objective (and
    optional anti-eager regularizer) as the per-step loss over a single step."""
    mask = torch.ones(pred.shape[0], 1, dtype=torch.bool, device=pred.device)
    xx = x.unsqueeze(1) if x is not None else None
    return step_loss(pred.unsqueeze(1), y.unsqueeze(1), mask, loss_kind, tau,
                     x=xx, eager_beta=eager_beta, eager_margin=eager_margin)


# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #
def fit(art, hp: dict, seed: int = SEED, model: "SeqNextLatent | None" = None):
    """Train the recurrent next-latent model on the artifact's TRAIN sessions
    and return the best (early-stopped) model, in eval mode.

    Pure training — no disk writes, no retrieval eval — so other predictors
    (e.g. the GRU x Markov blend) can reuse the EXACT training procedure
    (seeding, val split, per-step/last-item objective, early stopping) and get
    byte-identical weights. `train()` below is a thin wrapper: fit + eval +
    persist. `seed` defaults to the module SEED so existing behavior is
    unchanged.

    `model` lets a caller pass a PRE-BUILT / PRE-INITIALIZED SeqNextLatent (e.g.
    seq_ae_rnn transplants an autoencoder-pretrained encoder into it) so the
    fine-tune runs the identical objective/optimizer/early-stopping starting
    from those weights. When None (default) a fresh model is constructed exactly
    as before, so existing behavior is byte-identical."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

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
    topo = (f"{'bi' if hp['bidirectional'] else 'uni'}dir"
            f"{', residual' if hp['residual'] and hp['num_layers'] > 1 else ''}")
    emit({"kind": "log", "msg":
          f"{hp['arch']} hidden {hp['hidden']} x{hp['num_layers']} ({topo}), "
          f"loss {hp['loss']}, dropout {hp['dropout']}, lr {hp['lr']}, "
          f"batch {hp['batch_size']}, {hp['epochs']} epochs; objective "
          f"{'last-item (leak-free)' if hp['bidirectional'] else 'per-step teacher forcing'}"})

    if model is None:
        model = SeqNextLatent(D, hp["hidden"], hp["arch"], hp["num_layers"],
                              hp["dropout"], bidirectional=hp["bidirectional"],
                              residual=hp["residual"],
                              pre_hidden=hp.get("pre_hidden", 0),
                              pre_layers=hp.get("pre_layers", 1))
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    def batch_loss(x, y, mask, lengths):
        """Loss for one padded batch + its sample weight, branching on the
        training objective: per-step teacher forcing (unidirectional) or the
        leak-free last-item objective (bidirectional)."""
        beta = float(hp.get("eager_beta", 0.0))
        margin = float(hp.get("eager_margin", 0.0))
        if model.last_item_only:
            pred = model.predict_next(x, lengths)          # (B, D)
            idx = (lengths - 1).clamp(min=0)
            y_last = y[torch.arange(y.shape[0]), idx]      # (B, D) last-item latent
            x_last = x[torch.arange(x.shape[0]), idx]      # (B, D) current-item latent
            return last_item_loss(pred, y_last, hp["loss"], hp["tau"],
                                  x=x_last, eager_beta=beta, eager_margin=margin), y.shape[0]
        pred = model(x, lengths)                           # (B, T, D)
        return step_loss(pred, y, mask, hp["loss"], hp["tau"],
                         x=x, eager_beta=beta, eager_margin=margin), int(mask.sum())

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


def build_score_fn(model: "SeqNextLatent", latents: np.ndarray):
    """Full-vocab cosine score_fn(prefix) for a trained GRU/LSTM: the model's
    predicted next-latent for a prefix, cosine over the (pre-normalized) vocab.
    SHARED by train and predict so serving ranks byte-identically to eval."""
    item_norm = torch.nn.functional.normalize(torch.from_numpy(latents), dim=1)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(latents[prefix][None, :, :])  # (1, T, D)
        with torch.no_grad():
            pred = model.predict_next(x)                    # (1, D) prefix -> next
            pred = torch.nn.functional.normalize(pred, dim=1)
            scores = (pred @ item_norm.t()).squeeze(0)      # (n_items,)
        return scores.numpy()
    return score_fn


def load_model(model_dir: Path, art, hp: dict) -> "SeqNextLatent":
    """Reconstruct the trained model from a promoted dir's model.pt."""
    model = SeqNextLatent(art.latent_dim, hp["hidden"], hp["arch"], hp["num_layers"],
                          hp["dropout"], bidirectional=hp["bidirectional"],
                          residual=hp["residual"],
                          pre_hidden=hp.get("pre_hidden", 0),
                          pre_layers=hp.get("pre_layers", 1))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Serving-time ranking: reload the GRU/LSTM and rank the vocab for the
    caller's session prefix (see seq_common.predict_ranking)."""
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(model_dir)
    model = load_model(model_dir, art, hp)
    predict_ranking(art, build_score_fn(model, art.item_latents), input_dir, output, k=hp["k"])
    emit({"kind": "done"})


def _capture(model: "SeqNextLatent"):
    """Per-model-SAE activation tap: the recurrent encoder's per-step hidden
    state. Returns (layer_names, step_acts_fn) for seq_model_sae."""
    def step_acts_fn(seq_latents: np.ndarray):
        x = torch.from_numpy(seq_latents[None, :, :].astype(np.float32))  # (1,L,D)
        with torch.no_grad():
            acts = []
            if model.pre is not None:
                acts.append(model._pre_encode(x)[0, :-1].numpy())
            out = model._encode_seq(x, None)           # (1, L, hidden)
            acts.append(out[0, :-1].numpy())           # steps 0..L-2
        return acts

    names = (["pre"] if model.pre is not None else []) + ["recurrent"]
    return names, step_acts_fn


def model_sae(model_dir: Path, dataset: Path, output: Path, *, layers, n_atoms,
              l1, epochs, cache_dir, label_atoms, dataset_sae, topk=0) -> None:
    """Per-model SAE: dictionary-learn the GRU/LSTM's per-step hidden state and
    relate the atoms to next-item concepts (see seq_model_sae). ``topk>0`` uses a
    top-k SAE (l0==topk) instead of L1-induced sparsity."""
    import seq_model_sae
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(dataset)
    model = load_model(model_dir, art, hp)
    if model.bidirectional:
        raise SystemExit("model-sae: bidirectional models expose no per-step "
                         "hidden state (last-item objective); not supported")
    seq_model_sae.run(
        model_dir, dataset, output,
        capture=lambda _art: _capture(model),
        meta={"predictor": "seq-nexttrack",
              "hidden": ([hp["pre_hidden"]] if hp["pre_hidden"] > 0 else []) + [hp["hidden"]]},
        layers=layers, n_atoms=n_atoms, l1=l1, epochs=epochs,
        cache_dir=cache_dir, label_atoms=label_atoms, dropped=True, topk=topk)


def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    art = load_artifact(dataset)
    latents = art.item_latents

    model = fit(art, hp)
    run_dir.mkdir(parents=True, exist_ok=True)

    score_fn = build_score_fn(model, latents)
    metrics, predictions = eval_from_scores(
        art, score_fn, k=hp["k"],
        mmr_lambda=hp.get("mmr_lambda"), mmr_pool=int(hp.get("mmr_pool", 200)))
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
    pr = sub.add_parser("predict", help="rank the next track for a session prefix")
    pr.add_argument("--model", required=True, type=Path)
    pr.add_argument("--input", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    fx = sub.add_parser("fixture", help="write a synthetic SEQUENCE artifact")
    fx.add_argument("--output", required=True, type=Path)
    fx.add_argument("--n-items", type=int, default=20)
    fx.add_argument("--n-sessions", type=int, default=40)
    ms = sub.add_parser("model-sae", help="per-model SAE over the recurrent hidden state")
    ms.add_argument("--model", required=True, type=Path)
    ms.add_argument("--dataset", required=True, type=Path)
    ms.add_argument("--output", required=True, type=Path)
    ms.add_argument("--layers", default="")  # "1"; empty = every hidden layer
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
