#!/usr/bin/env python3
"""Next-track SEQUENCE predictor — TWO INDEPENDENT RNN TOWERS -> ONE MLP.

A dual-encoder: two recurrent towers (A and B) read the session prefix in
PARALLEL with completely separate weights (no sharing, no tying — each tower is
its own GRU/LSTM), and their per-step hidden states are CONCATENATED and fused
by a single MLP head that emits the next track's 64-dim latent. Retrieval then
ranks the whole vocab by cosine to that prediction — the same
`score_fn(prefix) -> full-vocab cosine vector` contract the rest of the
next-track family exposes (seq_nexttrack, seq_ann, the blend/stacker legs).

Where this sits relative to the existing family:

  * seq_nexttrack  = ONE recurrent tower + a LINEAR head.
  * seq_ann        = NO recurrence (bag-of-prefix pooling) + an MLP.
  * seq_blend      = two SEPARATELY-trained scorers combined post-hoc by a
                     fixed z-score blend.
  * seq_stacker    = separately-trained base legs + a learned XGBoost combiner
                     over score columns.
  * THIS model     = two towers trained JOINTLY, END-TO-END, fused inside the
                     network by a learned MLP over their hidden states. The
                     combiner sees the towers' REPRESENTATIONS, not their
                     scores, and gradients flow back into both towers, so the
                     towers can specialize against each other instead of each
                     being independently optimal.

Tower INPUT VIEWS — what makes the towers independent. By default both towers
read the same stream (the raw prefix latents), so they differ only by their
independent random init: the literal "two independent GRUs". Each tower's input
can instead be set to a different CAUSAL view of the prefix via `view_a` /
`view_b`, which gives the towers structurally complementary jobs:

  * `latent`  : the item latents x_t themselves — absolute position in the
                content space (what seq_nexttrack sees).
  * `delta`   : x_t - x_{t-1} (x_{-1} = 0) — the session's MOVEMENT / trajectory
                through the content space rather than where it is.
  * `cummean` : the causal cumulative mean of x_0..x_t — the slow-moving session
                centroid / "what this session is about so far" (the pooled view
                seq_ann built its whole representation from).

All three views are strictly causal (step t depends only on items 0..t), so the
per-step teacher forcing below stays leak-free. Every view has dim latent_dim,
so towers can be mixed freely.

Objective / training — the family's EXACT unidirectional recipe, reused verbatim
from seq_nexttrack (not reimplemented) so this model is comparable to the single
GRU on the same footing:
  * per-step teacher forcing — for a session [a,b,c,d] the supervised steps are
    b|[a], c|[a,b], d|[a,b,c],
  * the SAME masked cosine / in-batch-InfoNCE loss + optional anti-eager
    regularizer (`seq_nexttrack.step_loss`, `eager_beta` / `eager_margin`),
  * the SAME padded batching (`seq_nexttrack.make_batches`),
  * the SAME held-out-TRAIN-slice early stopping,
  * the SAME eval (`seq_common.eval_from_scores`, optional MMR re-rank).

There is no bidirectional mode: a backward pass over the prefix would let step t
see item t+1 and trivially copy the teacher-forcing target (see the LEAKAGE
caveat in seq_nexttrack). Both towers are causal by construction.

CLI (mirrors the rest of the family; the registry passes a `--hyperparams` FILE,
an inline JSON string is also accepted):

  seq_dualgru.py train --dataset <dir> --output <run_dir> --hyperparams <json|path>
  seq_dualgru.py predict --model <dir> --input <dir> --output <file>
  seq_dualgru.py model-sae --model <dir> --dataset <dir> --output <file>
  seq_dualgru.py fixture --output <dir>          # write a synthetic artifact

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
# family's (this is a new topology, not a new training recipe).
from seq_nexttrack import SEED, make_batches, step_loss

VIEWS = ("latent", "delta", "cummean")

DEFAULTS = {
    # Tower A / tower B — independent weights, independently sized.
    "hidden_a": 256,        # tower A recurrent width
    "hidden_b": 256,        # tower B recurrent width
    "arch_a": "gru",        # "gru" | "lstm"
    "arch_b": "gru",
    "view_a": "latent",     # "latent" | "delta" | "cummean"
    "view_b": "latent",     # same default => two independent same-input GRUs
    "layers_a": 1,          # stacked recurrent layers in tower A
    "layers_b": 1,
    # PRE-ENCODER: a per-step MLP in front of a tower's recurrence, so the tower
    # is "MLP + GRU" instead of a bare GRU. 0 = none (a bare GRU).
    #
    # This is a different KIND of asymmetry from `view_*`, and the reason matters.
    # The views (`delta`, `cummean`) are fixed, deterministic functions of the
    # same latent sequence, so a second tower reading one gets a lossier summary
    # of information the first tower already has — measured 2026-07-25: the
    # cummean tower added real linear directions (CKA 0.458) but ZERO next-item
    # concepts tower A lacked. A pre-MLP instead enlarges the tower's HYPOTHESIS
    # SPACE: a GRU's own input transform is linear (W_i·x per gate), so a
    # nonlinear re-embedding before the recurrence can represent input features
    # no single-layer GRU can. It is also the one direction with a track record
    # in this project — the champion's win was a *learned* projection of the
    # content space, not a hand-derived view of it.
    #
    # Defaults are 0/0 ON PURPOSE: `load_model` rebuilds the network from these
    # hyperparameters, so a non-zero default would change the architecture of
    # every already-promoted seq-dualgru model and break its checkpoint load.
    # The asymmetric "plain GRU + (MLP→GRU)" model is expressed explicitly.
    "pre_hidden_a": 0,      # tower A pre-MLP width (0 = bare GRU)
    "pre_hidden_b": 0,      # tower B pre-MLP width (0 = bare GRU)
    "pre_layers_a": 1,      # tower A pre-MLP hidden layers (ignored when width 0)
    "pre_layers_b": 1,
    # Fusion MLP over concat(h_a, h_b) -> latent_dim.
    "fusion_hidden": 256,   # width of each fusion hidden layer
    "fusion_layers": 1,     # hidden layers in the fusion MLP (0 = linear head)
    "dropout": 0.1,         # towers (inter-layer) + fusion MLP
    # Training (identical knobs/semantics to seq_nexttrack).
    "loss": "infonce",      # "cosine" | "infonce"
    "tau": 0.07,            # InfoNCE temperature
    "epochs": 40,
    "lr": 1e-3,
    "batch_size": 128,
    "patience": 6,          # early-stop patience (epochs; 0 = off)
    "val_fraction": 0.15,   # held-out slice of TRAIN sessions
    "seed": SEED,           # 1337 — deterministic
    "k": 10,                # Recall@k reported in metrics.recall_at_k
    # Anti-eager / diversity levers, same semantics as seq_nexttrack.
    "eager_beta": 0.0,      # weight of the anti-eager regularizer (0 = off)
    "eager_margin": 0.0,    # cos(pred, current) above this is penalized
    "mmr_lambda": None,     # MMR re-rank λ at eval (None/1.0 = off)
    "mmr_pool": 200,        # candidate pool the MMR re-rank operates over
    "artist_cap": None,   # HARD cap on top-k slots per artist (None/0 = off)
}


def load_hp(spec: str) -> dict:
    """Accept a path to a JSON file (registry convention) or an inline JSON
    string. Merges over DEFAULTS and validates."""
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    user = json.loads(raw)
    hp = {**DEFAULTS, **user}
    for key in ("hidden_a", "hidden_b", "layers_a", "layers_b",
                "pre_hidden_a", "pre_hidden_b", "pre_layers_a", "pre_layers_b",
                "fusion_hidden", "fusion_layers", "epochs", "batch_size", "seed"):
        hp[key] = int(hp[key])
    for side in ("a", "b"):
        assert hp[f"arch_{side}"] in ("gru", "lstm"), f"arch_{side} must be gru|lstm"
        assert hp[f"view_{side}"] in VIEWS, \
            f"view_{side} must be one of {'|'.join(VIEWS)}"
        assert hp[f"hidden_{side}"] >= 1, f"hidden_{side} must be >= 1"
        assert hp[f"layers_{side}"] >= 1, f"layers_{side} must be >= 1"
        assert hp[f"pre_hidden_{side}"] >= 0, f"pre_hidden_{side} must be >= 0"
        assert hp[f"pre_layers_{side}"] >= 1, f"pre_layers_{side} must be >= 1"
    assert hp["loss"] in ("cosine", "infonce"), "loss must be cosine|infonce"
    assert hp["fusion_layers"] >= 0, "fusion_layers must be >= 0"
    assert hp["epochs"] >= 1 and hp["batch_size"] >= 1
    return hp


# --------------------------------------------------------------------------- #
# Input views                                                                 #
# --------------------------------------------------------------------------- #
def make_view(x: torch.Tensor, kind: str) -> torch.Tensor:
    """A CAUSAL per-step view of the prefix latents x (B, T, D) -> (B, T, D).

    Every view's step t is a function of x[:, 0..t] only, so per-step teacher
    forcing stays leak-free. PADDED tails (x == 0 beyond the length) never
    corrupt a real step: `delta` at a real step t reads x[t-1], x[t] (both real),
    and `cummean`'s running sum only accumulates zeros AFTER the last real step."""
    if kind == "latent":
        return x
    if kind == "delta":
        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return x - prev
    if kind == "cummean":
        csum = torch.cumsum(x, dim=1)
        counts = torch.arange(1, x.size(1) + 1, device=x.device,
                              dtype=x.dtype).view(1, -1, 1)
        return csum / counts
    raise AssertionError(f"unknown view {kind!r}")


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class DualTowerNextLatent(nn.Module):
    """Two independent recurrent towers -> concat -> MLP -> next latent.

    Both towers are unidirectional and causal, and both emit a per-step hidden
    state; the fusion MLP is applied per step to concat(h_a[t], h_b[t]), so the
    module trains under the family's per-step teacher forcing and serves the
    last valid step at eval time.

    forward()      -> (B, T, latent_dim) per-step next-latent preds.
    predict_next() -> (B, latent_dim) final next-latent for a prefix."""

    def __init__(self, latent_dim: int, hp: dict):
        super().__init__()
        self.latent_dim = latent_dim
        self.view_a = hp["view_a"]
        self.view_b = hp["view_b"]
        self.hidden_a = hp["hidden_a"]
        self.hidden_b = hp["hidden_b"]
        self.arch_a = hp["arch_a"]
        self.arch_b = hp["arch_b"]
        dropout = float(hp["dropout"])

        def pre(width: int, n_layers: int) -> nn.Module | None:
            """Per-step nonlinear re-embedding in front of a tower's recurrence.

            Applied timestep-wise, so it stays strictly causal: step t's features
            depend on x[:, t] alone and the recurrence still carries all history.
            (A pre-MLP that mixed across time would leak the teacher-forcing
            target.)"""
            if width <= 0:
                return None
            mods: list[nn.Module] = []
            d_in = latent_dim
            for _ in range(n_layers):
                mods += [nn.Linear(d_in, width), nn.ReLU(), nn.Dropout(dropout)]
                d_in = width
            return nn.Sequential(*mods)

        self.pre_a = pre(hp["pre_hidden_a"], hp["pre_layers_a"])
        self.pre_b = pre(hp["pre_hidden_b"], hp["pre_layers_b"])
        # A tower's recurrent input width is its pre-MLP's output, or the raw
        # latent when it has none.
        self.in_a = hp["pre_hidden_a"] if self.pre_a is not None else latent_dim
        self.in_b = hp["pre_hidden_b"] if self.pre_b is not None else latent_dim

        def tower(arch: str, in_size: int, hidden: int, layers: int) -> nn.Module:
            cls = nn.GRU if arch == "gru" else nn.LSTM
            return cls(input_size=in_size, hidden_size=hidden,
                       num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)

        # Separate modules => separate parameters, drawn independently from the
        # seeded RNG stream. Nothing is shared between the towers.
        self.tower_a = tower(hp["arch_a"], self.in_a, hp["hidden_a"], hp["layers_a"])
        self.tower_b = tower(hp["arch_b"], self.in_b, hp["hidden_b"], hp["layers_b"])

        layers: list[nn.Module] = []
        width = hp["hidden_a"] + hp["hidden_b"]
        for _ in range(hp["fusion_layers"]):
            layers += [nn.Linear(width, hp["fusion_hidden"]), nn.ReLU(),
                       nn.Dropout(dropout)]
            width = hp["fusion_hidden"]
        layers += [nn.Linear(width, latent_dim)]
        self.fusion = nn.Sequential(*layers)
        # Index of every post-ReLU fusion activation (for the SAE tap).
        self.fusion_relu_idx = [3 * i + 1 for i in range(hp["fusion_layers"])]

    @staticmethod
    def _run(cell: nn.Module, x: torch.Tensor, lengths: torch.Tensor | None):
        """Run one tower, packing to skip padding when lengths are given."""
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = cell(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.size(1))
        else:
            out, _ = cell(x)
        return out

    def _tower_input(self, x: torch.Tensor, view: str,
                     pre: nn.Module | None) -> torch.Tensor:
        """A tower's recurrent input: its causal view, optionally re-embedded by
        that tower's own pre-MLP (applied per timestep — see `pre`)."""
        z = make_view(x, view)
        if pre is None:
            return z
        B, T, _ = z.shape
        return pre(z.reshape(B * T, -1)).reshape(B, T, -1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Per-step hidden states of both towers: (B, T, hidden_a), (B, T, hidden_b)."""
        h_a = self._run(self.tower_a,
                        self._tower_input(x, self.view_a, self.pre_a), lengths)
        h_b = self._run(self.tower_b,
                        self._tower_input(x, self.view_b, self.pre_b), lengths)
        return h_a, h_b

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Per-step next-latent predictions (B, T, D) for teacher forcing."""
        h_a, h_b = self.encode(x, lengths)
        h = torch.cat([h_a, h_b], dim=-1)                   # (B, T, Ha+Hb)
        B, T, _ = h.shape
        out = self.fusion(h.reshape(B * T, -1))             # (B*T, D)
        return out.reshape(B, T, self.latent_dim)

    def predict_next(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Final predicted next-latent (B, D) from the last valid prefix step."""
        h_a, h_b = self.encode(x, lengths)
        h = torch.cat([h_a, h_b], dim=-1)                   # (B, T, Ha+Hb)
        if lengths is not None:
            idx = (lengths.to(h.device) - 1).clamp(min=0)
            rep = h[torch.arange(h.size(0), device=h.device), idx]
        else:
            rep = h[:, -1, :]                                # full, unpadded prefix
        return self.fusion(rep)                              # (B, D)


def _shape(side: str, hp: dict) -> str:
    """One-line description of a tower, e.g. "mlp256x1->gru h256 x1 on 'latent'"."""
    pre = hp[f"pre_hidden_{side}"]
    head = (f"mlp{pre}x{hp[f'pre_layers_{side}']}->" if pre > 0 else "")
    return (f"{head}{hp[f'arch_{side}']} h{hp[f'hidden_{side}']} "
            f"x{hp[f'layers_{side}']} on '{hp[f'view_{side}']}'")


# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #
def fit(art, hp: dict, seed: int = SEED):
    """Train the dual-tower model on the artifact's TRAIN sessions and return the
    best (early-stopped) model in eval mode.

    Pure training — no disk writes, no retrieval eval — so an ensemble harness
    can reuse the EXACT procedure. Mirrors seq_nexttrack.fit / seq_ann.fit (same
    seeding, val split, per-step objective, early stopping); only the network
    differs."""
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
          f"dual-tower: A {_shape('a', hp)} | B {_shape('b', hp)} -> MLP "
          f"{hp['fusion_hidden']} x{hp['fusion_layers']}"})
    emit({"kind": "log", "msg":
          f"loss {hp['loss']}, dropout {hp['dropout']}, lr {hp['lr']}, "
          f"batch {hp['batch_size']}, {hp['epochs']} epochs; objective "
          f"per-step teacher forcing"})

    model = DualTowerNextLatent(D, hp)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    def batch_loss(x, y, mask, lengths):
        pred = model(x, lengths)                            # (B, T, D)
        return step_loss(pred, y, mask, hp["loss"], hp["tau"], x=x,
                         eager_beta=float(hp["eager_beta"]),
                         eager_margin=float(hp["eager_margin"])), int(mask.sum())

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


def build_score_fn(model: DualTowerNextLatent, latents: np.ndarray):
    """Full-vocab cosine score_fn(prefix) for a trained dual-tower model — the
    SAME interface the ensemble harness uses for the GRU / ANN legs. SHARED by
    train and predict so serving ranks byte-identically to eval."""
    item_norm = torch.nn.functional.normalize(torch.from_numpy(latents), dim=1)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(latents[prefix][None, :, :])  # (1, T, D)
        with torch.no_grad():
            pred = model.predict_next(x)                    # (1, D) prefix -> next
            pred = torch.nn.functional.normalize(pred, dim=1)
            scores = (pred @ item_norm.t()).squeeze(0)      # (n_items,)
        return scores.numpy()
    return score_fn


def load_model(model_dir: Path, art, hp: dict) -> "DualTowerNextLatent":
    """Reconstruct the trained dual-tower model from a promoted dir's model.pt."""
    model = DualTowerNextLatent(art.latent_dim, hp)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Serving-time ranking: reload the model and rank the vocab for the caller's
    session prefix (see seq_common.predict_ranking)."""
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(model_dir)
    model = load_model(model_dir, art, hp)
    predict_ranking(art, build_score_fn(model, art.item_latents), input_dir,
                    output, k=hp["k"], mmr_lambda=hp.get("mmr_lambda"),
                    mmr_pool=int(hp.get("mmr_pool", 200)),
                    artist_cap=hp.get("artist_cap"))
    emit({"kind": "done"})


def _capture(model: "DualTowerNextLatent"):
    """Per-model-SAE activation tap: EACH TOWER's per-step hidden state plus the
    post-ReLU fusion hidden layers. Tapping the towers separately is the point —
    it lets the SAE read whether the two towers actually capture DIFFERENT
    next-item concepts or duplicate each other. Returns (layer_names,
    step_acts_fn) for seq_model_sae."""
    # Pre-MLP taps come FIRST when present: the hypothesis for an (MLP→GRU) tower
    # is that a learned nonlinear re-embedding finds input features a bare GRU's
    # linear input transform cannot, so the question the SAE answers is whether
    # `pre_b` carries next-item concepts `tower_a` lacks. (The fixed-view variant
    # failed exactly there: new directions, zero new concepts.)
    names = []
    if model.pre_a is not None:
        names.append("pre_a")
    if model.pre_b is not None:
        names.append("pre_b")
    names += ["tower_a", "tower_b"]
    names += [f"fusion_{i + 1}" for i in range(len(model.fusion_relu_idx))]

    def step_acts_fn(seq_latents: np.ndarray):
        x = torch.from_numpy(seq_latents[None, :, :].astype(np.float32))  # (1,L,D)
        with torch.no_grad():
            acts = []
            for view, pre in ((model.view_a, model.pre_a),
                              (model.view_b, model.pre_b)):
                if pre is not None:
                    acts.append(model._tower_input(x, view, pre)[0])  # (L, pre)
            h_a, h_b = model.encode(x, None)               # (1, L, H*)
            h = torch.cat([h_a, h_b], dim=-1)[0]           # (L, Ha+Hb)
            acts += [h_a[0], h_b[0]]
            cur = h
            for i, layer in enumerate(model.fusion):
                cur = layer(cur)                            # Dropout = identity (eval)
                if i in model.fusion_relu_idx:
                    acts.append(cur)
        # Steps 0..L-2 have a next item; the last step has none.
        return [a[:-1].numpy() for a in acts]

    return names, step_acts_fn


def model_sae(model_dir: Path, dataset: Path, output: Path, *, layers, n_atoms,
              l1, epochs, cache_dir, label_atoms, dataset_sae, topk=0) -> None:
    """Per-model SAE: dictionary-learn each tower's hidden state (and the fusion
    layers) and relate the atoms to next-item concepts (see seq_model_sae).
    ``topk>0`` uses a top-k SAE (l0==topk) instead of L1-induced sparsity."""
    import seq_model_sae
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(dataset)
    model = load_model(model_dir, art, hp)
    hidden = []
    if hp["pre_hidden_a"] > 0:
        hidden.append(hp["pre_hidden_a"])
    if hp["pre_hidden_b"] > 0:
        hidden.append(hp["pre_hidden_b"])
    hidden += [hp["hidden_a"], hp["hidden_b"]]
    hidden += [hp["fusion_hidden"]] * hp["fusion_layers"]
    seq_model_sae.run(
        model_dir, dataset, output,
        capture=lambda _art: _capture(model),
        meta={"predictor": "seq-dualgru", "hidden": hidden},
        layers=layers, n_atoms=n_atoms, l1=l1, epochs=epochs,
        cache_dir=cache_dir, label_atoms=label_atoms, dropped=True, topk=topk)


def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    art = load_artifact(dataset)
    latents = art.item_latents

    model = fit(art, hp, seed=hp["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)

    score_fn = build_score_fn(model, latents)
    metrics, predictions = eval_from_scores(
        art, score_fn, k=hp["k"],
        mmr_lambda=hp.get("mmr_lambda"), mmr_pool=int(hp.get("mmr_pool", 200)),
        artist_cap=hp.get("artist_cap"))
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
    ms = sub.add_parser("model-sae",
                        help="per-model SAE over both towers + the fusion layers")
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
