#!/usr/bin/env python3
"""Next-track SEQUENCE predictor — a BANK of N PARALLEL bare recurrent towers.

Where `seq_dualgru` hard-wires exactly TWO towers (`view_a`/`view_b`) and `seq_stack`
generalises the SEQUENTIAL arrangement of stages, this model generalises the PARALLEL
one: N independent single-layer GRU towers read the prefix side by side over their own
causal views, their per-step hidden states are concatenated, and ONE LINEAR head maps
the concatenation to the next track's latent.

    views = "latent+latent+cummean"      # 3 towers: two on the raw latents, one on
                                         # the slow session centroid

so tower COUNT and the view MULTISET are one hyperparameter on one module, not N code
paths. That matters for the same reason it mattered for `seq_stack`: running N=2 on
`seq_dualgru` and N=3 on something new would confound topology with implementation, and
every arm's verdict could be the code. Every arm goes through this module, and the batch
carries its own in-batch control.

WHY THIS AXIS EXISTS AT ALL — parallelism is the one architecture direction on this
corpus with a measured win, and it is on the WALK surface, not the one-shot metric. The
N=2 dual-tower family was closed on recall (best f0 arm 0.11880 vs the h256 baseline's
0.12299, a tie) and then REOPENED once the generated walk was measured: `latent/latent`
gained vibe CI>0 and `latent/cummean` bought lower stride_err, less drift and more genre
variety at vibe held. The `latent/cummean` f0 arm is the deployed engine. The record's
own reading of the mechanism is that holding a mood needs a FIXED REFERENCE never
overwritten by recurrence — which is what a `cummean` tower approximates — so the open
question this module exists to answer is whether that mechanism is DOSE-RESPONSIVE.

WHAT IS DELIBERATELY NOT REACHABLE HERE, and why each was left out. This is the module's
main defence: every one of these is a *measured* regression on this corpus, so making
them unexpressible is stronger than leaving them at a safe default.

  * NO fusion MLP / `fusion_layers` — the head is hard-wired `nn.Linear(N*h, D)`.
    `fusion_layers=1` is a documented regression mode, confirmed three times
    (+0.016 / +0.020 / +0.0266 recovered by deleting it). The objective is cosine /
    InfoNCE retrieval in the item-latent space, so an extra nonlinear layer between the
    recurrent state and a metric read-out is a warp the loss then has to undo.
  * NO per-step pre-MLP / `pre_hidden` — refuted at N=1 (-0.01398 CI<0), refuted on the
    N=2 walk (Δvibe -0.0255 CI<0), and refuted in all five 4-stage stacks.
  * NO `layers` > 1 — every tower is exactly one recurrent layer. Depth inside the
    recurrent chain is the 2026-08-01 stage-stack cliff (55-77 hits of 1,431 lost).
  * NO `arch` / LSTM — 0.1188 vs the GRU's 0.12299 on this split, and its walk gains
    were the "better stride, worse vibe" trade, not a win.
  * NO `bidirectional` — a backward pass over the prefix lets step t see item t+1 and
    trivially copy the teacher-forcing target (see the LEAKAGE caveat in seq_nexttrack).
  * NO `eager_beta` / `eager_margin` — mis-specified against a calibrated baseline and
    refuted 2026-07-30. The keys exist in DEFAULTS at 0.0 only because `step_loss`
    reads them; they are NOT registry-declared, so no run can set them.

AT LEAST ONE `latent` TOWER IS REQUIRED, and `parse_views` raises without one. This is a
correctness guard, not a preference: the view-ablation arms that dropped the raw latents
entirely (`delta/delta`, `cummean/cummean`) cratered to recall 0.058-0.087, far below the
0.108 absolute floor, while their eagerness facets *improved* — exactly the shape that
looks interesting under a naive holisticness read and is worthless. Making those arms
unreachable removes the trap instead of documenting it.

DROPOUT — READ THIS BEFORE COMPARING TO THE DUAL-TOWER RECORD. This module applies
`seq_nexttrack`'s dropout semantics: one shared `nn.Dropout` on the concatenated
per-step hidden state, before the head. `seq_dualgru` at `fusion_layers=0` applies
**NO DROPOUT AT ALL** — its dropout only ever enters via a pre-MLP (absent at
`pre_hidden=0`), inter-layer inside a multi-layer tower (`dropout if layers > 1 else
0.0`), or inside the fusion MLP (which at `fusion_layers=0` is a bare
`Sequential(Linear)`). Two consequences, both material:

  1. The record's three-times-confirmed "f0 beats f1" measurement is CONFOUNDED with
     "no dropout beats dropout 0.1", and every f0 dual arm on record — including the
     deployed engine — trained unregularized. This module can measure that confound
     directly: two arms with byte-identical parameter counts differing only in
     `dropout` (0.0 vs 0.1) at `views="latent+latent"`.
  2. Because the f0 path has no active dropout, training it is DETERMINISTIC given init
     — which is why `dropout=0.0` here reproduces the historical dual-tower number to
     the last digit rather than landing in a band. Six independent runs of that config
     on record all return recall@10 0.1187980433263452.

TRAINING IS THE FAMILY'S, COPIED NOT DELEGATED. `fit()` below is a faithful copy of
`seq_dualgru.fit` — same seeding, same val split, same per-step teacher forcing, same
`seq_nexttrack.make_batches` / `step_loss`, same early stopping. It does NOT call
`seq_nexttrack.fit(art, hp, model=...)`: that re-seeds on entry, so handing it a
pre-built model offsets the dropout stream relative to the historical runs and degrades
both bit-identity gates below into mere bands. `seq_stack` had to accept exactly that
and settle for functional equivalence; here the stronger gate is available, so it is
taken.

TWO BIT-IDENTITY GATES (`verify`), both demanding max|Δ| == 0 with weights copied across
so the RNG is out of the picture entirely:

  * leg (a)  views="latent",        h256, dropout 0.1  ==  seq_nexttrack.SeqNextLatent
  * leg (b)  views="latent+latent", h256, dropout 0.0  ==  seq_dualgru.DualTowerNextLatent
                                                           at fusion_layers=0

Leg (a) proves the N=1 degenerate case IS the family's single GRU; leg (b) proves the
N=2 case IS the dual-tower f0 arm the record measured. Together they anchor the whole
N-axis to two independently-established points, so a difference at N>=3 is the topology
and not this file.

CAPACITY WARNING for whoever designs the campaign: N towers at width h is ~N x the
parameters of a single tower, and single-GRU width on PCA-192 is ALREADY CLOSED —
h425 (2.21x) and h537 (3.24x) both TIE the h256 baseline, and h297 (1.25x) actually
LOST (CI<0). Width points read 176 / 162 / 173 / 179 / 174 hits with no trend, so a
param-matched control is a WEAK instrument at this effect size: never rest a verdict on
the capacity control alone, and require the win to hold against BOTH the h256 baseline
and the param-matched control.

CLI (mirrors the family; the registry passes a `--hyperparams` FILE):

  seq_bank.py train     --dataset <dir> --output <run_dir> --hyperparams <json|path>
  seq_bank.py predict   --model <dir> --input <dir> --output <file>
  seq_bank.py verify    --dataset <dir>        # the two bit-identity gates
  seq_bank.py model-sae --model <dir> --dataset <dir> --output <file>
  seq_bank.py fixture   --output <dir>

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
# The causal views are reused VERBATIM from the dual-tower module — not reimplemented —
# so a bank arm reads exactly the stream the dual-tower record measured.
from seq_dualgru import VIEWS, make_view
# The family's EXACT batching + loss, so the objective is byte-for-byte the family's
# (this is a new topology, not a new training recipe).
from seq_nexttrack import SEED, make_batches, step_loss

# Parameter count of the family's h256 single-GRU reference, for the ratio in the log.
BASELINE_PARAMS = 394_944

DEFAULTS = {
    "views": "latent+latent",   # the tower bank, in order; "+" or "," joined
    "hidden": 256,              # width of EVERY tower (uniform by design)
    "dropout": 0.1,             # on the concatenated hidden state, before the head
    "loss": "infonce",
    "tau": 0.07,
    "epochs": 40,
    "patience": 6,
    "lr": 1e-3,
    "batch_size": 128,
    "val_fraction": 0.15,
    "seed": SEED,               # 1337 — deterministic
    "k": 10,
    # Family-standard retrieval policy, inert by default.
    "mmr_lambda": None,
    "mmr_pool": 200,
    "artist_cap": None,
    # Present ONLY so `step_loss` sees the keys it reads. Deliberately absent from the
    # registry block: the anti-eager levers are refuted and must not be settable.
    "eager_beta": 0.0,
    "eager_margin": 0.0,
}


def parse_views(spec) -> list[str]:
    """The tower bank as an ordered list of causal views.

    Raises rather than warns on an all-derived bank: a tower reading `delta` or
    `cummean` sees a lossy, deterministic function of the same latent sequence, so a
    bank without the raw latents has thrown away the only view that carries next-item
    identity. Measured: 0.058-0.087 recall, below every floor."""
    if isinstance(spec, (list, tuple)):
        out = [str(s).strip().lower() for s in spec]
    else:
        out = [s.strip().lower() for s in str(spec).replace(",", "+").split("+")]
    out = [s for s in out if s]
    if not out:
        raise SystemExit("views must name at least one tower")
    bad = [s for s in out if s not in VIEWS]
    if bad:
        raise SystemExit(f"unknown view(s) {bad}; allowed: {', '.join(VIEWS)}")
    if "latent" not in out:
        raise SystemExit(
            "views must contain at least one 'latent' tower — `delta` and `cummean` are "
            "deterministic lossy functions of the latent sequence, and banks without the "
            "raw view measured 0.058-0.087 recall@10, below the 0.108 absolute floor")
    return out


def load_hp(spec: str) -> dict:
    """Accept a path to a JSON file (registry convention) or an inline JSON string.
    Merges over DEFAULTS and validates."""
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    user = json.loads(raw)
    hp = {**DEFAULTS, **user}
    hp["views"] = parse_views(hp["views"])
    for key in ("hidden", "epochs", "patience", "batch_size", "seed", "k"):
        hp[key] = int(hp[key])
    hp["dropout"] = float(hp["dropout"])
    assert hp["loss"] in ("cosine", "infonce"), "loss must be cosine|infonce"
    assert hp["hidden"] >= 1, "hidden must be >= 1"
    assert 0.0 <= hp["dropout"] < 1.0, "dropout must be in [0, 1)"
    assert hp["epochs"] >= 1 and hp["batch_size"] >= 1
    return hp


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class BankNextLatent(nn.Module):
    """N independent bare GRU towers in parallel -> concat -> LINEAR head.

    Interface matches `SeqNextLatent` and `DualTowerNextLatent` exactly — `forward()`
    gives per-step predictions (B, T, latent_dim) for teacher forcing and
    `predict_next()` the final (B, latent_dim) — so the shared eval, the score-fn
    contract and the walk harness consume it unchanged.

    MODULE CREATION ORDER IS LOAD-BEARING: dropout, then the towers in `views` order,
    then the head. That is `SeqNextLatent`'s order (drop, [pre], rnn, head) and
    `DualTowerNextLatent`'s order ([pre_a], [pre_b], tower_a, tower_b, fusion), so at
    N=1 and N=2 this module draws its init from the seeded RNG in the same sequence with
    the same shapes as the module it must reproduce. Reordering these three lines would
    silently break both `verify` legs and, with them, the batch's reproduction gates."""

    def __init__(self, latent_dim: int, views: list[str], hidden: int, dropout: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.views = list(views)
        self.hidden = hidden
        # `fit` and the shared eval read these off the model.
        self.last_item_only = False
        self.bidirectional = False

        self.drop = nn.Dropout(dropout)
        # Separate modules => separate parameters, drawn independently from the seeded
        # RNG stream. Nothing is shared or tied between towers, including between two
        # towers that read the SAME view (that is the point of "N independent GRUs").
        self.towers = nn.ModuleList([
            nn.GRU(input_size=latent_dim, hidden_size=hidden, num_layers=1,
                   batch_first=True)
            for _ in self.views
        ])
        self.head = nn.Linear(len(self.views) * hidden, latent_dim)

    @staticmethod
    def _run(cell: nn.Module, x: torch.Tensor, lengths: torch.Tensor | None):
        """Run one tower, packing to skip padding when lengths are given.

        `total_length` is pinned to the input's T so every tower returns the same
        number of steps and the concatenation below cannot silently misalign when one
        tower's longest sequence differs from the batch's padded width."""
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = cell(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.size(1))
        else:
            out, _ = cell(x)
        return out

    def encode(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Per-step hidden states of every tower, concatenated: (B, T, N*hidden)."""
        outs = [self._run(cell, make_view(x, view), lengths)
                for view, cell in zip(self.views, self.towers)]
        return torch.cat(outs, dim=-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Per-step next-latent predictions (B, T, D) for teacher forcing."""
        return self.head(self.drop(self.encode(x, lengths)))

    def predict_next(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Final predicted next-latent (B, D) from the last valid prefix step."""
        h = self.encode(x, lengths)
        if lengths is not None:
            idx = (lengths.to(h.device) - 1).clamp(min=0)
            rep = h[torch.arange(h.size(0), device=h.device), idx]
        else:
            rep = h[:, -1, :]                                # full, unpadded prefix
        return self.head(self.drop(rep))


def build(art, hp: dict) -> BankNextLatent:
    return BankNextLatent(art.latent_dim, hp["views"], hp["hidden"], hp["dropout"])


def load_model(model_dir: Path, art, hp: dict) -> BankNextLatent:
    """Reconstruct the trained bank from a promoted dir's model.pt.

    `art` may come from the DATASET rather than the model dir (the walk harness does
    exactly that, after asserting the item space is byte-identical), so only
    `art.latent_dim` is used here."""
    m = build(art, hp)
    m.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    m.eval()
    return m


def build_score_fn(model: BankNextLatent, latents: np.ndarray):
    """Full-vocab cosine score_fn(prefix) for a trained bank — the SAME interface the
    blend/ensemble harness and the walk harness use for every other leg. SHARED by
    train and predict so serving ranks byte-identically to eval."""
    item_norm = torch.nn.functional.normalize(torch.from_numpy(latents), dim=1)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(latents[prefix][None, :, :])   # (1, T, D)
        with torch.no_grad():
            pred = model.predict_next(x)                    # (1, D) prefix -> next
            pred = torch.nn.functional.normalize(pred, dim=1)
            scores = (pred @ item_norm.t()).squeeze(0)      # (n_items,)
        return scores.numpy()
    return score_fn


# --------------------------------------------------------------------------- #
# the bit-identity gates                                                      #
# --------------------------------------------------------------------------- #
def _worst(ref: nn.Module, mine: nn.Module, D: int) -> tuple[float, float]:
    """max|Δ| on `forward` and `predict_next` over a fixed set of prefix lengths."""
    ref.eval(); mine.eval()
    worst_fwd = worst_next = 0.0
    for T in (1, 2, 5, 17, 40):
        x = torch.from_numpy(
            np.random.default_rng(T).standard_normal((3, T, D)).astype(np.float32))
        with torch.no_grad():
            worst_fwd = max(worst_fwd, float((ref(x) - mine(x)).abs().max()))
            worst_next = max(worst_next,
                             float((ref.predict_next(x) - mine.predict_next(x))
                                   .abs().max()))
        print(f"    T={T:3d}  forward max|Δ| {worst_fwd:.2e}   "
              f"predict_next max|Δ| {worst_next:.2e}")
    return worst_fwd, worst_next


def _params(m: nn.Module) -> int:
    return sum(v.numel() for v in m.state_dict().values())


def verify(dataset: Path) -> int:
    """The two bit-identity gates. Weights are COPIED across rather than
    re-initialised, so this isolates the computation from the RNG entirely: if either
    leg disagrees, the model class is wrong and nothing downstream is interpretable.

    Run this BEFORE the registry edit and BEFORE any server restart — a failure then
    costs nothing."""
    import seq_dualgru
    from seq_nexttrack import SeqNextLatent

    art = load_artifact(dataset)
    D = art.latent_dim
    ok = True

    # ---- leg (a): N=1 IS the family's single GRU ---------------------------- #
    print("  leg (a)  views='latent' h256 dropout 0.1  vs  SeqNextLatent")
    torch.manual_seed(SEED)
    ref_a = SeqNextLatent(D, 256, "gru", 1, 0.1, bidirectional=False, residual=False)
    mine_a = BankNextLatent(D, ["latent"], 256, 0.1)
    # name map: ref.rnn.* -> mine.towers.0.* ; heads share names
    sd = {k.replace("rnn.", "towers.0."): v for k, v in ref_a.state_dict().items()}
    res = mine_a.load_state_dict(sd, strict=False)
    if res.unexpected_keys or res.missing_keys:
        print(f"    unexpected {res.unexpected_keys}  missing {res.missing_keys}")
        ok = False
    wf, wn = _worst(ref_a, mine_a, D)
    n_ref, n_mine = _params(ref_a), _params(mine_a)
    print(f"    params  ref {n_ref:,}  mine {n_mine:,}  "
          f"{'match' if n_ref == n_mine else 'MISMATCH'}")
    leg_a = wf == 0.0 and wn == 0.0 and n_ref == n_mine == BASELINE_PARAMS
    print(f"    leg (a) {'PASS' if leg_a else 'FAIL'} "
          f"(requires max|Δ| exactly 0 and {BASELINE_PARAMS:,} params)")
    ok = ok and leg_a

    # ---- leg (b): N=2 IS the dual-tower f0 arm ------------------------------ #
    print("  leg (b)  views='latent+latent' h256 dropout 0.0  vs  DualTowerNextLatent f0")
    hp_dual = {**seq_dualgru.DEFAULTS, "view_a": "latent", "view_b": "latent",
               "hidden_a": 256, "hidden_b": 256, "fusion_layers": 0,
               "pre_hidden_a": 0, "pre_hidden_b": 0, "dropout": 0.0}
    torch.manual_seed(SEED)
    ref_b = seq_dualgru.DualTowerNextLatent(D, hp_dual)
    mine_b = BankNextLatent(D, ["latent", "latent"], 256, 0.0)
    # name map: tower_a.* -> towers.0.* ; tower_b.* -> towers.1.* ;
    #           fusion.0.* (the single linear layer at fusion_layers=0) -> head.*
    sd = {}
    for k, v in ref_b.state_dict().items():
        sd[k.replace("tower_a.", "towers.0.")
            .replace("tower_b.", "towers.1.")
            .replace("fusion.0.", "head.")] = v
    res = mine_b.load_state_dict(sd, strict=False)
    if res.unexpected_keys or res.missing_keys:
        print(f"    unexpected {res.unexpected_keys}  missing {res.missing_keys}")
        ok = False
    wf, wn = _worst(ref_b, mine_b, D)
    n_ref, n_mine = _params(ref_b), _params(mine_b)
    print(f"    params  ref {n_ref:,}  mine {n_mine:,}  "
          f"{'match' if n_ref == n_mine else 'MISMATCH'}")
    leg_b = wf == 0.0 and wn == 0.0 and n_ref == n_mine == 789_696
    print(f"    leg (b) {'PASS' if leg_b else 'FAIL'} "
          f"(requires max|Δ| exactly 0 and 789,696 params)")
    ok = ok and leg_b

    print("GATE PASS" if ok else "GATE FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #
def fit(art, hp: dict, seed: int = SEED):
    """Train the bank on the artifact's TRAIN sessions and return the best
    (early-stopped) model in eval mode.

    A FAITHFUL COPY of `seq_dualgru.fit` — same seeding, same val split, same per-step
    objective, same early stopping — deliberately NOT a delegation to
    `seq_nexttrack.fit(..., model=...)`, which re-seeds on entry and would offset the
    training stream relative to the runs this module has to reproduce. Pure training:
    no disk writes, no retrieval eval."""
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
          f"bank: {len(hp['views'])} parallel gru towers h{hp['hidden']} on "
          f"[{' | '.join(hp['views'])}] -> concat -> linear head"})
    emit({"kind": "log", "msg":
          f"loss {hp['loss']}, dropout {hp['dropout']}, lr {hp['lr']}, "
          f"batch {hp['batch_size']}, {hp['epochs']} epochs; objective "
          f"per-step teacher forcing"})

    model = build(art, hp)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    def batch_loss(x, y, mask, lengths):
        pred = model(x, lengths)                             # (B, T, D)
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


def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    art = load_artifact(dataset)
    views = hp["views"]

    torch.manual_seed(hp["seed"])          # mirror fit()'s seeding before the count
    n_par = sum(p.numel() for p in build(art, hp).parameters())
    emit({"kind": "log", "msg":
          f"bank [{'+'.join(views)}] h{hp['hidden']}  {n_par:,} params "
          f"({n_par / BASELINE_PARAMS:.2f}x the h256 baseline — single-GRU width is "
          f"already closed on this split, so read any N result against a param-matched "
          f"control AND the h256 baseline, never the control alone)"})

    model = fit(art, hp, seed=hp["seed"])
    score_fn = build_score_fn(model, art.item_latents)
    metrics, predictions = eval_from_scores(
        art, score_fn, k=hp["k"],
        mmr_lambda=hp.get("mmr_lambda"), mmr_pool=int(hp.get("mmr_pool", 200)),
        artist_cap=hp.get("artist_cap"))
    metrics["views"] = "+".join(views)
    metrics["n_towers"] = len(views)
    metrics["n_params"] = int(n_par)

    write_outputs(run_dir, metrics, predictions)
    torch.save(model.state_dict(), run_dir / "model.pt")
    hp_out = dict(hp); hp_out["views"] = "+".join(views)
    (run_dir / "hyperparams.json").write_text(json.dumps(hp_out))
    emit({"kind": "log", "msg":
          f"bank {'+'.join(views)} h{hp['hidden']}  "
          f"Recall@{hp['k']} {metrics['recall_at_k']:.5f}  "
          f"MRR {metrics['mrr']:.5f}  "
          f"H@{hp['k']} {metrics.get('holisticness_at_k', float('nan')):.6f}  "
          f"(n_test {metrics['n_test']})"})
    emit({"kind": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Serving-time ranking: reload the model and rank the vocab for the caller's
    session prefix (see seq_common.predict_ranking)."""
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(model_dir)
    model = load_model(model_dir, art, hp)
    predict_ranking(art, build_score_fn(model, art.item_latents), input_dir, output,
                    k=hp["k"], mmr_lambda=hp.get("mmr_lambda"),
                    mmr_pool=int(hp.get("mmr_pool", 200)),
                    artist_cap=hp.get("artist_cap"))
    emit({"kind": "done"})


# --------------------------------------------------------------------------- #
# Per-model SAE tap                                                           #
# --------------------------------------------------------------------------- #
def _capture(model: BankNextLatent):
    """Activation tap: EACH TOWER's per-step hidden state, separately.

    Tapping the towers individually is the whole point — it lets the SAE answer whether
    the extra towers capture next-item concepts the raw-`latent` tower lacks, or merely
    add linear directions. The N=2 measurement is the reason this axis needs the check:
    the `cummean` tower added real directions (CKA 0.458) but ZERO next-item concepts
    tower A did not already have. There is no fusion layer to tap — the head is linear
    by construction."""
    names = [f"tower_{i}" for i in range(len(model.views))]

    def step_acts_fn(seq_latents: np.ndarray):
        x = torch.from_numpy(seq_latents[None, :, :].astype(np.float32))   # (1, L, D)
        with torch.no_grad():
            acts = [model._run(cell, make_view(x, view), None)[0]
                    for view, cell in zip(model.views, model.towers)]
        # Steps 0..L-2 have a next item; the last step has none.
        return [a[:-1].numpy() for a in acts]

    return names, step_acts_fn


def model_sae(model_dir: Path, dataset: Path, output: Path, *, layers, n_atoms,
              l1, epochs, cache_dir, label_atoms, dataset_sae, topk=0) -> None:
    """Per-model SAE: dictionary-learn each tower's hidden state and relate the atoms
    to next-item concepts (see seq_model_sae). ``topk>0`` uses a top-k SAE (l0==topk)
    instead of L1-induced sparsity."""
    import seq_model_sae
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(dataset)
    model = load_model(model_dir, art, hp)
    seq_model_sae.run(
        model_dir, dataset, output,
        capture=lambda _art: _capture(model),
        meta={"predictor": "seq-bank",
              "hidden": [hp["hidden"]] * len(hp["views"])},
        layers=layers, n_atoms=n_atoms, l1=l1, epochs=epochs,
        cache_dir=cache_dir, label_atoms=label_atoms, dropped=True, topk=topk)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--dataset", required=True, type=Path)
    t.add_argument("--output", required=True, type=Path)
    t.add_argument("--hyperparams", required=True)   # file path or inline JSON
    p = sub.add_parser("predict"); p.add_argument("--model", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    v = sub.add_parser("verify", help="the two bit-identity gates")
    v.add_argument("--dataset", required=True, type=Path)
    f = sub.add_parser("fixture"); f.add_argument("--output", required=True, type=Path)
    f.add_argument("--n-items", type=int, default=20)
    f.add_argument("--n-sessions", type=int, default=40)
    ms = sub.add_parser("model-sae", help="per-model SAE over every tower")
    ms.add_argument("--model", required=True, type=Path)
    ms.add_argument("--dataset", required=True, type=Path)
    ms.add_argument("--output", required=True, type=Path)
    ms.add_argument("--layers", default="")          # "1,2"; empty = every tower
    ms.add_argument("--n-atoms", type=int, default=0)
    ms.add_argument("--l1", type=float, default=0.0015)
    ms.add_argument("--epochs", type=int, default=40)
    ms.add_argument("--topk", type=int, default=0,
                    help="top-k SAE: hard-cap active atoms per row to K (l0==K); 0 = L1-only")
    ms.add_argument("--cache-dir", type=Path, default=None)
    ms.add_argument("--label-atoms", action="store_true")
    ms.add_argument("--dataset-sae", type=Path, default=None)
    args = ap.parse_args()

    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    except Exception:
        pass

    if args.cmd == "train":
        args.output.mkdir(parents=True, exist_ok=True)
        train(args.dataset, args.output, args.hyperparams)
    elif args.cmd == "predict":
        predict(args.model, args.input, args.output)
    elif args.cmd == "verify":
        raise SystemExit(verify(args.dataset))
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
