#!/usr/bin/env python3
"""Next-track SEQUENCE predictor — an explicit STAGE STACK of MLPs and recurrences.

Where `seq_nexttrack` is [optional pre-MLP] -> [stacked RNN] and `seq_dualgru` is two
PARALLEL towers, this model is one SEQUENTIAL stack whose composition is spelled out:

    stages = ["ann", "gru", "gru", "ann"]      # MLP, GRU, GRU, MLP

so the four arrangements under test are a single hyperparameter, not four code paths:

    gru gru gru gru        stages=gru+gru+gru+gru
    ann gru ann gru        stages=ann+gru+ann+gru      <- not expressible before
    ann gru gru ann        stages=ann+gru+gru+ann      <- not expressible before
    ann gru gru gru        stages=ann+gru+gru+gru

ONE MODULE FOR EVERY ARM, DELIBERATELY. Two of these four were already reachable with
`seq_nexttrack`'s `pre_hidden`/`num_layers`, and two were not. Running half the arms on
one predictor and half on another would confound architecture with implementation — any
difference could be the code. So all four go through this module and the batch carries
its own in-batch control.

`ann` STAGES ARE STRICTLY PER-TIMESTEP. This is a correctness constraint, not a style
choice: `seq_nexttrack._pre_encode` warns that an MLP mixing timesteps "would leak the
teacher-forcing target". A per-step MLP preserves sequence length, so the next
recurrence still has a sequence to walk — which is the point of a sequential stack. The
nonlinearity is what makes it a real expressivity change: a purely linear stage would
fold into the next GRU's input transform (W_i·(Vx) = (W_iV)x).

TRAINING IS REUSED VERBATIM — `seq_nexttrack.fit(art, hp, model=...)`, the same
seeding, val split, per-step teacher forcing, objective and early stopping every other
member of the family uses. Nothing about the optimisation is new here; only the
topology is.

TWO GATES, because a single "bit-identical" claim is not available (see below):
  * FUNCTIONAL EQUIVALENCE (`verify` subcommand) — `stages=["gru"]` must compute the
    SAME FUNCTION as `seq_nexttrack.SeqNextLatent` given identical weights. Proves the
    model class is right, independent of any RNG.
  * IN-BATCH CONTROL — every campaign runs `stages=gru` through THIS module alongside
    the arms, so the arms are mutually comparable even though the RNG stream is offset
    from the historical h256 run (`fit` re-seeds on entry, so passing a pre-built model
    shifts the training stream by the init draws; the weights at init match, the dropout
    stream does not). Do NOT expect this module to reproduce 176/1431 exactly, and do
    not treat that as a failure — gate on `verify` plus the in-batch control instead.

CAPACITY WARNING for whoever designs the campaign: four stacked GRUs at h256 is ~4x the
parameters of the h256 baseline, and the record already measured h425 (2.2x params) as a
TIE (-0.0021, straddles). A depth result is uninterpretable without a param-matched
control in the same batch, or "deeper wins" is indistinguishable from "bigger wins" on
an axis that is already closed.

CLI (mirrors the family; the registry passes a `--hyperparams` FILE):

  seq_stack.py train  --dataset <dir> --output <run_dir> --hyperparams <json|path>
  seq_stack.py predict --model <dir> --input <dir> --output <file>
  seq_stack.py verify  --dataset <dir>        # the functional-equivalence gate
  seq_stack.py fixture --output <dir>

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
    write_outputs,
)
import seq_nexttrack
from seq_nexttrack import SEED, SeqNextLatent, build_score_fn, fit

STAGE_KINDS = ("ann", "gru", "lstm")

DEFAULTS = {
    "stages": "ann+gru+gru+ann",   # the stack, in order; "+" or "," joined
    "hidden": 256,                 # width of every recurrent stage
    "ann_hidden": 256,             # width of every per-step MLP stage
    "dropout": 0.1,
    "loss": "infonce",
    "tau": 0.07,
    "epochs": 40,
    "patience": 6,
    "lr": 1e-3,
    "batch_size": 128,
    "val_fraction": 0.15,
    "seed": SEED,
    "k": 10,
    # Family-standard retrieval policy, inert by default.
    "mmr_lambda": None,
    "mmr_pool": 200,
    "artist_cap": None,
    # Present so `fit` and the shared eval see the keys they expect.
    "bidirectional": False,
    "residual": False,
    "num_layers": 1,
    "arch": "gru",
    "eager_beta": 0.0,
    "eager_margin": 0.0,
}


def parse_stages(spec) -> list[str]:
    if isinstance(spec, (list, tuple)):
        out = [str(s).strip().lower() for s in spec]
    else:
        out = [s.strip().lower() for s in str(spec).replace(",", "+").split("+")]
    out = [s for s in out if s]
    if not out:
        raise SystemExit("stages must name at least one stage")
    bad = [s for s in out if s not in STAGE_KINDS]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; allowed: {', '.join(STAGE_KINDS)}")
    if not any(s in ("gru", "lstm") for s in out):
        raise SystemExit("stages must contain at least one recurrent stage "
                         "(an all-ann stack has no recurrence and cannot use the "
                         "family's per-step teacher forcing meaningfully)")
    return out


def load_hp(spec: str) -> dict:
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    user = json.loads(raw)
    hp = {**DEFAULTS, **user}
    hp["stages"] = parse_stages(hp["stages"])
    for key in ("hidden", "ann_hidden", "epochs", "patience", "batch_size", "seed", "k"):
        hp[key] = int(hp[key])
    assert hp["loss"] in ("cosine", "infonce"), "loss must be cosine|infonce"
    assert hp["epochs"] >= 1 and hp["batch_size"] >= 1
    return hp


class StackNextLatent(nn.Module):
    """A sequential stack of per-step MLP and recurrent stages -> next latent.

    Interface matches `SeqNextLatent` exactly — `forward()` gives per-step
    predictions (B, T, latent_dim) for teacher forcing and `predict_next()` the final
    (B, latent_dim) — so `seq_nexttrack.fit` and `build_score_fn` consume it unchanged.

    Module creation order mirrors SeqNextLatent's default path (dropout, then the
    parameterised stages, then the head) so a single-`gru` stack draws its init from the
    seeded RNG in the same order and with the same shapes.
    """

    def __init__(self, latent_dim: int, stages: list[str], hidden: int,
                 ann_hidden: int, dropout: float, arch_default: str = "gru"):
        super().__init__()
        self.latent_dim = latent_dim
        self.stages = list(stages)
        self.hidden = hidden
        # `fit` and the shared eval read these off the model.
        self.last_item_only = False
        self.bidirectional = False

        self.drop = nn.Dropout(dropout)

        mods: list[nn.Module] = []
        d = latent_dim
        for kind in self.stages:
            if kind == "ann":
                # Per-step MLP. Sequential over the LAST dim only; applied by
                # reshaping (B*T, d) so no information can cross timesteps.
                mods.append(nn.Sequential(nn.Linear(d, ann_hidden), nn.ReLU(),
                                          nn.Dropout(dropout)))
                d = ann_hidden
            else:
                cls = nn.GRU if kind == "gru" else nn.LSTM
                mods.append(cls(input_size=d, hidden_size=hidden, num_layers=1,
                                batch_first=True))
                d = hidden
        self.blocks = nn.ModuleList(mods)
        self.head = nn.Linear(d, latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, latent_dim) -> (B, T, d_final). Length is preserved throughout."""
        h = x
        for kind, blk in zip(self.stages, self.blocks):
            if kind == "ann":
                B, T, dd = h.shape
                h = blk(h.reshape(B * T, dd)).reshape(B, T, -1)
            else:
                h, _ = blk(h)
                h = self.drop(h)
        return h

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        return self.head(self.encode(x))

    def predict_next(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        h = self.encode(x)
        if lengths is not None:
            idx = (lengths.to(h.device) - 1).clamp(min=0)
            rep = h[torch.arange(h.size(0), device=h.device), idx]
        else:
            rep = h[:, -1, :]
        return self.head(rep)


def build(art, hp: dict) -> StackNextLatent:
    return StackNextLatent(art.latent_dim, hp["stages"], hp["hidden"],
                           hp["ann_hidden"], hp["dropout"])


def load_model(model_dir: Path, art, hp: dict) -> StackNextLatent:
    m = build(art, hp)
    m.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# the functional-equivalence gate                                             #
# --------------------------------------------------------------------------- #
def verify(dataset: Path) -> int:
    """`stages=["gru"]` must compute the SAME FUNCTION as SeqNextLatent.

    Weights are copied across rather than re-initialised, so this isolates the
    computation from the RNG entirely: if the two disagree, the model class is wrong.
    """
    art = load_artifact(dataset)
    D = art.latent_dim
    torch.manual_seed(SEED)
    ref = SeqNextLatent(D, 256, "gru", 1, 0.1, bidirectional=False, residual=False)
    mine = StackNextLatent(D, ["gru"], 256, 256, 0.1)
    # name map: ref.rnn.* -> mine.blocks.0.* ; heads share names
    sd = {}
    for k, v in ref.state_dict().items():
        sd[k.replace("rnn.", "blocks.0.")] = v
    missing = mine.load_state_dict(sd, strict=False)
    if missing.unexpected_keys:
        print(f"  unexpected keys: {missing.unexpected_keys}")
    ref.eval(); mine.eval()
    worst_fwd = worst_next = 0.0
    for T in (1, 2, 5, 17, 40):
        x = torch.from_numpy(
            np.random.default_rng(T).standard_normal((3, T, D)).astype(np.float32))
        with torch.no_grad():
            worst_fwd = max(worst_fwd, float((ref(x) - mine(x)).abs().max()))
            worst_next = max(worst_next,
                             float((ref.predict_next(x) - mine.predict_next(x)).abs().max()))
        print(f"  T={T:3d}  forward max|Δ| {worst_fwd:.2e}   predict_next max|Δ| {worst_next:.2e}")
    n_ref = sum(v.numel() for v in ref.state_dict().values())
    n_mine = sum(v.numel() for v in mine.state_dict().values())
    print(f"  params  ref {n_ref:,}  mine {n_mine:,}  {'match' if n_ref == n_mine else 'MISMATCH'}")
    ok = worst_fwd < 1e-6 and worst_next < 1e-6 and n_ref == n_mine
    print("GATE PASS" if ok else "GATE FAIL")
    return 0 if ok else 1


def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    art = load_artifact(dataset)
    stages = hp["stages"]
    torch.manual_seed(hp["seed"])          # mirror fit()'s seeding before init
    model = build(art, hp)
    n_par = sum(p.numel() for p in model.parameters())
    emit({"kind": "log", "msg":
          f"stack [{' -> '.join(stages)}]  hidden {hp['hidden']}  ann {hp['ann_hidden']}  "
          f"{n_par:,} params ({n_par / 394944:.2f}x the h256 baseline — read any depth "
          f"result against a param-matched control)"})

    model = fit(art, hp, seed=hp["seed"], model=model)
    score_fn = build_score_fn(model, art.item_latents)
    metrics, predictions = eval_from_scores(
        art, score_fn, k=hp["k"],
        mmr_lambda=hp.get("mmr_lambda"), mmr_pool=int(hp.get("mmr_pool", 200)),
        artist_cap=hp.get("artist_cap"))
    metrics["stages"] = "+".join(stages)
    metrics["n_params"] = int(n_par)

    write_outputs(run_dir, metrics, predictions)
    torch.save(model.state_dict(), run_dir / "model.pt")
    hp_out = dict(hp); hp_out["stages"] = "+".join(stages)
    (run_dir / "hyperparams.json").write_text(json.dumps(hp_out))
    emit({"kind": "log", "msg":
          f"stack {'+'.join(stages)}  Recall@{hp['k']} {metrics['recall_at_k']:.5f}  "
          f"MRR {metrics['mrr']:.5f}  H@{hp['k']} {metrics.get('holisticness_at_k', float('nan')):.6f}  "
          f"(n_test {metrics['n_test']})"})
    emit({"kind": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(model_dir)
    model = load_model(model_dir, art, hp)
    predict_ranking(art, build_score_fn(model, art.item_latents), input_dir, output,
                    k=hp["k"], mmr_lambda=hp.get("mmr_lambda"),
                    mmr_pool=int(hp.get("mmr_pool", 200)),
                    artist_cap=hp.get("artist_cap"))
    emit({"kind": "done"})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--dataset", required=True, type=Path)
    t.add_argument("--output", required=True, type=Path)
    t.add_argument("--hyperparams", required=True)
    p = sub.add_parser("predict"); p.add_argument("--model", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    v = sub.add_parser("verify"); v.add_argument("--dataset", required=True, type=Path)
    f = sub.add_parser("fixture"); f.add_argument("--output", required=True, type=Path)
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
    else:
        make_fixture(args.output)


if __name__ == "__main__":
    main()
