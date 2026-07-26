#!/usr/bin/env python3
"""Next-track SEQUENCE predictor with a LEARNED item representation.

Every other model in this family predicts into a FROZEN space: the item latents
come from an offline PCA/AE and the model only learns how to map a prefix to a
point in that fixed space. This one can learn the space itself — the item
embedding table is a parameter, used both as the GRU's input and as the retrieval
target, so the geometry is shaped by next-track adjacency rather than inherited
from a reconstruction objective.

That is the direction the record actually supports. The one architectural-family
success on record is a *learned* projection of the content space (InfoNCE metric
learning, +0.0259 CI>0, seed-robust, cold-reaching), and the 2026-07-18c study
measured the frozen PCA-192 geometry as a handicap at the raw level (six of eight
alternative spaces beat the frozen GRU by +0.025..+0.036 CI>0). Meanwhile five
consecutive campaigns adding machinery AROUND a single GRU over that frozen space
all nulled. Unfreezing the space is the untested axis.

THE SPECTRUM (`embed_mode`), because the endpoints are both informative:

  * `frozen`   — the table is the dataset's item latents, not trained. This
                 REPRODUCES seq-nexttrack and exists as a gate: if it does not
                 match the on-record baseline, the rest of this module is wrong.
  * `residual` — table = frozen latent + a trained residual. Cold-capable (an
                 unseen item still has its content vector) and the direct
                 generalization of the champion's post-hoc linear projection into
                 something learned jointly and nonlinearly.
  * `free`     — a fully learned table, optionally randomly initialized. The
                 textbook GRU4Rec/SASRec formulation, and the one this corpus is
                 least able to support (see DATA BUDGET).

DATA BUDGET — measured on the canonical split before writing this, because it
bounds what `free` can possibly do:

    73,632 training steps to fit 15,816 item vectors
    median per-item train frequency 1; only 3,294 items occur >= 5 times
    55.1% of test targets (788/1431) never appear in ANY training session

So `free` estimates a vector per item from a median of one observation, and can
never retrieve the cold 55% at all — its ceiling is recall@10 0.4493. That
ceiling is still 3.7x the current best single model (0.123) and 2.1x the champion
blend (0.212), so the trade "give up the cold half, be much better on the warm
half" is genuinely open. But read a `free` loss as a statement about DATA VOLUME,
not about whether co-occurrence carries signal.

COLLAPSE WARNING — why the loss default differs from the rest of the family. When
the retrieval space is FROZEN, `loss=cosine` is fine. When it is LEARNED, cosine
has a trivial optimum: drive every embedding to the same point and cos(pred,
target) = 1 everywhere, at zero information. Only a CONTRASTIVE objective avoids
it, because InfoNCE must separate the positive from in-batch negatives. Hence
`loss` defaults to infonce here and `cosine` is rejected outright for the trained
modes rather than silently collapsing.

Objective / training otherwise the family's, reused verbatim from seq_nexttrack
(`step_loss`): per-step teacher forcing, in-batch InfoNCE, the same early
stopping, the same eval. Only the batcher differs — it must yield item INDICES
rather than latents, since the embedding is what is being learned.

CLI (mirrors the family):

  seq_embed.py train --dataset <dir> --output <run_dir> --hyperparams <json|path>
  seq_embed.py predict --model <dir> --input <dir> --output <file>
  seq_embed.py model-sae --model <dir> --dataset <dir> --output <file>
  seq_embed.py fixture --output <dir>

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
from seq_nexttrack import SEED, step_loss

MODES = ("frozen", "residual", "free")
INITS = ("content", "random")

DEFAULTS = {
    "embed_mode": "residual",   # "frozen" | "residual" | "free"
    "embed_init": "content",    # "content" | "random" (random only with free)
    "embed_dim": 0,             # 0 = the dataset's latent_dim (required unless free)
    "residual_scale": 0.1,      # init magnitude of the residual, relative to |latent|
    "hidden": 256,
    "arch": "gru",              # "gru" | "lstm"
    "num_layers": 1,
    "dropout": 0.1,
    "loss": "infonce",          # see the COLLAPSE WARNING — cosine is unsafe here
    "tau": 0.07,
    "epochs": 40,
    "lr": 1e-3,
    "embed_lr": 0.0,            # 0 = same as lr; a separate rate for the table
    "batch_size": 128,
    "patience": 6,
    "val_fraction": 0.15,
    "seed": SEED,
    "k": 10,
    "eager_beta": 0.0,
    "eager_margin": 0.0,
    "mmr_lambda": None,
    "mmr_pool": 200,
}


def load_hp(spec: str) -> dict:
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    user = json.loads(raw)
    hp = {**DEFAULTS, **user}
    for key in ("embed_dim", "hidden", "num_layers", "epochs", "batch_size",
                "patience", "seed", "k"):
        hp[key] = int(hp[key])
    assert hp["embed_mode"] in MODES, f"embed_mode must be one of {MODES}"
    assert hp["embed_init"] in INITS, f"embed_init must be one of {INITS}"
    assert hp["arch"] in ("gru", "lstm"), "arch must be gru|lstm"
    assert hp["loss"] in ("cosine", "infonce"), "loss must be cosine|infonce"
    # Guard the collapse mode rather than letting it train to a useless optimum.
    if hp["embed_mode"] != "frozen" and hp["loss"] == "cosine":
        raise SystemExit(
            "loss=cosine with a LEARNED embedding has a trivial optimum (collapse "
            "every item to one point => cos == 1 everywhere at zero information). "
            "Use loss=infonce, or embed_mode=frozen if you really want cosine.")
    if hp["embed_init"] == "random" and hp["embed_mode"] != "free":
        raise SystemExit("embed_init=random only makes sense with embed_mode=free")
    if hp["embed_mode"] != "free" and hp["embed_dim"] not in (0,):
        # residual/frozen ride the dataset's own latent geometry.
        raise SystemExit("embed_dim must be 0 (= the dataset's latent_dim) unless "
                         "embed_mode=free")
    assert hp["epochs"] >= 1 and hp["batch_size"] >= 1
    return hp


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class EmbedNextItem(nn.Module):
    """Recurrent encoder over LEARNED item embeddings -> predicted next embedding.

    The embedding table is used twice — to embed the prefix and as the retrieval
    space the head predicts into (weight tying, as in GRU4Rec/SASRec). That is
    what makes the geometry trainable: a gradient step moves both the query and
    the targets."""

    def __init__(self, n_items: int, latents: np.ndarray, hp: dict):
        super().__init__()
        self.mode = hp["embed_mode"]
        dim = int(hp["embed_dim"]) or int(latents.shape[1])
        self.dim = dim

        base = torch.from_numpy(latents.astype(np.float32))
        if self.mode == "free" and hp["embed_init"] == "random":
            # Small random init, scaled like the content latents so the loss
            # starts in a comparable regime rather than saturating InfoNCE.
            scale = float(base.norm(dim=1).mean()) / max(dim ** 0.5, 1.0)
            self.embed = nn.Parameter(torch.randn(n_items, dim) * scale)
            self.base = None
        elif self.mode == "free":
            self.embed = nn.Parameter(base.clone())
            self.base = None
        elif self.mode == "residual":
            # Frozen content vector + a trained correction. Cold items still have
            # their content vector, so retrieval never loses the cold 55%.
            self.register_buffer("base", base)
            r = torch.randn(n_items, dim) * (
                float(base.norm(dim=1).mean()) * float(hp["residual_scale"])
                / max(dim ** 0.5, 1.0))
            self.embed = nn.Parameter(r)
        else:  # frozen — the reproduction gate
            self.register_buffer("base", base)
            self.embed = None

        rnn_cls = nn.GRU if hp["arch"] == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=dim, hidden_size=hp["hidden"],
                           num_layers=hp["num_layers"], batch_first=True,
                           dropout=hp["dropout"] if hp["num_layers"] > 1 else 0.0)
        self.drop = nn.Dropout(hp["dropout"])
        self.head = nn.Linear(hp["hidden"], dim)

    def table(self) -> torch.Tensor:
        """The current item representation — the retrieval space AND the input
        embedding. `residual` adds the trained correction to the frozen base."""
        if self.mode == "frozen":
            return self.base
        if self.mode == "residual":
            return self.base + self.embed
        return self.embed

    def _run(self, x: torch.Tensor, lengths: torch.Tensor | None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.size(1))
        else:
            out, _ = self.rnn(x)
        return out

    def forward(self, idx: torch.Tensor, lengths: torch.Tensor | None = None):
        """Per-step predicted next-embedding (B, T, dim) for teacher forcing."""
        x = self.table()[idx]                       # (B, T, dim)
        return self.head(self.drop(self._run(x, lengths)))

    def predict_next(self, idx: torch.Tensor, lengths: torch.Tensor | None = None):
        """Final predicted next-embedding (B, dim) for a prefix of item indices."""
        out = self._run(self.table()[idx], lengths)
        if lengths is not None:
            at = (lengths.to(out.device) - 1).clamp(min=0)
            last = out[torch.arange(out.size(0), device=out.device), at]
        else:
            last = out[:, -1, :]
        return self.head(self.drop(last))


# --------------------------------------------------------------------------- #
# Batching — INDICES, not latents                                             #
# --------------------------------------------------------------------------- #
def make_index_batches(seqs: list[np.ndarray], batch_size: int,
                       rng: np.random.Generator, shuffle: bool):
    """Padded teacher-forcing batches of item INDICES.

    seq_nexttrack.make_batches resolves indices to frozen latents inside the
    batcher, which is exactly what cannot happen here — the embedding is a
    parameter, so the lookup has to sit inside the graph. Everything else (the
    pairing, the padding, the mask, the lengths) matches it."""
    order = np.arange(len(seqs))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        batch = [seqs[i] for i in order[start:start + batch_size]]
        lengths = np.array([len(s) - 1 for s in batch], dtype=np.int64)
        tmax = int(lengths.max())
        xi = np.zeros((len(batch), tmax), dtype=np.int64)
        yi = np.zeros((len(batch), tmax), dtype=np.int64)
        mask = np.zeros((len(batch), tmax), dtype=bool)
        for r, seq in enumerate(batch):
            n = len(seq) - 1
            xi[r, :n] = seq[:-1]
            yi[r, :n] = seq[1:]
            mask[r, :n] = True
        yield (torch.from_numpy(xi), torch.from_numpy(yi),
               torch.from_numpy(mask), torch.from_numpy(lengths))


# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #
def fit(art, hp: dict, seed: int = SEED):
    """Train and return the best (early-stopped) model, in eval mode."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    all_train = training_pairs(art, art.train_sessions)
    assert all_train, "no usable train sessions (need length >= 2)"
    perm = rng.permutation(len(all_train))
    n_val = max(1, int(len(all_train) * hp["val_fraction"]))
    val_ids = set(perm[:n_val].tolist())
    tr_seqs = [s for i, s in enumerate(all_train) if i not in val_ids]
    va_seqs = [s for i, s in enumerate(all_train) if i in val_ids]

    # The data budget bounds what a free table can learn — surface it per run so
    # a null is read as data volume rather than as "co-occurrence has no signal".
    seen = set()
    for s in tr_seqs:
        seen.update(s.tolist())
    steps = int(sum(len(s) - 1 for s in tr_seqs))
    emit({"kind": "log", "msg":
          f"dataset {art.manifest['dataset_id']}: {art.n_items} items, "
          f"{len(tr_seqs)} train / {len(va_seqs)} val sessions, {steps} steps; "
          f"{len(seen)} items seen in train ({steps / max(len(seen), 1):.1f} "
          f"steps per seen item)"})
    elr = f" (embed_lr {hp['embed_lr']})" if hp["embed_lr"] else ""
    emit({"kind": "log", "msg":
          f"embed_mode {hp['embed_mode']} init {hp['embed_init']} dim "
          f"{hp['embed_dim'] or art.latent_dim}; {hp['arch']} h{hp['hidden']} "
          f"x{hp['num_layers']}, loss {hp['loss']}, lr {hp['lr']}{elr}, "
          f"{hp['epochs']} epochs"})

    model = EmbedNextItem(art.n_items, art.item_latents, hp)
    # A separate, usually smaller rate for the table is the standard remedy when
    # per-item data is thin: the encoder can move fast while 15.8k vectors seen a
    # median of once each move slowly.
    if hp["embed_lr"] and model.embed is not None:
        enc = [p for n, p in model.named_parameters() if n != "embed"]
        opt = torch.optim.Adam([{"params": enc, "lr": hp["lr"]},
                                {"params": [model.embed], "lr": hp["embed_lr"]}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    def batch_loss(xi, yi, mask, lengths):
        pred = model(xi, lengths)                       # (B, T, dim)
        tgt = model.table()[yi]                        # (B, T, dim) — in-graph
        return step_loss(pred, tgt, mask, hp["loss"], hp["tau"],
                         x=model.table()[xi],
                         eager_beta=float(hp["eager_beta"]),
                         eager_margin=float(hp["eager_margin"])), int(mask.sum())

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xi, yi, mask, lengths in make_index_batches(
                    va_seqs, hp["batch_size"], rng, shuffle=False):
                loss, w = batch_loss(xi, yi, mask, lengths)
                tot += float(loss) * w
                n += w
        return tot / max(n, 1)

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad = 0
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for xi, yi, mask, lengths in make_index_batches(
                tr_seqs, hp["batch_size"], rng, shuffle=True):
            opt.zero_grad()
            loss, w = batch_loss(xi, yi, mask, lengths)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * w
            n += w
        vl = val_loss()
        emit({"kind": "epoch", "epoch": epoch, "total_epochs": hp["epochs"],
              "loss": tot / max(n, 1), "val_loss": vl})
        if vl < best_val - 1e-6:
            best_val, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if hp["patience"] and bad >= hp["patience"]:
                emit({"kind": "log", "msg": f"early stop at epoch {epoch} "
                      f"(best val {best_val:.4f})"})
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def build_score_fn(model: EmbedNextItem):
    """Full-vocab cosine score_fn(prefix) over the LEARNED table — same contract
    as the rest of the family, so the blend/stacker harness and seq_extend can
    consume it unchanged."""
    with torch.no_grad():
        tbl = torch.nn.functional.normalize(model.table(), dim=1)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        idx = torch.from_numpy(np.asarray(prefix, dtype=np.int64)[None, :])
        with torch.no_grad():
            pred = torch.nn.functional.normalize(model.predict_next(idx), dim=1)
            return (pred @ tbl.t()).squeeze(0).numpy()
    return score_fn


def load_model(model_dir: Path, art, hp: dict) -> EmbedNextItem:
    model = EmbedNextItem(art.n_items, art.item_latents, hp)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(model_dir)
    model = load_model(model_dir, art, hp)
    predict_ranking(art, build_score_fn(model), input_dir, output, k=hp["k"])
    emit({"kind": "done"})


def _capture(model: EmbedNextItem):
    """SAE tap: the recurrent state, plus the LEARNED embedding of the current
    item — so an information-capture read can ask whether the trained table
    encodes next-item concepts the frozen latent did not."""
    names = ["embedding", "recurrent"]

    def step_acts_fn(seq_idx: np.ndarray):
        idx = torch.from_numpy(np.asarray(seq_idx, dtype=np.int64)[None, :])
        with torch.no_grad():
            emb = model.table()[idx][0]
            rec = model._run(model.table()[idx], None)[0]
        return [emb[:-1].numpy(), rec[:-1].numpy()]

    return names, step_acts_fn


def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    art = load_artifact(dataset)
    model = fit(art, hp, seed=hp["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics, predictions = eval_from_scores(
        art, build_score_fn(model), k=hp["k"],
        mmr_lambda=hp.get("mmr_lambda"), mmr_pool=int(hp.get("mmr_pool", 200)))
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
    tr.add_argument("--hyperparams", required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True, type=Path)
    pr.add_argument("--input", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    fx = sub.add_parser("fixture")
    fx.add_argument("--output", required=True, type=Path)
    fx.add_argument("--n-items", type=int, default=20)
    fx.add_argument("--n-sessions", type=int, default=40)
    ms = sub.add_parser("model-sae")
    ms.add_argument("--model", required=True, type=Path)
    ms.add_argument("--dataset", required=True, type=Path)
    ms.add_argument("--output", required=True, type=Path)
    ms.add_argument("--layers", default="")
    ms.add_argument("--n-atoms", type=int, default=0)
    ms.add_argument("--l1", type=float, default=0.0015)
    ms.add_argument("--epochs", type=int, default=40)
    ms.add_argument("--topk", type=int, default=0)
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
        import seq_model_sae
        hp = load_hp(str(args.model / "hyperparams.json"))
        art = load_artifact(args.dataset)
        model = load_model(args.model, art, hp)
        dim = hp["embed_dim"] or art.latent_dim
        seq_model_sae.run(
            args.model, args.dataset, args.output,
            capture=lambda _art: _capture(model),
            meta={"predictor": "seq-embed", "hidden": [dim, hp["hidden"]]},
            layers=[int(x) for x in args.layers.split(",") if x.strip()],
            n_atoms=args.n_atoms, l1=args.l1, epochs=args.epochs,
            cache_dir=args.cache_dir, label_atoms=args.label_atoms,
            dropped=True, topk=args.topk)
    else:
        make_fixture(args.output, n_items=args.n_items, n_sessions=args.n_sessions)


if __name__ == "__main__":
    main()
