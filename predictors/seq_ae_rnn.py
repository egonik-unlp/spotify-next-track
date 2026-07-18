#!/usr/bin/env python3
"""AE -> RNN next-track predictor: sequence-autoencoder pretraining, then the
standard recurrent next-latent fine-tune.

Motivation. The plain GRU (seq_nexttrack) learns its session encoder FROM
SCRATCH against the next-latent objective alone. This model first PRETRAINS the
same-shaped GRU encoder as the encoder of a seq2seq AUTOENCODER — reconstruct
each TRAIN session's latent sequence from a bottleneck context vector (MSE) —
and only then transplants those weights into a `SeqNextLatent` and fine-tunes
with the identical InfoNCE next-latent objective. It is the sequence analog of
this project's rotation `burn-ae-classifier` (AE pretrain 177->128->64, then
transplant the encoder into a fine-tuned classifier).

Hypothesis: reconstruction pretraining is self-supervised over the FULL session
(no next-item label needed), so the encoder sees far more structure — including
cold items that the next-latent objective barely touches — and starts the
fine-tune from a session representation that already captures co-listening
geometry. A better R (recurrent) leg should also lift the R x Markov blend.

Leakage discipline. Pretraining and fine-tuning use TRAIN sessions ONLY; test
sessions never enter either stage. The AE reconstructs its own input (allowed to
see the whole train sequence — that is what an autoencoder does); the downstream
next-latent fine-tune keeps the leak-free unidirectional per-step objective from
seq_nexttrack. The eval is the shared leave-last-out retrieval, byte-identical.

Transplant contract. The AE encoder is `nn.GRU(input_size=latent_dim,
hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=...)` — the
SAME construction as `SeqNextLatent.rnn` in the unidirectional (non-manual-stack)
path — so `encoder.state_dict()` loads key-for-key into `model.rnn`. Only the
prediction head is fresh. (bidirectional / residual manual-stack topologies are
NOT supported here: the transplant targets `model.rnn`.)

CLI mirrors seq_nexttrack:
  seq_ae_rnn.py train --dataset <dir> --output <run_dir> --hyperparams <json|path>

Run under the shared predictor venv (torch 2.12.0+cpu)."""

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
from seq_nexttrack import DEFAULTS as GRU_DEFAULTS, SeqNextLatent, fit

# AE-specific hyperparameters layered on top of the GRU fine-tune defaults.
AE_DEFAULTS = {
    **GRU_DEFAULTS,
    "hidden": 256,             # match the champion GRU width
    "loss": "infonce",         # fine-tune objective (champion setting)
    "pretrain_epochs": 30,     # seq2seq AE reconstruction epochs
    "pretrain_lr": 1e-3,
    "pretrain_patience": 5,    # early-stop the pretrain on held-out train recon
    "freeze_encoder": False,   # if True, fine-tune only the head (probe mode)
}
SEED = 1337


def load_hp(spec: str) -> dict:
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    user = json.loads(raw)
    hp = {**AE_DEFAULTS, **user}
    hp["num_layers"] = int(hp["num_layers"])
    hp["hidden"] = int(hp["hidden"])
    assert hp["arch"] in ("gru", "lstm"), "arch must be gru|lstm"
    assert hp["loss"] in ("cosine", "infonce"), "loss must be cosine|infonce"
    assert not hp.get("bidirectional", False), (
        "AE->RNN transplants into model.rnn; bidirectional not supported")
    assert not (hp.get("residual", False) and hp["num_layers"] > 1), (
        "AE->RNN transplants into model.rnn; manual residual stack not supported")
    assert hp["num_layers"] >= 1
    return hp


# --------------------------------------------------------------------------- #
# Sequence autoencoder                                                         #
# --------------------------------------------------------------------------- #
class SeqAE(nn.Module):
    """Seq2seq autoencoder over per-item latents. The ENCODER is shape-identical
    to SeqNextLatent.rnn (so it transplants). The decoder + output head exist
    only for reconstruction pretraining and are discarded afterwards.

    Encode: GRU over the full latent sequence -> final hidden state per layer
    (the bottleneck context). Decode: an autoregressive GRU initialized from the
    encoder's final hidden, teacher-forced on the input latents shifted by one
    (a zero start vector), reconstructing the input latent at every step."""

    def __init__(self, latent_dim: int, hidden: int, arch: str,
                 num_layers: int, dropout: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.arch = arch
        self.num_layers = num_layers
        rnn_cls = nn.GRU if arch == "gru" else nn.LSTM
        inter = dropout if num_layers > 1 else 0.0
        self.encoder = rnn_cls(input_size=latent_dim, hidden_size=hidden,
                               num_layers=num_layers, batch_first=True,
                               dropout=inter)
        self.decoder = rnn_cls(input_size=latent_dim, hidden_size=hidden,
                               num_layers=num_layers, batch_first=True,
                               dropout=inter)
        self.out = nn.Linear(hidden, latent_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # Encode the full sequence -> final hidden state (bottleneck).
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        # Decoder input: zero start token ++ x[:, :-1] (teacher forcing on the
        # true latents), so step t reconstructs x[:, t] from context + x[:, :t].
        B, T, D = x.shape
        start = torch.zeros(B, 1, D, dtype=x.dtype, device=x.device)
        dec_in = torch.cat([start, x[:, :-1, :]], dim=1)
        dec_packed = nn.utils.rnn.pack_padded_sequence(
            dec_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        dec_out, _ = self.decoder(dec_packed, state)
        dec_out, _ = nn.utils.rnn.pad_packed_sequence(
            dec_out, batch_first=True, total_length=T)
        return self.out(self.drop(dec_out))          # (B, T, D) reconstruction


def _ae_batches(seqs, latents, batch_size, rng, shuffle):
    """Full-sequence reconstruction batches: input == target == latents[seq]."""
    idx = np.arange(len(seqs))
    if shuffle:
        rng.shuffle(idx)
    D = latents.shape[1]
    for start in range(0, len(idx), batch_size):
        batch = [seqs[i] for i in idx[start:start + batch_size]]
        lengths = np.array([len(s) for s in batch], dtype=np.int64)
        Tmax = int(lengths.max())
        B = len(batch)
        x = np.zeros((B, Tmax, D), dtype=np.float32)
        mask = np.zeros((B, Tmax), dtype=bool)
        for r, seq in enumerate(batch):
            x[r, :len(seq)] = latents[seq]
            mask[r, :len(seq)] = True
        yield (torch.from_numpy(x), torch.from_numpy(mask),
               torch.from_numpy(lengths))


def pretrain_encoder(art, hp: dict, seed: int = SEED) -> SeqAE:
    """Pretrain the seq2seq AE on TRAIN sessions (reconstruction MSE), early
    stopping on a held-out train slice. Returns the AE (encoder is ready to
    transplant)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    D = art.latent_dim
    latents = art.item_latents

    # Reconstruct FULL train sessions (length >= 2). Reuse the same val split
    # convention as the fine-tune for consistency.
    sessions = training_pairs(art, art.train_sessions)
    perm = rng.permutation(len(sessions))
    n_val = max(1, int(len(sessions) * hp["val_fraction"]))
    val_ids = set(perm[:n_val].tolist())
    tr = [s for i, s in enumerate(sessions) if i not in val_ids]
    va = [s for i, s in enumerate(sessions) if i in val_ids]

    ae = SeqAE(D, hp["hidden"], hp["arch"], hp["num_layers"], hp["dropout"])
    opt = torch.optim.Adam(ae.parameters(), lr=hp["pretrain_lr"])
    emit({"kind": "log", "msg":
          f"AE pretrain: {len(tr)} train / {len(va)} val sessions, "
          f"{hp['arch']} enc/dec hidden {hp['hidden']} x{hp['num_layers']}, "
          f"{hp['pretrain_epochs']} epochs (recon MSE)"})

    def recon_loss(x, mask, lengths):
        rec = ae(x, lengths)                             # (B, T, D)
        m = mask.unsqueeze(-1)
        se = ((rec - x) ** 2 * m).sum()
        return se / m.sum().clamp(min=1), int(mask.sum())

    def val_loss():
        ae.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, mask, lengths in _ae_batches(va, latents, hp["batch_size"], rng, False):
                l, w = recon_loss(x, mask, lengths)
                tot += float(l) * w
                n += w
        return tot / max(n, 1)

    best = float("inf")
    best_state = {k: v.clone() for k, v in ae.state_dict().items()}
    bad = 0
    for epoch in range(1, hp["pretrain_epochs"] + 1):
        ae.train()
        tot, n = 0.0, 0
        for x, mask, lengths in _ae_batches(tr, latents, hp["batch_size"], rng, True):
            opt.zero_grad()
            l, w = recon_loss(x, mask, lengths)
            l.backward()
            opt.step()
            tot += float(l.detach()) * w
            n += w
        vl = val_loss()
        emit({"kind": "epoch", "epoch": epoch, "total_epochs": hp["pretrain_epochs"],
              "loss": tot / max(n, 1), "val_loss": vl})
        if vl < best - 1e-7:
            best = vl
            best_state = {k: v.clone() for k, v in ae.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if hp["pretrain_patience"] and bad >= hp["pretrain_patience"]:
                emit({"kind": "log", "msg":
                      f"AE early stop at epoch {epoch} (best recon {best:.5f})"})
                break
    ae.load_state_dict(best_state)
    ae.eval()
    emit({"kind": "log", "msg": f"AE pretrain done (best recon MSE {best:.5f})"})
    return ae


def build_pretrained_model(art, hp: dict, seed: int = SEED):
    """Pretrain the AE and transplant its encoder into a fresh SeqNextLatent."""
    ae = pretrain_encoder(art, hp, seed=seed)
    model = SeqNextLatent(art.latent_dim, hp["hidden"], hp["arch"],
                          hp["num_layers"], hp["dropout"],
                          bidirectional=False, residual=False)
    # Key-for-key transplant: SeqAE.encoder <-> SeqNextLatent.rnn (identical
    # nn.GRU construction). The head stays freshly initialized.
    missing, unexpected = model.rnn.load_state_dict(ae.encoder.state_dict(),
                                                    strict=True)
    if hp.get("freeze_encoder", False):
        for p in model.rnn.parameters():
            p.requires_grad = False
        emit({"kind": "log", "msg": "encoder FROZEN — fine-tuning head only"})
    emit({"kind": "log", "msg": "transplanted AE encoder -> SeqNextLatent.rnn"})
    return model


def fit_ae_rnn(art, hp: dict, seed: int = SEED):
    """Full AE->RNN: pretrain + transplant, then the standard next-latent
    fine-tune (seq_nexttrack.fit, model= the transplanted net)."""
    model = build_pretrained_model(art, hp, seed=seed)
    return fit(art, hp, seed=seed, model=model)


def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    art = load_artifact(dataset)
    latents = art.item_latents

    model = fit_ae_rnn(art, hp, seed=SEED)
    run_dir.mkdir(parents=True, exist_ok=True)

    item_norm = torch.nn.functional.normalize(torch.from_numpy(latents), dim=1)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(latents[prefix][None, :, :])
        with torch.no_grad():
            pred = torch.nn.functional.normalize(model.predict_next(x), dim=1)
            return (pred @ item_norm.t()).squeeze(0).numpy()

    metrics, predictions = eval_from_scores(art, score_fn, k=hp["k"])
    metrics["model"] = "ae-rnn"
    write_outputs(run_dir, metrics, predictions)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "hyperparams.json").write_text(json.dumps(hp))

    emit({"kind": "log", "msg":
          f"AE->RNN  Recall@{hp['k']} {metrics['recall_at_k']:.3f}  "
          f"Recall@20 {metrics['recall_at_20']:.3f}  MRR {metrics['mrr']:.3f}  "
          f"artist@10 {metrics['artist_recall_at_k']:.3f}  "
          f"genre@10 {metrics['genre_recall_at_k']:.3f}  "
          f"(n_test {metrics['n_test']})"})
    emit({"kind": "done"})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument("--output", required=True, type=Path)
    tr.add_argument("--hyperparams", required=True)
    fx = sub.add_parser("fixture", help="write a synthetic SEQUENCE artifact")
    fx.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    if args.cmd == "train":
        train(args.dataset, args.output, args.hyperparams)
    else:
        make_fixture(args.output)


if __name__ == "__main__":
    main()
