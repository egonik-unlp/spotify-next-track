#!/usr/bin/env python3
"""Export TWO `seq-bank` next-track models to ONNX so they can be heard, side by side
with the shipped engines, in the browser.

WHY THESE TWO, AND WHY THIS IS NOT A LEADERBOARD CLAIM. Neither model beat anything.
They are here to put the WALK METRICS THEMSELVES on trial by ear — `vibe`, `stride_err`
and `drift` are proxies, and no one has ever checked them against listening. The pair is
chosen to span the metrics' claims, so the ear can agree or disagree:

  * bank_v1  `latent+latent+cummean` h256      — the batch's hardest walk REFUTATION.
             Measured at a0.4/s0.3 vs the h256 control: Δvibe -0.0174 CI<0 and
             Δgenres -1.7250 CI<0. If `vibe` measures anything audible, this journey
             should wander off the seed's mood AND narrow in genre.
  * bank_e1  `latent+cummean` h256, dropout 0.1 — the regularized twin of the DEPLOYED
             engine (which, at `fusion_layers=0`, applies no dropout at all — a defect
             proven end to end: an explicit dropout=0.0 twin reproduces the deployed
             engine on 1,431/1,431 ranked lists). It PASSES the deployed-cell candidacy
             test (Δvibe +0.0024 straddles at a0.8/s0.5, `stride_err` better at the same
             time, so not the tiny-steps artifact) but costs mood-holding LATE
             (Δdrift -0.0226 CI<0 at a0.4/s0.3). It is NOT a recall improvement: the
             seed grid excluded that (+4/-4/-13 hits at seeds 1337/7/42).

So the interesting outcome is DISAGREEMENT. If `bank_v1` sounds fine, or `bank_e1`'s
late drift is inaudible, the walk surface is not tracking the experience — and three
campaigns' verdicts rest on that surface.

RETRIEVAL GEOMETRY IS UNCHANGED. Both models train on the same raw PCA-192 item space
as the shipped engines (`seq-20260715-131139`, no learned projection), so `latents.i16`
and `catalog.json` are untouched and no new large asset ships — only two small graphs.
This script ASSERTS that byte-for-byte rather than trusting it: a silent item-space
mismatch would rank the journey in the wrong vocabulary and return plausible garbage
instead of an error.

Interface is identical to `gru.onnx` / `dualgru.onnx`: prefix (1,T,192) -> next (1,192),
L2-normalized, so the browser's retrieval stays a plain dot product.

  predictors/.venv/bin/python clients/infinite-playlist/tools/export_bank_onnx.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
INSTANCE = HERE.parents[2]
sys.path.insert(0, str(INSTANCE / "predictors"))
import seq_bank                                                   # noqa: E402

D = 192
DATASET = INSTANCE / "data/datasets/seq-20260715-131139"
OUTDIR = HERE.parent / "public" / "model"

# key -> (model dir, onnx filename, anchor, stride, label, verdict)
# The (anchor, stride) cell is the DEPLOYED engine's a0.8/s0.5 for BOTH, deliberately:
# a listening comparison is only interpretable at a matched operating point. Note that
# bank_v1's CI<0 vibe refutation was measured at a0.4/s0.3 — the app's anchor/stride
# controls reach that cell if you want to hear the refuted configuration itself.
MODELS = {
    "bank_v1": (
        "bank-v1-llc256", "bankv1.onnx", 0.8, 0.5,
        "Bank ×3 + centroid — refuted at a0.4/s0.3, this cell unmeasured",
        "walk-REFUTED at a0.4/s0.3: vibe -0.0174 CI<0, genres -1.7250 CI<0; "
        "one-shot 162/1431, the batch's lowest. ⚠ BUT it is shipped here at a0.8/s0.5, "
        "a cell the campaign NEVER measured for this arm (only the two TRADE arms got "
        "the secondary cell) — and on a single-seed headless check at a0.8/s0.5 it "
        "scored the HIGHEST vibe of all four engines. So the refutation may be "
        "cell-specific. Unresolved; needs walk_headtohead at a0.8/s0.5 to settle.",
    ),
    "bank_e1": (
        "dropout-e1-latent-cummean-d01", "banke1.onnx", 0.8, 0.5,
        "Dual-tower, regularized — vibe held here, drifts late",
        "measured AT this cell: passes deployed-cell candidacy (vibe +0.0024 straddles "
        "at a0.8/s0.5 with stride_err better at the same time, so not the tiny-steps "
        "artifact); drift -0.0226 CI<0 at a0.4/s0.3. Recall gain EXCLUDED by the "
        "3-seed grid (+4/-4/-13 hits). Not a champion claim.",
    ),
}


class Wrap(torch.nn.Module):
    """predict_next -> L2-normalized next latent. No `lengths`: the browser always feeds
    the real, unpadded prefix, so predict_next takes h[:, -1, :]."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):                       # (1, T, D)
        return torch.nn.functional.normalize(self.m.predict_next(x), dim=1)


def export_one(key: str, spec: tuple, ds_latents: bytes) -> dict:
    model_dir, fname, anchor, stride, label, verdict = spec
    mdir = INSTANCE / "data/models" / model_dir
    if not mdir.is_dir():
        raise SystemExit(f"{key}: no promoted model dir {mdir}")

    # ---- item-space gate: the browser's latents.i16 was baked from the DATASET, so a
    # model trained in any other space cannot be scored against it. Same assertion
    # walk_headtohead.promoted_score makes before it will score a walk.
    mine = (mdir / "item_latents.f32").read_bytes()
    if mine != ds_latents:
        raise SystemExit(
            f"{key}: item_latents.f32 is NOT byte-identical to {DATASET.name} — the "
            "model lives in a different item space; refusing to export it against the "
            "browser's baked latents")

    hp = seq_bank.load_hp(str(mdir / "hyperparams.json"))
    art = seq_bank.load_artifact(mdir)
    model = seq_bank.load_model(mdir, art, hp)
    model.eval()
    w = Wrap(model).eval()

    out = OUTDIR / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        w, torch.randn(1, 5, D), str(out),
        input_names=["prefix"], output_names=["next"],
        dynamic_axes={"prefix": {1: "T"}}, opset_version=17,
        dynamo=False,   # legacy TorchScript exporter: honors dynamic_axes for the seq dim
    )
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n{key}: exported {out.name}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"  views {'+'.join(hp['views'])}  hidden {hp['hidden']}  "
          f"dropout {hp['dropout']}  {n_par:,} params")

    # ---- parity: onnxruntime vs torch across prefix lengths ----------------------
    # The `cummean` towers are a causal cumulative mean (torch.cumsum + arange), so
    # agreeing at T=1 AND at T=40 is the thing worth checking: a mis-traced cumsum or a
    # baked-in sequence length shows up ONLY as a length-dependent divergence, and would
    # otherwise sail through a single-length smoke test.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    worst = 0.0
    for T in (1, 2, 5, 13, 40, 97):
        x = np.random.default_rng(T).standard_normal((1, T, D)).astype(np.float32)
        o = sess.run(None, {"prefix": x})[0]
        with torch.no_grad():
            t = w(torch.from_numpy(x)).numpy()
        diff = float(np.abs(o - t).max())
        worst = max(worst, diff)
        print(f"    T={T:3d}  max|Δ| {diff:.2e}  |out| {float(np.linalg.norm(o)):.4f}")
    status = "OK" if worst < 1e-4 else f"WARN divergence {worst:.2e}"
    print(f"    parity {status}")
    if worst >= 1e-4:
        raise SystemExit(f"{key}: PARITY FAILED ({worst:.2e}) — not registering it")

    return {
        "file": out.name,
        "model": f"{model_dir} — seq-bank {'+'.join(hp['views'])} h{hp['hidden']} "
                 f"dropout {hp['dropout']}",
        "label": label,
        "verdict": verdict,
        "anchor": anchor, "stride": stride,
        "bytes": out.stat().st_size,
    }


def main() -> None:
    ds_latents = (DATASET / "item_latents.f32").read_bytes()
    print(f"item space: {DATASET.name}  ({len(ds_latents) / 1e6:.1f} MB reference)")

    blocks = {k: export_one(k, spec, ds_latents) for k, spec in MODELS.items()}

    man_path = OUTDIR / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man.update(blocks)
    man_path.write_text(json.dumps(man, indent=2))
    print(f"\nregistered {', '.join(blocks)} in {man_path}")
    print("NOTE: neither model is a champion claim — see each block's `verdict`.")


if __name__ == "__main__":
    main()
