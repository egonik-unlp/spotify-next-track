#!/usr/bin/env python3
"""Export the DUAL-TOWER next-track model to ONNX for in-browser generation.

Why this model. On the walk-surface head-to-head (40 identical held-out sessions,
paired bootstrap — `tools/walk_headtohead.py`), the dual-tower arms beat the shipped
single GRU on the objective this project actually states (sequences that hold a mood
while exploring cohesively), with NO CI-confirmed loss on any axis:

    dual latent/cummean f0, anchor 0.8 stride 0.5, vs the shipped GRU:
      stride_err  -0.255  CI<0      (journeys move like the user really does)
      genres      +1.50   CI>0      (more cohesive genre exploration)
      drift       +0.035  CI>0      (holds the seed's neighbourhood better)
      vibe        +0.001  straddles (held, not traded away)

The single GRU can be pushed to a realistic stride too, but only by giving up vibe
(-0.041, CI<0). The champion blend trades away far more (-0.133, CI<0).

Same retrieval geometry as the current engine — RAW PCA-192 item latents, no learned
projection — so `latents.i16` and `catalog.json` are unchanged and no new large asset
ships. Only the graph changes.

Interface is identical to gru.onnx: prefix (1,T,192) -> next (1,192), L2-normalized,
so the browser's retrieval stays a plain dot product.

  predictors/.venv/bin/python clients/infinite-playlist/tools/export_dualgru_onnx.py
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
import seq_dualgru                                              # noqa: E402

# latent/cummean, fusion_layers=0 — the head-to-head's best cohesive-exploration arm.
MODEL = INSTANCE / "data/models/best-seq-dualgru-20260725-143502-3e5d4"
OUT = HERE.parent / "public" / "model" / "dualgru.onnx"
D = 192


class Wrap(torch.nn.Module):
    """predict_next -> L2-normalized next latent. No `lengths`: the browser always
    feeds the real, unpadded prefix, so predict_next takes h[:, -1, :]."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):                       # (1, T, D)
        return torch.nn.functional.normalize(self.m.predict_next(x), dim=1)


def main() -> None:
    hp = seq_dualgru.load_hp(str(MODEL / "hyperparams.json"))
    art = seq_dualgru.load_artifact(MODEL)
    model = seq_dualgru.load_model(MODEL, art, hp)
    model.eval()
    w = Wrap(model).eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 5, D)
    torch.onnx.export(
        w, dummy, str(OUT),
        input_names=["prefix"], output_names=["next"],
        dynamic_axes={"prefix": {1: "T"}}, opset_version=17,
        dynamo=False,   # legacy TorchScript exporter: honors dynamic_axes for the seq dim
    )
    mb = OUT.stat().st_size / 1e6
    print(f"exported {OUT}  ({mb:.2f} MB)")
    print(f"  views {hp.get('view_a')}/{hp.get('view_b')}  hidden {hp.get('hidden_a')}/"
          f"{hp.get('hidden_b')}  fusion_layers {hp.get('fusion_layers')}")

    # ---- parity: onnxruntime vs torch, across several prefix lengths -------------
    # `cummean` is a causal cumulative mean (torch.cumsum), so getting the same answer
    # at T=1 and at T=40 is the thing worth checking — a mis-traced cumsum would only
    # show up as a length-dependent divergence.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    worst = 0.0
    for T in (1, 2, 5, 13, 40):
        x = np.random.default_rng(T).standard_normal((1, T, D)).astype(np.float32)
        o = sess.run(None, {"prefix": x})[0]
        with torch.no_grad():
            t = w(torch.from_numpy(x)).numpy()
        diff = float(np.abs(o - t).max())
        worst = max(worst, diff)
        print(f"  T={T:3d}  max|Δ| {diff:.2e}  |out| {float(np.linalg.norm(o)):.4f}")
    print("OK" if worst < 1e-4 else f"WARN: max divergence {worst:.2e}")

    # ---- register in the manifest the browser already reads ----------------------
    man_path = OUT.parent / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man["dualgru"] = {
        "file": OUT.name,
        "model": f"{MODEL.name} — dual-tower {hp.get('view_a')}/{hp.get('view_b')}, "
                 f"fusion_layers {hp.get('fusion_layers')}",
        "anchor": 0.8, "stride": 0.5,     # the head-to-head's recommended cell
        "bytes": OUT.stat().st_size,
    }
    man_path.write_text(json.dumps(man, indent=2))
    print(f"registered manifest.dualgru in {man_path}")


if __name__ == "__main__":
    main()
