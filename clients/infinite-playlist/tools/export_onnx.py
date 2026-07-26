#!/usr/bin/env python3
"""Export the anti-eager next-track GRU to ONNX for in-browser (onnxruntime-web)
autoregressive generation. Input: prefix latents (1, T, D) — the RAW PCA-192
item latents the model was trained on. Output: the L2-normalized predicted
next-latent (1, D), so retrieval in the app is a plain dot product = cosine.

Run: predictors/.venv/bin/python clients/infinite-playlist/tools/export_onnx.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
INSTANCE = HERE.parents[2]  # tools -> infinite-playlist -> clients -> spotify-next-track
sys.path.insert(0, str(INSTANCE / "predictors"))
from seq_nexttrack import SeqNextLatent  # noqa: E402

RUN = INSTANCE / "data/runs/run-20260723-002044-af11a-seq-nexttrack"  # β=0.1 anti-eager GRU
OUT = HERE.parent / "public" / "model" / "gru.onnx"
D = 192


class Wrap(torch.nn.Module):
    """predict_next → L2-normalized next-latent, no lengths (feed the real prefix)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):  # x: (1, T, D)
        return torch.nn.functional.normalize(self.m.predict_next(x), dim=1)


def main() -> None:
    hp = json.loads((RUN / "hyperparams.json").read_text())
    m = SeqNextLatent(D, hp["hidden"], hp.get("arch", "gru"), hp.get("num_layers", 1),
                      hp.get("dropout", 0.0), bidirectional=hp.get("bidirectional", False),
                      residual=hp.get("residual", False))
    m.load_state_dict(torch.load(RUN / "model.pt", map_location="cpu"))
    m.eval()
    w = Wrap(m).eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 5, D)
    torch.onnx.export(
        w, dummy, str(OUT),
        input_names=["prefix"], output_names=["next"],
        dynamic_axes={"prefix": {1: "T"}}, opset_version=17,
        dynamo=False,  # legacy TorchScript exporter: honors dynamic_axes for the seq dim
    )
    print(f"exported {OUT}  (arch={hp.get('arch')} hidden={hp['hidden']} beta={hp.get('eager_beta')})")

    # verify: onnxruntime output matches torch
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 9, D).astype(np.float32)
    onnx_out = sess.run(None, {"prefix": x})[0]
    with torch.no_grad():
        torch_out = w(torch.from_numpy(x)).numpy()
    diff = float(np.abs(onnx_out - torch_out).max())
    print(f"onnx vs torch max|Δ| = {diff:.2e}  (norm={float(np.linalg.norm(onnx_out)):.4f})")
    print("OK" if diff < 1e-4 else "WARN: large diff")


if __name__ == "__main__":
    main()
