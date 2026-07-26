#!/usr/bin/env python3
"""Bake the GRU retrieval assets: the RAW PCA-192 item latents the anti-eager
GRU was trained on (= its input space AND the retrieval space) + track metadata,
aligned to item index. int16-quantized (~7.4 MB). Pairs with gru.onnx.

  public/model/latents.i16   int16, row-major n*dim (dequant: v * scale)
  public/model/catalog.json  [{uri,name,artist,genre}, ...] (item-index order)
  public/model/manifest.json {n, dim, scale, latent, model}

Run: predictors/.venv/bin/python clients/infinite-playlist/tools/bake_gru.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
INSTANCE = HERE.parents[2]
DS = INSTANCE / "data" / "datasets" / "seq-20260715-131139"  # the GRU's training dataset
OUT = HERE.parent / "public" / "model"


def main() -> None:
    man = json.loads((DS / "sequence-manifest.json").read_text())
    n, dim = man["n_items"], man["latent_dim"]
    lat = np.fromfile(DS / "item_latents.f32", dtype="<f4").reshape(n, dim)
    items = json.loads((DS / "items.json").read_text())
    cat = [{"uri": items[str(i)].get("uri"), "name": items[str(i)].get("name"),
            "artist": items[str(i)].get("artist"), "genre": items[str(i)].get("genre")}
           for i in range(n)]

    mx = float(np.abs(lat).max())
    scale = mx / 32767.0
    q = np.clip(np.round(lat / scale), -32767, 32767).astype("<i2")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latents.i16").write_bytes(q.tobytes())
    (OUT / "catalog.json").write_text(json.dumps(cat, separators=(",", ":")))
    (OUT / "manifest.json").write_text(json.dumps({
        "n": n, "dim": dim, "scale": scale,
        "latent": "PCA-192 (seq-20260715-131139)",
        "model": "gru.onnx — seq-nexttrack GRU, eager_beta 0.1 (anti-eager)",
    }, indent=2))
    mb = (OUT / "latents.i16").stat().st_size / 1e6
    print(f"wrote {OUT}/ : {n} items dim {dim}, latents.i16 {mb:.2f} MB, scale {scale:.3e}")


if __name__ == "__main__":
    main()
