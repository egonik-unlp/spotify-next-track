#!/usr/bin/env python3
"""Isolate what a pre-MLP's RECTIFIER costs, from its affine map.

This is the control that made the 2026-07-26 pre-encoder finding conclusive, and
the SAE engine has no equivalent tap — hence it lives here rather than in a
scratchpad. It measures next-item genre decodability at ONE step from four taps
of the same `seq-nexttrack` model with `pre_hidden > 0`:

  (a) raw_latent  — the raw current-item latent (signed)
  (b) pre_linear  — the model's OWN affine map with every ReLU REMOVED
  (c) pre_relu    — the actual rectified pre-MLP activation
  (d) recurrent   — the GRU state (has history; context only, not a clean control)

(a) vs (b) isolates the affine map; (b) vs (c) isolates rectification ALONE. That
second contrast is the whole point: analytically a linear pre-encoder is
expressivity-neutral (`W_i·(Vx) = (W_iV)x`, unconstrained when
pre_hidden >= latent_dim), so the ReLU is the only expressivity change — and this
script is what turns that argument into a measurement.

MEASURED (2026-07-26, S1 = `MLP256→GRU256` on seq-20260715-131139, 11,184 probe
test rows, 25 genre classes):

    raw_latent   genre acc 0.4033 / AUC 0.7609
    pre_linear   genre acc 0.4084 / AUC 0.7667   (affine costs nothing, ~0.8 sigma)
    pre_relu     genre acc 0.3700 / AUC 0.7508   (the ReLU is the loss, ~5.9 sigma)

with `frac_exact_zero` = 0.5327 on (c) matching `frac_negative_coords` = 0.5327
on (b) to four digits — the rectifier deletes exactly the negative half, leaving
~120 live units to carry 192 signed PCA directions. Prescription: never widen a
pre-MLP below ~2x latent_dim, and the only variant worth retrying is
sign-preserving (a linear map, or `concat[relu(x), relu(-x)]`).

Also note the operational trap: the row-building phase thrashes badly if several
model-sae style processes run at once (3 concurrent did not finish row building in
12 min; alone it takes under a minute). Run one at a time.

    tools/rectifier_control.py <model_or_run_dir> <out.json> [dataset_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "predictors"))

import seq_model_sae as S  # noqa: E402
import seq_nexttrack as N  # noqa: E402
from seq_common import load_artifact  # noqa: E402

TAPS = ["raw_latent", "pre_linear", "pre_relu", "recurrent"]


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    model_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    dataset = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        ROOT / "data/datasets/seq-20260715-131139")

    hp = N.load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(dataset)
    model = N.load_model(model_dir, art, hp)
    if model.pre is None:
        sys.exit("model has no pre-MLP (pre_hidden = 0) — nothing to control for")

    def taps(seq: np.ndarray):
        x = torch.from_numpy(seq[None, :, :].astype(np.float32))
        with torch.no_grad():
            raw = x[0, :-1].numpy()
            pre = model._pre_encode(x)[0, :-1].numpy()
            # The counterfactual: compose ONLY the Linear layers of the model's
            # own pre-MLP, skipping every ReLU/Dropout. Same weights, no rectifier.
            lin = x
            for m in model.pre:
                if isinstance(m, torch.nn.Linear):
                    lin = m(lin)
            lin = lin[0, :-1].numpy()
            rec = model._encode_seq(x, None)[0, :-1].numpy()
        return [raw, lin, pre, rec]

    rows = S.build_rows(art, taps, TAPS)
    next_items = np.asarray(rows.next_items, dtype=np.int64)
    is_train = np.asarray(rows.is_train, dtype=bool)
    n = next_items.shape[0]
    print(f"rows {n} ({int(is_train.sum())} train / {int((~is_train).sum())} test)",
          flush=True)

    field = S.SEGMENT_FIELD
    row_class = S._relevance_ids(art, field)[next_items]
    rng = np.random.default_rng(0)
    sample = (np.sort(rng.choice(n, S.MAX_CONCEPT_ROWS, replace=False))
              if n > S.MAX_CONCEPT_ROWS else np.arange(n))
    top = S._top_classes(row_class, sample, sample.size)
    print(f"segment field {field}: {len(top)} top classes", flush=True)

    res = {}
    for name in TAPS:
        act = np.concatenate(rows.layers[name], axis=0).astype(np.float32)
        tr = act[is_train]                      # standardize on train, as run() does
        mu, sd = tr.mean(0), tr.std(0)
        sd[sd < 1e-8] = 1e-8
        d = S._decodability((act - mu) / sd, is_train, row_class, top, 0)
        d["dim"] = int(act.shape[1])
        # The pair that identifies the mechanism: on `pre_linear` this is the
        # fraction of coordinates the ReLU is about to delete, and on `pre_relu`
        # it should equal that fraction exactly.
        d["frac_negative_coords"] = round(float((act < 0).mean()), 4)
        d["frac_exact_zero"] = round(float((act == 0).mean()), 4)
        res[name] = d
        print(name, json.dumps(d), flush=True)

    out.write_text(json.dumps({
        "tool": "rectifier_control",
        "model_dir": str(model_dir),
        "dataset_id": art.manifest["dataset_id"],
        "n_rows": n, "segment_field": field, "n_classes": len(top),
        "taps": res,
    }, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
