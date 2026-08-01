#!/usr/bin/env python3
"""Bake `champion.bin` for the Infinite-Playlist Worker — the promoted blend
champion (`blend-gru-markov-content-proj`, R'+M+C') as a single binary.

This is a THIN WRAPPER around ../app's `tools/build_champion.py` rather than a
second implementation, deliberately: that builder is the one whose output is
pinned by `worker-core/tests/champion.rs` against the Python reference, and this
client vendors the same `champion.rs` that reads it. Two builders would mean two
things to keep in parity.

The bundle is read WORKER-side (the `[assets]` binding), so unlike the GRU engine's
latents.i16 + gru.onnx it is never downloaded by the browser. ~22.5 MiB, and
`clients/*/public/model/` is gitignored — regenerate rather than commit.

`markov_gate` and the per-artist cap are RUNTIME policy in champion.rs / walk.rs,
not baked, so this bundle serves both the gated and ungated configurations.

Run: predictors/.venv/bin/python clients/infinite-playlist/tools/bake_champion.py
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTANCE = HERE.parents[2]          # tools -> infinite-playlist -> clients -> instance
APP = INSTANCE.parent / "app"       # the sibling client that owns the builder
BUILDER = APP / "tools" / "build_champion.py"
OUT = HERE.parent / "public" / "model" / "champion.bin"


def main() -> None:
    if not BUILDER.is_file():
        sys.exit(
            f"builder not found: {BUILDER}\n"
            "This wrapper reuses ../app/tools/build_champion.py so the bundle stays "
            "byte-identical to the one worker-core/tests/champion.rs pins. If ../app "
            "is gone, vendor that script here before removing this wrapper."
        )
    sys.path.insert(0, str(INSTANCE / "predictors"))
    spec = importlib.util.spec_from_file_location("build_champion", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    mod.OUT = OUT                      # redirect the builder's output here
    mod.main()

    mb = OUT.stat().st_size / (1024 * 1024)
    print(f"\nbaked {OUT}  ({mb:.2f} MiB)")
    if mb > 25:
        print("!! over Cloudflare's 25 MiB per-file asset limit — will fail to deploy")


if __name__ == "__main__":
    main()
