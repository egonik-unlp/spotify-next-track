#!/usr/bin/env python3
"""Rescore stored `holisticness_at_k` under the CURRENT crown definition.

WHY THIS EXISTS. The crown is a composite whose definition has been revised
twice (3-factor → grounded 4-factor on 2026-07-25 → max-eagerness with a
mood-only floor on 2026-07-26). Stored values therefore split by the code
generation that wrote them, and because the grounding factor multiplies by ~0.3,
every OLD-generation run mechanically outranks every newer one regardless of
quality. Measured 2026-07-26: of 105 runs, 12 held 3-factor values (0.063-0.233),
10 held 4-factor/floor-0.185 values (0.044-0.047) and NONE held the live
definition (~0.024) — which put a capacity control and a refuted dual-tower arm
above the 0.212-recall champion in the best-models group, and would have locked
every good future run out of the group once its 12 slots filled.

This is safe to do because the crown was designed so that every input is stored
PER ROW in predictions.json, making any generation exactly recomputable. Keep it
that way: never add a crown factor that is not also emitted per row.

WHAT IT TOUCHES. `data/runs/<id>/meta.json` only — the RunMeta the Postgres
backfill upserts ("file wins", crates/lensing-db/src/backfill.rs). It does NOT
rewrite `metrics.json`, which is the predictor's own output and stays as the
historical record of what that run computed at the time.

AFTER RUNNING: `zig build migrate-data` to push the files into Postgres, then
`POST /api/best-models/recompute` so the group re-ranks on one generation.

    tools/rescore_holisticness.py [--apply]      (default is a dry run)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "predictors"))
from seq_common import MUSIC_FLOOR  # noqa: E402  the single source of truth

FACETS = ("music_sim", "mood_coh", "ild", "artist_adj")


def artist_map(dataset_id: str, cache: dict) -> dict | None:
    """item index -> artist, from the dataset's items.json (for artist_conc)."""
    if dataset_id in cache:
        return cache[dataset_id]
    out = None
    for base in (ROOT / "data/datasets", ROOT / "data/seq"):
        p = base / dataset_id / "items.json"
        if p.is_file():
            items = json.loads(p.read_text())
            out = {int(k): (v.get("artist") or f"?{k}") for k, v in items.items()}
            break
    cache[dataset_id] = out
    return out


def api_predictions(run_id: str, api: str) -> list | None:
    """Fetch predictions from the server for runs whose dir is not hydrated.

    Runs trained by a REMOTE worker upload their artifacts straight to Postgres,
    so `data/runs/<id>/predictions.json` may not exist locally even though the
    run is complete and served. Those are exactly the runs that kept legacy
    values after a file-only pass (measured 2026-07-26: 9 of them, one at
    holisticness 0.197 — high enough to top the board), so the fallback is what
    makes the rescore complete rather than partial."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"{api}/api/runs/{run_id}/predictions", timeout=90) as r:
            d = json.load(r)
        return d if isinstance(d, list) else d.get("predictions")
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def rescore(run_dir: Path, cache: dict, api: str) -> tuple[str, dict | None]:
    meta_p = run_dir / "meta.json"
    preds_p = run_dir / "predictions.json"
    if not meta_p.is_file():
        return "no meta.json", None
    meta = json.loads(meta_p.read_text())
    metrics = meta.get("metrics") or {}
    if preds_p.is_file():
        rows = json.loads(preds_p.read_text())
    else:
        rows = api_predictions(meta.get("run_id") or run_dir.name, api)
        if rows is None:
            return "no predictions (local or API)", None
    if not isinstance(rows, list) or not rows:
        return "empty predictions", None

    vals = {k: [r[k] for r in rows if r.get(k) is not None] for k in FACETS}
    if not all(vals[k] for k in FACETS):
        return "missing facets (pre-holisticness run)", None
    means = {k: float(np.mean(v)) for k, v in vals.items()}

    # artist_conc: stored per row on runs from 2026-07-26 on; reconstruct it from
    # top_k_ids + items.json for older runs so one definition covers everything.
    conc = [r["artist_conc"] for r in rows if r.get("artist_conc") is not None]
    if not conc:
        amap = artist_map(meta.get("dataset_id") or "", cache)
        if amap is None:
            return "no items.json for artist_conc", None
        for r in rows:
            top = (r.get("top_k_ids") or [])[:10]
            if len(top) < 2:
                continue
            names = [amap.get(int(i), "?") for i in top]
            n = len(names)
            counts = np.array([names.count(x) for x in set(names)], dtype=float)
            hhi = float(np.sum((counts / n) ** 2))
            conc.append((hhi - 1.0 / n) / (1.0 - 1.0 / n))
    if not conc:
        return "could not derive artist_conc", None
    ac = float(np.mean(conc))

    rel = min(max((means["music_sim"] - MUSIC_FLOOR) / (1.0 - MUSIC_FLOOR), 0.0), 1.0)
    eager = max(means["artist_adj"], ac)
    h = (max(means["mood_coh"], 0.0) * means["ild"]
         * (1.0 - min(max(eager, 0.0), 1.0)) * rel)

    return "ok", {
        "meta": meta, "path": meta_p,
        "old": metrics.get("holisticness_at_k"),
        "new": h, "artist_conc": ac,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write meta.json (default: dry run)")
    ap.add_argument("--api", default="http://localhost:8096",
                    help="server used to fetch predictions for un-hydrated runs")
    args = ap.parse_args()

    cache: dict = {}
    changed, skipped = [], {}
    for run_dir in sorted((ROOT / "data/runs").iterdir()):
        if not run_dir.is_dir():
            continue
        why, res = rescore(run_dir, cache, args.api)
        if res is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        changed.append((run_dir.name, res))

    print(f"rescorable: {len(changed)}   (MUSIC_FLOOR={MUSIC_FLOOR})")
    for why, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {n:>3}  {why}")
    print()
    print(f"{'run':46} {'stored':>9} {'live':>9} {'a_conc':>7}")
    for name, r in sorted(changed, key=lambda kv: -(kv[1]["new"])):
        old = f"{r['old']:.5f}" if r["old"] is not None else "    —    "
        print(f"{name[:46]:46} {old:>9} {r['new']:9.5f} {r['artist_conc']:7.4f}")

    if not args.apply:
        print("\nDRY RUN — re-run with --apply to write meta.json")
        return
    for _, r in changed:
        m = r["meta"]
        m.setdefault("metrics", {})
        m["metrics"]["holisticness_at_k"] = r["new"]
        m["metrics"]["artist_conc_at_k"] = r["artist_conc"]
        r["path"].write_text(json.dumps(m, indent=2))
    print(f"\nAPPLIED to {len(changed)} meta.json files.")
    print("Next: `zig build migrate-data`, then POST /api/best-models/recompute")


if __name__ == "__main__":
    main()
