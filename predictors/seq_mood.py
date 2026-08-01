#!/usr/bin/env python3
"""MOOD-COHERENT session generator (the `seq-mood` predictor).

Not a next-track predictor — it does not try to guess the true next track (recall
is incidental). It generates a session that HOLDS THE SEED'S VIBE: rank every
candidate by cosine to the session prefix's MOOD CENTROID in the intrinsic
content-metric space (`spotify_tracks_content_metric`, the same space music@10 /
mood_coh@10 use), then subtract a penalty from tracks sharing the SEED's artist
so it doesn't just replay the current artist. Objective = high mood_coh@10 (stays
in the neighbourhood) + low artist_adj@10 (not eager) + reasonable ild@10.

Deterministic (no training / no weights) — like the baselines, it rebuilds its
scorer from the baked sequence artifact at serve time, and needs the
content-metric Qdrant collection reachable (QDRANT_URL, default localhost:6337).
Run under the shared predictor venv (numpy only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seq_common import (  # noqa: E402
    emit, load_artifact, eval_from_scores, predict_ranking, write_outputs,
    _load_music_vectors, _relevance_ids,
)

DEFAULTS = {
    "artist_penalty": 0.5,   # subtracted from tracks sharing the SEED's artist (anti-eager)
    "k": 10,                 # top-k reported / served
    "mmr_lambda": None,      # optional light spread (1.0/None = off); mood ranking rarely needs it
    "mmr_pool": 100,
    "artist_cap": None,   # HARD cap on top-k slots per artist (None/0 = off)
}
NEG = -1e18


def load_hp(spec: str) -> dict:
    """Path to a JSON file or an inline JSON string, merged over DEFAULTS."""
    p = Path(spec)
    raw = json.loads(p.read_text()) if p.exists() else json.loads(spec)
    hp = dict(DEFAULTS)
    hp.update(raw or {})
    return hp


def mood_scorer(art, artist_penalty: float):
    """score_fn(prefix) → per-item cosine to the prefix mood centroid, minus the
    same-seed-artist penalty. Items without a content-metric vector stay at -inf."""
    M, mask = _load_music_vectors(art)
    if M is None:
        raise SystemExit("seq-mood: content-metric vectors unavailable "
                         "(is the spotify_tracks_content_metric Qdrant collection up? set QDRANT_URL)")
    artist_ids = _relevance_ids(art, "artist")
    n = art.n_items

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        pm = prefix[mask[prefix]]
        s = np.full(n, NEG, dtype=np.float64)
        if pm.size == 0:
            return s
        c = M[pm].mean(axis=0)
        cn = float(np.linalg.norm(c))
        if cn == 0:
            return s
        c = c / cn
        sims = M @ c
        s[mask] = sims[mask]
        seed = int(prefix[-1])
        a = int(artist_ids[seed])
        if a >= 0 and artist_penalty:
            same = artist_ids == a
            s[same] = np.where(s[same] > NEG / 2, s[same] - float(artist_penalty), s[same])
        return s
    return score_fn


def train(dataset: Path, output: Path, hp: dict) -> None:
    art = load_artifact(dataset)
    emit({"kind": "log", "msg":
          f"seq-mood on {art.manifest['dataset_id']}: {art.n_items} items, "
          f"{int(art.test_sessions.size)} test sessions; artist_penalty {hp['artist_penalty']}"})
    score_fn = mood_scorer(art, hp["artist_penalty"])
    metrics, predictions = eval_from_scores(
        art, score_fn, k=hp["k"],
        mmr_lambda=hp.get("mmr_lambda"), mmr_pool=int(hp.get("mmr_pool", 100)),
        artist_cap=hp.get("artist_cap"))
    metrics["artist_penalty"] = hp["artist_penalty"]
    output.mkdir(parents=True, exist_ok=True)
    write_outputs(output, metrics, predictions)
    (output / "hyperparams.json").write_text(json.dumps(hp))
    emit({"kind": "log", "msg":
          f"seq-mood  mood_coh@{hp['k']} {metrics.get('mood_coh_at_k', float('nan')):.3f}  "
          f"artist_adj {metrics.get('artist_adj_at_k', float('nan')):.3f}  "
          f"ild {metrics.get('ild_at_k', float('nan')):.3f}  recall {metrics['recall_at_k']:.3f}"})
    emit({"kind": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Serving: rebuild the mood scorer from the promoted model's baked sequence
    artifact and rank for the caller's prefix (see seq_common.predict_ranking)."""
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(model_dir)
    score_fn = mood_scorer(art, hp["artist_penalty"])
    predict_ranking(art, score_fn, input_dir, output, k=hp["k"],
                    mmr_lambda=hp.get("mmr_lambda"),
                    mmr_pool=int(hp.get("mmr_pool", 200)),
                    artist_cap=hp.get("artist_cap"))
    emit({"kind": "done"})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument("--output", required=True, type=Path)
    tr.add_argument("--hyperparams", default="{}")
    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True, type=Path)
    pr.add_argument("--input", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.dataset, a.output, load_hp(a.hyperparams))
    else:
        predict(a.model, a.input, a.output)


if __name__ == "__main__":
    main()
