#!/usr/bin/env python3
"""Non-learned baselines for the next-track SEQUENCE task.

All three are evaluated on the exact same leave-last-out retrieval protocol as
the GRU (seq_common.eval_from_scores): rank the whole vocab by a per-item
score, exclude items already in the prefix (next-distinct), report Recall@k,
Recall@20, MRR and hit-rate@10 into a lensing-shaped metrics.json.

  B0 popularity  — context-free; score every item by its TRAIN-set play
                   frequency (falls back to items.json play_count if a session
                   count is unavailable).
  B1 recency     — score by recency inside the prefix; with the current track
                   excluded (next-distinct), this effectively predicts the
                   2nd-most-recent distinct item, then 3rd, ... Non-prefix
                   items get popularity as a weak tail so the ranking is total.
  B2 markov      — first-order Markov with artist/genre back-off, the affinity
                   formula from app/worker-core/src/transit.rs (backoff_k=8,
                   artist=0.6, genre=0.4). Transitions are rebuilt from TRAIN
                   sessions only (leak-free) exactly as pipeline/corpus/
                   transitions.py mines them (consecutive within-session pairs,
                   self-loops dropped); with no reason_end in the artifact each
                   pair carries unit weight. THE bar to beat.

CLI:
  seq_baselines.py --dataset <dir> --output <run_dir> --mode popularity|recent|markov
                   [--hyperparams <json|path>]

Run under the shared predictor venv."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from seq_common import (
    SeqArtifact,
    emit,
    eval_from_scores,
    load_artifact,
    predict_ranking,
    write_outputs,
)

# Affinity constants, copied from app/worker-core/src/transit.rs so B2 matches
# the pathfinder's Markov scorer exactly.
BACKOFF_K = 8.0
ARTIST_WEIGHT = 0.6
GENRE_WEIGHT = 0.4


# --------------------------------------------------------------------------- #
# Train-set statistics                                                        #
# --------------------------------------------------------------------------- #
def train_play_counts(art: SeqArtifact) -> np.ndarray:
    """Per-item occurrence count across TRAIN sessions (context-free
    popularity). Items never seen in train fall back to items.json play_count
    so every item has a defined, positive-ish score."""
    counts = np.zeros(art.n_items, dtype=np.float64)
    for s in art.train_sessions:
        seq = art.session(int(s))
        for i in seq:
            counts[int(i)] += 1.0
    if counts.sum() == 0:  # degenerate; use metadata play_count
        for k, v in art.items.items():
            counts[int(k)] = float(v.get("play_count", 0))
    return counts


def build_train_transitions(art: SeqArtifact):
    """Rebuild the first-order transition tables from TRAIN sessions only,
    mirroring pipeline/corpus/transitions.py: for each consecutive
    within-session pair (u, v) with u != v, accumulate a unit-weighted bigram
    at track, artist and genre level (the artifact has no reason_end, so the
    satisfaction weight collapses to 1.0). Artist/genre conditionals are
    normalized to probabilities over observed targets."""
    def artist(i: int) -> str:
        return art.items.get(str(int(i)), {}).get("artist", "?")

    def genre(i: int) -> str:
        return art.items.get(str(int(i)), {}).get("genre", "unknown")

    track_bigram: dict[int, Counter] = defaultdict(Counter)
    artist_bigram: dict[str, Counter] = defaultdict(Counter)
    genre_bigram: dict[str, Counter] = defaultdict(Counter)
    n_pairs = 0
    for s in art.train_sessions:
        seq = art.session(int(s))
        for a, b in zip(seq[:-1], seq[1:]):
            a, b = int(a), int(b)
            if a == b:
                continue  # self-loops carry no routing signal
            track_bigram[a][b] += 1.0
            artist_bigram[artist(a)][artist(b)] += 1.0
            genre_bigram[genre(a)][genre(b)] += 1.0
            n_pairs += 1

    def conditional(bigram: dict[str, Counter]) -> dict[str, dict[str, float]]:
        cond = {}
        for src, tgts in bigram.items():
            tot = sum(tgts.values())
            if tot > 0:
                cond[src] = {d: w / tot for d, w in tgts.items()}
        return cond

    return {
        "track_bigram": {u: dict(c) for u, c in track_bigram.items()},
        "artist_cond": conditional(artist_bigram),
        "genre_cond": conditional(genre_bigram),
        "n_pairs": n_pairs,
        "artist": artist,
        "genre": genre,
    }


# --------------------------------------------------------------------------- #
# Score functions                                                             #
# --------------------------------------------------------------------------- #
def popularity_scorer(art: SeqArtifact):
    counts = train_play_counts(art)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        return counts  # context-free: identical for every prefix
    return score_fn


def recency_scorer(art: SeqArtifact):
    """Recency inside the prefix; ties broken by (a small amount of)
    popularity so non-prefix items still get a total order. Prefix items are
    later excluded by the shared eval, so this predicts the most-recent
    *distinct* item that isn't the current track."""
    counts = train_play_counts(art)
    pop = counts / (counts.max() + 1e-9)  # in [0,1], weak tail

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        scores = pop.copy()  # baseline tail so the ranking is total
        # More-recent prefix positions score higher; last position highest.
        # (The current/last item is dropped by the next-distinct exclusion.)
        for rank, item in enumerate(prefix):
            recency = float(rank + 1)  # earlier -> smaller, latest -> largest
            scores[int(item)] = 1.0 + recency  # dominate the popularity tail
        return scores
    return score_fn


def markov_scorer(art: SeqArtifact):
    tr = build_train_transitions(art)
    track_bigram = tr["track_bigram"]
    artist_cond = tr["artist_cond"]
    genre_cond = tr["genre_cond"]
    artist_of = tr["artist"]
    genre_of = tr["genre"]
    counts = train_play_counts(art)
    pop = counts / (counts.max() + 1e-9)  # tie-break tail (Markov-with-pop-backoff)
    n_items = art.n_items

    # Integer-code each item's artist/genre so the artist/genre back-off is a
    # vectorized gather rather than a per-item dict lookup (n_items ~ 20k).
    artist_ids: dict[str, int] = {}
    genre_ids: dict[str, int] = {}
    item_artist_code = np.empty(n_items, dtype=np.int64)
    item_genre_code = np.empty(n_items, dtype=np.int64)
    for i in range(n_items):
        a = artist_of(i)
        g = genre_of(i)
        item_artist_code[i] = artist_ids.setdefault(a, len(artist_ids))
        item_genre_code[i] = genre_ids.setdefault(g, len(genre_ids))

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        u = int(prefix[-1])  # current track = last item of the prefix
        scores = np.zeros(n_items, dtype=np.float64)

        # Track-level P(v | u) and its support n_u.
        tb = track_bigram.get(u, {})
        n_u = float(sum(tb.values()))
        if n_u > 0:
            vs = np.fromiter(tb.keys(), dtype=np.int64, count=len(tb))
            scores[vs] = np.fromiter(tb.values(), dtype=np.float64, count=len(tb)) / n_u
        trust = n_u / (n_u + BACKOFF_K)

        # Artist / genre back-off, gathered onto every item by its code.
        a_row = artist_cond.get(artist_of(u), {})
        g_row = genre_cond.get(genre_of(u), {})
        a_by_code = np.zeros(len(artist_ids), dtype=np.float64)
        for name, p in a_row.items():
            if name in artist_ids:
                a_by_code[artist_ids[name]] = p
        g_by_code = np.zeros(len(genre_ids), dtype=np.float64)
        for name, p in g_row.items():
            if name in genre_ids:
                g_by_code[genre_ids[name]] = p
        backoff = (ARTIST_WEIGHT * a_by_code[item_artist_code]
                   + GENRE_WEIGHT * g_by_code[item_genre_code])

        scores = trust * scores + (1.0 - trust) * backoff
        # Popularity tie-break so items with no Markov signal are still ordered
        # sensibly (tiny epsilon keeps it strictly below any real affinity).
        return scores + 1e-6 * pop
    return score_fn


SCORERS = {
    "popularity": popularity_scorer,
    "recent": recency_scorer,
    "markov": markov_scorer,
}


def run(dataset: Path, run_dir: Path, mode: str, k: int) -> None:
    art = load_artifact(dataset)
    emit({"kind": "log", "msg":
          f"baseline {mode} on {art.manifest['dataset_id']}: "
          f"{art.n_items} items, test sessions {int(art.test_sessions.size)}"})
    score_fn = SCORERS[mode](art)
    metrics, predictions = eval_from_scores(art, score_fn, k=k)
    metrics["baseline"] = mode
    write_outputs(run_dir, metrics, predictions)
    emit({"kind": "log", "msg":
          f"[{mode}] Recall@{k} {metrics['recall_at_k']:.3f}  "
          f"Recall@20 {metrics['recall_at_20']:.3f}  "
          f"MRR {metrics['mrr']:.3f}  hit@10 {metrics['hit_rate']:.3f}  "
          f"(n_test {metrics['n_test']})"})
    emit({"kind": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path, mode: str) -> None:
    """Serving-time ranking: rebuild the train-only baseline scorer for `mode`
    from the baked sequence artifact and rank the vocab for the caller's session
    prefix (see seq_common.predict_ranking). No checkpoint — the statistics are
    reconstructed from the snapshotted train sessions."""
    art = load_artifact(model_dir)
    predict_ranking(art, SCORERS[mode](art), input_dir, output)
    emit({"kind": "done"})


def main() -> None:
    ap = argparse.ArgumentParser()
    # Back-compat: the historical flat form (no subcommand) IS the run form, so
    # accept `--dataset/--output/--mode` at the top level and default cmd=run.
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--mode", choices=list(SCORERS))
    ap.add_argument("--hyperparams", default="{}")  # file path or inline JSON
    sub = ap.add_subparsers(dest="cmd")
    pr = sub.add_parser("predict", help="rank the next track for a session prefix")
    pr.add_argument("--model", required=True, type=Path)
    pr.add_argument("--input", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    pr.add_argument("--mode", required=True, choices=list(SCORERS))
    args = ap.parse_args()

    if args.cmd == "predict":
        predict(args.model, args.input, args.output, args.mode)
        return
    assert args.dataset and args.output and args.mode, \
        "run mode needs --dataset, --output, --mode"
    p = Path(args.hyperparams)
    hp = json.loads(p.read_text() if p.exists() else args.hyperparams)
    k = int(hp.get("k", 10))
    run(args.dataset, args.output, args.mode, k)


if __name__ == "__main__":
    main()
