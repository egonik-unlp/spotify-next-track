#!/usr/bin/env python3
"""Shared plumbing for the next-track SEQUENCE predictors (GRU + baselines).

This module owns everything that is model-agnostic:

  * reading a SEQUENCE artifact (the contract documented in the Phase-2.5
    plan): sequence-manifest.json + sessions.u32 / offsets.u32 /
    item_latents.f32 / train_sessions.u32 / test_sessions.u32 / items.json,
  * the leave-last-out retrieval evaluation (rank the whole vocab by cosine to
    a predicted next-latent, exclude items already in the prefix — the
    "next-distinct" / Markov-parity rule) and the lensing-shaped metrics,
  * writing metrics.json / predictions.json,
  * a tiny synthetic fixture generator that emits an artifact matching the
    contract byte-for-byte, so the models can be developed/tested with no
    dependency on the (possibly still-running) real producer.

Nothing here imports torch — the baselines and the fixture generator run on
numpy alone; only seq_nexttrack.py pulls in torch for the GRU.

Run under the shared predictor venv (torch 2.12.0+cpu / numpy / sklearn)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def emit(obj: dict) -> None:
    """Stream a progress event as one line of JSON on stdout (best-effort)."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Artifact contract                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class SeqArtifact:
    """An in-memory view of a SEQUENCE dataset directory."""

    manifest: dict
    sessions: np.ndarray          # (n_items_total,) uint32, concatenated
    offsets: np.ndarray           # (n_sessions + 1,) uint32 prefix offsets
    item_latents: np.ndarray      # (n_items, latent_dim) float32
    train_sessions: np.ndarray    # (n_train,) uint32 session indices
    test_sessions: np.ndarray     # (n_test,) uint32 session indices
    items: dict                   # {"<item_index>": {uri,name,artist,genre,play_count}}

    @property
    def n_sessions(self) -> int:
        return int(self.offsets.shape[0] - 1)

    @property
    def n_items(self) -> int:
        return int(self.item_latents.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.item_latents.shape[1])

    def session(self, s: int) -> np.ndarray:
        """The ordered item-index sequence for session `s` (last item = label)."""
        return self.sessions[self.offsets[s]:self.offsets[s + 1]]


def load_artifact(dataset: Path) -> SeqArtifact:
    """Read a SEQUENCE artifact directory. All binaries are little-endian."""
    manifest = json.loads((dataset / "sequence-manifest.json").read_text())
    assert manifest.get("kind") == "sequence", "not a sequence artifact"
    latent_dim = int(manifest["latent_dim"])

    sessions = np.fromfile(dataset / "sessions.u32", dtype="<u4")
    offsets = np.fromfile(dataset / "offsets.u32", dtype="<u4")
    latents = np.fromfile(dataset / "item_latents.f32", dtype="<f4")
    train_sessions = np.fromfile(dataset / "train_sessions.u32", dtype="<u4")
    test_sessions = np.fromfile(dataset / "test_sessions.u32", dtype="<u4")
    items = json.loads((dataset / "items.json").read_text())

    n_items = int(manifest["n_items"])
    assert latents.size == n_items * latent_dim, (
        f"item_latents.f32 has {latents.size} floats, manifest says "
        f"{n_items}x{latent_dim}")
    item_latents = latents.reshape(n_items, latent_dim)

    n_sessions = int(manifest["n_sessions"])
    assert offsets.size == n_sessions + 1, (
        f"offsets.u32 has {offsets.size} entries, expected n_sessions+1="
        f"{n_sessions + 1}")
    assert int(offsets[-1]) == sessions.size, (
        "offsets tail must equal len(sessions)")

    return SeqArtifact(
        manifest=manifest,
        sessions=sessions,
        offsets=offsets,
        item_latents=item_latents,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
        items=items,
    )


def training_pairs(art: SeqArtifact, session_ids: np.ndarray) -> list[np.ndarray]:
    """Return, for each session, its ordered item-index array (length >= 2).

    Every session yields all next-step pairs by teacher forcing downstream:
    for [a,b,c,d] the supervised steps are b|[a], c|[a,b], d|[a,b,c]."""
    seqs = []
    for s in session_ids:
        seq = art.session(int(s))
        if seq.shape[0] >= 2:
            seqs.append(seq.astype(np.int64))
    return seqs


# --------------------------------------------------------------------------- #
# Leave-last-out retrieval evaluation                                         #
# --------------------------------------------------------------------------- #
def rank_metrics(ranks: list[int | None], k: int = 10) -> dict:
    """Aggregate per-session truth ranks (1-based; None = truth unrankable,
    e.g. excluded because it was already in the prefix) into lensing-shaped
    retrieval metrics.

    With exactly one relevant item per session, Recall@K == hit-rate@K, so
    `recall_at_k` and `hit_rate` coincide here (both = "truth in top-K")."""
    n = len(ranks)
    if n == 0:
        return {"n_test": 0, "recall_at_k": 0.0, "recall_at_10": 0.0,
                "recall_at_20": 0.0, "mrr": 0.0, "hit_rate": 0.0}

    def recall_at(kk: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= kk) / n

    mrr = sum((1.0 / r) for r in ranks if r is not None) / n
    return {
        "n_test": n,
        "recall_at_k": recall_at(k),      # Recall@k (k configurable, default 10)
        "recall_at_10": recall_at(10),
        "recall_at_20": recall_at(20),
        "mrr": mrr,                       # MRR over the full ranking
        "hit_rate": recall_at(10),        # hit-rate@10
        "k": k,
    }


def _relevance_ids(art: SeqArtifact, field: str) -> np.ndarray:
    """Map each item index → an integer class id for a graded-relevance field
    (e.g. "artist" or "genre"), reading the label off `art.items` (which is the
    dataset's items.json verbatim). Missing / unknown labels get a unique
    negative id so they never spuriously match another item's label.

    Returns an int64 array of length n_items (item index → label id)."""
    n_items = art.n_items
    ids = np.full(n_items, -1, dtype=np.int64)
    vocab: dict[str, int] = {}
    for key, meta in art.items.items():
        i = int(key)
        if not (0 <= i < n_items):
            continue
        val = meta.get(field)
        if val is None:
            continue
        cid = vocab.get(val)
        if cid is None:
            cid = len(vocab)
            vocab[val] = cid
        ids[i] = cid
    return ids


def eval_from_scores(
    art: SeqArtifact,
    score_fn,
    k: int = 10,
) -> tuple[dict, list[dict]]:
    """Run the leave-last-out retrieval eval over TEST sessions.

    `score_fn(prefix: np.ndarray) -> np.ndarray` returns a length-`n_items`
    score vector (higher = more likely next) for a given prefix (all but the
    last item). This function applies the shared ranking rules identically for
    every model:
      * candidates = all vocab items EXCEPT those already in the prefix
        (next-distinct / Markov-parity),
      * truth = the session's last item,
      * rank = 1-based position of truth in the descending score order
        (None if truth was excluded because it also appears in the prefix).

    In addition to the EXACT next-track metrics (recall_at_k / mrr / hit_rate
    on the item id), it also computes GRADED-RELEVANCE metrics that credit
    landing the right band/vibe even when the exact track is wrong:
      * artist_recall_at_k = fraction of test sessions where SOME top-k
        candidate shares the truth's artist,
      * genre_recall_at_k  = likewise on genre,
      * artist_mrr = mean reciprocal rank of the FIRST same-artist candidate
        over the full descending ranking.
    The relevance fields are hardcoded to "artist"/"genre" (the item metadata
    keys in items.json); domain.toml documents them under
    [sequence].relevance_fields — this module reads the artifact only, not the
    domain config, so the two must stay in sync.

    Returns (metrics, predictions) where predictions matches the lensing
    predictions.json shape."""
    n_items = art.n_items
    artist_ids = _relevance_ids(art, "artist")
    genre_ids = _relevance_ids(art, "genre")

    ranks: list[int | None] = []
    artist_hits: list[float] = []
    genre_hits: list[float] = []
    artist_rrs: list[float] = []
    predictions: list[dict] = []

    for s in art.test_sessions:
        s = int(s)
        seq = art.session(s)
        if seq.shape[0] < 2:
            continue
        prefix = seq[:-1].astype(np.int64)
        truth = int(seq[-1])

        scores = np.asarray(score_fn(prefix), dtype=np.float64)
        assert scores.shape[0] == n_items, "score_fn must cover the full vocab"

        # Exclude items already in the prefix (next-distinct rule). Pushing
        # them to -inf keeps them out of the ranking entirely.
        scores = scores.copy()
        scores[prefix] = -np.inf

        # Descending order over the surviving candidates.
        order = np.argsort(-scores, kind="stable")
        top_k = order[:10].tolist()

        # 1-based rank of the truth (None if it was excluded / -inf).
        if scores[truth] == -np.inf:
            rank: int | None = None
        else:
            rank = int(np.where(order == truth)[0][0]) + 1
        ranks.append(rank)

        # ---- graded relevance over the SAME ranking (top-k / full order) ----
        topk_idx = order[:k]
        a_true = int(artist_ids[truth])
        g_true = int(genre_ids[truth])
        artist_hits.append(
            1.0 if a_true >= 0 and bool((artist_ids[topk_idx] == a_true).any())
            else 0.0)
        genre_hits.append(
            1.0 if g_true >= 0 and bool((genre_ids[topk_idx] == g_true).any())
            else 0.0)
        if a_true >= 0:
            same_artist = artist_ids[order] == a_true  # descending order
            if same_artist.any():
                first = int(np.argmax(same_artist))    # first True position
                artist_rrs.append(1.0 / (first + 1))
            else:
                artist_rrs.append(0.0)
        else:
            artist_rrs.append(0.0)

        predictions.append({
            "row_id": s,                       # session index (u64)
            "actual": float(truth),            # true item index (f64)
            "predicted": float(top_k[0]),      # top-1 item index (f64)
            "top_k_ids": [int(i) for i in top_k],  # top-10 item indices (u64)
        })

    metrics = rank_metrics(ranks, k=k)
    n = len(ranks)
    metrics["artist_recall_at_k"] = (sum(artist_hits) / n) if n else 0.0
    metrics["genre_recall_at_k"] = (sum(genre_hits) / n) if n else 0.0
    metrics["artist_mrr"] = (sum(artist_rrs) / n) if n else 0.0
    return metrics, predictions


def write_outputs(run_dir: Path, metrics: dict, predictions: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "predictions.json").write_text(json.dumps(predictions))


# --------------------------------------------------------------------------- #
# Synthetic fixture generator                                                 #
# --------------------------------------------------------------------------- #
def make_fixture(
    output: Path,
    n_items: int = 20,
    n_sessions: int = 40,
    latent_dim: int = 64,
    min_len: int = 2,
    max_len: int = 6,
    seed: int = 7,
) -> None:
    """Write a tiny synthetic SEQUENCE artifact that matches the contract
    byte-for-byte, for offline development/testing.

    Latents are random unit-ish vectors; to make the task learnable (so the
    GRU and Markov baselines can actually score above chance) sessions are
    generated as short random walks over a fixed successor map, and each
    item's latent is nudged toward its most common successor's latent."""
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=True)

    # Base random latents.
    latents = rng.standard_normal((n_items, latent_dim)).astype(np.float32)

    # A deterministic "preferred successor" per item induces routing signal.
    succ = np.array([(i * 7 + 3) % n_items for i in range(n_items)], dtype=np.int64)
    # Bleed a little of the successor's latent into each item so that a model
    # that predicts "next latent" from the current one can retrieve it.
    latents = latents + 0.6 * latents[succ]
    latents = latents.astype(np.float32)

    sessions: list[int] = []
    offsets: list[int] = [0]
    for _ in range(n_sessions):
        length = int(rng.integers(min_len, max_len + 1))
        cur = int(rng.integers(0, n_items))
        seq = [cur]
        for _ in range(length - 1):
            # Follow the preferred successor most of the time, else jump.
            if rng.random() < 0.75:
                cur = int(succ[cur])
            else:
                cur = int(rng.integers(0, n_items))
            seq.append(cur)
        sessions.extend(seq)
        offsets.append(len(sessions))

    n_sess = len(offsets) - 1
    # Chronological-style split: first 75% train, last 25% test.
    cut = int(n_sess * 0.75)
    train_sessions = np.arange(0, cut, dtype="<u4")
    test_sessions = np.arange(cut, n_sess, dtype="<u4")

    manifest = {
        "dataset_id": "seq-fixture",
        "created_at": "2026-07-12T00:00:00+00:00",
        "kind": "sequence",
        "n_sessions": n_sess,
        "n_items": n_items,
        "latent_dim": latent_dim,
        "latent_source": "synthetic-fixture",
        "split": {
            "strategy": "index_first_75",
            "cut": str(cut),
            "n_train_sessions": int(train_sessions.size),
            "n_test_sessions": int(test_sessions.size),
        },
    }
    items = {
        str(i): {
            "uri": f"spotify:track:fixture{i:05d}",
            "name": f"item {i}",
            "artist": f"artist {i % 5}",
            "genre": f"genre {i % 3}",
            "play_count": int(1 + (i % 4)),
        }
        for i in range(n_items)
    }

    np.asarray(sessions, dtype="<u4").tofile(output / "sessions.u32")
    np.asarray(offsets, dtype="<u4").tofile(output / "offsets.u32")
    latents.astype("<f4").tofile(output / "item_latents.f32")
    train_sessions.tofile(output / "train_sessions.u32")
    test_sessions.tofile(output / "test_sessions.u32")
    (output / "items.json").write_text(json.dumps(items))
    (output / "sequence-manifest.json").write_text(json.dumps(manifest, indent=2))

    emit({"kind": "fixture", "output": str(output), "n_sessions": n_sess,
          "n_items": n_items, "n_train": int(train_sessions.size),
          "n_test": int(test_sessions.size)})
