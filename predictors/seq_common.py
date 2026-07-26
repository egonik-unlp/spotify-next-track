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


# How many of the most-recent session-prefix items to record per prediction as
# the "query context" (what the model predicted FROM). Bounds predictions.json;
# the full length rides alongside as prefix_len.
PREFIX_KEEP = 12

# music@k floor used to rescale the GROUNDING factor of the holisticness crown
# (see the composite at the end of eval_from_scores). Credit only musical
# relevance ABOVE THE MOOD-ONLY BASELINE, i.e. above what a ranker gets by
# serving mood-matched but otherwise unrelated music.
#
# Raised 0.185 -> 0.34 on 2026-07-26. 0.185 was the context-free `seq-popularity`
# score, which turned out far too permissive: mood coherence is ALREADY paid for
# by the mood_coh and ild factors, so a floor below the mood-only level lets the
# composite double-count mood and rewards models that abandon relevance for it.
# Measured consequence at 0.185 — the degenerate `cummean/cummean` arm led the
# whole crown at recall@10 0.066 (95 correct of 1,431). 0.34 sits just above the
# measured music@10 of the mood-only generator `mood-session` (0.3384 on the
# canonical split), which therefore scores exactly 0: a mood-only ranker earns
# nothing on a metric defined as relevance beyond mood-only. Re-derive this
# constant if the corpus or the musical-distance space changes.
MUSIC_FLOOR = 0.34


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


def _album_ids(art: SeqArtifact) -> np.ndarray:
    """Map each item index → an integer album-class id, keyed by the
    (artist, album) tuple so identically-named albums ("Greatest Hits") by
    different artists never merge into one album.

    Items lacking an `album` field — every dataset built before the album
    emitter landed in pipeline/corpus/sequences.py — get id -1, so the
    album_adj@k metric is omitted gracefully rather than fabricated.

    Returns an int64 array of length n_items (item index → album id)."""
    n_items = art.n_items
    ids = np.full(n_items, -1, dtype=np.int64)
    vocab: dict[str, int] = {}
    for key, meta in art.items.items():
        i = int(key)
        if not (0 <= i < n_items):
            continue
        album = meta.get("album")
        if album is None:
            continue
        val = f"{meta.get('artist')}\x1f{album}"
        cid = vocab.get(val)
        if cid is None:
            cid = len(vocab)
            vocab[val] = cid
        ids[i] = cid
    return ids


def _load_music_vectors(art: "SeqArtifact"):
    """Load the MUSICAL-DISTANCE vectors for the vocab from the Qdrant collection
    built by pipeline/build_content_metric.py, aligned to item index.

    Returns (M[n_items, D] float32, L2-normalized; mask[n_items] bool) or
    (None, None) if the index is unavailable — the music@k metric is then simply
    omitted, leaving every other metric untouched. Fully optional / best-effort:
    the eval never fails because the sonic index is missing or Qdrant is down.

    Config via env (defaults suit the local instance):
      LENSING_MUSIC_METRIC_COLLECTION (default spotify_tracks_content_metric)
      LENSING_MUSIC_METRIC_DIAL       (default balanced)
      QDRANT_URL / PATHFINDER_QDRANT_URL (default http://localhost:6337)
    """
    import os
    if os.environ.get("LENSING_MUSIC_METRIC_DISABLE"):
        return None, None
    collection = os.environ.get("LENSING_MUSIC_METRIC_COLLECTION", "spotify_tracks_content_metric")
    dial = os.environ.get("LENSING_MUSIC_METRIC_DIAL", "balanced")
    url = (os.environ.get("QDRANT_URL")
           or os.environ.get("PATHFINDER_QDRANT_URL", "http://localhost:6337"))
    try:
        import hashlib
        from qdrant_client import QdrantClient

        n = art.n_items
        uris: list[str | None] = [None] * n
        for key, meta in art.items.items():
            i = int(key)
            if 0 <= i < n:
                uris[i] = meta.get("uri")

        def to_id(u: str) -> int:  # matches pipeline/corpus/ids.py
            return int.from_bytes(hashlib.sha256(u.encode()).digest()[:8], "little")

        ids = [to_id(u) if u else None for u in uris]
        idmap = {pid: i for i, pid in enumerate(ids) if pid is not None}
        client = QdrantClient(url=url, timeout=60)
        M = None
        mask = np.zeros(n, dtype=bool)
        want = [pid for pid in ids if pid is not None]
        for b in range(0, len(want), 512):
            recs = client.retrieve(collection, ids=want[b:b + 512],
                                   with_vectors=[dial], with_payload=False)
            for r in recs:
                v = r.vector[dial] if isinstance(r.vector, dict) else r.vector
                if M is None:
                    M = np.zeros((n, len(v)), dtype=np.float32)
                j = idmap.get(r.id)
                if j is not None:
                    M[j] = v
                    mask[j] = True
        if M is None:
            return None, None
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emit({"event": "log",
              "msg": f"music@k: loaded {int(mask.sum())}/{n} {dial!r} sonic vectors "
                     f"from {collection!r}"})
        return M / norms, mask
    except Exception as e:  # noqa: BLE001 — best-effort; never break the eval
        emit({"event": "log",
              "msg": f"music@k: musical-distance index unavailable ({e}); metric skipped"})
        return None, None


def mmr_rerank_order(order: np.ndarray, scores: np.ndarray,
                     music_vecs, music_mask, k: int, lam: float,
                     pool: int = 200) -> np.ndarray:
    """MMR (maximal-marginal-relevance) re-rank of the top-`pool` candidates in
    the content-metric ("sonic/mood") space, trading relevance for diversity.

    Greedily selects items maximizing `λ·rel(i) − (1−λ)·max_{j∈selected}
    cos(M[i], M[j])`, where rel is the min-max-normalized base score over the
    candidate pool. Returns a FULL permutation (reranked head + remaining pool in
    score order + untouched tail) so downstream truth-rank lookup still works.

    λ=1.0 recovers the pure-score ranking exactly (identity); lower λ diversifies
    (anti-eager). No-op when music vectors are unavailable. Phase-2 lever."""
    if music_vecs is None or lam is None or lam >= 1.0:
        return order
    cand = [int(i) for i in order[:pool] if np.isfinite(scores[i])]
    if len(cand) < 2:
        return order
    raw = np.array([scores[i] for i in cand], dtype=np.float64)
    lo, hi = float(raw.min()), float(raw.max())
    rel = (raw - lo) / (hi - lo) if hi > lo else np.ones_like(raw)
    relmap = {c: float(r) for c, r in zip(cand, rel)}
    selected: list[int] = []
    remaining = list(cand)
    budget = min(k, len(cand))
    while remaining and len(selected) < budget:
        sel_cov = [s for s in selected if music_mask[s]]
        best, best_val = remaining[0], -1e18
        for c in remaining:
            if sel_cov and music_mask[c]:
                div = float(np.max(music_vecs[c] @ music_vecs[sel_cov].T))
            else:
                div = 0.0
            val = lam * relmap[c] - (1.0 - lam) * div
            if val > best_val:
                best_val, best = val, c
        selected.append(best)
        remaining.remove(best)
    sel_set = set(selected)
    rest = [c for c in cand if c not in sel_set]
    tail = [int(i) for i in order[pool:]]
    return np.array(selected + rest + tail, dtype=order.dtype)


def eval_from_scores(
    art: SeqArtifact,
    score_fn,
    k: int = 10,
    mmr_lambda: float | None = None,
    mmr_pool: int = 200,
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
    album_ids = _album_ids(art)
    music_vecs, music_mask = _load_music_vectors(art)

    ranks: list[int | None] = []
    artist_hits: list[float] = []
    genre_hits: list[float] = []
    artist_rrs: list[float] = []
    music_sims: list[float] = []
    # Session-holisticness diagnostics (display-only; see aggregation below).
    artist_adjs: list[float] = []   # top-k share of the SEED's artist (lower better)
    artist_concs: list[float] = []  # WITHIN-list artist concentration (lower better)
    album_adjs: list[float] = []    # top-k share of the SEED's album  (lower better)
    mood_cohs: list[float] = []     # top-k cosine to the prefix mood centroid
    ilds: list[float] = []          # intra-list diversity of the top-k
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
        # Optional MMR diversity/mood re-rank (Phase 2; identity when off).
        if mmr_lambda is not None and mmr_lambda < 1.0:
            order = mmr_rerank_order(order, scores, music_vecs, music_mask,
                                     k, mmr_lambda, mmr_pool)
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

        # ---- musical distance: best sonic match in the top-k vs the truth ----
        # music@k = mean over sessions of max cosine (in the balanced content-
        # metric space) between the truth and the top-k candidates — a CONTINUOUS
        # graded-relevance metric: credits landing something that SOUNDS like the
        # truth even when the exact track / artist / genre is missed. A top-k that
        # contains the exact truth scores 1.0 (self-similarity), so music@k is a
        # smooth superset of recall@k. Higher is better.
        row_music = None
        if music_vecs is not None and music_mask[truth]:
            cand = topk_idx[music_mask[topk_idx]]
            if cand.size:
                row_music = float(np.max(music_vecs[cand] @ music_vecs[truth]))
                music_sims.append(row_music)

        # ---- session-holisticness diagnostics (display-only) ----
        # Anti-eagerness: what fraction of the top-k just parrots the SEED (the
        # "current" track = last prefix item) by artist / album. High values are
        # the "predicts the rest of the album" symptom, so these are LOWER-better.
        # These reference the SEED, unlike artist@k/genre@k which credit sharing
        # the TRUTH's artist/genre.
        seed = int(prefix[-1])
        a_seed = int(artist_ids[seed])
        row_artist_adj = None
        if a_seed >= 0:
            row_artist_adj = float(np.mean(artist_ids[topk_idx] == a_seed))
            artist_adjs.append(row_artist_adj)

        # WITHIN-LIST artist concentration — the blind spot artist_adj cannot see.
        # artist_adj only counts the SEED's artist, so a top-k made of ten tracks
        # by one OTHER artist scores a perfect 0.0. Measured 2026-07-26:
        # `mood-session` scores artist_adj 0.0036 (essentially never the seed's
        # artist) while its within-list concentration is 0.3291 — a 90x gap — and
        # the degenerate cummean/cummean arm reads 0.19 vs 0.29. The blind spot is
        # widest precisely on the pathological models the crown must reject.
        # Normalized Herfindahl over the top-k artists: 0 = every slot a different
        # artist, 1 = the whole list is one artist. LOWER is better.
        row_artist_conc = None
        if topk_idx.size > 1:
            a_top = artist_ids[topk_idx]
            _, counts = np.unique(a_top, return_counts=True)
            n_top = float(a_top.size)
            hhi = float(np.sum((counts / n_top) ** 2))
            row_artist_conc = (hhi - 1.0 / n_top) / (1.0 - 1.0 / n_top)
            artist_concs.append(row_artist_conc)
        al_seed = int(album_ids[seed])
        row_album_adj = None
        if al_seed >= 0:
            row_album_adj = float(np.mean(album_ids[topk_idx] == al_seed))
            album_adjs.append(row_album_adj)

        # Mood coherence + intra-list diversity in the content-metric space.
        # mood_coh = mean cosine of the covered top-k to the PREFIX mood centroid
        # (does the list stay in the established sonic neighborhood — references
        # the whole prefix context, not the single truth like music@k). ild =
        # mean pairwise sonic distance within the top-k; low ild = a duplicative,
        # album-eager list. Read mood_coh ALONGSIDE ild: mood_coh alone is maxed
        # by album-repeats, ild is the counter-signal.
        row_mood_coh = None
        row_ild = None
        if music_vecs is not None:
            pm = prefix[music_mask[prefix]]
            cand = topk_idx[music_mask[topk_idx]]
            if pm.size and cand.size:
                centroid = music_vecs[pm].mean(axis=0)
                cnorm = float(np.linalg.norm(centroid))
                if cnorm > 0:
                    centroid = centroid / cnorm
                    row_mood_coh = float(np.mean(music_vecs[cand] @ centroid))
                    mood_cohs.append(row_mood_coh)
            if cand.size >= 2:
                sims = music_vecs[cand] @ music_vecs[cand].T
                iu = np.triu_indices(cand.shape[0], k=1)
                row_ild = float(np.mean(1.0 - sims[iu]))
                ilds.append(row_ild)

        # The QUERY: the ordered session prefix the model predicted FROM (the
        # last element is the "current" track → predict-next). Capped to the most
        # recent PREFIX_KEEP items to bound predictions.json; prefix_len records
        # the true length so the UI can say "+N earlier". Lets the runs view show
        # WHAT was played, not just the truth + suggestions.
        prefix_tail = prefix[-PREFIX_KEEP:]
        pred = {
            "row_id": s,                       # session index (u64)
            "actual": float(truth),            # true item index (f64)
            "predicted": float(top_k[0]),      # top-1 item index (f64)
            "top_k_ids": [int(i) for i in top_k],  # top-10 item indices (u64)
            "prefix_ids": [int(i) for i in prefix_tail],  # query context, oldest→newest
            "prefix_len": int(prefix.shape[0]),           # full prefix length
        }
        # Per-row music@k: the best (max) musical-distance cosine between the truth
        # and the top-k candidates — this row's contribution to the run's music@k.
        # A hit scores 1.0 (truth is in top-k); a sonically-close MISS still scores
        # high (e.g. same-artist/vibe), so the UI can show a "miss" wasn't a miss
        # musically. Absent when the metric index is unavailable.
        if row_music is not None:
            pred["music_sim"] = round(row_music, 4)
        # Per-row holisticness (this row's contribution to the run aggregates).
        if row_artist_adj is not None:
            pred["artist_adj"] = round(row_artist_adj, 4)
        if row_artist_conc is not None:
            pred["artist_conc"] = round(row_artist_conc, 4)
        if row_album_adj is not None:
            pred["album_adj"] = round(row_album_adj, 4)
        if row_mood_coh is not None:
            pred["mood_coh"] = round(row_mood_coh, 4)
        if row_ild is not None:
            pred["ild"] = round(row_ild, 4)
        predictions.append(pred)

    metrics = rank_metrics(ranks, k=k)
    n = len(ranks)
    metrics["artist_recall_at_k"] = (sum(artist_hits) / n) if n else 0.0
    metrics["genre_recall_at_k"] = (sum(genre_hits) / n) if n else 0.0
    metrics["artist_mrr"] = (sum(artist_rrs) / n) if n else 0.0
    if music_vecs is not None and music_sims:
        metrics["music_at_k"] = sum(music_sims) / len(music_sims)
    # Session-holisticness diagnostics — each aggregated only over the sessions
    # where its signal was available (guarded, like music@k), so a missing
    # album field or absent content-metric index just omits that one metric.
    if artist_adjs:
        metrics["artist_adj_at_k"] = sum(artist_adjs) / len(artist_adjs)
    if artist_concs:
        metrics["artist_conc_at_k"] = sum(artist_concs) / len(artist_concs)
    if album_adjs:
        metrics["album_adj_at_k"] = sum(album_adjs) / len(album_adjs)
    if mood_cohs:
        metrics["mood_coh_at_k"] = sum(mood_cohs) / len(mood_cohs)
    if ilds:
        metrics["ild_at_k"] = sum(ilds) / len(ilds)
    # Composite HOLISTICNESS crown = coherent × varied × not-eager × GROUNDED.
    # Only defined when all four facets are present. Every factor is clamped into
    # [0,1] so degenerate cases score ~0 instead of gaming the crown.
    #
    # The first three facets are all INTERNAL to the returned list — they measure
    # the top-k against the prefix's mood centroid and against itself, and none
    # requires any relationship to what the user actually played next. A ranker
    # that confidently serves pleasant, varied, artist-diverse music while
    # IGNORING the user therefore maximized the old 3-factor crown (measured
    # 2026-07-25: `seq-mood` won it at holisticness 0.233 on recall@10 0.024 /
    # music@10 0.338, and a dual-tower `cummean/cummean` arm took rank 2 at
    # recall 0.066 — the ranking ran near-perfectly ANTI-correlated with music@10).
    #
    # `music_rel` is the GROUNDING factor: music@k (mean best cosine, in the
    # intrinsic musical-distance space, between the TRUE next track and the
    # top-k) rescaled so 0 = MUSIC_FLOOR (the context-free popularity baseline)
    # and 1 = a perfect match. It is deliberately NOT exact recall — it credits
    # recommending something that merely *sounds like* the real continuation,
    # which is the relevance notion a session-coherence goal wants. The rescaling
    # is what gives it leverage: music@k's RAW spread across models (~0.34..0.46)
    # is far narrower than the 3-factor product's (~0.13..0.23), so multiplying
    # in the raw value does NOT stop the ignore-the-user failure mode (verified:
    # the plain product still ranks `seq-mood` first). Above the floor the
    # spreads are comparable, so the grounded term actually binds.
    #
    # EAGERNESS is the WORSE of the two repetition notions, not just seed echo:
    #   artist_adj  — the top-k repeats the SEED's artist (album-eager)
    #   artist_conc — the top-k repeats ONE artist, whoever it is (list-eager)
    # Taking max() closes artist_adj's blind spot without adding a fifth factor
    # (a further factor would compress the composite's already-narrow range and
    # cost discrimination). Measured 2026-07-26: `mood-session` reads artist_adj
    # 0.0036 — "perfectly non-eager" — at artist_conc 0.3291.
    #
    # NOTE ON COMPARABILITY: this composite has been revised twice (3-factor →
    # grounded 4-factor on 2026-07-25 → this eagerness/floor revision on
    # 2026-07-26), so stored `holisticness_at_k` is only comparable within a code
    # generation. That is survivable because EVERY input is also stored per row in
    # predictions.json (`music_sim`, `mood_coh`, `ild`, `artist_adj`,
    # `artist_conc`), so any version is exactly recomputable for any past run.
    # Keep it that way: never add a crown factor that isn't also emitted per row.
    mc = metrics.get("mood_coh_at_k")
    il = metrics.get("ild_at_k")
    aa = metrics.get("artist_adj_at_k")
    ac = metrics.get("artist_conc_at_k")
    mu = metrics.get("music_at_k")
    if mc is not None and il is not None and aa is not None and mu is not None:
        music_rel = (mu - MUSIC_FLOOR) / (1.0 - MUSIC_FLOOR)
        eager = aa if ac is None else max(aa, ac)
        metrics["holisticness_at_k"] = (
            max(mc, 0.0)
            * il
            * (1.0 - min(max(eager, 0.0), 1.0))
            * min(max(music_rel, 0.0), 1.0)
        )
    return metrics, predictions


def write_outputs(run_dir: Path, metrics: dict, predictions: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "predictions.json").write_text(json.dumps(predictions))


# --------------------------------------------------------------------------- #
# Serving-time ranking (predict). Mirrors eval_from_scores' ranking rule for a #
# single caller-supplied session prefix instead of the held-out test split.   #
# --------------------------------------------------------------------------- #
def resolve_prefix(art: SeqArtifact, tokens: list) -> tuple[np.ndarray, list]:
    """Map serving prefix tokens to vocab item indices, preserving order.

    A token may be a full Spotify URI (`spotify:track:<id>`), a bare track id,
    a Spotify URL (`https://open.spotify.com/track/<id>?...`), or the item's
    integer vocab index (as int or string). Unknown tokens (cold items not in
    the model's vocab) are dropped and returned separately so the caller can
    warn. Returns (indices int64, unknown tokens)."""
    n_items = art.n_items
    uri_to_idx: dict[str, int] = {}
    id_to_idx: dict[str, int] = {}
    for key, meta in art.items.items():
        i = int(key)
        uri = meta.get("uri")
        if uri:
            uri_to_idx[uri] = i
            id_to_idx[uri.rsplit(":", 1)[-1]] = i  # bare id after the last ':'

    def bare_id(tok: str) -> str:
        # https://open.spotify.com/track/<id>?si=...  ->  <id>
        tok = tok.split("?", 1)[0].rstrip("/")
        return tok.rsplit("/", 1)[-1].rsplit(":", 1)[-1]

    idxs: list[int] = []
    unknown: list = []
    for tok in tokens:
        # Bare integer vocab index (int, or a purely-numeric string).
        if isinstance(tok, int) or (isinstance(tok, str) and tok.strip().isdigit()):
            i = int(tok)
            (idxs.append(i) if 0 <= i < n_items else unknown.append(tok))
            continue
        t = str(tok).strip()
        idx = uri_to_idx.get(t)
        if idx is None:
            idx = id_to_idx.get(bare_id(t))
        (idxs.append(idx) if idx is not None else unknown.append(tok))
    return np.asarray(idxs, dtype=np.int64), unknown


def rank_topk(art: SeqArtifact, score_fn, prefix_idx: np.ndarray,
              k: int = 10) -> list[int]:
    """Score the full vocab for one prefix, exclude the prefix items
    (next-distinct rule, identical to eval_from_scores), return the top-k item
    indices best-first."""
    scores = np.asarray(score_fn(prefix_idx), dtype=np.float64).copy()
    assert scores.shape[0] == art.n_items, "score_fn must cover the full vocab"
    if prefix_idx.size:
        scores[prefix_idx] = -np.inf
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order[:k]]


def load_prefix_request(input_dir: Path) -> tuple[list, int | None]:
    """Read the server-written ranking input (`prefix.json`): an ordered list of
    prefix tokens + an OPTIONAL requested k (None when the caller did not
    override the model's trained k). Also accepts a bare JSON list of tokens."""
    req = json.loads((input_dir / "prefix.json").read_text())
    if isinstance(req, list):
        return req, None
    k = req.get("k")
    return list(req.get("prefix", [])), (int(k) if k is not None else None)


def predict_ranking(art: SeqArtifact, score_fn, input_dir: Path,
                    output: Path, k: int | None = None) -> None:
    """Full serving-time predict: read the prefix request, rank the vocab, and
    write the lensing predictions.json shape ([{row_id, predicted, top_k_ids}]).
    One query (prefix) per call → a single-element list. Precedence for the
    result size: caller-requested k (prefix.json) > the model's trained k
    (passed as `k`) > 10."""
    tokens, req_k = load_prefix_request(input_dir)
    kk = int(req_k if req_k is not None else (k if k is not None else 10))
    prefix_idx, unknown = resolve_prefix(art, tokens)
    if unknown:
        emit({"event": "log",
              "msg": f"predict: {len(unknown)} prefix token(s) not in vocab "
                     f"(cold, dropped): {unknown[:5]}"})
    if prefix_idx.size == 0:
        raise SystemExit("predict: no prefix token resolved to a known vocab "
                         "item — cannot rank a next track")
    top = rank_topk(art, score_fn, prefix_idx, k=kk)
    preds = [{
        "row_id": 0,                              # single query
        "predicted": float(top[0]) if top else -1.0,
        "top_k_ids": [int(i) for i in top],
    }]
    output.write_text(json.dumps(preds))
    emit({"event": "log",
          "msg": f"ranked top-{kk} over {art.n_items} items from a "
                 f"{prefix_idx.size}-item prefix"})


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
