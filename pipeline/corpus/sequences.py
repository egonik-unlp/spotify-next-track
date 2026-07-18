"""SEQUENCE dataset producer (#2.4) — ordered per-session track sequences +
per-item song-AE latents for a next-track ranking model.

Unlike `transitions.py` (which mines *aggregate* Markov statistics), this
materializes the raw ordered material a sequence model trains on: for every
kept listening session, the ordered list of item indices, plus the 64-dim
song-AE latent for each item. The artifact contract below is FIXED — a Rust
loader `SequenceDataset::load` reads exactly these files, so field names,
byte layouts (all little-endian) and the manifest shape must not drift.

Pipeline:
  1. Keep non-skip plays (ms_played >= SKIP_THRESHOLD_MS); within each session
     (ordered by ts) drop consecutive self-loops (same uri twice in a row).
  2. Item vocabulary = distinct kept uris that ALSO have a point (latent) in
     the `spotify_tracks_song_ae` collection (join via id_map). Plays whose
     track has no latent are dropped.
  3. Keep sessions with >= 2 items (need a prefix + a label).
  4. Chronological split by session start time: earliest 80% of sessions
     (by first-play ts) -> train, latest 20% -> test; record the cut ts.
  5. Write the binary + json artifact under data/seq/<dataset_id>/.

The ONLY correct latent source is `spotify_tracks_song_ae` (64-dim song-AE
latent). `spotify_tracks` is a 200-dim *behavioral* embedding that soft-leaks
replay and must never be used here.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

from .config import (
    ID_MAP_JSON,
    QDRANT_URL,
    SESSIONS_PARQUET,
    SKIP_THRESHOLD_MS,
)

# Song-AE latent collection — 64-dim, point id == the u64 from id_map.
# Hardcoded (NOT config.QDRANT_COLLECTION, which points at the behavioral
# `spotify_tracks` embedding) because this artifact is only ever correct
# against the song-AE latent.
LATENT_COLLECTION = "spotify_tracks_song_ae"
LATENT_DIM = 64
TRAIN_FRAC = 0.80
RETRIEVE_BATCH = 500

# Artifact root: <instance>/data/seq/  (corpus -> pipeline -> instance root).
INSTANCE_ROOT = Path(__file__).resolve().parents[2]
SEQ_ROOT = INSTANCE_ROOT / "data" / "seq"


def _drop_consecutive_loops(uris: list[str]) -> list[str]:
    """Collapse runs of the same uri (repeat-one) to a single occurrence —
    self-loops carry no next-track routing signal."""
    out: list[str] = []
    for u in uris:
        if not out or out[-1] != u:
            out.append(u)
    return out


def _fetch_latents(client: QdrantClient, ids: list[int],
                   collection: str = LATENT_COLLECTION) -> dict[int, list[float]]:
    """Retrieve item latent vectors for the given point ids, batched. Ids without
    a point (no latent) are simply absent from the returned dict."""
    vecs: dict[int, list[float]] = {}
    for i in range(0, len(ids), RETRIEVE_BATCH):
        batch = ids[i : i + RETRIEVE_BATCH]
        points = client.retrieve(
            collection_name=collection,
            ids=batch,
            with_vectors=True,
            with_payload=False,
        )
        for p in points:
            if p.vector is not None:
                vecs[int(p.id)] = p.vector
    return vecs


def run(latent_collection: str = LATENT_COLLECTION,
        latent_dim: int = LATENT_DIM) -> dict:
    # ---- 1. Load + non-skip filter ----
    plays = pd.read_parquet(SESSIONS_PARQUET)
    plays["ts"] = pd.to_datetime(plays["ts"], utc=True)
    kept = plays[plays["ms_played"] >= SKIP_THRESHOLD_MS].sort_values("ts").copy()

    with open(ID_MAP_JSON) as f:
        uri_to_id = json.load(f)["uri_to_id"]

    # ---- 2. Item vocabulary (kept uris that have a song-AE latent) ----
    candidate_uris = kept["spotify_track_uri"].dropna().unique().tolist()
    # uris present in the id_map -> candidate point ids in the latent collection
    cand_id_by_uri = {u: uri_to_id[u] for u in candidate_uris if u in uri_to_id}

    client = QdrantClient(url=QDRANT_URL)
    # Derive the true latent dim from the collection config; treat the caller's
    # --latent-dim as an assertion so a mismatched flag fails loud, not silent.
    actual_dim = client.get_collection(latent_collection).config.params.vectors.size
    assert actual_dim == latent_dim, (
        f"--latent-dim {latent_dim} != collection {latent_collection!r} dim {actual_dim}"
    )
    id_to_vec = _fetch_latents(client, list(cand_id_by_uri.values()), latent_collection)

    # Keep only uris whose id resolved to an actual vector. Sort for a stable,
    # reproducible item-index assignment.
    vocab_uris = sorted(u for u, pid in cand_id_by_uri.items() if pid in id_to_vec)
    item_index = {u: i for i, u in enumerate(vocab_uris)}
    n_items = len(vocab_uris)

    # ---- item_latents matrix, row = item index ----
    latents = np.zeros((n_items, latent_dim), dtype=np.float32)
    for u, i in item_index.items():
        latents[i] = np.asarray(id_to_vec[cand_id_by_uri[u]], dtype=np.float32)

    # Drop plays whose track has no latent, then build ordered per-session
    # sequences (self-loops collapsed) of item indices; keep sessions >= 2.
    kept = kept[kept["spotify_track_uri"].isin(item_index)]

    session_seqs: list[list[int]] = []
    session_starts: list[pd.Timestamp] = []
    for _, g in kept.groupby("session_id", sort=False):
        uris = _drop_consecutive_loops(g["spotify_track_uri"].tolist())
        if len(uris) < 2:
            continue
        session_seqs.append([item_index[u] for u in uris])
        session_starts.append(g["ts"].iloc[0])

    n_sessions = len(session_seqs)

    # ---- 4. Chronological split by session start time ----
    order = sorted(range(n_sessions), key=lambda s: session_starts[s])
    n_train = int(round(TRAIN_FRAC * n_sessions))
    train_idx = sorted(order[:n_train])
    test_idx = sorted(order[n_train:])
    # cut = start ts of the earliest test session (the train/test boundary).
    cut = session_starts[order[n_train]] if test_idx else session_starts[order[-1]]

    # ---- 5. Serialize the artifact ----
    dataset_id = datetime.now(timezone.utc).strftime("seq-%Y%m%d-%H%M%S")
    created_at = datetime.now(timezone.utc).isoformat()
    out_dir = SEQ_ROOT / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flat concatenated sequences + prefix offsets (offsets[s]..offsets[s+1]).
    flat = np.concatenate([np.asarray(s, dtype=np.uint32) for s in session_seqs])
    offsets = np.zeros(n_sessions + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum([len(s) for s in session_seqs], dtype=np.uint32)

    (out_dir / "sessions.u32").write_bytes(flat.astype("<u4").tobytes())
    (out_dir / "offsets.u32").write_bytes(offsets.astype("<u4").tobytes())
    (out_dir / "item_latents.f32").write_bytes(latents.astype("<f4").tobytes())
    (out_dir / "train_sessions.u32").write_bytes(
        np.asarray(train_idx, dtype="<u4").tobytes()
    )
    (out_dir / "test_sessions.u32").write_bytes(
        np.asarray(test_idx, dtype="<u4").tobytes()
    )

    # items.json — display/debug metadata keyed by string item index. Pulled
    # from the song-AE payload (nested under "metadata").
    id_to_uri = {pid: u for u, pid in cand_id_by_uri.items()}
    meta_by_id: dict[int, dict] = {}
    vocab_ids = [cand_id_by_uri[u] for u in vocab_uris]
    for i in range(0, len(vocab_ids), RETRIEVE_BATCH):
        batch = vocab_ids[i : i + RETRIEVE_BATCH]
        for p in client.retrieve(
            collection_name=latent_collection,
            ids=batch,
            with_vectors=False,
            with_payload=True,
        ):
            meta_by_id[int(p.id)] = (p.payload or {}).get("metadata", {})

    items: dict[str, dict] = {}
    for u, i in item_index.items():
        m = meta_by_id.get(cand_id_by_uri[u], {})
        items[str(i)] = {
            "uri": u,
            "name": m.get("track_name"),
            "artist": m.get("artist"),
            "genre": m.get("genre_primary"),
            "play_count": m.get("play_count"),
        }
    with open(out_dir / "items.json", "w") as f:
        json.dump(items, f)

    manifest = {
        "dataset_id": dataset_id,
        "created_at": created_at,
        "kind": "sequence",
        "n_sessions": n_sessions,
        "n_items": n_items,
        "latent_dim": latent_dim,
        "latent_source": latent_collection,
        "split": {
            "strategy": "chronological_by_session_start",
            "cut": cut.isoformat(),
            "n_train_sessions": len(train_idx),
            "n_test_sessions": len(test_idx),
        },
    }
    with open(out_dir / "sequence-manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- 6. Summary + cold-item rate ----
    train_items = set()
    for s in train_idx:
        train_items.update(session_seqs[s])
    n_cold = sum(
        1 for s in test_idx if session_seqs[s][-1] not in train_items
    )
    cold_rate = n_cold / len(test_idx) if test_idx else 0.0
    mean_len = float(flat.size / n_sessions) if n_sessions else 0.0

    print(f"sequences: {n_sessions} sessions, {n_items} items")
    print(
        f"sequences: split {len(train_idx)} train / {len(test_idx)} test "
        f"(cut {cut.isoformat()})"
    )
    print(f"sequences: mean session length {mean_len:.2f} items")
    print(
        f"sequences: cold-item rate {cold_rate:.4f} "
        f"({n_cold}/{len(test_idx)} test-label items never seen in a train session)"
    )
    print(f"sequences: artifact -> {out_dir}")
    return manifest


def _validate(dataset_id: str | None = None) -> None:
    """Reload the artifact with numpy and assert the contract holds."""
    if dataset_id is None:
        dataset_id = sorted(p.name for p in SEQ_ROOT.iterdir() if p.is_dir())[-1]
    d = SEQ_ROOT / dataset_id
    man = json.load(open(d / "sequence-manifest.json"))
    n_items = man["n_items"]
    latent_dim = man["latent_dim"]

    sessions = np.frombuffer((d / "sessions.u32").read_bytes(), dtype="<u4")
    offsets = np.frombuffer((d / "offsets.u32").read_bytes(), dtype="<u4")
    latents = np.frombuffer((d / "item_latents.f32").read_bytes(), dtype="<f4")

    assert offsets[0] == 0, "offsets[0] must be 0"
    assert offsets[-1] == len(sessions), "offsets[-1] must equal len(sessions)"
    assert len(offsets) == man["n_sessions"] + 1, "offsets length mismatch"
    assert sessions.max() < n_items, "item index out of range"
    lengths = np.diff(offsets)
    assert (lengths >= 2).all(), "every session must have >= 2 items"
    assert latents.size == n_items * latent_dim, "item_latents size mismatch"

    train = np.frombuffer((d / "train_sessions.u32").read_bytes(), dtype="<u4")
    test = np.frombuffer((d / "test_sessions.u32").read_bytes(), dtype="<u4")
    assert len(train) + len(test) == man["n_sessions"], "split does not cover all sessions"
    assert len(set(train.tolist()) & set(test.tolist())) == 0, "train/test overlap"
    print(f"sequences: validation OK ({dataset_id})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent-collection", default=LATENT_COLLECTION,
                    help="Qdrant collection of per-item latents (default "
                         "spotify_tracks_song_ae; must be a leak-safe item space, "
                         "NEVER the 200-dim behavioral spotify_tracks)")
    ap.add_argument("--latent-dim", type=int, default=LATENT_DIM,
                    help="expected latent dim; asserted against the collection config")
    a = ap.parse_args()
    m = run(latent_collection=a.latent_collection, latent_dim=a.latent_dim)
    _validate(m["dataset_id"])
