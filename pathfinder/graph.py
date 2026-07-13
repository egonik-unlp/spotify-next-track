"""Track graph: vectors + metadata from Qdrant, kNN edges over the corpus.

Every track on the AE song-embedding has a real learned vector, so the WHOLE
collection is the walkable manifold (the old `embedding_source=="learned"`
gating was a co-listening-era artifact — on co-listening only ~2.9k tracks had
real positions and the rest were artist/genre centroids usable only as
endpoints). `nearest_learned` is now a no-op for in-corpus tracks.
"""
from dataclasses import dataclass, field

import numpy as np
from qdrant_client import QdrantClient

from .config import KNN_K, QDRANT_COLLECTION, QDRANT_URL


@dataclass
class TrackGraph:
    ids: list[int]                       # qdrant point ids, row-aligned with vectors
    vectors: np.ndarray                  # unit-normalized, (n, dim)
    meta: dict[int, dict]                # point id -> metadata payload
    learned_ids: set[int]
    neighbors: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    _row: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self._row = {pid: i for i, pid in enumerate(self.ids)}

    def vec(self, pid: int) -> np.ndarray:
        return self.vectors[self._row[pid]]

    def cos_dist(self, a: int, b: int) -> float:
        return float(1.0 - self.vec(a) @ self.vec(b))

    def find(self, query: str) -> int:
        """Resolve a track URI or (partial) 'name' / 'name - artist' string."""
        q = query.strip().lower()
        for pid, m in self.meta.items():
            if m["track_uri"].lower() == q:
                return pid
        matches = [
            (pid, m) for pid, m in self.meta.items()
            if q in f"{m['track_name']} - {m['artist']}".lower()
        ]
        if not matches:
            raise KeyError(f"no track matches {query!r}")
        # Prefer the most-played match (likely the canonical version)
        matches.sort(key=lambda x: -x[1].get("play_count", 0))
        if len(matches) > 1:
            top = ", ".join(
                f"{m['track_name']} — {m['artist']}" for _, m in matches[:3]
            )
            print(f"note: {len(matches)} matches for {query!r}; using most-played of: {top}")
        return matches[0][0]

    def label(self, pid: int) -> str:
        m = self.meta[pid]
        return f"{m['track_name']} — {m['artist']} [{m.get('genre_primary', '?')}]"


def load_graph() -> TrackGraph:
    client = QdrantClient(url=QDRANT_URL)
    ids, vectors, meta = [], [], {}
    offset = None
    while True:
        points, offset = client.scroll(
            QDRANT_COLLECTION, limit=2048, offset=offset,
            with_payload=True, with_vectors=True,
        )
        for p in points:
            ids.append(p.id)
            vectors.append(p.vector)
            meta[p.id] = p.payload["metadata"]
        if offset is None:
            break

    vecs = np.asarray(vectors, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    # The AE embedding gives every track a real position, so the whole corpus is
    # navigable. (On a co-listening collection, gate on embedding_source=="learned"
    # instead — only those tracks have true positions.) kNN is then m×m over all
    # tracks; for ~12k×64 that's a ~0.6 GB transient at graph build, which is fine.
    learned = set(ids)

    g = TrackGraph(ids=ids, vectors=vecs, meta=meta, learned_ids=learned)
    _build_knn(g)
    return g


def _build_knn(g: TrackGraph, block: int = 2048) -> None:
    """Brute-force kNN over the learned subset.

    Edges are symmetrized (union of kNN relations): a directed kNN graph
    leaves low-in-degree tracks unreachable as path destinations.

    Similarities are computed in row-blocks rather than as one m×m matrix: a
    full m×m float32 buffer (plus the negated copy `np.argpartition` needs) is
    ~2·m²·4 bytes, which OOMs once the corpus grows past ~15k tracks. Blocking
    caps the transient at ~2·block·m·4 bytes regardless of m.
    """
    learned_rows = [g._row[pid] for pid in g.ids if pid in g.learned_ids]
    learned_pids = [g.ids[r] for r in learned_rows]
    sub = g.vectors[learned_rows]              # (m, dim), unit norm
    m = sub.shape[0]
    k = min(KNN_K, m - 1)

    edges: dict[int, dict[int, float]] = {pid: {} for pid in learned_pids}
    for b in range(0, m, block):
        e = min(b + block, m)
        sims = sub[b:e] @ sub.T                # (blk, m)
        for r in range(e - b):
            sims[r, b + r] = -np.inf           # exclude self (global row b+r)
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(e - b):
            pid = learned_pids[b + r]
            for j in top[r]:
                other = learned_pids[j]
                dist = float(1.0 - sims[r, j])
                edges[pid][other] = dist
                edges[other][pid] = dist
    for pid, nbrs in edges.items():
        g.neighbors[pid] = sorted(nbrs.items(), key=lambda t: t[1])


def nearest_learned(g: TrackGraph, pid: int) -> int:
    """Anchor an arbitrary track to its nearest learned-manifold node."""
    if pid in g.learned_ids:
        return pid
    learned_rows = [g._row[p] for p in g.ids if p in g.learned_ids]
    sims = g.vectors[learned_rows] @ g.vec(pid)
    best = int(np.argmax(sims))
    return g.ids[learned_rows[best]]
