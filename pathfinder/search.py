"""A* playlist pathfinding over the co-listening kNN graph.

cost(u->v) = W_DIST * cos_dist(u, v)
           + W_FIT  * (1 - habitfit_norm(v))
           + W_DIV  * genre_jump_penalty
           + W_TRANS* (1 - transition_norm(u->v))   # learned next-track (#7)
           + W_CTX  * (1 - context_norm(v | ctx))    # time-of-day / shuffle (#8)
Hard constraints (checked against the reconstructed path during expansion):
no repeats, <=2 consecutive same-artist tracks, per-artist total cap.
Heuristic: cosine distance to the end vector (keeps search goal-directed).

The transition and context terms are min-max normalized *across the candidate
next hops at each expansion*: they express which of the reachable neighbors you
are most likely to play next / which best fit the moment, not an absolute
scale. Both are inactive when no transition model is supplied (model=None),
recovering the original distance+fit+diversity search exactly.
"""
import heapq
import itertools
import math

from .config import (
    ARTIST_SHARE_DIVISOR,
    GENRE_JUMP_PENALTY,
    MAX_CONSECUTIVE_ARTIST,
    W_CTX,
    W_DIST,
    W_DIV,
    W_FIT,
    W_TRANS,
)
from .graph import TrackGraph
from .transitions import Context, TransitionModel


def _minmax(raw: dict[int, float]) -> dict[int, float]:
    """Normalize candidate scores to [0,1]; all-equal (or empty) -> neutral 0.5."""
    if not raw:
        return {}
    lo, hi = min(raw.values()), max(raw.values())
    span = hi - lo
    if span <= 0:
        return {k: 0.5 for k in raw}
    return {k: (v - lo) / span for k, v in raw.items()}


def _violates(g: TrackGraph, path: list[int], candidate: int, length_hint: int) -> bool:
    if candidate in path:
        return True
    artist = g.meta[candidate]["artist"]
    # consecutive-artist cap
    run = 1
    for pid in reversed(path):
        if g.meta[pid]["artist"] == artist:
            run += 1
        else:
            break
    if run > MAX_CONSECUTIVE_ARTIST:
        return True
    # total per-artist cap
    cap = math.ceil(max(length_hint, len(path) + 1) / ARTIST_SHARE_DIVISOR)
    total = sum(1 for pid in path if g.meta[pid]["artist"] == artist) + 1
    return total > cap


def find_path(
    g: TrackGraph,
    start: int,
    end: int,
    scores: dict[int, float],
    length_hint: int = 12,
    max_expansions: int = 200_000,
    model: TransitionModel | None = None,
    ctx: Context | None = None,
) -> list[int] | None:
    """A* from start to end over learned-manifold neighbors."""
    counter = itertools.count()  # tie-breaker; avoids comparing paths
    open_heap = [(g.cos_dist(start, end), next(counter), 0.0, start, [start])]
    best_g: dict[int, float] = {start: 0.0}

    for _ in range(max_expansions):
        if not open_heap:
            return None
        _, _, g_cost, node, path = heapq.heappop(open_heap)
        if node == end:
            return path
        if g_cost > best_g.get(node, float("inf")):
            continue

        mu = g.meta[node]
        prev_genre = mu.get("genre_primary")
        cands = [
            (nbr, dist)
            for nbr, dist in g.neighbors.get(node, [])
            if nbr == end or not _violates(g, path, nbr, length_hint)
        ]

        # Behavioral terms, normalized across this expansion's candidate set.
        trans_n: dict[int, float] = {}
        ctx_n: dict[int, float] = {}
        if model is not None and cands:
            trans_n = _minmax({nbr: model.affinity(mu, g.meta[nbr]) for nbr, _ in cands})
            if ctx is not None:
                ctx_n = _minmax({nbr: model.context_fit(g.meta[nbr], ctx) for nbr, _ in cands})

        for nbr, dist in cands:
            step = W_DIST * dist
            step += W_FIT * (1.0 - scores.get(nbr, 0.5))
            if g.meta[nbr].get("genre_primary") != prev_genre:
                step += W_DIV * GENRE_JUMP_PENALTY
            if trans_n:
                step += W_TRANS * (1.0 - trans_n.get(nbr, 0.5))
            if ctx_n:
                step += W_CTX * (1.0 - ctx_n.get(nbr, 0.5))
            new_g = g_cost + step
            # Path-dependent constraints make strict dominance pruning unsound,
            # but it keeps the search tractable and works well in practice.
            if new_g >= best_g.get(nbr, float("inf")):
                continue
            best_g[nbr] = new_g
            h = g.cos_dist(nbr, end)
            heapq.heappush(open_heap, (new_g + h, next(counter), new_g, nbr, path + [nbr]))
    return None


def _violates_insertion(g: TrackGraph, path: list[int], i: int, candidate: int, length_hint: int) -> bool:
    """Constraint check for inserting `candidate` between path[i] and path[i+1]:
    unlike _violates, it sees both sides of the insertion point."""
    if candidate in path:
        return True
    artist = g.meta[candidate]["artist"]
    run = 1
    for pid in reversed(path[: i + 1]):
        if g.meta[pid]["artist"] == artist:
            run += 1
        else:
            break
    for pid in path[i + 1 :]:
        if g.meta[pid]["artist"] == artist:
            run += 1
        else:
            break
    if run > MAX_CONSECUTIVE_ARTIST:
        return True
    cap = math.ceil(max(length_hint, len(path) + 1) / ARTIST_SHARE_DIVISOR)
    total = sum(1 for pid in path if g.meta[pid]["artist"] == artist) + 1
    return total > cap


def densify(
    g: TrackGraph,
    path: list[int],
    scores: dict[int, float],
    target_length: int,
    model: TransitionModel | None = None,
    ctx: Context | None = None,
) -> list[int]:
    """Grow the A* skeleton toward the target length by repeatedly splitting
    the widest hop with the learned track that best bridges it (small detour
    cost, high habit-fit, fits the transition flow a->c->b and the context,
    constraints respected)."""
    path = list(path)
    while len(path) < target_length:
        # Widest remaining hop
        gaps = [(g.cos_dist(a, b), i) for i, (a, b) in enumerate(zip(path, path[1:]))]
        gaps.sort(reverse=True)
        inserted = False
        for _, i in gaps:
            a, b = path[i], path[i + 1]
            ma, mb = g.meta[a], g.meta[b]
            candidates = {pid for pid, _ in g.neighbors.get(a, [])} | {
                pid for pid, _ in g.neighbors.get(b, [])
            }
            gap = g.cos_dist(a, b)
            # Collect geometrically-valid bridges, then score with normalized
            # behavioral terms across exactly that set.
            valid = []
            for c in candidates:
                if _violates_insertion(g, path, i, c, target_length):
                    continue
                # Between-ness guard: the insert must refine the gap, not pull
                # the path sideways toward an unrelated high-fit cluster.
                if g.cos_dist(a, c) >= gap or g.cos_dist(c, b) >= gap:
                    continue
                detour = g.cos_dist(a, c) + g.cos_dist(c, b) - gap
                valid.append((c, detour))
            if not valid:
                continue

            trans_n: dict[int, float] = {}
            ctx_n: dict[int, float] = {}
            if model is not None:
                # A natural bridge follows a *and* leads into b.
                trans_n = _minmax({
                    c: model.affinity(ma, g.meta[c]) + model.affinity(g.meta[c], mb)
                    for c, _ in valid
                })
                if ctx is not None:
                    ctx_n = _minmax({c: model.context_fit(g.meta[c], ctx) for c, _ in valid})

            best, best_cost = None, float("inf")
            for c, detour in valid:
                cost = W_DIST * detour + W_FIT * (1.0 - scores.get(c, 0.5))
                if trans_n:
                    cost += W_TRANS * (1.0 - trans_n.get(c, 0.5))
                if ctx_n:
                    cost += W_CTX * (1.0 - ctx_n.get(c, 0.5))
                if cost < best_cost:
                    best, best_cost = c, cost
            if best is not None:
                path.insert(i + 1, best)
                inserted = True
                break
        if not inserted:
            break  # nothing insertable anywhere — accept the shorter path
    return path
