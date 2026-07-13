"""Next-track transition scoring + listening-context fit for the pathfinder.

Consumes pipeline/out/transitions.json (built by `python -m pipeline.transitions`):

  - affinity(meta_u, meta_v) -> float in [0,1]
        A backed-off, satisfaction-weighted estimate of "how naturally v
        follows u in your listening": track-level bigram evidence where it
        exists, deferring to artist- then genre-level transition tendencies
        when it's thin. Directional — affinity(u,v) != affinity(v,u).

  - context_fit(meta_v, ctx) -> float (>0, ~1.0 == neutral)
        How over-represented v's genre (and v itself) is in the chosen
        time-of-day / shuffle context. >1 means "fits this moment".

Degrades gracefully: if the tables are missing, `load()` returns None and the
pathfinder simply drops the two terms (a warning is printed by the caller).
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass

from .config import (
    TRANSITION_ARTIST_WEIGHT,
    TRANSITION_BACKOFF_K,
    TRANSITION_GENRE_WEIGHT,
    TRANSITIONS_JSON,
)

# Boundaries identical to pipeline/config.py:TOD_BUCKETS — keep in sync.
_TOD = [
    ("night", range(0, 6)),
    ("morning", range(6, 12)),
    ("afternoon", range(12, 18)),
    ("evening", range(18, 24)),
]


def tod_bucket(hour: int) -> str:
    for name, hours in _TOD:
        if hour in hours:
            return name
    return "night"


@dataclass
class Context:
    """A resolved listening context: the lift sub-tables to apply, plus a
    human label for display. Empty sub-tables == 'don't condition on this'."""
    genre_tod: dict[str, float]
    genre_shuf: dict[str, float]
    track_tod: dict[str, float]
    label: str


@dataclass
class TransitionModel:
    track_bigram: dict[str, dict[str, float]]
    artist_cond: dict[str, dict[str, float]]
    genre_cond: dict[str, dict[str, float]]
    ctx_genre_lift: dict[str, dict[str, float]]
    ctx_track_lift: dict[str, dict[str, float]]
    tod_buckets: list[str]

    # ---- #7: directional transition affinity -------------------------------
    def affinity(self, mu: dict, mv: dict) -> float:
        u, v = mu["track_uri"], mv["track_uri"]
        tb = self.track_bigram.get(u)
        track_p, n_u = 0.0, 0.0
        if tb:
            n_u = sum(tb.values())
            if n_u > 0:
                track_p = tb.get(v, 0.0) / n_u

        a_p = self.artist_cond.get(mu.get("artist", "?"), {}).get(mv.get("artist", "?"), 0.0)
        g_p = self.genre_cond.get(
            mu.get("genre_primary", "unknown"), {}
        ).get(mv.get("genre_primary", "unknown"), 0.0)
        backoff = TRANSITION_ARTIST_WEIGHT * a_p + TRANSITION_GENRE_WEIGHT * g_p

        trust = n_u / (n_u + TRANSITION_BACKOFF_K)  # 0 when unseen, ->1 when well-observed
        return trust * track_p + (1.0 - trust) * backoff

    # ---- #8: context fit ----------------------------------------------------
    def context_fit(self, mv: dict, ctx: Context) -> float:
        genre = mv.get("genre_primary", "unknown")
        lift = ctx.genre_tod.get(genre, 1.0) * ctx.genre_shuf.get(genre, 1.0)
        tl = ctx.track_tod.get(mv["track_uri"])
        if tl is not None:
            lift *= tl
        return lift

    def resolve_context(self, tod: str | None = "now", shuffle: bool | None = None) -> Context:
        """Build a Context from a spec.

        tod:     'now' (wall-clock bucket), an explicit bucket name, or None
                 to not condition on time of day.
        shuffle: True / False to condition on shuffle state, or None to ignore.
        """
        labels = []
        if tod == "now":
            tod = tod_bucket(datetime.datetime.now().hour)
        genre_tod, track_tod = {}, {}
        if tod:
            genre_tod = self.ctx_genre_lift.get(f"tod:{tod}", {})
            track_tod = self.ctx_track_lift.get(tod, {})
            labels.append(tod)
        genre_shuf = {}
        if shuffle is not None:
            state = "shuffle" if shuffle else "linear"
            genre_shuf = self.ctx_genre_lift.get(f"shuf:{state}", {})
            labels.append(state)
        return Context(
            genre_tod=genre_tod,
            genre_shuf=genre_shuf,
            track_tod=track_tod,
            label=" · ".join(labels) if labels else "any",
        )


def load() -> TransitionModel | None:
    if not TRANSITIONS_JSON.exists():
        return None
    with open(TRANSITIONS_JSON) as f:
        d = json.load(f)
    return TransitionModel(
        track_bigram=d["track_bigram"],
        artist_cond=d["artist_cond"],
        genre_cond=d["genre_cond"],
        ctx_genre_lift=d["ctx_genre_lift"],
        ctx_track_lift=d["ctx_track_lift"],
        tod_buckets=d.get("meta", {}).get("tod_buckets", [b for b, _ in _TOD]),
    )
