"""Learned next-track transitions + listening-context tables.

Two products, both mined from the sessionized play log and consumed by the
pathfinder (never by lensing — this is a sequence artifact, not a per-track
regression target):

  1. Transition model (#7) — directional, satisfaction-weighted Markov
     statistics over *consecutive* non-skip plays within a session:
       - track_bigram[u][v]  : satisfaction-weighted count of v following u
       - artist_cond[a][b]   : P(next artist = b | current artist = a)
       - genre_cond[g][h]    : P(next genre  = h | current genre  = g)
     The pathfinder backs off track -> artist -> genre, so the sparse
     track-level evidence is used where it exists and the dense artist/genre
     tendencies cover everything else.

  2. Context model (#8) — how taste shifts with *when/how* you listen, as a
     lift (over-representation vs the global base rate):
       - ctx_genre_lift["tod:<bucket>"][genre]  and ["shuf:<state>"][genre]
       - ctx_track_lift["<bucket>"][uri]        (popular tracks only)
     Lift > 1 => that genre/track is over-represented in the context.

Transitions are weighted by how the *destination* play ended (reason_end):
landing on a track you let finish or replayed is a good transition; landing on
one you skipped forward off is not.
"""
import json
from collections import Counter, defaultdict

import pandas as pd

from .config import (
    CONTEXT_MIN_SUPPORT,
    CONTEXT_TRACK_MIN_PLAYS,
    NEUTRAL_SAT,
    REASON_END_SATISFACTION,
    SESSIONS_PARQUET,
    SKIP_THRESHOLD_MS,
    TOD_BUCKETS,
    TRACK_FEATURES_PARQUET,
    TRANSITIONS_JSON,
)

LIFT_CLIP = (0.2, 5.0)   # keep a single thin bucket from inventing a 30x pull


def tod_bucket(hour: int) -> str:
    for name, hours in TOD_BUCKETS:
        if hour in hours:
            return name
    return "night"  # unreachable given full 0..23 coverage


def _shrink_lift(observed: float, base: float, support: int) -> float:
    """Lift = P(x|ctx)/P(x), shrunk toward 1.0 (no signal) when the context
    bucket has little support, then clipped to a sane range."""
    if base <= 0:
        return 1.0
    lift = observed / base
    trust = min(1.0, support / CONTEXT_MIN_SUPPORT)
    lift = 1.0 + (lift - 1.0) * trust
    return float(min(LIFT_CLIP[1], max(LIFT_CLIP[0], lift)))


def _conditional(bigram: dict[str, Counter]) -> dict[str, dict[str, float]]:
    """Normalize a weighted bigram into P(next | current) over observed targets."""
    cond: dict[str, dict[str, float]] = {}
    for src, targets in bigram.items():
        total = sum(targets.values())
        if total <= 0:
            continue
        cond[src] = {dst: w / total for dst, w in targets.items()}
    return cond


def run() -> dict:
    plays = pd.read_parquet(SESSIONS_PARQUET)
    feats = pd.read_parquet(TRACK_FEATURES_PARQUET)

    uri_artist = dict(zip(feats["track_uri"], feats["artist"]))
    uri_genre = dict(zip(feats["track_uri"], feats["genre_primary"]))

    # Skips/previews break the "I chose to keep listening" adjacency.
    nz = plays[plays["ms_played"] >= SKIP_THRESHOLD_MS].sort_values("ts").copy()
    nz["ts"] = pd.to_datetime(nz["ts"], utc=True)
    nz["sat"] = nz["reason_end"].map(
        lambda r: REASON_END_SATISFACTION.get(r, NEUTRAL_SAT)
    )

    # ---- 1. Transition bigrams over consecutive within-session pairs ----
    track_bigram: dict[str, Counter] = defaultdict(Counter)
    artist_bigram: dict[str, Counter] = defaultdict(Counter)
    genre_bigram: dict[str, Counter] = defaultdict(Counter)
    n_pairs = 0
    for _, g in nz.groupby("session_id", sort=False):
        uris = g["spotify_track_uri"].tolist()
        sats = g["sat"].tolist()
        for i in range(len(uris) - 1):
            u, v = uris[i], uris[i + 1]
            if u == v:
                continue  # self-loops (repeat-one) carry no routing signal
            w = sats[i + 1]  # weight by how the *destination* play ended
            track_bigram[u][v] += w
            artist_bigram[uri_artist.get(u, "?")][uri_artist.get(v, "?")] += w
            genre_bigram[uri_genre.get(u, "unknown")][uri_genre.get(v, "unknown")] += w
            n_pairs += 1

    track_bigram_out = {u: dict(c) for u, c in track_bigram.items()}
    artist_cond = _conditional(artist_bigram)
    genre_cond = _conditional(genre_bigram)

    # ---- 2. Context lift tables ----
    nz["tod"] = nz["ts"].dt.hour.map(tod_bucket)
    nz["shuf"] = nz["shuffle"].map(lambda s: "shuffle" if bool(s) else "linear")

    total = len(nz)
    nz["genre"] = nz["spotify_track_uri"].map(lambda u: uri_genre.get(u, "unknown"))
    genre_base = (nz["genre"].value_counts() / total).to_dict()

    ctx_genre_lift: dict[str, dict[str, float]] = {}
    for dim, col in (("tod", "tod"), ("shuf", "shuf")):
        for bucket, sub in nz.groupby(col):
            n_bucket = len(sub)
            shares = sub["genre"].value_counts()
            table = {}
            for genre, cnt in shares.items():
                table[genre] = _shrink_lift(cnt / n_bucket, genre_base.get(genre, 0.0), int(cnt))
            ctx_genre_lift[f"{dim}:{bucket}"] = table

    # Track-level time-of-day lift, popular tracks only.
    track_total = nz["spotify_track_uri"].value_counts()
    popular = set(track_total[track_total >= CONTEXT_TRACK_MIN_PLAYS].index)
    track_base = (track_total / total).to_dict()
    ctx_track_lift: dict[str, dict[str, float]] = {}
    for bucket, sub in nz.groupby("tod"):
        n_bucket = len(sub)
        shares = sub[sub["spotify_track_uri"].isin(popular)]["spotify_track_uri"].value_counts()
        table = {}
        for uri, cnt in shares.items():
            table[uri] = _shrink_lift(cnt / n_bucket, track_base.get(uri, 0.0), int(cnt))
        ctx_track_lift[bucket] = table

    out = {
        "track_bigram": track_bigram_out,
        "artist_cond": artist_cond,
        "genre_cond": genre_cond,
        "ctx_genre_lift": ctx_genre_lift,
        "ctx_track_lift": ctx_track_lift,
        "meta": {
            "n_pairs": n_pairs,
            "tod_buckets": [name for name, _ in TOD_BUCKETS],
            "shuffle_states": ["shuffle", "linear"],
            "neutral_sat": NEUTRAL_SAT,
        },
    }
    with open(TRANSITIONS_JSON, "w") as f:
        json.dump(out, f)

    print(
        f"transitions: {n_pairs} consecutive non-skip pairs -> "
        f"{len(track_bigram_out)} source tracks, "
        f"{len(artist_cond)} artists, {len(genre_cond)} genres"
    )
    print(
        f"transitions: context — {len(ctx_genre_lift)} genre-lift buckets, "
        f"{sum(len(t) for t in ctx_track_lift.values())} track-lift entries "
        f"({len(popular)} popular tracks across {len(ctx_track_lift)} time buckets)"
    )
    return out


if __name__ == "__main__":
    run()
