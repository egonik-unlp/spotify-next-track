"""Train co-listening track embeddings (word2vec on sessionized play sequences).

Tracks with >= min_count non-skip plays get a learned vector; sparse tracks get
a fallback (same-artist mean, then genre centroid, then global mean) and are
flagged so the pathfinder can exclude them as interior waypoints.

Run with --check to print nearest-neighbor sanity probes instead of retraining.
"""
import argparse
import json

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from .config import (
    OUT_DIR,
    SESSIONS_PARQUET,
    SKIP_THRESHOLD_MS,
    TRACK_FEATURES_PARQUET,
    TRACK_VECTOR_URIS_JSON,
    TRACK_VECTORS_NPY,
    W2V_PARAMS,
)

MODEL_PATH = OUT_DIR / "w2v.model"


def build_sentences() -> list[list[str]]:
    df = pd.read_parquet(SESSIONS_PARQUET)
    # Skips/previews pollute co-listening adjacency
    df = df[df["ms_played"] >= SKIP_THRESHOLD_MS]
    sentences = (
        df.sort_values("ts")
        .groupby("session_id")["spotify_track_uri"]
        .agg(list)
    )
    sentences = [s for s in sentences if len(s) >= 2]
    return sentences


def train() -> None:
    sentences = build_sentences()
    n_tokens = sum(len(s) for s in sentences)
    print(f"embed: {len(sentences)} sessions, {n_tokens} play tokens")

    model = Word2Vec(sentences=sentences, **W2V_PARAMS)
    model.save(str(MODEL_PATH))
    print(f"embed: learned vectors for {len(model.wv)} tracks (min_count={W2V_PARAMS['min_count']})")

    tracks = pd.read_parquet(TRACK_FEATURES_PARQUET)
    dim = W2V_PARAMS["vector_size"]

    learned_uris = set(model.wv.key_to_index)
    tracks["learned"] = tracks["track_uri"].isin(learned_uris)

    # Centroids for fallbacks, computed from learned vectors only
    learned = tracks[tracks["learned"]]
    artist_centroids = {
        artist: np.mean([model.wv[u] for u in grp["track_uri"]], axis=0)
        for artist, grp in learned.groupby("artist")
    }
    genre_centroids = {
        genre: np.mean([model.wv[u] for u in grp["track_uri"]], axis=0)
        for genre, grp in learned.groupby("genre_primary")
    }
    global_centroid = model.wv.vectors.mean(axis=0)

    vectors = np.zeros((len(tracks), dim), dtype=np.float32)
    sources = []
    for i, row in enumerate(tracks.itertuples()):
        if row.learned:
            vectors[i] = model.wv[row.track_uri]
            sources.append("learned")
        elif row.artist in artist_centroids:
            vectors[i] = artist_centroids[row.artist]
            sources.append("fallback")
        elif row.genre_primary in genre_centroids:
            vectors[i] = genre_centroids[row.genre_primary]
            sources.append("fallback")
        else:
            vectors[i] = global_centroid
            sources.append("fallback")

    np.save(TRACK_VECTORS_NPY, vectors)
    with open(TRACK_VECTOR_URIS_JSON, "w") as f:
        json.dump(
            {"uris": tracks["track_uri"].tolist(), "sources": sources, "dim": dim},
            f,
        )
    n_learned = sources.count("learned")
    print(
        f"embed: wrote {len(tracks)} vectors "
        f"({n_learned} learned, {len(tracks) - n_learned} fallback)"
    )


def check() -> None:
    """Sanity probes: anchor neighbors should be genre/era-coherent, and
    within-artist similarity must dominate random-pair similarity."""
    model = Word2Vec.load(str(MODEL_PATH))
    tracks = pd.read_parquet(TRACK_FEATURES_PARQUET)
    by_uri = tracks.set_index("track_uri")
    learned = tracks[tracks["track_uri"].isin(model.wv.key_to_index)]

    def label(uri: str) -> str:
        if uri in by_uri.index:
            r = by_uri.loc[uri]
            return f"{r['track_name']} — {r['artist']} [{r['genre_primary']}]"
        return uri

    # Anchors: most-played learned track per genre, across the top genres
    anchors = (
        learned.sort_values("play_count", ascending=False)
        .groupby("genre_primary")
        .head(1)
        .sort_values("play_count", ascending=False)
        .head(8)
    )
    for row in anchors.itertuples():
        print(f"\n=== {label(row.track_uri)} (plays={row.play_count}) ===")
        for uri, sim in model.wv.most_similar(row.track_uri, topn=5):
            print(f"  {sim:.3f}  {label(uri)}")

    # Quantitative: within-artist vs random cosine
    rng = np.random.default_rng(42)
    within = []
    for _, grp in learned.groupby("artist"):
        uris = grp["track_uri"].tolist()
        if len(uris) >= 2:
            pairs = min(5, len(uris) - 1)
            for i in range(pairs):
                within.append(model.wv.similarity(uris[i], uris[i + 1]))
    keys = list(model.wv.key_to_index)
    random_pairs = []
    for _ in range(2000):
        a, b = rng.choice(len(keys), 2, replace=False)
        random_pairs.append(model.wv.similarity(keys[a], keys[b]))
    print(f"\nwithin-artist cosine: mean={np.mean(within):.3f} (n={len(within)})")
    print(f"random-pair cosine:   mean={np.mean(random_pairs):.3f} (n={len(random_pairs)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="run sanity probes on the saved model")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        train()
