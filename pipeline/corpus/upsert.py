"""Assemble Qdrant points (co-listening vector + lensing-shaped payload) and upsert.

Payload layout matches lensing's domain config: metadata_root="metadata",
content_field="content". The 200-dim vector is authoritative; `content` is a
human-readable display string only.
"""
import json
import math

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import (
    EMBEDDING_DIM,
    ID_MAP_JSON,
    QDRANT_COLLECTION,
    QDRANT_URL,
    TRACK_FEATURES_PARQUET,
    TRACK_VECTOR_URIS_JSON,
    TRACK_VECTORS_NPY,
    UPSERT_BATCH_SIZE,
)


def none_if_nan(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def int_or_none(v):
    """Coerce a possibly-NaN numeric (float64 column with gaps) to a clean int."""
    v = none_if_nan(v)
    return None if v is None else int(v)


def build_points() -> list[PointStruct]:
    tracks = pd.read_parquet(TRACK_FEATURES_PARQUET)
    vectors = np.load(TRACK_VECTORS_NPY)
    with open(TRACK_VECTOR_URIS_JSON) as f:
        vec_meta = json.load(f)
    with open(ID_MAP_JSON) as f:
        uri_to_id = json.load(f)["uri_to_id"]

    assert vec_meta["uris"] == tracks["track_uri"].tolist(), "vector/feature row order mismatch"
    assert vectors.shape == (len(tracks), EMBEDDING_DIM)

    points = []
    for i, row in enumerate(tracks.itertuples()):
        genre = row.genre_primary
        content = f"{row.track_name} — {row.artist} — {row.album} [{genre}]"
        metadata = {
            "engagement": round(float(row.engagement), 4),
            "rotation": float(row.rotation),  # default binary target (play_count >= 2)
            "play_count": int(row.play_count),
            "completion_ratio": round(float(row.completion_ratio), 4),
            "skip_rate": round(float(row.skip_rate), 4),
            "shuffle_rate": round(float(row.shuffle_rate), 4),
            "distinct_sessions": int(row.distinct_sessions),
            "artist": row.artist,
            "album": row.album,
            "genre_primary": genre,
            "artist_popularity": none_if_nan(row.artist_popularity),
            "track_popularity": none_if_nan(row.track_popularity),
            "artist_followers": int_or_none(row.artist_followers),
            "release_year": int_or_none(row.release_year),
            "album_type": row.album_type,
            "artist_count": int_or_none(row.artist_count),
            "is_saved": int(row.is_saved),
            "in_playlist_count": int(row.in_playlist_count),
            "on_repeat_count": int(row.on_repeat_count),
            "top_platform": row.top_platform,
            "embedding_source": vec_meta["sources"][i],
            "track_name": row.track_name,
            "track_uri": row.track_uri,
            "first_played": row.first_played.isoformat(),
            "last_played": row.last_played.isoformat(),
        }
        # Qdrant payloads should omit nulls rather than carry JSON null
        metadata = {k: v for k, v in metadata.items() if v is not None}
        points.append(
            PointStruct(
                id=uri_to_id[row.track_uri],
                vector=vectors[i].tolist(),
                payload={"content": content, "metadata": metadata},
            )
        )
    return points


def run() -> None:
    client = QdrantClient(url=QDRANT_URL)
    points = build_points()

    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"upsert: created collection {QDRANT_COLLECTION!r}")

    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(QDRANT_COLLECTION, points[start : start + UPSERT_BATCH_SIZE], wait=True)

    count = client.count(QDRANT_COLLECTION, exact=True).count
    print(f"upsert: collection {QDRANT_COLLECTION!r} now holds {count} points")
    assert count == len(points), f"expected {len(points)} points, found {count}"


if __name__ == "__main__":
    run()
