#!/usr/bin/env python3
"""Content-embedding pipeline (Option A) for spotify-predict-engagement.

Reads every point from the source collection (the sibling project's behavioral
co-listening corpus), builds a short INTRINSIC-METADATA document per track, and
writes a NEW collection whose vectors are a multilingual sentence embedding of
that document. Same point ids and same payloads — only the vector changes — so
existing dataset recipes work unchanged by pointing `collection` at the new one.

Why: the source vectors encode *how tracks were played together* (behavioral,
target-adjacent). These vectors encode *what the track is* (title, artist,
album, genre, era) — non-behavioral, and additive to the structured one-hots
(it gives the long `artist=__other__` tail real coordinates instead of one
bucket).

Run:  predictors/.venv/bin/python pipeline/embed_content.py
Env:  QDRANT_URL (default http://localhost:6335)
      SRC_COLLECTION (default spotify_tracks)
      DST_COLLECTION (default spotify_tracks_content)
      MODEL (default sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
"""
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6335")
SRC = os.environ.get("SRC_COLLECTION", "spotify_tracks")
DST = os.environ.get("DST_COLLECTION", "spotify_tracks_content")
MODEL = os.environ.get(
    "MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
BATCH = 256


def content_doc(meta: dict) -> str:
    """An intrinsic-metadata document. Deliberately excludes every behavioral /
    target field (play_count, completion_ratio, engagement, popularity,
    followers, curation, platform) — only what the track *is*."""
    track = (meta.get("track_name") or "").strip()
    artist = (meta.get("artist") or "").strip()
    album = (meta.get("album") or "").strip()
    genre = (meta.get("genre_primary") or "").strip()
    album_type = (meta.get("album_type") or "").strip()
    year = meta.get("release_year")
    n_artists = meta.get("artist_count")

    parts = []
    if track:
        parts.append(track + ".")
    if artist:
        credited = f"{artist}"
        if isinstance(n_artists, (int, float)) and n_artists and n_artists > 1:
            credited += f" (with {int(n_artists) - 1} other artist(s))"
        parts.append(f"Artist: {credited}.")
    if album:
        meta_bits = [b for b in (album_type, str(year) if year else "") if b]
        suffix = f" ({', '.join(meta_bits)})" if meta_bits else ""
        parts.append(f"Album: {album}{suffix}.")
    if genre:
        parts.append(f"Genre: {genre}.")
    return " ".join(parts) if parts else "Unknown track."


def main() -> None:
    client = QdrantClient(url=QDRANT_URL, timeout=120)
    print(f"loading model {MODEL} ...")
    model = SentenceTransformer(MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"embedding dim = {dim}")

    # Scroll the whole source collection (payload only; we recompute vectors).
    ids, payloads, docs = [], [], []
    offset = None
    while True:
        points, offset = client.scroll(
            SRC, limit=2048, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            meta = (p.payload or {}).get("metadata", {}) or {}
            ids.append(p.id)
            payloads.append(p.payload)
            docs.append(content_doc(meta))
        if offset is None:
            break
    print(f"read {len(ids)} points from {SRC!r}")
    print("sample doc:", repr(docs[0]))

    # (Re)create the destination collection with the model's dim + cosine.
    client.recreate_collection(
        DST, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    print(f"(re)created collection {DST!r} (size={dim}, cosine)")

    # Embed + upsert in batches.
    n = len(ids)
    for start in range(0, n, BATCH):
        chunk = docs[start : start + BATCH]
        vecs = model.encode(
            chunk, normalize_embeddings=True, show_progress_bar=False
        )
        client.upsert(
            DST,
            points=[
                PointStruct(id=ids[start + i], vector=vecs[i].tolist(),
                            payload=payloads[start + i])
                for i in range(len(chunk))
            ],
        )
        print(f"  upserted {min(start + BATCH, n)}/{n}", flush=True)

    info = client.get_collection(DST)
    print(f"done: {DST!r} now has {info.points_count} points, dim {dim}")


if __name__ == "__main__":
    main()
