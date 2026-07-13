"""Stable u64 point IDs: first 8 bytes of SHA-256(track URI), little-endian.

Deterministic across reruns, no counter to sync. Collision probability over
~12k tracks is ~4e-12, but we assert distinctness anyway — a silent collision
would overwrite a track in Qdrant.
"""
import hashlib
import json

import pandas as pd

from .config import ID_MAP_JSON, TRACK_FEATURES_PARQUET


def uri_to_id(uri: str) -> int:
    return int.from_bytes(hashlib.sha256(uri.encode()).digest()[:8], "little")


def run() -> dict[str, int]:
    tracks = pd.read_parquet(TRACK_FEATURES_PARQUET)
    uris = tracks["track_uri"].tolist()
    uri_to = {u: uri_to_id(u) for u in uris}

    assert len(set(uri_to.values())) == len(uri_to), "u64 ID collision — aborting"

    with open(ID_MAP_JSON, "w") as f:
        json.dump(
            {
                "uri_to_id": uri_to,
                "id_to_uri": {str(i): u for u, i in uri_to.items()},
            },
            f,
        )
    print(f"ids: {len(uri_to)} distinct u64 IDs, no collisions")
    return uri_to


if __name__ == "__main__":
    run()
