"""Taste-fit (rotation) scores from the promoted lensing model.

Batch-scores point ids via POST /api/models/<name>/predict and min-max
normalizes to [0,1]. Degrades gracefully: if the lensing server or model is
unavailable, every track scores 0.5 and the pathfinder effectively paths on
embedding distance + diversity alone.
"""
import requests

from .config import LENSING_API, MODEL_NAME

BATCH = 512


def fetch_scores(point_ids: list[int]) -> dict[int, float]:
    raw: dict[int, float] = {}
    try:
        for start in range(0, len(point_ids), BATCH):
            batch = point_ids[start : start + BATCH]
            resp = requests.post(
                f"{LENSING_API}/api/models/{MODEL_NAME}/predict",
                json={"point_ids": batch},
                timeout=120,
            )
            resp.raise_for_status()
            for p in resp.json()["predictions"]:
                raw[int(p["row_id"])] = float(p["predicted"])
    except (requests.RequestException, KeyError) as e:
        print(f"warning: habit-fit scoring unavailable ({e}); using neutral scores")
        return {pid: 0.5 for pid in point_ids}

    # The rotation-fit model predicts a calibrated P(enters rotation), already
    # in [0,1] and well-spread, so no log compression is needed (the old
    # engagement target was heavy-tailed and required it). Min-max across the
    # scored set to use the full [0,1] fit range.
    lo, hi = min(raw.values()), max(raw.values())
    span = (hi - lo) or 1.0
    return {pid: (v - lo) / span for pid, v in raw.items()}
