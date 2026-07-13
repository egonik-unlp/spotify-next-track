"""Build master_df.csv from the raw Spotify "Extended streaming history" export.

Concatenates every play-history JSON in the export directory into one
row-per-stream CSV — the canonical input the rest of the pipeline (`clean`
onward) reads.

Both export filename shapes are accepted (see config.ENDSONG_GLOBS): the older
`endsong_*.json` and the newer `Streaming_History_Audio_*.json` — same rich
schema (ts, ms_played, track URI, reason_start/reason_end, shuffle, platform,
…), just renamed across Spotify export vintages. The lean "Account data"
StreamingHistory*.json is NOT a play source — it lacks the track URI and
play-reason fields the embeddings, target and transitions need.

To inject newer data: replace the export under `extended-2026/` (a fresh
export's extended-history folder holds the COMPLETE history), point at one
elsewhere with `--source /path/to/export`, or set LENSING_CORPUS_SOURCE.
Idempotent: every run rebuilds master_df.csv from whatever files it finds.

Usage: python -m pipeline.corpus.ingest [--source DIR]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ENDSONG_GLOBS, EXTENDED_DIR, MASTER_CSV

# Episode/video rows carry no track URI and are dropped downstream anyway; the
# globs exclude endvideo.json / Streaming_History_Video_*.json.


def _find_files(source: Path) -> list[Path]:
    seen: dict[str, Path] = {}
    for pattern in ENDSONG_GLOBS:
        for f in source.glob(pattern):
            seen[f.name] = f
    return [seen[k] for k in sorted(seen)]


def run(source: Path | None = None) -> pd.DataFrame:
    source = Path(source) if source else EXTENDED_DIR
    files = _find_files(source)
    if not files:
        raise FileNotFoundError(
            f"no {ENDSONG_GLOBS!r} files in {source} — point --source at an "
            f"unzipped Spotify 'Extended streaming history' export"
        )

    records: list[dict] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            records.extend(json.load(fh))

    df = pd.DataFrame.from_records(records)
    n_raw = len(df)

    # Spotify re-ships the full history across exports, so concatenating
    # overlapping export sets can duplicate rows — drop exact dupes here; the
    # play-level dedup in clean.py is a second line of defense.
    df = df.drop_duplicates().reset_index(drop=True)

    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_CSV, index=False)
    span = f"{df['ts'].min().date()} – {df['ts'].max().date()}"
    print(
        f"ingest: {len(files)} history files -> {n_raw} rows "
        f"({n_raw - len(df)} exact dupes dropped) -> {len(df)} plays "
        f"spanning {span}; wrote {MASTER_CSV.name}"
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="dir holding endsong_*.json / Streaming_History_Audio_*.json "
        "(default: extended-2026/ or $LENSING_CORPUS_SOURCE)",
    )
    args = parser.parse_args()
    run(source=args.source)
