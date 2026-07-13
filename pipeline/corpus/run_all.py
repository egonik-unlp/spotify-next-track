"""Run the full pipeline in order. Idempotent — every stage rewrites its output.

End to end:
    ingest    raw endsong export        -> master_df.csv
    enrich    Spotify API metadata      -> track_data_block_1.json, data.json  (incremental)
    clean / sessionize / aggregates / embed / transitions / ids
    upsert    -> Qdrant "spotify_tracks"

To inject newer Spotify data: replace the export under `extended-2026/` (or
pass --source DIR / set LENSING_CORPUS_SOURCE), then run this. ingest rebuilds
master_df.csv, enrich fetches only the new tracks/artists, and the rest
recomputes. After it finishes run the AE chain and rebuild the lensing dataset
(see pipeline/refresh_corpus.sh and CLAUDE.md "Refreshing data").

Usage:
    python -m pipeline.corpus.run_all [--source DIR] [--skip-enrich] [--skip-upsert]
"""
import argparse

from . import aggregates, clean, embed, enrich, ids, ingest, sessionize, transitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="export dir for ingest (default: extended-2026/ or $LENSING_CORPUS_SOURCE)")
    parser.add_argument("--skip-enrich", action="store_true", help="skip the Spotify API metadata refresh (reuse caches)")
    parser.add_argument("--skip-upsert", action="store_true", help="stop before writing to Qdrant")
    args = parser.parse_args()

    ingest.run(source=args.source)
    if not args.skip_enrich:
        enrich.run()
    clean.run()
    sessionize.run()
    aggregates.run()
    embed.train()
    transitions.run()  # needs sessions + per-track genre/artist from aggregates
    ids.run()
    if not args.skip_upsert:
        from . import upsert  # deferred: requires qdrant-client + a reachable Qdrant

        upsert.run()


if __name__ == "__main__":
    main()
