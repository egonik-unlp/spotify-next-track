"""Clean master_df.csv: drop episodes and duplicates, parse timestamps, flag skips."""
import pandas as pd

from .config import MASTER_CSV, OUT_DIR, PLAYS_PARQUET, SKIP_THRESHOLD_MS

KEEP_COLUMNS = [
    "ts", "spotify_track_uri", "master_metadata_track_name",
    "master_metadata_album_artist_name", "master_metadata_album_album_name",
    "ms_played", "reason_start", "reason_end", "skipped", "shuffle", "platform",
]


def run() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    n_raw = len(df)

    # Episode (podcast) rows have no track URI
    df = df[df["spotify_track_uri"].notna()]
    n_episodes = n_raw - len(df)

    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()

    df = df[KEEP_COLUMNS].copy()
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df["ms_played"] = df["ms_played"].astype("int64")

    # `skipped` is NaN for ~91% of rows, so the robust skip signal is the
    # 30s stream threshold; explicit skipped==True is folded in where present.
    df["is_skip"] = (df["ms_played"] < SKIP_THRESHOLD_MS) | (df["skipped"] == True)  # noqa: E712

    df = df.sort_values("ts").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PLAYS_PARQUET, index=False)
    print(
        f"clean: {n_raw} raw rows -> {len(df)} plays "
        f"(dropped {n_episodes} episode rows, {n_dupes} duplicates); "
        f"{int(df['is_skip'].sum())} skips ({df['is_skip'].mean():.1%})"
    )
    return df


if __name__ == "__main__":
    run()
