"""Split the play stream into listening sessions on a 30-minute inactivity gap."""
import pandas as pd

from .config import PLAYS_PARQUET, SESSIONS_PARQUET, SESSION_GAP_MINUTES


def run() -> pd.DataFrame:
    df = pd.read_parquet(PLAYS_PARQUET).sort_values("ts").reset_index(drop=True)

    gap = df["ts"].diff() > pd.Timedelta(minutes=SESSION_GAP_MINUTES)
    df["session_id"] = gap.cumsum().astype("int64")

    df.to_parquet(SESSIONS_PARQUET, index=False)
    n_sessions = df["session_id"].nunique()
    plays_per = len(df) / n_sessions
    print(f"sessionize: {len(df)} plays -> {n_sessions} sessions (mean {plays_per:.1f} plays/session)")
    return df


if __name__ == "__main__":
    run()
