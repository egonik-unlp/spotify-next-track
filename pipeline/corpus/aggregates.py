"""Per-track aggregates + artist genre/popularity join.

Target for lensing: engagement = play_count * completion_ratio (raw value;
lensing applies the log1p transform declared in domain.toml).
"""
import json
from collections import Counter

import pandas as pd

from .config import (
    ARTIST_JSON,
    LIBRARY_JSON,
    ONREPEAT_JSON,
    PLAYLISTS_JSON,
    SESSIONS_PARQUET,
    TRACK_DATA_JSON,
    TRACK_FEATURES_PARQUET,
)


def coarse_platform(platform: str) -> str:
    p = str(platform).lower()
    if "android" in p:
        return "android"
    if "windows" in p:
        return "windows"
    if "linux" in p:
        return "linux"
    if "ios" in p or "iphone" in p or "ipad" in p:
        return "ios"
    if "os x" in p or "mac" in p:
        return "macos"
    if "web" in p:
        return "web"
    return "other"


def load_track_durations() -> dict[str, dict]:
    """uri -> track metadata from the /v1/tracks dumps.

    Carries duration/popularity/explicit (already used downstream) plus the
    previously-unused release_year, album_type and artist_count.
    """
    with open(TRACK_DATA_JSON) as f:
        blocks = json.load(f)
    out = {}
    for block in blocks.values():
        for t in block.get("tracks", []):
            if t is None:
                continue
            album = t.get("album") or {}
            release_date = album.get("release_date") or ""
            release_year = int(release_date[:4]) if release_date[:4].isdigit() else None
            out[t["uri"]] = {
                "duration_ms": t["duration_ms"],
                "track_popularity": t.get("popularity"),
                "explicit": t.get("explicit", False),
                "release_year": release_year,
                "album_type": album.get("album_type"),
                "artist_count": len(t.get("artists") or []) or None,
            }
    return out


def load_saved_uris() -> set[str]:
    """Set of track URIs the user has saved (YourLibrary 'Liked Songs' + saves)."""
    with open(LIBRARY_JSON) as f:
        lib = json.load(f)
    saved = {t["uri"] for t in lib.get("tracks", []) if t.get("uri")}
    print(f"aggregates: library: {len(saved)} saved tracks")
    return saved


def load_playlist_counts() -> Counter:
    """track URI -> number of the user's own playlists that contain it."""
    with open(PLAYLISTS_JSON) as f:
        data = json.load(f)
    counts: Counter = Counter()
    for pl in data.get("playlists", []):
        for item in pl.get("items", []):
            track = item.get("track") or {}
            uri = track.get("trackUri")
            if uri:
                counts[uri] += 1
    print(f"aggregates: playlists: {len(counts)} distinct tracks across user playlists")
    return counts


def load_onrepeat_counts() -> Counter:
    """track URI -> number of On-Repeat snapshots the track appears in.

    On-Repeat is Spotify's own most-replayed surface; behaviorally close to a
    play_count proxy, so treat the resulting feature with leakage suspicion.
    """
    with open(ONREPEAT_JSON) as f:
        snapshots = json.load(f)
    counts: Counter = Counter()
    for snap in snapshots:
        for uri in snap.get("message_contents", []):
            counts[uri] += 1
    print(f"aggregates: on-repeat: {len(counts)} distinct tracks across {len(snapshots)} snapshots")
    return counts


def load_artist_info() -> dict[str, dict]:
    """artist name (as queried) -> {genres, popularity, followers} from best search match."""
    with open(ARTIST_JSON) as f:
        data = json.load(f)
    out = {}
    name_mismatches = 0
    for query_name, result in data.items():
        items = result.get("artists", {}).get("items") or []
        if not items:
            continue
        best = items[0]
        if best["name"].lower() != query_name.lower():
            name_mismatches += 1
        out[query_name] = {
            "genres": best.get("genres") or [],
            "artist_popularity": best.get("popularity"),
            "artist_followers": (best.get("followers") or {}).get("total"),
        }
    print(f"aggregates: artist join table: {len(out)} artists ({name_mismatches} fuzzy name matches)")
    return out


def run() -> pd.DataFrame:
    plays = pd.read_parquet(SESSIONS_PARQUET)
    durations = load_track_durations()
    artists = load_artist_info()
    saved = load_saved_uris()
    playlist_counts = load_playlist_counts()
    onrepeat_counts = load_onrepeat_counts()

    plays["duration_ms"] = plays["spotify_track_uri"].map(
        lambda u: durations.get(u, {}).get("duration_ms")
    )
    # Fallback duration: longest observed play of the track
    max_played = plays.groupby("spotify_track_uri")["ms_played"].transform("max")
    plays["duration_ms"] = plays["duration_ms"].fillna(max_played).clip(lower=1)
    plays["completion"] = (plays["ms_played"] / plays["duration_ms"]).clip(upper=1.0)
    plays["platform_coarse"] = plays["platform"].map(coarse_platform)

    g = plays.groupby("spotify_track_uri")
    tracks = pd.DataFrame({
        "track_name": g["master_metadata_track_name"].first(),
        "artist": g["master_metadata_album_artist_name"].first(),
        "album": g["master_metadata_album_album_name"].first(),
        "play_count": g.size(),
        "total_ms": g["ms_played"].sum(),
        "completion_ratio": g["completion"].mean(),
        "skip_rate": g["is_skip"].mean(),
        "shuffle_rate": g["shuffle"].mean(),
        "distinct_sessions": g["session_id"].nunique(),
        "first_played": g["ts"].min(),
        "last_played": g["ts"].max(),
        "top_platform": g["platform_coarse"].agg(lambda s: s.mode().iloc[0]),
    }).reset_index(names="track_uri")

    tracks["engagement"] = tracks["play_count"] * tracks["completion_ratio"]

    # rotation = "did the user come back to this track" (play_count >= 2). This
    # is the default lensing target (domain.toml [target] field = "rotation",
    # task = "binary"); the server reads it raw from the payload, so it must be
    # materialized here rather than derived at dataset-build time.
    tracks["rotation"] = (tracks["play_count"] >= 2).astype(float)

    # Track-level API metadata
    tracks["track_popularity"] = tracks["track_uri"].map(
        lambda u: durations.get(u, {}).get("track_popularity")
    )
    tracks["release_year"] = tracks["track_uri"].map(
        lambda u: durations.get(u, {}).get("release_year")
    )
    tracks["album_type"] = tracks["track_uri"].map(
        lambda u: durations.get(u, {}).get("album_type") or "unknown"
    )
    tracks["artist_count"] = tracks["track_uri"].map(
        lambda u: durations.get(u, {}).get("artist_count")
    )

    # Deliberate-curation signals (default-on features; on_repeat_count is the
    # most entangled with the target — see domain.toml leakage note).
    tracks["is_saved"] = tracks["track_uri"].isin(saved).astype(int)
    tracks["in_playlist_count"] = (
        tracks["track_uri"].map(playlist_counts).fillna(0).astype(int)
    )
    tracks["on_repeat_count"] = (
        tracks["track_uri"].map(onrepeat_counts).fillna(0).astype(int)
    )

    # Artist-level join
    def artist_field(name, field, default):
        return artists.get(name, {}).get(field) or default

    tracks["genre_primary"] = tracks["artist"].map(
        lambda a: (artist_field(a, "genres", []) or ["unknown"])[0]
    )
    tracks["genres"] = tracks["artist"].map(lambda a: artist_field(a, "genres", []))
    tracks["artist_popularity"] = tracks["artist"].map(
        lambda a: artist_field(a, "artist_popularity", None)
    )
    tracks["artist_followers"] = tracks["artist"].map(
        lambda a: artist_field(a, "artist_followers", None)
    )

    unmatched = (tracks["genre_primary"] == "unknown").mean()
    no_duration = tracks["track_uri"].map(lambda u: u not in durations).mean()

    tracks.to_parquet(TRACK_FEATURES_PARQUET, index=False)
    no_release = tracks["release_year"].isna().mean()
    print(
        f"aggregates: {len(tracks)} tracks "
        f"(genre unknown: {unmatched:.1%}, duration fallback: {no_duration:.1%}, "
        f"release_year missing: {no_release:.1%}); "
        f"engagement p50={tracks['engagement'].median():.2f} "
        f"p95={tracks['engagement'].quantile(0.95):.1f} max={tracks['engagement'].max():.0f}; "
        f"rotation pos_rate={tracks['rotation'].mean():.3f}"
    )
    print(
        f"aggregates: curation — saved={int(tracks['is_saved'].sum())}, "
        f"in≥1 playlist={int((tracks['in_playlist_count'] > 0).sum())}, "
        f"on-repeat={int((tracks['on_repeat_count'] > 0).sum())}"
    )
    return tracks


if __name__ == "__main__":
    run()
