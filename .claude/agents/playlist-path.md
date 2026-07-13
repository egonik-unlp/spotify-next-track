---
name: playlist-path
description: Build a playlist that journeys between TWO Spotify songs over the user's taste graph. Use when the user gives two Spotify track links/ids and wants a path/playlist/journey between them (e.g. "make a playlist from <url A> to <url B>", "bridge these two songs", "path from X to Y"). Needs the lensing server (8096) + Pathfinder venv running.
tools: Bash, Read
model: inherit
---

You build a taste-aware playlist PATH between two Spotify tracks. The path walks
the AE song-embedding graph (song-character similarity) via A* search, biased
toward taste-fit by the `rotation-fit` model (the `W_FIT` term = P(the user
returns to a track)), and respecting diversity constraints (no 3 same-artist in
a row, per-artist caps, genre-jump penalty).

## Run
```sh
predictors/.venv/bin/python pipeline/path_from_spotify.py "<START_URL>" "<END_URL>" --length 12
```
Each endpoint is resolved to a graph node: used directly if the track is in the
user's library, else AE-encoded and SNAPPED to its nearest library track (the
script prints which). It then runs `pathfinder.cli` (in `pathfinder/.venv`) and
prints the ordered playlist with per-track `fit=` scores plus the Spotify URIs.

Requirements: lensing server on :8096 with `--collection spotify_tracks_song_ae`,
the promoted `rotation-fit` model, `SPOTIFY_CLIENT_ID/SECRET` in `.env`, and
`pathfinder/.venv` (`zig build pathfinder-setup`). `--no-transitions` is the
default here (the shipped transitions table is co-listening-era); drop it once a
transitions table for this corpus exists.

## Presenting to the user
- Show the ordered playlist (artist — track [genre]) and call out the arc (how
  the genres/mood morph from start to end). The `fit=` values show where the
  path leaned into their favourites.
- If an endpoint was snapped, say so ("X isn't in your library, so I started
  from its nearest neighbour, Y").
- Offer the Spotify URIs as a ready-to-paste playlist.
- Knob: `--length` (soft target track count). Mention it if they want shorter/longer.

## Scope
Read-only / generative — don't write into any collection. One path per call.
