#!/usr/bin/env python3
"""Publish the PUBLIC Spotify client id to the playlist lab.

The lab's save-to-Spotify flow is browser PKCE, which needs the client id (public
by design — the same value the deployed infinite-playlist Worker hands out at
`/api/config`) but NOT the secret. lensing-server serves clients/playlist-lab/public
verbatim, so a generated `spotify.json` reaches the page without adding an endpoint
to the server — which would mean a rebuild and a restart, and restarts kill live
training runs.

The file is gitignored: it is derived from .env, which is where the account lives.

  python3 scripts/lab_spotify_config.py

Then register the lab's redirect URI in the Spotify app dashboard. Spotify accepts
loopback redirects only on an explicit IP literal, so it must be
`http://127.0.0.1:8096/lab/` — not `localhost`.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "clients/playlist-lab/public/spotify.json"
REDIRECT = "http://127.0.0.1:8096/lab/"


def main() -> int:
    env_path = ROOT / ".env"
    if not env_path.exists():
        print(f"no {env_path} — nothing to publish", file=sys.stderr)
        return 1
    env = dict(re.findall(r"^([A-Z_]+)=(.*)$", env_path.read_text(), re.M))
    cid = env.get("SPOTIFY_CLIENT_ID", "").strip()
    if not cid:
        print("SPOTIFY_CLIENT_ID missing from .env", file=sys.stderr)
        return 1
    OUT.write_text(json.dumps({"client_id": cid}, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} (client id {cid[:4]}…{cid[-4:]})")
    print(f"register this redirect URI in the Spotify dashboard: {REDIRECT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
