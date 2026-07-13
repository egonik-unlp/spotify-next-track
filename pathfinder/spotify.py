"""Spotify export: Authorization Code OAuth + playlist creation.

Writing a playlist into the user's account needs user-authorized OAuth (the
`playlist-modify-*` scopes) — the app-only Client Credentials flow can't do it.
This module implements the Authorization Code flow with `requests` (no spotipy
dependency), caches the refresh token to disk, and creates playlists.

Credentials come from the environment (the lensing-server loads them from .env
and the sidecar inherits them):

    SPOTIFY_CLIENT_ID         (required to enable export)
    SPOTIFY_CLIENT_SECRET     (required)
    SPOTIFY_REDIRECT_URI      (optional; defaults to the lensing-server callback)

The redirect URI must be registered verbatim in the Spotify app dashboard.
Spotify's http loopback exemption requires the literal 127.0.0.1 host, which is
exactly what the lensing-server passes us, so the default just works.
"""
import base64
import json
import os
import threading
import time
from pathlib import Path

import requests

CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
# The public URL Spotify redirects to after consent (hits the lensing-server,
# which proxies the callback to this sidecar). Defaults off the server URL the
# sidecar was handed at spawn.
_LENSING_API = os.environ.get("PATHFINDER_LENSING_API", "http://127.0.0.1:8095")
REDIRECT_URI = os.environ.get(
    "SPOTIFY_REDIRECT_URI", f"{_LENSING_API}/api/pathfinder/spotify/callback"
)
SCOPES = "playlist-modify-private playlist-modify-public"

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"

# Refresh token cached next to the package (gitignored).
TOKEN_CACHE = Path(__file__).resolve().parent / ".spotify-token.json"

_lock = threading.Lock()
_pending_states: set[str] = set()


class SpotifyError(Exception):
    """Surfaced to the caller as a 4xx/5xx with a readable message."""


def is_configured() -> bool:
    return bool(CLIENT_ID and CLIENT_SECRET)


def _require_configured() -> None:
    if not is_configured():
        raise SpotifyError(
            "Spotify export is not configured — set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in .env (and register the redirect URI "
            f"{REDIRECT_URI!r} in your Spotify app), then restart the server."
        )


def _basic_auth() -> str:
    raw = f"{CLIENT_ID}:{CLIENT_SECRET}".encode()
    return "Basic " + base64.b64encode(raw).decode()


# ---- catalog search (client-credentials; no user login needed) -----------
_client_tok: dict = {"value": None, "exp": 0.0}


def _client_token() -> str:
    """App-level token for catalog reads (search). Unlike the export flow this
    needs no user authorization — just the app's client credentials."""
    import time
    if _client_tok["value"] and _client_tok["exp"] > time.time() + 30:
        return _client_tok["value"]
    _require_configured()
    resp = requests.post(TOKEN_URL, headers={"Authorization": _basic_auth()},
                         data={"grant_type": "client_credentials"}, timeout=20)
    if resp.status_code != 200:
        raise SpotifyError(f"client-credentials token failed ({resp.status_code}): {resp.text[:200]}")
    j = resp.json()
    _client_tok["value"] = j["access_token"]
    _client_tok["exp"] = time.time() + j.get("expires_in", 3600)
    return _client_tok["value"]


def search_catalog(q: str, limit: int = 12) -> list[dict]:
    """Spotify catalog track search → normalized candidates."""
    if not q.strip():
        return []
    resp = requests.get(f"{API}/search",
                        headers={"Authorization": f"Bearer {_client_token()}"},
                        params={"q": q, "type": "track", "limit": limit}, timeout=20)
    if resp.status_code != 200:
        raise SpotifyError(f"search failed ({resp.status_code}): {resp.text[:200]}")
    out = []
    for t in resp.json().get("tracks", {}).get("items", []):
        if not t:
            continue
        imgs = ((t.get("album") or {}).get("images") or [])
        out.append({
            "spotify_id": t["id"],
            "uri": t.get("uri") or f"spotify:track:{t['id']}",
            "name": t.get("name"),
            "artist": ", ".join(a["name"] for a in t.get("artists", [])),
            "album": (t.get("album") or {}).get("name"),
            "art": (imgs[-1].get("url") if imgs else None),
        })
    return out


# ---- token cache ---------------------------------------------------------

def _load_token() -> dict | None:
    if not TOKEN_CACHE.exists():
        return None
    try:
        with open(TOKEN_CACHE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_token(tok: dict) -> None:
    # Keep the long-lived refresh_token across refreshes that omit it.
    existing = _load_token() or {}
    if "refresh_token" not in tok and "refresh_token" in existing:
        tok["refresh_token"] = existing["refresh_token"]
    tok["expires_at"] = time.time() + float(tok.get("expires_in", 3600))
    tmp = TOKEN_CACHE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(tok, f)
    os.replace(tmp, TOKEN_CACHE)


def _access_token() -> str:
    """A valid access token, refreshing transparently. Raises if not authorized."""
    tok = _load_token()
    if not tok or "refresh_token" not in tok:
        raise SpotifyError("not authorized with Spotify yet — connect your account first")
    if time.time() < float(tok.get("expires_at", 0)) - 60:
        return tok["access_token"]
    # Refresh.
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]},
        headers={"Authorization": _basic_auth()},
        timeout=30,
    )
    if resp.status_code != 200:
        raise SpotifyError(f"token refresh failed ({resp.status_code}): {resp.text[:200]}")
    new = resp.json()
    _save_token(new)
    return new["access_token"]


# ---- OAuth flow ----------------------------------------------------------

def authorize_url(state: str) -> str:
    _require_configured()
    with _lock:
        _pending_states.add(state)
    from urllib.parse import urlencode

    q = urlencode({
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
    })
    return f"{AUTH_URL}?{q}"


def exchange_code(code: str, state: str) -> None:
    _require_configured()
    with _lock:
        known = state in _pending_states
        _pending_states.discard(state)
    if not known:
        raise SpotifyError("OAuth state mismatch — restart the connection from the app")
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
        },
        headers={"Authorization": _basic_auth()},
        timeout=30,
    )
    if resp.status_code != 200:
        raise SpotifyError(f"token exchange failed ({resp.status_code}): {resp.text[:200]}")
    _save_token(resp.json())


def status() -> dict:
    """{configured, authorized, user} — never raises (drives the UI button)."""
    if not is_configured():
        return {"configured": False, "authorized": False, "user": None}
    try:
        token = _access_token()
    except SpotifyError:
        return {"configured": True, "authorized": False, "user": None}
    try:
        me = requests.get(
            f"{API}/me", headers={"Authorization": f"Bearer {token}"}, timeout=30
        )
        if me.status_code == 200:
            d = me.json()
            return {
                "configured": True,
                "authorized": True,
                "user": d.get("display_name") or d.get("id"),
            }
    except requests.RequestException:
        pass
    return {"configured": True, "authorized": False, "user": None}


# ---- playlist creation ---------------------------------------------------

def create_playlist(name: str, uris: list[str], description: str = "") -> dict:
    """Create a private playlist for the authorized user and add `uris`.
    Returns {url, name, n}. Raises SpotifyError on any failure."""
    _require_configured()
    uris = [u for u in uris if u and u.startswith("spotify:track:")]
    if not uris:
        raise SpotifyError("no Spotify track URIs to export")
    token = _access_token()
    auth = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    me = requests.get(f"{API}/me", headers=auth, timeout=30)
    if me.status_code != 200:
        raise SpotifyError(f"could not read Spotify profile ({me.status_code})")
    user_id = me.json()["id"]

    created = requests.post(
        f"{API}/users/{user_id}/playlists",
        headers=auth,
        json={"name": name or "taste path", "public": False,
              "description": description or "Generated by the lensing taste-path pathfinder."},
        timeout=30,
    )
    if created.status_code not in (200, 201):
        raise SpotifyError(f"playlist create failed ({created.status_code}): {created.text[:200]}")
    pl = created.json()
    pid = pl["id"]

    # Add in batches of 100 (Spotify's per-request cap).
    for i in range(0, len(uris), 100):
        batch = uris[i : i + 100]
        add = requests.post(
            f"{API}/playlists/{pid}/tracks", headers=auth, json={"uris": batch}, timeout=30
        )
        if add.status_code not in (200, 201):
            raise SpotifyError(f"adding tracks failed ({add.status_code}): {add.text[:200]}")

    return {
        "url": pl.get("external_urls", {}).get("spotify", f"https://open.spotify.com/playlist/{pid}"),
        "name": pl.get("name", name),
        "n": len(uris),
    }
