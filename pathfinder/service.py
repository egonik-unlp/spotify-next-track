"""API-only pathfinder service, spawned as a sidecar by lensing-server.

    python -m pathfinder.service        # binds 127.0.0.1:$PATHFINDER_PORT

Exposes the same JSON contract the original standalone web UI used, minus the
embedded HTML page — the lensing React UI (ui/src/views/PathfinderView.tsx)
is the front end now, reaching these endpoints through the lensing-server
proxy at /api/pathfinder/*.

    GET  /health                       -> {"ready": bool, "tracks": int}
    GET  /api/search?q=<query>         -> [track, ...]
    GET  /api/path?start=&end=&length=&context=&shuffle=
                                       -> {"tracks": [track, ...], "context": str|null}
    GET  /api/spotify/status           -> {configured, authorized, user}
    GET  /api/spotify/login            -> {authorize_url}
    GET  /api/spotify/callback?code=&state=   -> HTML (closes the popup)
    POST /api/spotify/export {name, uris}     -> {url, name, n}

The graph loads once before the socket is bound, so a successful connection
means the service is ready; habit-fit scores and the transition model load
lazily on the first /api/path request.
"""
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import spotify
from . import transitions as transitions_mod
from .config import MODEL_NAME, SERVICE_PORT
from .graph import TrackGraph, load_graph, nearest_learned
from .score import fetch_scores
from .search import densify, find_path

_graph: TrackGraph | None = None
_scores: dict[int, float] | None = None
_model: transitions_mod.TransitionModel | None = None
_model_loaded = False
_lock = threading.Lock()


def graph() -> TrackGraph:
    global _graph
    with _lock:
        if _graph is None:
            print("loading graph from Qdrant...", flush=True)
            _graph = load_graph()
            print(f"graph ready: {len(_graph.ids)} tracks, {len(_graph.learned_ids)} learned", flush=True)
        return _graph


def scores() -> dict[int, float]:
    global _scores
    g = graph()
    with _lock:
        if _scores is None:
            print(f"fetching habit-fit scores from model {MODEL_NAME!r}...", flush=True)
            _scores = fetch_scores(sorted(g.learned_ids))
        return _scores


def transition_model() -> transitions_mod.TransitionModel | None:
    global _model, _model_loaded
    with _lock:
        if not _model_loaded:
            _model = transitions_mod.load()
            print(
                "transition model loaded" if _model is not None
                else "no transitions.json — next-track + context steering disabled",
                flush=True,
            )
            _model_loaded = True
        return _model


def track_json(g: TrackGraph, pid: int, fit: float | None = None) -> dict:
    m = g.meta[pid]
    return {
        "id": str(pid),  # u64s overflow JS Number — ship as strings
        "name": m["track_name"],
        "artist": m["artist"],
        "genre": m.get("genre_primary", "unknown"),
        "plays": m.get("play_count", 0),
        "learned": m.get("embedding_source") == "learned",
        "uri": m["track_uri"],
        "spotify_url": "https://open.spotify.com/track/" + m["track_uri"].rsplit(":", 1)[-1],
        "fit": fit,
    }


def api_search(q: str, limit: int = 12) -> list[dict]:
    g = graph()
    needle = q.strip().lower()
    if not needle:
        return []
    hits = [
        (pid, m) for pid, m in g.meta.items()
        if needle in f"{m['track_name']} {m['artist']}".lower()
    ]
    hits.sort(key=lambda x: -x[1].get("play_count", 0))
    return [track_json(g, pid) for pid, _ in hits[:limit]]


def api_spotify_search(q: str, limit: int = 12) -> list[dict]:
    """Spotify CATALOG search (any track), each annotated with whether it's in
    the user's library (and its corpus id, usable directly as a path endpoint)."""
    cands = spotify.search_catalog(q, limit)
    g = graph()
    uri2pid = {m["track_uri"]: pid for pid, m in g.meta.items()}
    for c in cands:
        pid = uri2pid.get(c["uri"])
        c["in_library"] = pid is not None
        c["id"] = str(pid) if pid is not None else None
        c["plays"] = g.meta[pid].get("play_count", 0) if pid is not None else 0
        c["spotify_url"] = "https://open.spotify.com/track/" + c["spotify_id"]
    return cands


def api_path(start_id: int, end_id: int, length: int,
             context: str = "now", shuffle: str = "any") -> dict:
    g = graph()
    s = scores()
    model = transition_model()
    ctx = None
    if model is not None:
        tod = None if context == "any" else context
        shuf = {"any": None, "shuffle": True, "linear": False}.get(shuffle)
        ctx = model.resolve_context(tod=tod, shuffle=shuf)
    start = nearest_learned(g, start_id)
    end = nearest_learned(g, end_id)
    path = find_path(g, start, end, s, length_hint=length, model=model, ctx=ctx)
    if path is None:
        return {"error": "no path found between these tracks"}
    if len(path) < length:
        path = densify(g, path, s, length, model=model, ctx=ctx)
    if start_id != start:
        path = [start_id] + path
    if end_id != end:
        path = path + [end_id]
    return {
        "tracks": [track_json(g, pid, s.get(pid)) for pid in path],
        "context": ctx.label if ctx is not None else None,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, body: bytes, code: int, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), code, "application/json")

    def _html(self, html: str, code: int = 200) -> None:
        self._send(html.encode(), code, "text/html; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/health":
                self._json({"ready": _graph is not None,
                            "tracks": len(_graph.ids) if _graph else 0})
            elif url.path == "/api/search":
                self._json(api_search(q.get("q", [""])[0]))
            elif url.path == "/api/spotify/search":
                self._json(api_spotify_search(q.get("q", [""])[0]))
            elif url.path == "/api/path":
                self._json(api_path(
                    int(q["start"][0]), int(q["end"][0]),
                    max(2, min(50, int(q.get("length", ["12"])[0]))),
                    context=q.get("context", ["now"])[0],
                    shuffle=q.get("shuffle", ["any"])[0],
                ))
            elif url.path == "/api/spotify/status":
                self._json(spotify.status())
            elif url.path == "/api/spotify/login":
                state = secrets.token_urlsafe(24)
                self._json({"authorize_url": spotify.authorize_url(state)})
            elif url.path == "/api/spotify/callback":
                self._html(self._spotify_callback(q))
            else:
                self._json({"error": "not found"}, 404)
        except spotify.SpotifyError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # surface errors to the caller, don't kill the service
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/spotify/export":
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = spotify.create_playlist(
                    name=payload.get("name", ""),
                    uris=payload.get("uris", []),
                    description=payload.get("description", ""),
                )
                self._json(result)
            else:
                self._json({"error": "not found"}, 404)
        except spotify.SpotifyError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _spotify_callback(self, q: dict) -> str:
        """Exchange the auth code, then return a tiny page that notifies the
        opener window and closes itself. Same-origin with the app (both served
        by the lensing-server), so postMessage + close work."""
        err = q.get("error", [None])[0]
        if err:
            return _CALLBACK_PAGE.format(ok="false", msg=f"Spotify denied access: {err}")
        code = q.get("code", [None])[0]
        state = q.get("state", [""])[0]
        if not code:
            return _CALLBACK_PAGE.format(ok="false", msg="no authorization code returned")
        try:
            spotify.exchange_code(code, state)
        except spotify.SpotifyError as e:
            return _CALLBACK_PAGE.format(ok="false", msg=str(e))
        return _CALLBACK_PAGE.format(ok="true", msg="Connected to Spotify — you can close this tab.")


_CALLBACK_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Spotify</title><style>
body{{font:15px system-ui,sans-serif;background:#0e0f13;color:#e8e9ed;
display:grid;place-items:center;height:100vh;margin:0}}
.box{{text-align:center;max-width:24rem;padding:2rem}}
.dot{{color:#1db954;font-size:2rem}}</style></head>
<body><div class="box"><div class="dot">●</div><p>{msg}</p></div>
<script>
  try {{ if (window.opener) window.opener.postMessage(
    {{ type: 'spotify-auth', ok: {ok} }}, '*'); }} catch (e) {{}}
  setTimeout(function(){{ window.close(); }}, {ok} ? 1200 : 6000);
</script></body></html>"""


def main() -> None:
    graph()  # load before accepting requests: a live socket means ready
    server = ThreadingHTTPServer(("127.0.0.1", SERVICE_PORT), Handler)
    print(f"pathfinder service: http://127.0.0.1:{SERVICE_PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
