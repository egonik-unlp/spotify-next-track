#!/usr/bin/env python3
"""Place a track that is NOT in the corpus into the model's own latent space.

The lab drops seed tracks it cannot resolve (`seq_common.resolve_prefix`), so a
pasted playlist of 52 songs with one library track generates from that one track.
The infinite-playlist product instead cold-starts the strangers — but it must do so
in a browser, through `projector.bin`: a bge-m3 → MLP *surrogate* trained to imitate
the real projection, because a Worker cannot run MiniLM plus a 539×192 basis.

The server has no such constraint, so it can run the REAL projection:

    Spotify /tracks + /artists   (metadata, multi-genre)
    ReccoBeats /audio-features   (acoustics; best-effort, missing-flagged)
    content_doc -> MiniLM 384    (pipeline/embed_content.py's exact document)
        -> the 539-d leak-safe feature row (pipeline/build_song_pca.py's blocks)
        -> FROZEN song_pca192 basis  -> the 192-d latent the models were trained on

Nothing here is fitted: every statistic (z-score means/stds, the 120-genre vocab,
album types, block spans, the text model name) comes out of the stored preprocess
artifact, and the projection is the same components/mean the corpus collection was
built with. The output is therefore in-space by construction rather than by
approximation.

MEASURED FIDELITY (`verify`): round-tripping corpus tracks — discarding everything
the corpus knows, rebuilding from the live APIs, projecting — reproduces the trained
latent at cosine ~0.90 (0.83–0.95 over 8 tracks), against 0.003 for a random corpus
pair. The residual is real-world drift (popularity/followers move between corpus
ingest and now) plus tracks ReccoBeats has no acoustics for, not projection error:
the basis carries EVR 0.992.

  predictors/.venv/bin/python predictors/seq_coldstart.py verify [--n 8]
  predictors/.venv/bin/python predictors/seq_coldstart.py show <url|uri|id> ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "pipeline/artifacts"
DEFAULT_BASIS = "song_pca192"
UA = {"User-Agent": "lensing-seq-coldstart/1.0", "Accept": "application/json"}
SPOTIFY_API = "https://api.spotify.com/v1"
RECCOBEATS_API = "https://api.reccobeats.com/v1"
# Spotify caps ids per request; ReccoBeats is undocumented but 40 is what the
# product's worker settled on.
SP_BATCH, RB_BATCH = 50, 40


class ColdStartUnavailable(RuntimeError):
    """Raised when the cold-start chain cannot run at all (no creds, no basis, no
    text model). The caller falls back to dropping unknown seeds — degraded, but
    never a failed journey."""


@dataclass
class ColdTrack:
    """One cold-started track: where it sits, and what it is."""
    uri: str
    latent: np.ndarray                  # (192,) L2-normalized, model latent space
    name: str | None = None
    artist: str | None = None
    genre: str | None = None
    meta: dict = field(default_factory=dict)     # the raw assembled metadata
    acoustics: dict = field(default_factory=dict)
    doc: str = ""

    @property
    def mood(self) -> dict:
        """The per-stop mood fields, in `seq_extend.MOOD_FIELDS` terms. Available
        for a cold track without any corpus lookup — these come from the same
        Spotify/ReccoBeats fetch the latent was built from."""
        af = self.acoustics
        return {
            "energy": af.get("af_energy"), "valence": af.get("af_valence"),
            "tempo": af.get("af_tempo"), "danceability": af.get("af_danceability"),
            "acousticness": af.get("af_acousticness"),
            "year": self.meta.get("release_year"),
        }


# --------------------------------------------------------------------------- #
# The frozen basis + every preprocessing statistic                             #
# --------------------------------------------------------------------------- #
class Basis:
    """`song_pca<N>.npz` + `song_pca<N>_preprocess.json`, loaded once."""

    def __init__(self, name: str = DEFAULT_BASIS, artifacts: Path = ARTIFACTS):
        npz_path, pre_path = artifacts / f"{name}.npz", artifacts / f"{name}_preprocess.json"
        if not npz_path.exists() or not pre_path.exists():
            raise ColdStartUnavailable(
                f"no frozen basis at {npz_path} / {pre_path} — cold start needs the "
                f"projection the dataset's latents were built with")
        z = np.load(npz_path)
        self.components = np.asarray(z["components"], dtype=np.float64)   # (L, din)
        self.mean = np.asarray(z["mean"], dtype=np.float64)               # (din,)
        pre = json.loads(pre_path.read_text())
        self.pre = pre
        self.numerics: list = pre["numerics"]
        self.log_numerics = set(pre.get("log_numerics") or [])
        self.num_stats: dict = pre["num_stats"]
        self.acoustics: list = pre["acoustics"]
        self.ac_stats: dict = pre["ac_stats"]
        self.genre_vocab: list = pre["genre_vocab"]
        self.album_types: list = pre["album_types"]
        self.text_model: str = pre["text_model"]
        self.din: int = int(pre["din"])
        self.latent_dim: int = int(pre["latent"])
        self.evr: float = float(pre.get("cumulative_evr", float("nan")))
        self.name = name

    def project(self, x: np.ndarray) -> np.ndarray:
        """539-d feature row → L2-normalized latent, exactly as the collection was
        built (PCA transform, then the builder's L2 normalization)."""
        if x.shape[-1] != self.din:
            raise ValueError(f"feature row is {x.shape[-1]}-d, basis expects {self.din}")
        z = (x.astype(np.float64) - self.mean) @ self.components.T
        n = np.linalg.norm(z, axis=-1, keepdims=True)
        return np.asarray(z / np.where(n > 0, n, 1.0), dtype=np.float32)


_TEXT_MODEL_CACHE: dict = {}


def text_encoder(model_name: str):
    """The MiniLM the corpus was embedded with, from the local HF cache. Lazy: a
    lab run that never cold-starts must not pay for loading it."""
    if model_name in _TEXT_MODEL_CACHE:
        return _TEXT_MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:                                    # pragma: no cover
        raise ColdStartUnavailable(f"sentence-transformers not installed: {exc}") from exc
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:                                      # noqa: BLE001
        raise ColdStartUnavailable(
            f"could not load the text model {model_name!r} ({exc}) — it must be in "
            f"the local HF cache; cold start does not download at request time") from exc
    _TEXT_MODEL_CACHE[model_name] = model
    return model


# --------------------------------------------------------------------------- #
# Fetch: Spotify metadata + ReccoBeats acoustics                               #
# --------------------------------------------------------------------------- #
def _env(path: Path | None = None) -> dict:
    out = {}
    p = path or (ROOT / ".env")
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _get_json(url: str, headers: dict | None = None, data: bytes | None = None,
              timeout: int = 45):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def spotify_token() -> str:
    """Client-credentials token. Same creds seq_extend uses to expand playlists."""
    env = _env()
    cid = os.environ.get("SPOTIFY_CLIENT_ID") or env.get("SPOTIFY_CLIENT_ID")
    sec = os.environ.get("SPOTIFY_CLIENT_SECRET") or env.get("SPOTIFY_CLIENT_SECRET")
    if not cid or not sec:
        raise ColdStartUnavailable(
            "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET missing (.env or environment) "
            "— cold start needs them to read the track's metadata")
    body = urllib.parse.urlencode({"grant_type": "client_credentials",
                                   "client_id": cid, "client_secret": sec}).encode()
    return _get_json("https://accounts.spotify.com/api/token", data=body)["access_token"]


def bare_id(token: str) -> str | None:
    """`spotify:track:<id>` / an open.spotify.com URL / a bare id → the id."""
    t = str(token).split("?", 1)[0].rstrip("/")
    t = t.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return t if re.fullmatch(r"[A-Za-z0-9]{22}", t) else None


def fetch_metadata(ids: list[str], token: str) -> tuple[dict, dict]:
    """(track by id, artist by id) — batched, the artists in one follow-up call."""
    tracks: dict = {}
    for i in range(0, len(ids), SP_BATCH):
        chunk = ids[i:i + SP_BATCH]
        got = _get_json(f"{SPOTIFY_API}/tracks?ids={','.join(chunk)}",
                        {"Authorization": f"Bearer {token}"})
        for t in got.get("tracks") or []:
            if t and t.get("id"):
                tracks[t["id"]] = t
    artist_ids = sorted({a["id"] for t in tracks.values()
                         for a in (t.get("artists") or []) if a.get("id")})
    artists: dict = {}
    for i in range(0, len(artist_ids), SP_BATCH):
        chunk = artist_ids[i:i + SP_BATCH]
        got = _get_json(f"{SPOTIFY_API}/artists?ids={','.join(chunk)}",
                        {"Authorization": f"Bearer {token}"})
        for a in got.get("artists") or []:
            if a and a.get("id"):
                artists[a["id"]] = a
    return tracks, artists


def fetch_acoustics(ids: list[str]) -> dict:
    """ReccoBeats af_* by Spotify id. BEST EFFORT: the acoustic block carries its
    own missing-flags in the feature row, so a track with no acoustics still gets a
    valid (slightly blurrier) latent rather than none at all."""
    out: dict = {}
    sid_uuid: dict = {}
    for i in range(0, len(ids), RB_BATCH):
        try:
            got = _get_json(f"{RECCOBEATS_API}/track?ids={','.join(ids[i:i + RB_BATCH])}")
        except Exception:                                          # noqa: BLE001
            continue
        for t in got.get("content") or []:
            sid = (t.get("href") or "").rstrip("/").rsplit("/", 1)[-1]
            if sid and t.get("id"):
                sid_uuid[sid] = t["id"]
    uuids = list(sid_uuid.values())
    uuid_sid = {v: k for k, v in sid_uuid.items()}
    for i in range(0, len(uuids), RB_BATCH):
        try:
            got = _get_json(f"{RECCOBEATS_API}/audio-features?ids={','.join(uuids[i:i + RB_BATCH])}")
        except Exception:                                          # noqa: BLE001
            continue
        for f in got.get("content") or []:
            sid = ((f.get("href") or "").rstrip("/").rsplit("/", 1)[-1]
                   or uuid_sid.get(f.get("id")))
            if not sid:
                continue
            out[sid] = {f"af_{k}": v for k, v in f.items()
                        if isinstance(v, (int, float))}
    return out


# --------------------------------------------------------------------------- #
# Assemble: the leak-safe 539-d row                                            #
# --------------------------------------------------------------------------- #
def build_meta(track: dict, artists: dict) -> dict:
    """The payload shape `build_song_pca.py` reads, from a Spotify track object.
    Mirrors pipeline/fetch_spotify_meta.py's mapping."""
    arts = [a for a in (track.get("artists") or []) if a.get("id")]
    first = artists.get(arts[0]["id"], {}) if arts else {}
    genres = sorted({g for a in arts for g in (artists.get(a["id"], {}).get("genres") or [])})
    album = track.get("album") or {}
    released = album.get("release_date") or ""
    return {
        "track_uri": f"spotify:track:{track.get('id')}",
        "track_name": track.get("name"),
        "album": album.get("name"),
        "artist": arts[0].get("name") if arts else None,
        "artist_count": len(track.get("artists") or []),
        "track_popularity": track.get("popularity"),
        "artist_popularity": first.get("popularity"),
        "artist_followers": (first.get("followers") or {}).get("total"),
        "album_type": album.get("album_type"),
        "release_year": int(released[:4]) if re.match(r"^\d{4}", released) else None,
        "sp_duration_ms": track.get("duration_ms"),
        "sp_explicit": 1 if track.get("explicit") else 0,
        "sp_track_number": track.get("track_number"),
        "sp_n_markets": len(track.get("available_markets") or []),
        "sp_genres": genres,
        "sp_genre_count": len(genres),
        "genre_primary": genres[0] if genres else "unknown",
    }


def content_doc(meta: dict) -> str:
    """VERBATIM from pipeline/embed_content.py. If this drifts, the text block lands
    off-distribution and every cold latent is quietly wrong — so it is duplicated
    deliberately and must be changed in lockstep."""
    track = (meta.get("track_name") or "").strip()
    artist = (meta.get("artist") or "").strip()
    album = (meta.get("album") or "").strip()
    genre = (meta.get("genre_primary") or "").strip()
    album_type = (meta.get("album_type") or "").strip()
    year = meta.get("release_year")
    n_artists = meta.get("artist_count")

    parts = []
    if track:
        parts.append(track + ".")
    if artist:
        credited = artist
        if isinstance(n_artists, (int, float)) and n_artists and n_artists > 1:
            credited += f" (with {int(n_artists) - 1} other artist(s))"
        parts.append(f"Artist: {credited}.")
    if album:
        meta_bits = [b for b in (album_type, str(year) if year else "") if b]
        suffix = f" ({', '.join(meta_bits)})" if meta_bits else ""
        parts.append(f"Album: {album}{suffix}.")
    if genre:
        parts.append(f"Genre: {genre}.")
    return " ".join(parts) or "Unknown track."


def feature_row(meta: dict, text_vec: np.ndarray, acoustics: dict, basis: Basis) -> np.ndarray:
    """text 384 ⊕ numeric 10 (z, log1p on two) ⊕ acoustic 11+11 flags ⊕ genre 120
    multi-hot ⊕ album_type 3 — block for block as build_song_pca.py assembles it,
    using the STORED statistics so a single row standardizes like the corpus did."""
    nums = []
    for k in basis.numerics:
        s = basis.num_stats[k]
        v = meta.get(k)
        v = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else math.nan
        if s.get("log"):
            v = math.log1p(v) if v >= 0 else math.nan
        if not math.isfinite(v):
            v = s["mean"]                       # the builder's nan_to_num(mean)
        nums.append((v - s["mean"]) / s["std"])

    ac_vals, ac_flags = [], []
    for k in basis.acoustics:
        s = basis.ac_stats[k]
        v = acoustics.get(k)
        v = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else math.nan
        ok = math.isfinite(v)
        ac_flags.append(0.0 if ok else 1.0)
        ac_vals.append(((v if ok else s["mean"]) - s["mean"]) / s["std"])

    genres = set(meta.get("sp_genres") or [])
    if meta.get("genre_primary"):
        genres.add(meta["genre_primary"])
    gvec = [1.0 if g in genres else 0.0 for g in basis.genre_vocab]
    album_type = meta.get("album_type") or "__other__"
    avec = [1.0 if a == album_type else 0.0 for a in basis.album_types]

    row = np.concatenate([
        np.asarray(text_vec, dtype=np.float64),
        np.asarray(nums), np.asarray(ac_vals), np.asarray(ac_flags),
        np.asarray(gvec), np.asarray(avec),
    ])
    if row.shape[0] != basis.din:
        raise ValueError(f"assembled {row.shape[0]}-d row, basis expects {basis.din} "
                         f"— the feature spec has drifted from the stored artifact")
    return row


# --------------------------------------------------------------------------- #
# The entry point the lab calls                                                #
# --------------------------------------------------------------------------- #
def cold_start(tokens: list, *, basis_name: str = DEFAULT_BASIS,
               artifacts: Path = ARTIFACTS, emit=None) -> dict[str, ColdTrack]:
    """Cold-start every resolvable token. Returns {track_uri: ColdTrack}.

    Tokens that Spotify does not know, or that are not track references at all, are
    simply absent from the result — the caller reports them as still-unknown. Raises
    ColdStartUnavailable only when the chain cannot run for ANY token (missing creds,
    basis or text model), which is the one case the caller must fall back on.
    """
    def say(msg: str) -> None:
        if emit:
            emit(msg)

    ids, seen = [], set()
    for tok in tokens:
        sid = bare_id(tok)
        if sid and sid not in seen:
            seen.add(sid)
            ids.append(sid)
    if not ids:
        return {}

    b = Basis(basis_name, artifacts)
    encoder = text_encoder(b.text_model)          # raises if unavailable, before any I/O
    say(f"cold start: {len(ids)} unknown track(s) → {b.name} ({b.latent_dim}-d)")

    token = spotify_token()
    tracks, artists = fetch_metadata(ids, token)
    say(f"cold start: Spotify returned {len(tracks)}/{len(ids)} track(s)")
    acoustics = fetch_acoustics([i for i in ids if i in tracks])
    say(f"cold start: acoustics for {len(acoustics)}/{len(tracks)} (missing is flagged, not fatal)")

    metas = {sid: build_meta(t, artists) for sid, t in tracks.items()}
    order = [sid for sid in ids if sid in metas]
    if not order:
        return {}
    vecs = encoder.encode([content_doc(metas[sid]) for sid in order],
                          normalize_embeddings=False, show_progress_bar=False)

    out: dict[str, ColdTrack] = {}
    for sid, tvec in zip(order, vecs):
        m = metas[sid]
        af = acoustics.get(sid, {})
        latent = b.project(feature_row(m, tvec, af, b))
        out[m["track_uri"]] = ColdTrack(
            uri=m["track_uri"], latent=latent, name=m.get("track_name"),
            artist=m.get("artist"), genre=m.get("genre_primary"),
            meta=m, acoustics=af, doc=content_doc(m))
    say(f"cold start: placed {len(out)} track(s) in the model's latent space")
    return out


# --------------------------------------------------------------------------- #
# verify — the fidelity gate                                                   #
# --------------------------------------------------------------------------- #
def verify(dataset: Path, n: int, basis_name: str = DEFAULT_BASIS) -> int:
    """Round-trip tracks that ARE in the dataset: discard what the corpus knows,
    rebuild from the live APIs, project, and compare with the trained latent.

    This is the claim the whole module rests on, so it is checkable rather than
    asserted. Reported against the random-corpus-pair cosine, because "0.90" only
    means something next to what an unrelated track scores."""
    manifest = json.loads((dataset / "sequence-manifest.json").read_text())
    items = json.loads((dataset / "items.json").read_text())
    n_items, dim = int(manifest["n_items"]), int(manifest["latent_dim"])
    lat = np.fromfile(dataset / "item_latents.f32", dtype=np.float32).reshape(n_items, dim)
    rows = items if isinstance(items, list) else [items[str(i)] for i in range(n_items)]

    rng = np.random.default_rng(7)
    pick = [int(i) for i in rng.choice(n_items, size=n * 3, replace=False)
            if str(rows[i].get("uri") or "").startswith("spotify:track:")][:n]
    cold = cold_start([rows[i]["uri"] for i in pick], basis_name=basis_name,
                      emit=lambda m: print(f"  {m}"))
    if not cold:
        print("verify: nothing cold-started (no creds / no network?)", file=sys.stderr)
        return 1

    print(f"\n{'track':44} {'cos(cold, trained)':>19} {'acoustics':>10}")
    print("-" * 78)
    cosines = []
    for i in pick:
        ct = cold.get(rows[i]["uri"])
        if ct is None:
            continue
        a = ct.latent / (np.linalg.norm(ct.latent) or 1.0)
        b = lat[i] / (np.linalg.norm(lat[i]) or 1.0)
        c = float(a @ b)
        cosines.append(c)
        label = f"{(ct.name or '?')[:25]} — {(ct.artist or '?')[:15]}"
        print(f"{label:44} {c:19.4f} {'yes' if ct.acoustics else 'MISSING':>10}")
    if not cosines:
        print("verify: no overlap between the picks and the cold-start result", file=sys.stderr)
        return 1

    cos = np.asarray(cosines)
    pairs = rng.choice(n_items, size=(400, 2))
    norms = np.linalg.norm(lat, axis=1)
    norms[norms == 0] = 1.0
    base = np.einsum("ij,ij->i", lat[pairs[:, 0]], lat[pairs[:, 1]]) / (
        norms[pairs[:, 0]] * norms[pairs[:, 1]])
    print("-" * 78)
    print(f"n={cos.size}  mean {cos.mean():.4f}  median {np.median(cos):.4f}  "
          f"min {cos.min():.4f}  max {cos.max():.4f}")
    print(f"random corpus pair: mean {base.mean():.4f}  p95 {np.quantile(base, 0.95):.4f}")
    # A cold latent must be far closer to its own trained latent than an unrelated
    # track is — that is the whole claim, and it fails loudly if the chain drifts.
    ok = cos.mean() >= 0.75 and cos.min() >= 0.6
    print("GATE PASS" if ok else "GATE FAIL (mean < 0.75 or min < 0.6)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="round-trip corpus tracks through the chain")
    v.add_argument("--dataset", type=Path,
                   default=ROOT / "data/datasets/seq-20260715-131139")
    v.add_argument("--n", type=int, default=8)
    v.add_argument("--basis", default=DEFAULT_BASIS)
    s = sub.add_parser("show", help="cold-start tracks and print where they land")
    s.add_argument("tokens", nargs="+")
    s.add_argument("--basis", default=DEFAULT_BASIS)
    args = ap.parse_args()

    if args.cmd == "verify":
        return verify(args.dataset, args.n, args.basis)
    cold = cold_start(args.tokens, basis_name=args.basis, emit=lambda m: print(f"  {m}"))
    for uri, ct in cold.items():
        print(f"\n{uri}\n  {ct.name} — {ct.artist} · {ct.genre}")
        print(f"  doc: {ct.doc}")
        print(f"  latent[:6]: {np.round(ct.latent[:6], 4).tolist()}  |v| {np.linalg.norm(ct.latent):.4f}")
        print(f"  acoustics: {'yes' if ct.acoustics else 'MISSING (flagged)'}")
    return 0 if cold else 1


if __name__ == "__main__":
    raise SystemExit(main())
