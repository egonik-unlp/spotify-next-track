#!/usr/bin/env python3
"""Discovery pipeline: embed tracks OUTSIDE the user's library into the app's
PCA-192 retrieval pool so infinite-playlist can surface UNHEARD songs.

Sources external candidates via Spotify search (seeded on the user's own
top genres), dedupes them against the known corpus, fetches Spotify meta +
ReccoBeats acoustics, builds the content-text embedding, and runs the PROVEN
featurizer (reproduces spotify_tracks_song_pca192 at cosine 1.0) to get a
PCA-192 latent per candidate.  Then it APPENDS the discoveries to the app's
model bundle under clients/infinite-playlist/public/model/:

  * catalog.json  - {uri,name,artist,genre,il}; existing 19,402 library rows
                    keep their index (the GRU indexes into them), get il=1;
                    discoveries appended after index 19,401 with il=0.
  * latents.i16   - int16 PCA-192 latents; the whole UNION is re-quantized with
                    scale = max|lat|/32767 and rewritten (old rows byte-order
                    identical in MEANING, only re-quantized).
  * manifest.json - n, scale updated; n_library added; dim/latent/model kept.

MUST run under predictors/.venv (SentenceTransformer model lives there):
  predictors/.venv/bin/python clients/infinite-playlist/tools/build_discovery.py

Re-run / raise the cap:  --cap 3000   (bounded because ReccoBeats is ~1 call/
track).  --market controls the Spotify search market.  Writes ONLY under
clients/infinite-playlist/; never touches data/ or Qdrant.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

import numpy as np

# --- repo paths (this file lives at clients/infinite-playlist/tools/) --------
HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.dirname(HERE)                     # clients/infinite-playlist
REPO = os.path.dirname(os.path.dirname(CLIENT_DIR))    # instance root
MODEL_DIR = os.path.join(CLIENT_DIR, "public", "model")
ART = os.path.join(REPO, "pipeline", "artifacts")
SCRATCH = os.environ.get(
    "DISCOVERY_CACHE",
    "/tmp/claude-1000/-home-gonik-Documents-git-snappler-lensing-workspace-"
    "lensing-instances-spotify-next-track/7aa27ef1-66b6-403d-b4ab-ce6d128782af/"
    "scratchpad/discovery_cache",
)
os.makedirs(SCRATCH, exist_ok=True)

sys.path.insert(0, REPO)
from pipeline.embed_content import content_doc  # noqa: E402
from pipeline.fetch_audio_features import get_json as rb_get_json  # noqa: E402
from pipeline.fetch_spotify_meta import Spotify, load_env, year_of  # noqa: E402

QDRANT_URL = "http://localhost:6337"
CONTENT = "spotify_tracks_content"
API = "https://api.spotify.com/v1"
RB_FEATURES = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "instrumentalness", "loudness", "speechiness", "liveness", "key", "mode",
]
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# --- Spotify search (client-credentials; reuses the corpus Spotify client) ---
class SpotifySearch(Spotify):
    def search(self, q, market, limit=50, offset=0):
        qs = urllib.parse.urlencode(
            {"q": q, "type": "track", "limit": limit, "offset": offset,
             "market": market})
        for _ in range(5):
            req = urllib.request.Request(
                f"{API}/search?{qs}",
                headers={"Authorization": f"Bearer {self.tok}"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    self._auth(); continue
                if e.code == 429:
                    time.sleep(int(e.headers.get("Retry-After", "2")) + 1); continue
                if e.code == 404:
                    return {}
                raise
            except Exception:  # noqa: BLE001
                time.sleep(1); continue
        return {}


# --- featurizer (adapted from proof.py: takes (meta_dict, text_vec_384)) -----
def load_featurizer():
    pp = json.load(open(os.path.join(ART, "song_pca192_preprocess.json")))
    npz = np.load(os.path.join(ART, "song_pca192.npz"))
    components = npz["components"]           # (192, 539)
    mean = npz["mean"]                       # (539,)
    NUMERICS = pp["numerics"]
    LOG_NUMERICS = set(pp["log_numerics"])
    num_stats = pp["num_stats"]
    ACOUSTICS = pp["acoustics"]
    ac_stats = pp["ac_stats"]
    genre_vocab = pp["genre_vocab"]; gidx = {g: i for i, g in enumerate(genre_vocab)}
    album_types = pp["album_types"]; amap = {a: i for i, a in enumerate(album_types)}

    def featurize(m, text):
        text = np.asarray(text, dtype=np.float32)               # 384-d content emb
        # numeric
        nums = []
        for k in NUMERICS:
            v = m.get(k)
            v = float(v) if isinstance(v, (int, float)) else math.nan
            if k in LOG_NUMERICS:
                v = math.log1p(v) if (v == v and v >= 0) else math.nan
            mu, sd = num_stats[k]["mean"], num_stats[k]["std"]
            v = mu if not (v == v) else v
            nums.append((v - mu) / sd)
        numeric = np.asarray(nums, dtype=np.float32)
        # acoustic 11 z + 11 flags
        acz, acf = [], []
        for k in ACOUSTICS:
            v = m.get(k)
            v = float(v) if isinstance(v, (int, float)) else math.nan
            mu, sd = ac_stats[k]["mean"], ac_stats[k]["std"]
            finite = (v == v)
            acz.append(((mu if not finite else v) - mu) / sd)
            acf.append(0.0 if finite else 1.0)
        acoustic = np.asarray(acz + acf, dtype=np.float32)
        # categorical: genre multi-hot (sp_genres + genre_primary) + album_type
        gs = list(m.get("sp_genres") or [])
        gp = m.get("genre_primary")
        if gp:
            gs.append(gp)
        gmat = np.zeros(len(genre_vocab), np.float32)
        for g in gs:
            if g in gidx:
                gmat[gidx[g]] = 1.0
        amat = np.zeros(len(album_types), np.float32)
        at = m.get("album_type") or "__other__"
        if at in amap:
            amat[amap[at]] = 1.0
        categorical = np.concatenate([gmat, amat])
        X = np.concatenate([text, numeric, acoustic, categorical]).astype(np.float32)
        return X

    def project(X):
        Z = (X - mean) @ components.T
        Z = Z / (np.linalg.norm(Z) + 1e-9)
        return Z.astype(np.float32)

    return featurize, project, pp["din"]


# --- map a Spotify /tracks + /artists response to the base meta keys ---------
def map_meta(track, artists_by_id):
    arts = [a for a in (track.get("artists") or []) if a.get("id")]
    first = arts[0] if arts else {}
    first_full = artists_by_id.get(first.get("id"), {})
    genres = sorted({g for a in arts for g in (artists_by_id.get(a.get("id"), {}).get("genres") or [])})
    album = track.get("album") or {}
    followers = ((first_full.get("followers") or {}).get("total"))
    return {
        "track_uri": track.get("uri"),
        "track_name": track.get("name"),
        "artist": (first.get("name") or ""),
        "album": album.get("name"),
        "track_popularity": track.get("popularity"),
        "artist_popularity": first_full.get("popularity"),
        "artist_followers": followers,
        "release_year": year_of(album.get("release_date", "")),
        "artist_count": len(arts),
        "sp_duration_ms": track.get("duration_ms"),
        "sp_n_markets": len(track.get("available_markets") or []),
        "sp_track_number": track.get("track_number"),
        "sp_explicit": 1 if track.get("explicit") else 0,
        "sp_genres": genres,
        "sp_genre_count": len(genres),
        "genre_primary": (genres[0] if genres else "unknown"),
        "album_type": album.get("album_type"),
    }


def cache_load(name):
    p = os.path.join(SCRATCH, name)
    return json.load(open(p)) if os.path.exists(p) else None


def cache_save(name, obj):
    json.dump(obj, open(os.path.join(SCRATCH, name), "w"))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=1500,
                    help="max unheard candidates to embed (ReccoBeats is ~1 call/track)")
    ap.add_argument("--n-genres", type=int, default=25)
    ap.add_argument("--market", default="AR")
    ap.add_argument("--pages", type=int, default=2, help="search pages per query (50/page)")
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    qc = QdrantClient(url=QDRANT_URL, timeout=180)

    # 1) known corpus uris (content collection + GRU catalog) + top genres ----
    known = set()
    gp_c, gs_c = Counter(), Counter()
    off = None
    while True:
        pts, off = qc.scroll(
            CONTENT, limit=2048, offset=off,
            with_payload=["metadata.track_uri", "metadata.genre_primary",
                          "metadata.sp_genres"], with_vectors=False)
        for p in pts:
            m = (p.payload or {}).get("metadata", {}) or {}
            if m.get("track_uri"):
                known.add(m["track_uri"])
            gpv = m.get("genre_primary")
            if gpv and gpv != "unknown":
                gp_c[gpv] += 1
            for g in (m.get("sp_genres") or []):
                gs_c[g] += 1
        if off is None:
            break
    catalog = json.load(open(os.path.join(MODEL_DIR, "catalog.json")))
    n_library = len(catalog)
    for row in catalog:
        known.add(row["uri"])
    print(f"[known] content uris + GRU catalog -> {len(known)} known uris "
          f"({n_library} in library)")

    # seed genres from the user's own taste (primary + multi-genre tags)
    merged = Counter()
    for g, c in gp_c.items():
        merged[g] += c
    for g, c in gs_c.items():
        merged[g] += c
    seed_genres = [g for g, _ in merged.most_common(args.n_genres)]
    print(f"[seed] top {len(seed_genres)} genres: {seed_genres}")

    sp = SpotifySearch(load_env(os.path.join(REPO, ".env"))["SPOTIFY_CLIENT_ID"],
                       load_env(os.path.join(REPO, ".env"))["SPOTIFY_CLIENT_SECRET"])

    # 2) source external candidates via Spotify search -----------------------
    # round-robin across genres so the pool isn't dominated by one genre.
    per_genre = {}          # genre -> ordered list of unheard uris (deduped)
    uri_meta_seed = {}      # uri -> (name, artist) from search (fallback)
    sourced = 0
    for g in seed_genres:
        found = []
        seen_local = set()
        for q in (f'genre:"{g}"', g):
            for page in range(args.pages):
                r = sp.search(q, args.market, limit=50, offset=page * 50)
                items = (r.get("tracks") or {}).get("items") or []
                if not items:
                    break
                for t in items:
                    u = t.get("uri")
                    if not u or u in seen_local:
                        continue
                    seen_local.add(u)
                    sourced += 1
                    if u in known:
                        continue
                    found.append(u)
                    a = t.get("artists") or [{}]
                    uri_meta_seed[u] = (t.get("name"), (a[0].get("name") if a else ""))
                time.sleep(0.1)
        per_genre[g] = found
    # round-robin flatten to cap
    candidates = []
    seen = set()
    idx = 0
    while len(candidates) < args.cap:
        progressed = False
        for g in seed_genres:
            lst = per_genre.get(g, [])
            if idx < len(lst):
                progressed = True
                u = lst[idx]
                if u not in seen:
                    seen.add(u)
                    candidates.append(u)
                    if len(candidates) >= args.cap:
                        break
        idx += 1
        if not progressed:
            break
    print(f"[source] sourced {sourced} search hits -> "
          f"{sum(len(v) for v in per_genre.values())} unheard (pre-cap, w/ dupes) -> "
          f"{len(candidates)} unique unheard candidates (cap {args.cap})")

    # 3) batch-fetch Spotify meta + artists ----------------------------------
    sp_meta_cache = cache_load("spotify_meta.json") or {}
    sids = [u.split(":")[-1] for u in candidates]
    need = [s for s in sids if s not in sp_meta_cache]
    tracks_raw = {}
    artist_ids = set()
    for i in range(0, len(need), 50):
        chunk = need[i:i + 50]
        for t in (sp.get("tracks", chunk).get("tracks") or []):
            if not t:
                continue
            tracks_raw[t["id"]] = t
            for a in (t.get("artists") or []):
                if a.get("id"):
                    artist_ids.add(a["id"])
        time.sleep(0.1)
        print(f"  [spotify tracks] {min(i + 50, len(need))}/{len(need)}", flush=True)
    artists_by_id = {}
    aids = sorted(artist_ids)
    for i in range(0, len(aids), 50):
        chunk = aids[i:i + 50]
        for a in (sp.get("artists", chunk).get("artists") or []):
            if a:
                artists_by_id[a["id"]] = a
        time.sleep(0.1)
        print(f"  [spotify artists] {min(i + 50, len(aids))}/{len(aids)}", flush=True)
    for sid, t in tracks_raw.items():
        sp_meta_cache[sid] = map_meta(t, artists_by_id)
    cache_save("spotify_meta.json", sp_meta_cache)

    # keep only candidates we actually resolved
    metas = []
    for u, s in zip(candidates, sids):
        m = sp_meta_cache.get(s)
        if m and m.get("track_uri"):
            metas.append(m)
    print(f"[spotify] resolved meta for {len(metas)}/{len(candidates)} candidates")

    # 4) ReccoBeats acoustics (best-effort, ~1 call/track batched) -----------
    rb_cache = cache_load("reccobeats.json") or {}
    rb_meta_sids = [m["track_uri"].split(":")[-1] for m in metas]
    need_rb = [s for s in rb_meta_sids if s not in rb_cache]
    BATCH = 40
    # resolve sid -> uuid
    sid_uuid = {}
    for i in range(0, len(need_rb), BATCH):
        chunk = need_rb[i:i + BATCH]
        try:
            content = rb_get_json("track", chunk).get("content", [])
            for t in content:
                href = t.get("href", "")
                sid = href.rstrip("/").split("/")[-1] if href else ""
                if sid and t.get("id"):
                    sid_uuid[sid] = t["id"]
        except Exception as e:  # noqa: BLE001
            print(f"  [rb resolve] batch {i} skip: {e}")
        time.sleep(0.3)
    uuid_sid = {v: k for k, v in sid_uuid.items()}
    uuids = list(uuid_sid)
    for i in range(0, len(uuids), BATCH):
        chunk = uuids[i:i + BATCH]
        try:
            content = rb_get_json("audio-features", chunk).get("content", [])
            for f in content:
                href = f.get("href", "")
                sid = (href.rstrip("/").split("/")[-1] if href else "") or uuid_sid.get(f.get("id"), "")
                if not sid:
                    continue
                vals = {f"af_{k}": f[k] for k in RB_FEATURES if f.get(k) is not None}
                if len(vals) >= 8:
                    rb_cache[sid] = vals
        except Exception as e:  # noqa: BLE001
            print(f"  [rb feats] batch {i} skip: {e}")
        time.sleep(0.3)
        print(f"  [reccobeats] {min(i + BATCH, len(uuids))}/{len(uuids)}", flush=True)
    # mark sids we queried but got nothing for, so cache is honest (no re-query)
    for s in need_rb:
        rb_cache.setdefault(s, {})
    cache_save("reccobeats.json", rb_cache)
    rb_hits = sum(1 for m in metas if rb_cache.get(m["track_uri"].split(":")[-1]))
    print(f"[reccobeats] acoustic coverage {rb_hits}/{len(metas)} "
          f"({(rb_hits/len(metas)*100 if metas else 0):.1f}%)")

    # merge af_* into metas
    for m in metas:
        af = rb_cache.get(m["track_uri"].split(":")[-1]) or {}
        m.update(af)

    # 5) embed content docs (load model ONCE) --------------------------------
    from sentence_transformers import SentenceTransformer
    print(f"[embed] loading {EMB_MODEL} ...")
    model = SentenceTransformer(EMB_MODEL)
    docs = [content_doc(m) for m in metas]
    embs = model.encode(docs, normalize_embeddings=True, batch_size=256,
                        show_progress_bar=False)

    # 6) featurize -> PCA-192 latent -----------------------------------------
    featurize, project, din = load_featurizer()
    new_latents = np.zeros((len(metas), 192), dtype=np.float32)
    for i, m in enumerate(metas):
        X = featurize(m, embs[i])
        assert X.shape[0] == din, (X.shape, din)
        new_latents[i] = project(X)
    print(f"[featurize] embedded {len(metas)} discoveries -> PCA-192 latents")

    # 7) verification: known corpus track through the FULL external path ------
    verify_cos = verify_pipeline(qc, sp, rb_cache, model, featurize, project, args.market)

    # 8) reload existing latents (dequant with current scale) ----------------
    manifest = json.load(open(os.path.join(MODEL_DIR, "manifest.json")))
    old_scale = float(manifest["scale"])
    old_raw = np.fromfile(os.path.join(MODEL_DIR, "latents.i16"), dtype="<i2")
    old_lat = old_raw.reshape(n_library, 192).astype(np.float32) * old_scale
    print(f"[reload] old latents {old_lat.shape} dequantized @ scale {old_scale:.6e}")

    # 9) union + re-quantize + atomic write ----------------------------------
    union = np.concatenate([old_lat, new_latents], axis=0)
    new_scale = float(np.max(np.abs(union)) / 32767.0)
    quant = np.round(union / new_scale).astype("<i2")
    n_total = union.shape[0]

    # build new catalog with il on ALL rows
    new_catalog = []
    for row in catalog:
        r = dict(row); r["il"] = 1; new_catalog.append(r)
    for m in metas:
        new_catalog.append({
            "uri": m["track_uri"],
            "name": m.get("track_name") or "",
            "artist": m.get("artist") or "",
            "genre": m.get("genre_primary") or "unknown",
            "il": 0,
        })
    assert len(new_catalog) == n_total, (len(new_catalog), n_total)

    new_manifest = dict(manifest)
    new_manifest["n"] = n_total
    new_manifest["dim"] = 192
    new_manifest["n_library"] = n_library
    new_manifest["scale"] = new_scale

    # write once, at the end
    quant.tofile(os.path.join(MODEL_DIR, "latents.i16"))
    json.dump(new_catalog, open(os.path.join(MODEL_DIR, "catalog.json"), "w"),
              ensure_ascii=False)
    json.dump(new_manifest, open(os.path.join(MODEL_DIR, "manifest.json"), "w"),
              indent=2, ensure_ascii=False)

    # 10) report -------------------------------------------------------------
    print("\n==================== DISCOVERY REPORT ====================")
    print(f"candidates sourced (search hits):  {sourced}")
    print(f"unheard after dedup vs corpus/cap: {len(candidates)}")
    print(f"successfully embedded:             {len(metas)}")
    print(f"ReccoBeats acoustic coverage:      {rb_hits}/{len(metas)} "
          f"({(rb_hits/len(metas)*100 if metas else 0):.1f}%)")
    print(f"NEW pool size:                     {n_total} "
          f"(n_library {n_library} + n_discovery {len(metas)})")
    print(f"union re-quant scale:              {new_scale:.6e} (was {old_scale:.6e})")
    print(f"featurizer self-consistency cos:   {verify_cos}")
    print("\nsanity: 3 discovery tracks (NOT in corpus):")
    shown = 0
    for m in metas:
        if m["track_uri"] not in known:
            print(f"  - {m.get('track_name')!r} by {m.get('artist')!r} "
                  f"[{m.get('genre_primary')}]  in_corpus={m['track_uri'] in known}")
            shown += 1
            if shown >= 3:
                break
    print("\nwrote:", MODEL_DIR)
    print("=========================================================")


def verify_pipeline(qc, sp, rb_cache, model, featurize, project, market):
    """Embed a KNOWN corpus track through the external fetch path (Spotify +
    ReccoBeats + embed), NOT from the content collection, and report cosine to
    its baked pca192 vector. Won't be 1.0 (different feature provenance)."""
    try:
        pts, _ = qc.scroll(CONTENT, limit=1, with_payload=["metadata.track_uri"],
                           with_vectors=False)
        if not pts:
            return "n/a (no corpus track)"
        uri = ((pts[0].payload or {}).get("metadata", {}) or {}).get("track_uri")
        sid = uri.split(":")[-1]
        pid = pts[0].id
        # fetch via external path
        tr = (sp.get("tracks", [sid]).get("tracks") or [None])[0]
        if not tr:
            return "n/a (spotify miss)"
        aids = [a["id"] for a in (tr.get("artists") or []) if a.get("id")]
        abyid = {a["id"]: a for a in (sp.get("artists", aids).get("artists") or []) if a}
        m = map_meta(tr, abyid)
        # reccobeats (best effort)
        try:
            content = rb_get_json("track", [sid]).get("content", [])
            uuid = next((t["id"] for t in content
                         if (t.get("href", "").rstrip("/").split("/")[-1] == sid)), None)
            if uuid:
                fc = rb_get_json("audio-features", [uuid]).get("content", [])
                for f in fc:
                    vals = {f"af_{k}": f[k] for k in RB_FEATURES if f.get(k) is not None}
                    if len(vals) >= 8:
                        m.update(vals)
        except Exception:  # noqa: BLE001
            pass
        emb = model.encode(content_doc(m), normalize_embeddings=True)
        Z = project(featurize(m, emb))
        baked = np.asarray(
            qc.retrieve("spotify_tracks_song_pca192", ids=[pid], with_vectors=True)[0].vector,
            dtype=np.float32)
        cos = float(Z @ baked / (np.linalg.norm(Z) * np.linalg.norm(baked) + 1e-12))
        return f"{cos:.4f} (track {m.get('track_name')!r})"
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


if __name__ == "__main__":
    main()
