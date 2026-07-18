#!/usr/bin/env python3
"""One-shot engagement prediction for a Spotify track URL, on the AE embedding.

This is the cold-start use case: score a song with ZERO play history. It mirrors
the lensing listing->predict path, but supplies the vector the stock path can't
(the AE latent isn't a text embedding):

  URL -> track id
      -> fetch intrinsic data: Spotify /tracks + /artists (metadata, multi-genre)
                               ReccoBeats /audio-features (acoustics)
      -> assemble the AE input row (same preprocessing as training, from
         pipeline/artifacts/song_ae_preprocess.json)
      -> AE encoder -> 64-d latent  (the metadata is baked into the latent)
      -> project through the trained dataset's frozen PCA basis
      -> xgboost (the lensing-trained model) -> expm1 -> predicted engagement
      -> + nearest neighbours in AE space (songs in your library it's most like)

Usage:
  predict_spotify_url.py <spotify_url_or_id> --dataset <ds-id> --model <model.ubj>
"""
import argparse
import json
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6337"
AE_COLLECTION = "spotify_tracks_song_ae"
ART = Path("pipeline/artifacts")
UA = {"User-Agent": "Mozilla/5.0 lensing/1.0", "Accept": "application/json"}


# ---- fetch helpers ---------------------------------------------------------
def env(path=".env"):
    e = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); e[k] = v
    return e


def sp_token(cid, csec):
    body = urllib.parse.urlencode({"grant_type": "client_credentials",
                                   "client_id": cid, "client_secret": csec}).encode()
    with urllib.request.urlopen(urllib.request.Request(
            "https://accounts.spotify.com/api/token", data=body), timeout=30) as r:
        return json.load(r)["access_token"]


def sp_get(tok, path):
    req = urllib.request.Request(f"https://api.spotify.com/v1/{path}",
                                 headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def recco_features(sid):
    """ReccoBeats: resolve Spotify id -> uuid, then audio-features."""
    try:
        req = urllib.request.Request(
            f"https://api.reccobeats.com/v1/track?ids={sid}", headers=UA)
        content = json.load(urllib.request.urlopen(req, timeout=30)).get("content", [])
        if not content:
            return {}
        uuid = content[0]["id"]
        req = urllib.request.Request(
            f"https://api.reccobeats.com/v1/audio-features?ids={uuid}", headers=UA)
        feats = json.load(urllib.request.urlopen(req, timeout=30)).get("content", [])
        return feats[0] if feats else {}
    except Exception:
        return {}


def content_doc(m):
    parts = []
    if m.get("track_name"): parts.append(m["track_name"] + ".")
    if m.get("artist"):
        c = m["artist"]
        if (m.get("artist_count") or 1) > 1:
            c += f" (with {int(m['artist_count']) - 1} other artist(s))"
        parts.append(f"Artist: {c}.")
    if m.get("album"):
        bits = [b for b in (m.get("album_type", ""), str(m.get("release_year") or "")) if b]
        parts.append(f"Album: {m['album']}" + (f" ({', '.join(bits)})." if bits else "."))
    if m.get("genre_primary"): parts.append(f"Genre: {m['genre_primary']}.")
    return " ".join(parts) or "Unknown track."


# ---- feature assembly (must match build_song_ae.py) ------------------------
def build_input_row(meta, text_emb, pre):
    z = lambda v, st: (((math.log1p(v) if st.get("log") and v is not None and v >= 0 else v)
                        if v is not None else st["mean"]) - st["mean"]) / st["std"]
    num = [z(meta.get(k), pre["num_stats"][k]) for k in pre["numerics"]]
    ac, flag = [], []
    for k in pre["acoustics"]:
        v = meta.get(k); st = pre["ac_stats"][k]
        present = isinstance(v, (int, float))
        ac.append(((v if present else st["mean"]) - st["mean"]) / st["std"])
        flag.append(0.0 if present else 1.0)
    genres = set(meta.get("sp_genres") or [])
    if meta.get("genre_primary"): genres.add(meta["genre_primary"])
    gmat = [1.0 if g in genres else 0.0 for g in pre["genre_vocab"]]
    amat = [1.0 if meta.get("album_type") == a else 0.0 for a in pre["album_types"]]
    row = np.concatenate([text_emb, np.array(num + ac + flag + gmat + amat, dtype=np.float32)])
    assert row.shape[0] == pre["din"], f"row {row.shape[0]} != din {pre['din']}"
    return row.astype(np.float32)


class Encoder(nn.Module):
    def __init__(self, din, hidden, latent):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, latent))

    def forward(self, x):
        return self.enc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--dataset", required=True, help="ds-id whose PCA basis the model used")
    ap.add_argument("--model", required=True, help="path to trained model.ubj")
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    sid = re.sub(r".*[/:]", "", args.url.split("?")[0])
    e = env()
    tok = sp_token(e["SPOTIFY_CLIENT_ID"], e["SPOTIFY_CLIENT_SECRET"])

    tr = sp_get(tok, f"tracks/{sid}")
    artist_ids = [a["id"] for a in tr.get("artists", []) if a.get("id")]
    arts = sp_get(tok, "artists?ids=" + ",".join(artist_ids[:50])).get("artists", []) if artist_ids else []
    sp_genres = sorted({g for a in arts for g in (a.get("genres") or [])})
    af = recco_features(sid)

    meta = {
        "track_name": tr.get("name"), "album": (tr.get("album") or {}).get("name"),
        "artist": (tr.get("artists") or [{}])[0].get("name"),
        "artist_count": len(tr.get("artists", []) or []),
        "track_popularity": tr.get("popularity"),
        "artist_popularity": arts[0].get("popularity") if arts else None,
        "artist_followers": (arts[0].get("followers") or {}).get("total") if arts else None,
        "album_type": (tr.get("album") or {}).get("album_type"),
        "release_year": int(((tr.get("album") or {}).get("release_date") or "0")[:4] or 0) or None,
        "sp_duration_ms": tr.get("duration_ms"), "sp_explicit": 1 if tr.get("explicit") else 0,
        "sp_track_number": tr.get("track_number"),
        "sp_n_markets": len(tr.get("available_markets", []) or []),
        "sp_genres": sp_genres, "sp_genre_count": len(sp_genres),
        "genre_primary": sp_genres[0] if sp_genres else None,
        "af_danceability": af.get("danceability"), "af_energy": af.get("energy"),
        "af_valence": af.get("valence"), "af_tempo": af.get("tempo"),
        "af_acousticness": af.get("acousticness"), "af_instrumentalness": af.get("instrumentalness"),
        "af_loudness": af.get("loudness"), "af_speechiness": af.get("speechiness"),
        "af_liveness": af.get("liveness"), "af_key": af.get("key"), "af_mode": af.get("mode"),
    }

    pre = json.load(open(ART / "song_ae_preprocess.json"))
    st_model = SentenceTransformer(pre["text_model"])
    text_emb = st_model.encode([content_doc(meta)], normalize_embeddings=True)[0]
    row = build_input_row(meta, text_emb, pre)

    enc = Encoder(pre["din"], pre["hidden"], pre["latent"])
    sd = torch.load(ART / "song_ae.pt", map_location="cpu")
    enc.load_state_dict({k: v for k, v in sd.items() if k.startswith("enc.")})
    enc.eval()
    with torch.no_grad():
        latent = enc(torch.from_numpy(row).unsqueeze(0)).numpy()[0]
    latent = latent / (np.linalg.norm(latent) + 1e-9)

    # project through the dataset's frozen PCA basis, then predict
    man = json.load(open(f"data/datasets/{args.dataset}/manifest.json"))
    mean = np.array(man["pca"]["mean"], dtype=np.float32)
    k, d = man["pca"]["components_shape"]
    comp = np.fromfile(f"data/datasets/{args.dataset}/pca_components.f32", dtype="<f4").reshape(k, d)
    feat = (latent - mean) @ comp.T
    bst = xgb.Booster(); bst.load_model(args.model)
    log_pred = bst.predict(xgb.DMatrix(feat.reshape(1, -1)))[0]
    engagement = math.expm1(log_pred) if man.get("target", {}).get("transform") == "log1p" else log_pred

    # nearest neighbours in AE space
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    hits = client.query_points(AE_COLLECTION, query=latent.tolist(), limit=args.topk + 1,
                               with_payload=["metadata.track_name", "metadata.artist", "metadata.engagement"]).points

    print(f"\n=== {meta['artist']} — {meta['track_name']} ===")
    print(f"  genres: {sp_genres}")
    print(f"  acoustics: energy {meta['af_energy']}, valence {meta['af_valence']}, "
          f"dance {meta['af_danceability']}, tempo {meta['af_tempo']}"
          + ("  (audio features MISSING)" if not af else ""))
    tgt = man.get("target", {}).get("field", "").split(".")[-1]
    if tgt == "rotation":
        print(f"\n  TASTE-FIT — P(enters rotation): {engagement:.0%}")
        print("  (modeled probability you'd return to this track, i.e. play it >=2x;")
        print("   this is the per-track desirability the Pathfinder W_FIT term consumes)")
    else:
        print(f"\n  PREDICTED {tgt.upper() or 'TARGET'}: {engagement:.2f}")
    print(f"\n  nearest songs in your library (AE-embedding cosine):")
    for h in hits:
        m = h.payload["metadata"]
        if m.get("track_name") == meta["track_name"]:
            continue
        print(f"    {h.score:.3f}  {m.get('artist')} — {m.get('track_name')}  "
              f"(actual engagement {m.get('engagement', '?')})")


if __name__ == "__main__":
    main()
