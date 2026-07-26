#!/usr/bin/env python3
"""Autoregressive SESSION EXTENDER — the playlist-lab generation backend.

Turns any promoted sequence model into an *infinite playlist generator*: given a
seed prefix, repeatedly score the whole vocab, pick a next track under a SHARED
retrieval policy, append it to the prefix, and repeat. One process generates the
whole journey, which is the entire point — `POST /api/models/{name}/predict`
re-loads the baked artifact on every call (~3s), so N single-step HTTP calls cost
N×3s while this costs 3s + N×(a full-vocab matvec).

WHY THIS EXISTS SEPARATELY FROM THE PREDICTORS

Every next-track predictor already exposes the same contract — a
`score_fn(prefix) -> full-vocab score vector`. This module does NOT reimplement
any model: it dispatches to each predictor's OWN scorer (see DISPATCH below) and
owns only the two things a fair comparison requires:

  1. **One retrieval policy for every model.** Whether a journey feels good
     depends heavily on the non-model knobs (artist penalty, mood anchor, MMR,
     sampling temperature). If each front-end reimplemented them — as the
     `clients/infinite-playlist` app.js does in JavaScript — then comparing two
     models would silently compare two policies. Here the policy is one
     parameterized code path, so models compete under identical rules.
  2. **Journey-level diagnostics + INTENT.** The same facets the crown is built
     from (mood_coh / ild / artist_adj / music_rel, see
     `seq_common.eval_from_scores`), computed over the GENERATED journey instead
     of a held-out test split — so a listener can check whether the metric
     agrees with their ears.

INTENT ("the shape of the next track", beyond naming it) is derived from the
SCORE DISTRIBUTION, not from the model's internals. That is deliberate: it makes
the readout work identically for a GRU, a blend and a Markov baseline, none of
which share an internal representation. Per step we report, over the top-M
candidates: the softmax score mass aggregated BY GENRE and BY ARTIST (what kind
of thing the model wants next, and how concentrated), the top-1 margin, and the
distribution entropy (is the model committed or wandering?). Models that expose
a predicted next-LATENT additionally get that latent's nearest genre prototypes
in the musical-distance space — the model's target *point* rather than its
ranked candidates.

CLI:

  seq_extend.py extend --model <promoted_model_dir> --predictor <registry name>
                       --output <json> [--params <json|path>]

`--predictor` is required because a promoted model dir does not record which
predictor produced it; the server knows and passes it through.

Run under the shared predictor venv (torch 2.12.0+cpu / numpy)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np

from seq_common import (
    _load_music_vectors,
    _relevance_ids,
    emit,
    load_artifact,
    resolve_prefix,
)

# EVERY policy/intent knob below is expressed in Z-SCORE UNITS of the model's own
# score distribution, never in raw score units. This is load-bearing for a
# comparison tool: raw score scales differ wildly by family (a cosine scorer
# lives in [-1,1] with vocab std ~0.1; a blend sums z-normalized legs and spreads
# several times wider), so an absolute `artist_penalty` that de-duplicates a GRU
# is negligible against a blend, and an absolute intent temperature that resolves
# a GRU's candidates saturates to entropy 0 on a blend. Measured before the fix:
# the champion blend returned 5 distinct artists in 12 stops under the same
# `artist_penalty=0.5` that gave the GRU 12/12, and its intent read entropy 0.0 /
# artist_conc 1.0 at every step. Standardizing first makes one policy mean one
# thing for all models — otherwise the lab compares policies, not algorithms.
DEFAULTS = {
    "seed": [],             # ordered prefix tokens (URIs / ids / vocab indices)
    "steps": 30,            # how many tracks to generate
    # --- retrieval policy (shared by every model; units = pool-local score z) ---
    "policy_pool": 200,     # candidates defining the local scale (see local_z)
    "artist_penalty": 2.0,  # z subtracted from candidates whose artist was used
    "artist_window": 0,     # 0 = penalize any already-used artist; N = last N only
    # HARD constraints. The soft penalty above is in score-z units and is
    # therefore scale-fragile: with the mood anchor on, an artist sitting at the
    # seed centroid outscores the penalty and the journey locks. Measured on a
    # 93-track playlist seed: 8 consecutive Floating Points tracks at
    # artist_penalty 2.0. A cooldown cannot be outscored, which is what a
    # listenable playlist actually needs; the penalty then shapes the choice
    # *within* what the cooldown allows.
    "artist_cooldown": 5,   # an artist cannot repeat within N stops (0 = off)
    "dedupe_titles": True,  # drop a (title, artist) already played — the vocab
                            # holds the same recording under several releases, so
                            # next-distinct-by-index alone replays songs
                            # ("Del Oro" twice in 8 stops)
    "anchor_lambda": 1.0,   # weight on z(cos(cand, seed-core centroid))
    "anchor_min_core": 5,   # anchor only when the seed core has >= this many rows
    "temperature": 0.0,     # 0 = greedy argmax; >0 = sample (z units)
    "sample_top_k": 20,     # candidate pool sampled from when temperature > 0
    "seed_rng": 1337,       # sampling RNG seed (reproducible journeys)
    # --- intent readout ---
    "intent_top_m": 50,     # candidates summarized in the per-step intent
    "intent_tau": 0.5,      # softmax temperature over z for intent score mass
    "intent_genres": 5,     # genres reported per step
}


def local_z(v: np.ndarray, allowed: np.ndarray, pool_n: int = 200) -> np.ndarray:
    """Standardize a score vector against its PLAUSIBLE-CANDIDATE pool, so one
    policy unit means the same displacement for every model.

    Whole-vocab standardization is not enough. Measured on one step from the same
    seed (top-200 spread in vocab-z / top1-top2 gap in vocab-z):

        GRU cosine scorer : 0.263 / 0.12
        champion z-blend  : 0.634 / 1.30      (2.4x and 11x wider)

    The blend is far peakier *locally*, so a penalty sized in vocab-z units barely
    moves its leader while flattening the GRU's shortlist — which is why the blend
    kept repeating one artist under the same `artist_penalty` that gave the GRU 12
    distinct artists. Centering on the pool MEDIAN and scaling by the pool STD
    makes "1.0" mean "one plausible-candidate spread" everywhere. Median rather
    than mean because these tails are heavy (Markov bigram spikes)."""
    v = np.asarray(v, dtype=np.float64)
    idx = np.nonzero(allowed)[0]
    if idx.size == 0:
        return np.zeros_like(v)
    top = idx[np.argsort(-v[idx])[:max(2, int(pool_n))]]
    centre = float(np.median(v[top]))
    scale = float(np.std(v[top]))
    return (v - centre) / scale if scale > 1e-12 else (v - centre)

# Which module + entry points serve each registry predictor. Each pair is
# (loader, scorer-builder); the builder's signature differs per family so the
# dispatch below adapts rather than forcing a false uniformity.
DISPATCH = ("seq-nexttrack", "seq-dualgru", "seq-ann", "seq-blend",
            "seq-markov", "seq-popularity", "seq-recency")


# --------------------------------------------------------------------------- #
# Seed expansion — accept a PLAYLIST / ALBUM, not just individual tracks       #
# --------------------------------------------------------------------------- #
# The lab's real workflow (same as the infinite-playlist product) is "paste a
# playlist, hear where my library goes": more seed tracks = more context for the
# model. `seq_common.resolve_prefix` only understands single TRACKS, so a pasted
# playlist URL resolved to nothing at all. We expand container URLs here, before
# resolution, via Spotify's client-credentials API (stdlib only — this must run
# under predictors/.venv, which has no `requests`).
#
# Out-of-vocab tracks are NOT a problem to solve here: the vocab IS the user's
# library, so a seed track the model has never seen is simply dropped and
# reported in `seed.unknown`. Seeding with anything and journeying through the
# library is the intended shape, not a degradation.
SPOTIFY_API = "https://api.spotify.com/v1"
_CONTAINER = ("playlist", "album")

# Minimum members for a genre's mean latent to be used as a PROTOTYPE when
# naming a direction (the eigen axis poles, the predicted-latent `aim`).
# Measured on the champion split: 866 genres, MEDIAN SIZE 3 — so a ≥3 floor
# names directions off 3-track averages and the extremes of a few hundred such
# prototypes are pure sampling noise (it produced poles like
# "rap politico ←→ drone" from an ambient/idm seed). ≥15 keeps 200 genres,
# which is more resolution than a readout needs, with stable means.
GENRE_PROTO_MIN = 15


def _spotify_creds() -> tuple[str, str]:
    """Client id/secret from the environment, falling back to the repo `.env`.

    The predictor is spawned by lensing-server, which may not have sourced
    `.env` (it does not need Spotify itself), so relying on the inherited
    environment alone would make playlist seeding fail depending on how the
    server was started."""
    cid = os.environ.get("SPOTIFY_CLIENT_ID")
    sec = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if cid and sec:
        return cid, sec
    for base in (Path.cwd(), *Path(__file__).resolve().parents):
        env = base / ".env"
        if not env.is_file():
            continue
        vals = {}
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip("'\"")
        cid = cid or vals.get("SPOTIFY_CLIENT_ID")
        sec = sec or vals.get("SPOTIFY_CLIENT_SECRET")
        if cid and sec:
            return cid, sec
    raise SystemExit(
        "seed expansion needs Spotify credentials: set SPOTIFY_CLIENT_ID and "
        "SPOTIFY_CLIENT_SECRET in the environment or in the repo .env "
        "(individual track seeds work without them)")


def _spotify_token() -> str:
    import base64
    import urllib.request
    cid, sec = _spotify_creds()
    basic = base64.b64encode(f"{cid}:{sec}".encode()).decode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=b"grant_type=client_credentials",
        headers={"Authorization": f"Basic {basic}",
                 "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())["access_token"]


def _spotify_get(path: str, token: str) -> dict:
    import urllib.request
    req = urllib.request.Request(
        path if path.startswith("http") else f"{SPOTIFY_API}{path}",
        headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def parse_container(token: str) -> tuple[str, str] | None:
    """Recognize a playlist/album URL or URI → (kind, id), else None."""
    t = str(token).strip()
    for kind in _CONTAINER:
        if t.startswith(f"spotify:{kind}:"):
            return kind, t.split(":")[-1]
        for host in ("https://open.spotify.com/", "http://open.spotify.com/"):
            if t.startswith(f"{host}{kind}/"):
                rest = t[len(host) + len(kind) + 1:]
                return kind, rest.split("?")[0].split("/")[0]
    return None


def expand_seed(tokens: list) -> tuple[list, dict]:
    """Expand any playlist/album tokens into their track URIs, in order.

    Returns (tokens, info). Individual track tokens pass through untouched, so a
    seed mixing a playlist and a few tracks works. Paginates fully — a truncated
    seed silently changes the experiment (the infinite-playlist product lost 83
    of 102 tracks this way and produced garbage from the fragment)."""
    containers = [(i, c) for i, t in enumerate(tokens)
                  if (c := parse_container(t)) is not None]
    if not containers:
        return list(tokens), {}

    token = _spotify_token()
    out: list = []
    info = {"expanded": []}
    for i, t in enumerate(tokens):
        c = parse_container(t)
        if c is None:
            out.append(t)
            continue
        kind, cid = c
        uris: list[str] = []
        name = None
        if kind == "playlist":
            meta = _spotify_get(
                f"/playlists/{cid}?fields=name,tracks(total)", token)
            name = meta.get("name")
            url = (f"/playlists/{cid}/tracks"
                   "?fields=next,items(track(uri,type,is_local))&limit=100")
            while url:
                page = _spotify_get(url, token)
                for it in page.get("items", []):
                    tr = it.get("track") or {}
                    if tr.get("uri") and tr.get("type") == "track" \
                            and not tr.get("is_local"):
                        uris.append(tr["uri"])
                url = page.get("next")
        else:
            meta = _spotify_get(f"/albums/{cid}", token)
            name = meta.get("name")
            url = f"/albums/{cid}/tracks?limit=50"
            while url:
                page = _spotify_get(url, token)
                uris.extend(tr["uri"] for tr in page.get("items", [])
                            if tr.get("uri"))
                url = page.get("next")
        out.extend(uris)
        info["expanded"].append({"kind": kind, "id": cid, "name": name,
                                 "tracks": len(uris)})
        emit({"kind": "log",
              "msg": f"expanded {kind} {name or cid}: {len(uris)} tracks"})
    return out, info


def load_params(spec: str | None) -> dict:
    """Accept a path to a JSON file, an inline JSON string, or None."""
    if not spec:
        return dict(DEFAULTS)
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    user = json.loads(raw)
    prm = {**DEFAULTS, **user}
    prm["steps"] = max(1, int(prm["steps"]))
    prm["intent_top_m"] = max(1, int(prm["intent_top_m"]))
    prm["sample_top_k"] = max(1, int(prm["sample_top_k"]))
    assert float(prm["temperature"]) >= 0.0, "temperature must be >= 0"
    return prm


# --------------------------------------------------------------------------- #
# Model dispatch — reuse each predictor's OWN scorer, never reimplement one    #
# --------------------------------------------------------------------------- #
def build_scorer(predictor: str, model_dir: Path, art):
    """Return (score_fn, predict_latent_fn|None, info dict) for a promoted model.

    `score_fn(prefix_idx) -> full-vocab float score vector` is the universal
    contract. `predict_latent_fn(prefix_idx) -> (D,) latent` is OPTIONAL and only
    exists for the models that predict a next-latent directly (the recurrent /
    feed-forward families); it powers the richer intent readout. Blends and the
    count-based baselines return scores only, so theirs is None."""
    if predictor not in DISPATCH:
        raise SystemExit(f"extend: unsupported predictor {predictor!r} "
                         f"(supported: {', '.join(DISPATCH)})")

    latents = art.item_latents
    info: dict = {"predictor": predictor}

    if predictor in ("seq-markov", "seq-popularity", "seq-recency"):
        # Count-based baselines: the scorers are built from the baked artifact's
        # TRAIN sessions alone, exactly as seq_baselines.predict does.
        import seq_baselines
        if predictor == "seq-markov":
            score_fn = seq_baselines.markov_scorer(art)
        elif predictor == "seq-popularity":
            score_fn = seq_baselines.popularity_scorer(art)
        else:
            score_fn = seq_baselines.recency_scorer(art)
        return score_fn, None, info

    # Neural / blend families all keep their config in the promoted dir.
    if predictor == "seq-nexttrack":
        import seq_nexttrack as mod
        hp = mod.load_hp(str(model_dir / "hyperparams.json"))
        model = mod.load_model(model_dir, art, hp)
        score_fn = mod.build_score_fn(model, latents)
    elif predictor == "seq-dualgru":
        import seq_dualgru as mod
        hp = mod.load_hp(str(model_dir / "hyperparams.json"))
        model = mod.load_model(model_dir, art, hp)
        score_fn = mod.build_score_fn(model, latents)
    elif predictor == "seq-ann":
        import seq_ann as mod
        hp = mod.load_hp(str(model_dir / "hyperparams.json"))
        model = mod.load_model(model_dir, art, hp)
        score_fn = mod.ann_score_fn(model, latents)
    else:  # seq-blend
        import seq_blend as mod
        hp = mod.load_hp(str(model_dir / "hyperparams.json"))
        model = mod.load_model(model_dir, art, hp)
        score_fn = mod.build_blend_score_fn(art, model, hp)
        info["hyperparams"] = {k: hp[k] for k in
                              ("alpha", "content", "projection") if k in hp}
        return score_fn, None, info

    info["hyperparams"] = {k: v for k, v in hp.items()
                           if k in ("hidden", "hidden_a", "hidden_b", "arch",
                                    "arch_a", "arch_b", "view_a", "view_b",
                                    "fusion_layers", "loss")}

    # The next-latent families expose predict_next(x) -> (1, D).
    def predict_latent_fn(prefix_idx: np.ndarray):
        import torch
        x = torch.from_numpy(latents[prefix_idx][None, :, :])
        with torch.no_grad():
            pred = model.predict_next(x)
            pred = torch.nn.functional.normalize(pred, dim=1)
        return pred.numpy()[0]

    return score_fn, predict_latent_fn, info


# --------------------------------------------------------------------------- #
# Seed core + mood anchor (ported from the infinite-playlist app's policy)     #
# --------------------------------------------------------------------------- #
def dominant_core(idx: np.ndarray, vecs: np.ndarray | None):
    """The coherent SUB-CLUSTER of a seed, as the app's `dominantCore` computes it.

    A pasted playlist is often diffuse (multi-genre); anchoring generation to the
    centroid of the WHOLE seed then aims at a meaningless average. Instead: take
    the medoid (the row most similar to all others), keep the rows whose
    similarity to it is at least the mean, and use those. Validated in the
    2026-07-23 anchoring work — it does not improve seed cohesion's effect on
    its own, but it makes the retrieval ANCHOR meaningfully on-mood.

    Returns (core_idx, anchor_unit_vector|None)."""
    if vecs is None or idx.size == 0:
        return idx, None
    ok = [i for i in idx.tolist() if i < vecs.shape[0]]
    if len(ok) < 2:
        return idx, None
    V = vecs[ok]                                   # already L2-normalized
    sims = V @ V.T
    np.fill_diagonal(sims, 0.0)
    medoid = int(np.argmax(sims.sum(axis=1)))
    to_med = V @ V[medoid]
    keep_mask = to_med >= float(np.mean(to_med))
    keep = [ok[i] for i in range(len(ok)) if keep_mask[i]]
    if len(keep) < max(2, int(0.4 * len(ok))):     # floor: 40% of the seed
        keep = ok
    centroid = vecs[keep].mean(axis=0)
    n = float(np.linalg.norm(centroid))
    anchor = (centroid / n) if n > 0 else None
    return np.array(keep, dtype=np.int64), anchor


# --------------------------------------------------------------------------- #
# Intent — "the shape of the next track(s)" from the score distribution        #
# --------------------------------------------------------------------------- #
def step_intent(scores: np.ndarray, allowed: np.ndarray, artist_ids: np.ndarray,
                genre_names: dict, prm: dict, latents: np.ndarray | None = None,
                genre_ids: np.ndarray | None = None) -> dict:
    """Summarize WHAT KIND of track the model wants next, without naming one.

    Works for every model because it reads only the score vector. Over the top-M
    allowed candidates we softmax the scores (temperature `intent_tau`) into a
    probability mass and aggregate it by genre and by artist. Reported:

      * `genres`     : the top genres by score mass — the model's stylistic pull
      * `entropy`    : Shannon entropy (bits) of the top-M mass, normalized to
                       [0,1] by log2(M). Low = the model is committed to a narrow
                       pocket; high = it is wandering / indifferent.
      * `margin`     : top1 − top2 raw score gap (how decisive the pick is)
      * `artist_conc`: the largest single artist's share of the mass (is the model
                       reaching for ONE artist — the album-eager failure mode —
                       or for a style?)"""
    pool = allowed[np.argsort(-scores[allowed])[:int(prm["intent_top_m"])]]
    if pool.size == 0:
        return {}
    s = scores[pool].astype(np.float64)
    tau = max(float(prm["intent_tau"]), 1e-6)
    z = (s - s.max()) / tau
    w = np.exp(z)
    w /= w.sum()

    by_genre: dict[str, float] = {}
    by_artist: dict[int, float] = {}
    for i, item in enumerate(pool.tolist()):
        g = genre_names.get(item) or "—"
        by_genre[g] = by_genre.get(g, 0.0) + float(w[i])
        a = int(artist_ids[item])
        by_artist[a] = by_artist.get(a, 0.0) + float(w[i])

    ent = float(-np.sum(w * np.log2(np.clip(w, 1e-12, None))))
    ent_norm = ent / math.log2(max(pool.size, 2))
    top_genres = sorted(by_genre.items(), key=lambda kv: -kv[1])[:int(prm["intent_genres"])]
    srt = np.sort(s)[::-1]
    out = {
        "genres": [{"genre": g, "mass": round(m, 4)} for g, m in top_genres],
        "entropy": round(ent_norm, 4),
        "margin": round(float(srt[0] - srt[1]), 5) if srt.size > 1 else None,
        "artist_conc": round(max(by_artist.values()), 4),
        "pool": int(pool.size),
    }
    if latents is not None and genre_ids is not None:
        out.update(candidate_eigen(pool, w, latents, genre_ids, genre_names))
    return out


def candidate_eigen(pool: np.ndarray, weights: np.ndarray, latents: np.ndarray,
                    genre_ids: np.ndarray, genre_names: dict,
                    keep: int = 4) -> dict:
    """EIGEN-SHAPE of the model's next-track belief — the geometry of what it is
    considering, which the ranked list and the genre histogram both hide.

    Take the top-M candidates' latents, weight each by its share of score mass,
    centre them, and take the SVD. The squared singular values are the
    eigenvalues of the weighted candidate covariance, i.e. how the model's
    plausible continuations are distributed in the content space:

      * `eigen`    : the leading eigenvalues as VARIANCE SHARES (they sum to 1).
                     A dominant first share means the candidates lie along ONE
                     axis — the model has a direction of travel and is choosing
                     *how far*, not *where*. Comparable shares mean the belief is
                     isotropic: many unrelated continuations score alike.
      * `pr`       : participation ratio 1/Σsᵢ² — the EFFECTIVE NUMBER of
                     directions the model is spreading over (1 = a single axis,
                     M = fully diffuse). This is the honest "how many different
                     things could come next" number; entropy over items conflates
                     "many candidates" with "many kinds of candidate", and a
                     tight cluster of 50 near-identical tracks has high entropy
                     but pr ≈ 1.
      * `axis`     : the leading eigenvector's genre poles — the two ends of that
                     principal axis, so the direction of travel is nameable
                     ("ambient ←→ classic dubstep") rather than abstract.

    Model-agnostic: it reads only scores + item latents, so a blend and a GRU are
    described in the same terms even though they share no internals."""
    if pool.size < 3:
        return {}
    X = latents[pool].astype(np.float64)                # (M, D)
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    mu = (w[:, None] * X).sum(axis=0)
    Xc = (X - mu) * np.sqrt(w)[:, None]
    try:
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return {}
    ev = s ** 2
    tot = float(ev.sum())
    if tot <= 1e-18:
        return {}
    shares = ev / tot
    pr = float(1.0 / max(float(np.sum(shares ** 2)), 1e-12))

    out = {
        "eigen": [round(float(x), 4) for x in shares[:keep]],
        "pr": round(pr, 2),
    }
    # Name the principal axis from the CANDIDATES THEMSELVES, not from the global
    # genre prototypes. The axis is the direction along which *this shortlist*
    # spreads, so its poles are only meaningful in terms of the shortlist's own
    # contents. Scanning all prototypes instead returns whichever global genre
    # mean happens to project furthest — which produced poles like
    # "baroque ←→ city pop" for an ambient/idm shortlist: real arithmetic,
    # no interpretive value. Here each pole is the genre whose candidates sit
    # furthest along that end, so "ambient ←→ classic dubstep" means the model is
    # genuinely torn between those two within its plausible set.
    # Eigenvector sign is arbitrary, hence a pair rather than a direction.
    axis = Vt[0]
    proj = (X - mu) @ axis                              # (M,) per-candidate
    by_g: dict[str, list[float]] = {}
    for i, item in enumerate(pool.tolist()):
        by_g.setdefault(genre_names.get(item) or "—", []).append(float(proj[i]))
    means = sorted(((g, sum(v) / len(v)) for g, v in by_g.items()),
                   key=lambda kv: kv[1])
    if len(means) > 1:
        out["axis"] = [means[-1][0], means[0][0]]
        out["axis_spread"] = round(float(means[-1][1] - means[0][1]), 4)
    elif means:
        # The whole shortlist is ONE genre — there is no axis to name because the
        # model sees no stylistic choice at all. Reported explicitly rather than
        # as a missing field: combined with a high leading eigenvalue and pr→1
        # this is the album-lock signature (measured on the champion blend:
        # pr 1.25, leading share 0.89, single-genre pool).
        out["axis_collapsed"] = means[0][0]
    return out


def latent_intent(pred_latent: np.ndarray, latents: np.ndarray,
                  genre_ids: np.ndarray, genre_names: dict, top: int = 5) -> dict:
    """Richer intent for models that predict a next-LATENT: where the model's
    TARGET POINT sits, independent of which catalog items happen to be near it.

    We L2-normalize the predicted latent and report its cosine to each genre's
    PROTOTYPE (that genre's mean item latent). This is the model's aim rather
    than its ranked shortlist — the two differ when the target lands in a sparse
    region of the catalog, which is exactly when a journey drifts."""
    p = pred_latent / max(float(np.linalg.norm(pred_latent)), 1e-9)
    out = []
    seen: dict[int, str] = {}
    for item, name in genre_names.items():
        seen[int(genre_ids[item])] = name
    for gid, name in seen.items():
        members = np.nonzero(genre_ids == gid)[0]
        if members.size < GENRE_PROTO_MIN:
            continue
        proto = latents[members].mean(axis=0)
        n = float(np.linalg.norm(proto))
        if n <= 0:
            continue
        out.append((name, float(p @ (proto / n))))
    out.sort(key=lambda kv: -kv[1])
    return {"target_genres": [{"genre": g, "cos": round(c, 4)} for g, c in out[:top]]}


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #
def extend(model_dir: Path, predictor: str, prm: dict) -> dict:
    """Generate a journey and return the full lab payload (stops + diagnostics)."""
    art = load_artifact(model_dir)
    latents = art.item_latents
    artist_ids = _relevance_ids(art, "artist")
    genre_ids = _relevance_ids(art, "genre")
    genre_names = {int(k): (v.get("genre") or "—") for k, v in art.items.items()
                   if int(k) < art.n_items}
    music_vecs, music_mask = _load_music_vectors(art)

    # Expand playlists/albums BEFORE resolution — resolve_prefix only knows
    # single tracks, so a pasted playlist URL would resolve to nothing.
    tokens, seed_info = expand_seed(list(prm["seed"]))
    seed_idx, unknown = resolve_prefix(art, tokens)
    if seed_idx.size == 0:
        raise SystemExit(
            f"extend: none of the {len(tokens)} seed track(s) are in this "
            f"model's vocabulary ({art.n_items} items). The vocab is the "
            f"library the model was trained on; seed with tracks from it, or "
            f"with a playlist that overlaps it.")

    score_fn, predict_latent_fn, info = build_scorer(predictor, model_dir, art)
    core_idx, anchor = dominant_core(seed_idx, music_vecs)
    use_anchor = (anchor is not None
                  and float(prm["anchor_lambda"]) > 0
                  and core_idx.size >= int(prm["anchor_min_core"]))

    # The DIAGNOSTICS reference is deliberately NOT the retrieval anchor. The
    # anchor is an opt-in lever gated on a big enough coherent core; the mood
    # reference must exist for ANY seed (including a single track, where it is
    # just that track's vector) or mood_coh/holisticness would be unmeasurable
    # exactly in the most common lab case. `seq_common.eval_from_scores` scores
    # mood_coh against the whole prefix centroid, and this matches it.
    mood_ref = None
    if music_vecs is not None and music_mask is not None:
        cov = seed_idx[music_mask[seed_idx]]
        if cov.size:
            c = music_vecs[cov].mean(axis=0)
            n = float(np.linalg.norm(c))
            if n > 0:
                mood_ref = c / n

    emit({"kind": "log", "msg":
          f"extend {predictor}: seed {seed_idx.size} resolved "
          f"({len(unknown)} unknown), core {core_idx.size}, "
          f"anchor {'on' if use_anchor else 'off'}, steps {prm['steps']}"})

    rng = np.random.default_rng(int(prm["seed_rng"]))
    # Generation runs from the CORE when anchoring (the app's `begin()` rule), but
    # every loaded seed counts as used so nothing is replayed.
    seq = (core_idx if use_anchor else seed_idx).astype(np.int64).tolist()
    used = set(seed_idx.tolist())
    used_artists: list[int] = [int(artist_ids[i]) for i in seed_idx.tolist()]

    # Title identity for the dedupe constraint: the same recording appears in the
    # vocab under multiple releases (single + album + edit), each its own item
    # index, so index-level next-distinct does not stop a replay. Key on
    # (normalized title, artist id).
    title_key = np.full(art.n_items, -1, dtype=np.int64)
    _tvocab: dict[tuple, int] = {}
    for key, meta in art.items.items():
        i = int(key)
        if not (0 <= i < art.n_items):
            continue
        name = (meta.get("name") or "").strip().lower()
        if not name:
            continue
        # Strip the usual release-variant suffixes so "X" and "X - edit" collide.
        base = re.split(r"\s*[-–(\[]\s*", name)[0].strip() or name
        k = (base, int(artist_ids[i]))
        cid = _tvocab.setdefault(k, len(_tvocab))
        title_key[i] = cid
    played_titles: set[int] = {int(title_key[i]) for i in seed_idx.tolist()
                               if title_key[i] >= 0}

    anchor_sims = None
    if use_anchor:
        anchor_sims = np.full(art.n_items, 0.0, dtype=np.float64)
        mm = music_mask if music_mask is not None else np.ones(art.n_items, bool)
        anchor_sims[mm] = music_vecs[mm] @ anchor

    stops = []
    for step in range(int(prm["steps"])):
        prefix = np.array(seq, dtype=np.int64)
        raw = np.asarray(score_fn(prefix), dtype=np.float64)

        # --- shared retrieval policy -----------------------------------------
        allowed = np.ones(art.n_items, dtype=bool)
        allowed[list(used)] = False            # next-distinct: never replay
        # Hard constraints, applied to ELIGIBILITY rather than to the score, so
        # no amount of model confidence or anchor pull can override them.
        cooldown = int(prm["artist_cooldown"])
        if cooldown > 0 and used_artists:
            recent_a = [a for a in used_artists[-cooldown:] if a >= 0]
            if recent_a:
                allowed &= ~np.isin(artist_ids, recent_a)
        if prm["dedupe_titles"] and played_titles:
            allowed &= ~np.isin(title_key, list(played_titles))
        if not allowed.any():
            # Constraints exhausted the vocab — relax them for this step rather
            # than truncating the journey, and say so.
            emit({"kind": "log", "msg":
                  f"step {step + 1}: constraints left no candidate; relaxing "
                  f"the artist cooldown for this step"})
            allowed = np.ones(art.n_items, dtype=bool)
            allowed[list(used)] = False
            if not allowed.any():
                break

        # Standardize FIRST, against the plausible-candidate pool, so every knob
        # below is in comparable units across model families.
        pool_n = int(prm["policy_pool"])
        scores = local_z(raw, allowed, pool_n)

        if use_anchor:
            scores = scores + float(prm["anchor_lambda"]) * local_z(
                anchor_sims, allowed, pool_n)
        ap = float(prm["artist_penalty"])
        if ap:
            win = int(prm["artist_window"])
            recent = used_artists[-win:] if win > 0 else used_artists
            bad = set(a for a in recent if a >= 0)
            if bad:
                pen = np.isin(artist_ids, list(bad))
                scores = scores - ap * pen

        pool_idx = np.nonzero(allowed)[0]
        intent = step_intent(scores, pool_idx, artist_ids, genre_names, prm,
                             latents=latents, genre_ids=genre_ids)
        if predict_latent_fn is not None:
            try:
                intent.update(latent_intent(predict_latent_fn(prefix), latents,
                                            genre_ids, genre_names))
            except Exception as exc:                # never fail a journey on this
                intent["target_genres_error"] = str(exc)

        # --- pick ------------------------------------------------------------
        temp = float(prm["temperature"])
        if temp > 0:
            top = pool_idx[np.argsort(-scores[pool_idx])[:int(prm["sample_top_k"])]]
            z = (scores[top] - scores[top].max()) / temp
            p = np.exp(z)
            p /= p.sum()
            choice = int(rng.choice(top, p=p))
        else:
            choice = int(pool_idx[np.argmax(scores[pool_idx])])

        meta = art.items.get(str(choice), {})
        stop = {
            "step": step + 1,
            "item_index": choice,
            "uri": meta.get("uri"),
            "name": meta.get("name"),
            "artist": meta.get("artist"),
            "genre": meta.get("genre"),
            # Both scales: z is what the policy acted on and is comparable across
            # models; raw is the model's own number, which is not.
            "score_z": round(float(scores[choice]), 5),
            "score_raw": round(float(raw[choice]), 5),
            "intent": intent,
        }
        # Per-stop grounding: how close this pick is to the seed's mood centroid.
        if mood_ref is not None and music_mask is not None and music_mask[choice]:
            stop["mood_sim"] = round(float(music_vecs[choice] @ mood_ref), 4)
        stops.append(stop)

        seq.append(choice)
        used.add(choice)
        used_artists.append(int(artist_ids[choice]))
        if title_key[choice] >= 0:
            played_titles.add(int(title_key[choice]))

    return {
        "model_dir": str(model_dir),
        "predictor": predictor,
        "info": info,
        "seed": {
            "resolved": int(seed_idx.size),
            "submitted": len(tokens),
            "unknown": unknown[:20],
            "unknown_count": len(unknown),
            "core": int(core_idx.size),
            "anchored": bool(use_anchor),
            **seed_info,
        },
        "params": prm,
        "stops": stops,
        "diagnostics": journey_diagnostics(stops, seed_idx, art, artist_ids,
                                           music_vecs, music_mask, mood_ref),
    }


def journey_diagnostics(stops, seed_idx, art, artist_ids, music_vecs,
                        music_mask, mood_ref) -> dict:
    """The crown's facets, computed over the GENERATED journey.

    These mirror `seq_common.eval_from_scores` — same definitions — but scored on
    what the model actually produced for this seed rather than on a held-out
    split, which is what makes the lab's numbers comparable to the leaderboard's
    while still describing the thing you are listening to:

      * `mood_coh`    : mean cosine of the stops to the seed mood centroid
      * `ild`         : mean pairwise DISTANCE among the stops (intra-list
                        diversity — higher = less duplicative)
      * `artist_adj`  : fraction of stops sharing the SEED's artist (eagerness;
                        LOWER is better)
      * `music_rel`   : mood_coh rescaled above the popularity floor, the same
                        grounding factor the crown now multiplies in
      * `holisticness`: clamp(mood_coh,0) × ild × (1−artist_adj) × music_rel —
                        the grounded composite, on this journey
      * `distinct_artists` / `distinct_genres`: plain listenability counts"""
    import seq_common
    out: dict = {"n_stops": len(stops)}
    if not stops:
        return out
    idx = np.array([s["item_index"] for s in stops], dtype=np.int64)

    seed_artist = int(artist_ids[seed_idx[-1]]) if seed_idx.size else -1
    if seed_artist >= 0:
        out["artist_adj"] = round(float(np.mean(artist_ids[idx] == seed_artist)), 4)
    out["distinct_artists"] = int(len(set(artist_ids[idx].tolist())))

    # JOURNEY-INTERNAL repetition. The crown's `artist_adj` measures the share of
    # top-k sharing the SEED's artist — the right notion for one-shot next-track
    # eval, but blind to what actually ruins an infinite playlist: the generator
    # looping on an artist that isn't the seed's. Measured 2026-07-25: the
    # champion blend played three consecutive Caribou tracks from a Tim Hecker
    # seed at artist_adj 0.0, i.e. "perfectly non-eager" by the crown's facet.
    #   max_artist_run : longest consecutive same-artist streak (2+ = audible)
    #   artist_hhi     : Herfindahl concentration over artists, 1/n_stops (flat)
    #                    .. 1.0 (one artist for the whole journey)
    a_seq = artist_ids[idx].tolist()
    run = best = 1
    for i in range(1, len(a_seq)):
        run = run + 1 if a_seq[i] == a_seq[i - 1] else 1
        best = max(best, run)
    out["max_artist_run"] = int(best)
    counts = np.array([a_seq.count(a) for a in set(a_seq)], dtype=np.float64)
    out["artist_hhi"] = round(float(np.sum((counts / len(a_seq)) ** 2)), 4)
    out["distinct_genres"] = int(len({(art.items.get(str(i), {}) or {}).get("genre")
                                      for i in idx.tolist()}))

    if music_vecs is not None and music_mask is not None:
        cov = idx[music_mask[idx]]
        if cov.size:
            V = music_vecs[cov]
            if mood_ref is not None:
                out["mood_coh"] = round(float(np.mean(V @ mood_ref)), 4)
            if cov.size > 1:
                sims = V @ V.T
                iu = np.triu_indices(cov.size, k=1)
                out["ild"] = round(float(np.mean(1.0 - sims[iu])), 4)

    mc, il, aa = out.get("mood_coh"), out.get("ild"), out.get("artist_adj")
    if mc is not None:
        rel = (mc - seq_common.MUSIC_FLOOR) / (1.0 - seq_common.MUSIC_FLOOR)
        out["music_rel"] = round(min(max(rel, 0.0), 1.0), 4)
        if il is not None and aa is not None:
            out["holisticness"] = round(
                max(mc, 0.0) * il * (1.0 - min(max(aa, 0.0), 1.0))
                * out["music_rel"], 5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extend", help="autoregressively extend a seed session")
    ex.add_argument("--model", required=True, type=Path)
    ex.add_argument("--predictor", required=True,
                    help=f"registry predictor name ({', '.join(DISPATCH)})")
    ex.add_argument("--output", required=True, type=Path)
    ex.add_argument("--params", default=None,
                    help="JSON file path or inline JSON (seed, steps, policy)")
    args = ap.parse_args()

    try:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    except Exception:
        pass

    prm = load_params(args.params)
    payload = extend(args.model, args.predictor, prm)
    args.output.write_text(json.dumps(payload))
    d = payload["diagnostics"]
    emit({"kind": "log", "msg":
          f"{len(payload['stops'])} stops; "
          f"mood_coh {d.get('mood_coh')} ild {d.get('ild')} "
          f"artist_adj {d.get('artist_adj')} holisticness {d.get('holisticness')} "
          f"({d.get('distinct_artists')} artists / {d.get('distinct_genres')} genres)"})
    emit({"kind": "done"})


if __name__ == "__main__":
    main()
