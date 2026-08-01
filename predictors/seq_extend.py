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
  3. **A LEGIBLE mood report** (`mood_report`, see below). The crown's facets are
     cosines in a 517-d fused space: comparable across models, but impossible to
     confirm by ear, and therefore useless for deciding which generator holds a
     vibe better. The report card restates the same journey on axes a listener
     can check — valence / energy / tempo / era / their own play history — and
     calibrates each one against the bands THEIR OWN REAL SESSIONS occupy, so
     every number arrives as a verdict ("choppier than your listening") instead
     of a bare float.

INTENT ("the shape of the next track", beyond naming it) is derived from the
SCORE DISTRIBUTION, not from the model's internals. That is deliberate: it makes
the readout work identically for a GRU, a blend and a Markov baseline, none of
which share an internal representation. Per step we report, over the top-M
candidates: the softmax score mass aggregated BY GENRE and BY ARTIST (what kind
of thing the model wants next, and how concentrated), the top-1 margin, the
distribution entropy (is the model committed or wandering?), the NAMED stylistic
axis the shortlist spreads along, and `passed_over` — the runner-up tracks it
turned down, by name, with the count of candidates that were genuinely in
contention. That last one replaced an eigen-decomposition readout of the
candidate cloud: same question ("how much choice did the model have"), but a
skipped track you can play is checkable and a variance share is not. Models that
expose a predicted next-LATENT additionally get that latent's nearest genre
prototypes in the musical-distance space — the model's target *point* rather
than its ranked candidates.

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
DISPATCH = ("seq-nexttrack", "seq-dualgru", "seq-ann", "seq-blend", "seq-embed",
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
    elif predictor == "seq-embed":
        # Retrieves in its OWN LEARNED table, not the dataset's frozen latents —
        # so its score_fn takes no latents argument, and the intent readout's
        # latent-space extras below would be measuring the wrong space.
        import seq_embed as mod
        hp = mod.load_hp(str(model_dir / "hyperparams.json"))
        model = mod.load_model(model_dir, art, hp)
        score_fn = mod.build_score_fn(model)
        info["hyperparams"] = {k: hp[k] for k in
                               ("embed_mode", "embed_init", "embed_dim", "hidden")
                               if k in hp}
        return score_fn, None, info
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
        out.update(candidate_axis(pool, w, latents, genre_ids, genre_names))
    return out


def candidate_axis(pool: np.ndarray, weights: np.ndarray, latents: np.ndarray,
                   genre_ids: np.ndarray, genre_names: dict) -> dict:
    """NAME the stylistic choice this step is facing: the two genre poles of the
    axis along which the shortlist spreads ("ambient ←→ classic dubstep").

    Mechanically this is still the leading eigenvector of the weighted candidate
    covariance, but only its NAMED poles survive into the payload. The variance
    shares and participation ratio that used to ship alongside are gone: they
    described the same geometry in units nobody can hear, and `passed_over`
    answers "how much choice was there" concretely instead — with the count of
    real contenders and the skipped tracks' names.

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
    if float((s ** 2).sum()) <= 1e-18:
        return {}

    out: dict = {}
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


def passed_over(scores: np.ndarray, pool_idx: np.ndarray, chosen: int, art,
                mood: dict, top: int = 3, near: float = 0.5) -> dict:
    """THE ROAD NOT TAKEN — the runner-up tracks this step passed over, by name.

    This replaces the eigen/participation-ratio readout that used to describe the
    candidate cloud's geometry. Both answer "how much choice did the model have",
    but only one of them can be checked by a listener: an eigenvalue bar cannot
    tell you whether the track it skipped was the better call, and the named
    alternative can. Reported:

      * `over`       : the top runner-ups (name / artist / genre / era), each with
                       its `gap` = how far below the pick it scored, in the same
                       pool-local z units the policy acted on. A NEGATIVE gap
                       means it outscored the pick — which happens exactly when
                       temperature > 0 and the step sampled instead of taking the
                       argmax, so the sign is the sampling audit trail.
      * `contenders` : how many candidates sat within `near` z of the pick — the
                       honest "how many real options were there" count. 0 means
                       the model had one obvious answer; 20 means the pick was
                       nearly arbitrary and a re-roll would sound different.
      * `choices`    : distinct genres / artists / decades among the top-50, so a
                       shortlist that is 50 tracks of one artist reads as the
                       single choice it actually is."""
    s = scores[pool_idx]
    win = float(scores[chosen])
    order = np.argsort(-s)
    runners = []
    for j in order[: top + 6].tolist():
        i = int(pool_idx[j])
        if i == chosen:
            continue
        meta = art.items.get(str(i), {})
        yr = mood["year"][i]
        runners.append({
            "name": meta.get("name"),
            "artist": meta.get("artist"),
            "genre": meta.get("genre"),
            "gap": round(win - float(scores[i]), 3),
            "year": int(yr) if np.isfinite(yr) else None,
        })
        if len(runners) >= top:
            break

    short = pool_idx[order[:50]]
    decades = {int(y // 10 * 10) for y in mood["year"][short] if np.isfinite(y)}
    return {
        "over": runners,
        "contenders": max(int(np.sum(s >= win - near)) - 1, 0),
        "choices": {
            "genres": len({(art.items.get(str(int(i)), {}) or {}).get("genre")
                           for i in short.tolist()}),
            "artists": len({(art.items.get(str(int(i)), {}) or {}).get("artist")
                            for i in short.tolist()}),
            "decades": len(decades),
        },
    }


# --------------------------------------------------------------------------- #
# Legible mood — the vocabulary a LISTENER can actually check                  #
# --------------------------------------------------------------------------- #
# The crown's facets (mood_coh / ild / music_rel) are cosines in a 517-d fused
# space: correct, comparable, and unreadable at the point of listening. "mood_coh
# 0.42" cannot be confirmed or refuted by ear, so it cannot settle which
# generator holds a vibe better. These features can: valence/energy/tempo are the
# axes people describe mood WITH, and release_year / play_count / skip_rate are
# facts about this listener's own history with the track.
#
# Source is the content-metric collection's payload (copied from
# spotify_tracks_content), NOT the corpus collection — `spotify_tracks` carries no
# af_* at all. Coverage is partial (measured 2026-07-26: 17455/23529 = 74% have
# af_*; release_year / skip_rate / play_count are complete), and the acoustics are
# ReccoBeats-derived and occasionally wrong (Muse's "Intro" is logged at valence
# 0.0 / energy 0.03). Both facts are REPORTED rather than hidden: every aggregate
# carries the item count it was computed from, so a thin axis is visibly thin.
# They are weak as predictors — that is on the record — but this is description,
# not prediction.
MOOD_FIELDS = {
    "energy": "af_energy",
    "valence": "af_valence",
    "tempo": "af_tempo",
    "acousticness": "af_acousticness",
    "danceability": "af_danceability",
    "instrumentalness": "af_instrumentalness",
    "year": "release_year",
    "skip_rate": "skip_rate",
    "completion": "completion_ratio",
    "popularity": "track_popularity",
    "saved": "is_saved",
    "on_repeat": "on_repeat_count",
}
# Per-stop chips, in display order (key, label, format).
STOP_MOOD = [("energy", "energy", "unit"), ("valence", "valence", "unit"),
             ("tempo", "bpm", "bpm"), ("acousticness", "acoustic", "unit"),
             ("year", "year", "year")]


def load_mood_table(art) -> dict:
    """Human-legible per-item descriptors, aligned to item index.

    Returns a dict of float arrays (NaN = unknown) keyed by the MOOD_FIELDS names,
    plus `familiarity` (the play-count percentile within this vocabulary, so
    "top 8% most-played of your library" rather than a raw count), `loved`
    (saved or on-repeat, 0/1) and `_acoustic` (bool mask: does this item have
    af_* at all). Best-effort — a missing collection yields an all-NaN table and
    every mood axis then reports itself as uncovered instead of failing."""
    n = art.n_items
    out = {k: np.full(n, np.nan, dtype=np.float64) for k in MOOD_FIELDS}
    out["_acoustic"] = np.zeros(n, dtype=bool)

    # play_count rides items.json, so familiarity is available even with Qdrant
    # down. Midrank percentile: raw counts are extremely tied (most tracks played
    # once), and a plain argsort rank would order those ties arbitrarily.
    pc = np.zeros(n, dtype=np.float64)
    for key, meta in art.items.items():
        i = int(key)
        if 0 <= i < n:
            pc[i] = float(meta.get("play_count") or 0)
    uniq, inv, counts = np.unique(pc, return_inverse=True, return_counts=True)
    below = np.cumsum(counts) - counts
    out["play_count"] = pc
    out["familiarity"] = 100.0 * (below + counts / 2.0)[inv] / max(n, 1)

    if os.environ.get("LENSING_MUSIC_METRIC_DISABLE"):
        out["loved"] = np.full(n, np.nan)
        return out
    collection = os.environ.get("LENSING_MUSIC_METRIC_COLLECTION",
                                "spotify_tracks_content_metric")
    url = (os.environ.get("QDRANT_URL")
           or os.environ.get("PATHFINDER_QDRANT_URL", "http://localhost:6337"))
    try:
        import hashlib

        from qdrant_client import QdrantClient

        uris: list[str | None] = [None] * n
        for key, meta in art.items.items():
            i = int(key)
            if 0 <= i < n:
                uris[i] = meta.get("uri")

        def to_id(u: str) -> int:          # matches pipeline/corpus/ids.py
            return int.from_bytes(hashlib.sha256(u.encode()).digest()[:8], "little")

        ids = [to_id(u) if u else None for u in uris]
        idmap = {pid: i for i, pid in enumerate(ids) if pid is not None}
        client = QdrantClient(url=url, timeout=60)
        want = [pid for pid in ids if pid is not None]
        # Nested field selectors, not the whole payload: these points also carry
        # the content text and sp_genres, which we never read (measured 2.7x
        # faster per batch on the local instance).
        fields = [f"metadata.{f}" for f in MOOD_FIELDS.values()]
        for b in range(0, len(want), 512):
            for r in client.retrieve(collection, ids=want[b:b + 512],
                                     with_payload=fields, with_vectors=False):
                i = idmap.get(r.id)
                if i is None:
                    continue
                md = (r.payload or {}).get("metadata") or {}
                for key, field in MOOD_FIELDS.items():
                    v = md.get(field)
                    if isinstance(v, (int, float)):
                        out[key][i] = float(v)
                out["_acoustic"][i] = isinstance(md.get("af_energy"), (int, float))
        emit({"kind": "log", "msg":
              f"mood: {int(out['_acoustic'].sum())}/{n} items carry acoustics "
              f"from {collection!r}"})
    except Exception as exc:               # noqa: BLE001 — descriptive, never fatal
        emit({"kind": "log", "msg":
              f"mood: legible features unavailable ({exc}); axes will be empty"})

    saved, rep = out["saved"], out["on_repeat"]
    loved = np.where(np.isnan(saved) & np.isnan(rep), np.nan,
                     ((np.nan_to_num(saved) > 0)
                      | (np.nan_to_num(rep) > 0)).astype(np.float64))
    out["loved"] = loved
    return out


# Axes calibrated as LEVELS (a journey's mean) and as JUMPS (mean |Δ| between
# consecutive tracks). The jump family is the one that answers the actual
# question — "does this algorithm conserve mood-like behaviour" is a statement
# about transitions, not about averages: a journey can average the seed's energy
# exactly while alternating 0.2 and 0.9 track by track, which is precisely the
# lurch a listener hears and the mean hides.
LEVEL_KEYS = ("energy", "valence", "tempo", "acousticness", "familiarity",
              "skip_rate", "loved", "year")
JUMP_KEYS = ("energy", "valence", "tempo", "year")


def natural_bands(art, mood: dict, music_vecs, music_mask,
                  genre_ids: np.ndarray, max_sessions: int = 1200,
                  cap: int = 30000, seed: int = 7) -> dict:
    """Calibrate every axis against THIS LISTENER'S OWN REAL SESSIONS.

    A raw number ("step-to-step energy jump 0.19") is unevaluable; the same number
    against the band the listener's real listening occupies ("your own sessions
    run 0.08–0.24") is a verdict. So we walk the artifact's TRAIN sessions — real
    listening histories, the same data the models were fit on — and collect the
    identical statistics we compute on a generated journey. Everything reported to
    the UI is then a percentile within that reference.

    Sampling is capped (`max_sessions` sessions, `cap` samples per measure) to
    keep this a few hundred milliseconds; the bands are wide statistics and do not
    move meaningfully with more."""
    rng = np.random.default_rng(seed)
    tr = np.asarray(art.train_sessions, dtype=np.int64)
    if tr.size == 0:
        return {}
    pick = tr if tr.size <= max_sessions else rng.choice(tr, max_sessions, replace=False)

    lev: dict[str, list] = {k: [] for k in LEVEL_KEYS}
    jmp: dict[str, list] = {k: [] for k in (*JUMP_KEYS, "sonic")}
    churn: list[float] = []
    drift: list[float] = []
    used = 0
    for s in pick.tolist():
        seq = art.session(int(s)).astype(np.int64)
        if seq.size < 3:
            continue
        used += 1
        for k in LEVEL_KEYS:
            v = mood[k][seq]
            v = v[np.isfinite(v)]
            if v.size and len(lev[k]) < cap:
                lev[k].append(float(v.mean()))
        a, b = seq[:-1], seq[1:]
        for k in JUMP_KEYS:
            d = np.abs(mood[k][b] - mood[k][a])
            d = d[np.isfinite(d)]
            if d.size and len(jmp[k]) < cap:
                jmp[k].append(float(d.mean()))
        if music_vecs is not None and music_mask is not None:
            ok = music_mask[a] & music_mask[b]
            if ok.any() and len(jmp["sonic"]) < cap:
                cos = np.sum(music_vecs[a[ok]] * music_vecs[b[ok]], axis=1)
                jmp["sonic"].append(float(np.mean(1.0 - cos)))
            d = _drift(seq, music_vecs, music_mask)
            if d is not None:
                drift.append(d)
        g = genre_ids[seq]
        churn.append(10.0 * float(np.mean(g[1:] != g[:-1])))

    out = {"levels": {k: np.asarray(v) for k, v in lev.items() if v},
           "jumps": {k: np.asarray(v) for k, v in jmp.items() if v},
           "churn": np.asarray(churn), "drift": np.asarray(drift),
           "n_sessions": used}
    emit({"kind": "log", "msg":
          f"mood: natural bands from {used} of your real sessions"})
    return out


def _drift(seq: np.ndarray, music_vecs, music_mask) -> float | None:
    """Signed DRIFT: how much closer to (or further from) its own opening the
    back third of a sequence sits. Negative = it walked away from where it
    started; ~0 = it stayed in the neighbourhood. Measured against the opening
    rather than against a fixed seed so it applies to real sessions too, which is
    what makes it calibratable."""
    ok = seq[music_mask[seq]]
    if ok.size < 6:
        return None
    third = max(ok.size // 3, 2)
    head = music_vecs[ok[:third]].mean(axis=0)
    nrm = float(np.linalg.norm(head))
    if nrm <= 0:
        return None
    head = head / nrm
    first = float(np.mean(music_vecs[ok[:third]] @ head))
    last = float(np.mean(music_vecs[ok[-third:]] @ head))
    return last - first


def _finite(v) -> float | None:
    return float(v) if v is not None and np.isfinite(v) else None


def _pctile(sample, v) -> float | None:
    """Where `v` falls inside the natural reference, 0–100."""
    if sample is None or len(sample) < 20 or v is None or not np.isfinite(v):
        return None
    return round(100.0 * float(np.mean(np.asarray(sample) <= v)), 1)


def _band(sample) -> list | None:
    if sample is None or len(sample) < 20:
        return None
    return [round(float(x), 4) for x in np.percentile(np.asarray(sample), [10, 50, 90])]


def _mean(arr: np.ndarray, idx: np.ndarray) -> tuple[float | None, int]:
    v = arr[idx]
    v = v[np.isfinite(v)]
    return (float(v.mean()) if v.size else None), int(v.size)


def _step_jump(arr: np.ndarray, path: np.ndarray) -> tuple[float | None, int]:
    d = np.abs(arr[path[1:]] - arr[path[:-1]])
    d = d[np.isfinite(d)]
    return (float(d.mean()) if d.size else None), int(d.size)


def _axis(key, group, label, value, *, n=0, seed=None, band=None, pct=None,
          fmt="unit", verdict=None, hint=None, tone=None) -> dict:
    """One report-card row. `tone` is deliberately sparse: sitting outside your
    natural band is INTERESTING, not wrong (a smoother-than-life journey may be
    exactly what you wanted), so only axes with an unambiguous bad direction ever
    carry a warning and everything else is left for the listener to judge."""
    return {"key": key, "group": group, "label": label,
            "value": None if value is None else round(float(value), 4),
            "seed": None if seed is None else round(float(seed), 4),
            "band": band, "pct": pct, "fmt": fmt, "n": int(n),
            "verdict": verdict, "hint": hint,
            "tone": tone, "out": bool(pct is not None and (pct < 10 or pct > 90))}


def _vs_seed(value, seed, band, higher: str, lower: str) -> str | None:
    """Verdict for a LEVEL axis: is the journey where the seed was? Tolerance is
    a fraction of the natural spread rather than a fixed epsilon, so "the same
    energy" means the same thing on a 0–1 axis and on a BPM axis."""
    if value is None or seed is None:
        return None
    tol = 0.15 * (band[2] - band[0]) if band else 0.05
    d = value - seed
    if abs(d) <= max(tol, 1e-9):
        return "holds the seed's level"
    return f"{higher if d > 0 else lower} than the seed"


def _vs_band(pct, low: str, mid: str, high: str) -> str | None:
    if pct is None:
        return None
    return low if pct < 10 else (high if pct > 90 else mid)


def mood_report(stops, seed_idx, art, mood, bands, music_vecs, music_mask,
                genre_ids) -> dict:
    """The report card: every axis a listener can verify, each against the band
    their own sessions occupy. Verdicts are computed HERE, not in the UI, so the
    client stays a renderer and the wording lives with the arithmetic."""
    path = np.array([s["item_index"] for s in stops], dtype=np.int64)
    if path.size == 0:
        return {}
    seed_idx = np.asarray(seed_idx, dtype=np.int64)
    lb, jb = bands.get("levels", {}), bands.get("jumps", {})
    # The seed's own last track is where the first generated step is heard FROM,
    # so transitions are measured over seed-tail + journey: a jarring opening jump
    # is the most audible failure and would otherwise go unmeasured.
    full = np.concatenate([seed_idx[-1:], path]) if seed_idx.size else path
    axes = []

    # --- 1. acoustic mood: the axes people describe mood WITH ----------------
    for key, label, fmt, hi, lo in (
            ("energy", "Energy", "unit", "hotter", "calmer"),
            ("valence", "Valence", "unit", "brighter", "darker"),
            ("tempo", "Tempo", "bpm", "faster", "slower"),
            ("acousticness", "Acoustic", "unit", "more acoustic", "more electric")):
        v, n = _mean(mood[key], path)
        sv, _ = _mean(mood[key], seed_idx)
        band = _band(lb.get(key))
        axes.append(_axis(key, "mood", label, v, n=n, seed=sv, band=band, fmt=fmt,
                          pct=_pctile(lb.get(key), v),
                          verdict=_vs_seed(v, sv, band, hi, lo),
                          hint=f"mean {MOOD_FIELDS.get(key, key)} over the "
                               f"{n} stop(s) that carry it"))

    # --- 2. continuity: the transition statistics, which is what "conserves
    #        mood-like behaviour" actually means ------------------------------
    if music_vecs is not None and music_mask is not None:
        ok = music_mask[full]
        w = full[ok]
        sj = None
        if w.size > 1:
            cos = np.sum(music_vecs[w[1:]] * music_vecs[w[:-1]], axis=1)
            sj = float(np.mean(1.0 - cos))
        pct = _pctile(jb.get("sonic"), sj)
        axes.append(_axis("sonic_jump", "flow", "Sonic jump / step", sj,
                          n=max(int(w.size) - 1, 0), band=_band(jb.get("sonic")),
                          pct=pct, fmt="unit",
                          verdict=_vs_band(pct, "smoother than you ever listen",
                                           "natural transitions",
                                           "choppier than your listening"),
                          hint="mean cosine distance between CONSECUTIVE stops in "
                               "the content-metric space, against the same "
                               "statistic on your real sessions"))
    for key, label, fmt in (("energy", "Energy jolt / step", "unit"),
                            ("tempo", "BPM jump / step", "bpm")):
        v, n = _step_jump(mood[key], full)
        pct = _pctile(jb.get(key), v)
        axes.append(_axis(f"{key}_jump", "flow", label, v, n=n,
                          band=_band(jb.get(key)), pct=pct, fmt=fmt,
                          verdict=_vs_band(pct, "flatter than your listening",
                                           "natural", "lurching"),
                          hint=f"mean |Δ{key}| from one stop to the next"))
    dv = _drift(full, music_vecs, music_mask) if music_vecs is not None else None
    pct = _pctile(bands.get("drift"), dv)
    axes.append(_axis("drift", "flow", "Drift from the opening", dv,
                      n=int(path.size), band=_band(bands.get("drift")), pct=pct,
                      fmt="signed",
                      verdict=None if dv is None else (
                          "holds the neighbourhood" if dv > -0.02 else
                          ("wanders off" if pct is not None and pct < 10
                           else "drifts, as your sessions do")),
                      hint="how much closer to its own opening the back third "
                           "sits than the front third; negative = walked away"))

    # --- 3. your own history with these tracks ------------------------------
    v, n = _mean(mood["familiarity"], path)
    sv, _ = _mean(mood["familiarity"], seed_idx)
    pct = _pctile(lb.get("familiarity"), v)
    axes.append(_axis("familiarity", "history", "Familiarity", v, n=n, seed=sv,
                      band=_band(lb.get("familiarity")), pct=pct, fmt="pctile",
                      verdict=_vs_band(pct, "deeper cuts than you ever queue",
                                       "typical depth", "your greatest hits"),
                      hint="mean play-count percentile within your library — 50 is "
                           "a median track, 90 one of your most-played. The band "
                           "sits HIGH by construction (a track you played 50 times "
                           "appears in 50 real sessions), so read this A-vs-B "
                           "rather than as a pass/fail"))
    v, n = _mean(mood["loved"], path)
    axes.append(_axis("loved", "history", "Saved / on repeat", v, n=n,
                      band=_band(lb.get("loved")),
                      pct=_pctile(lb.get("loved"), v), fmt="share",
                      verdict=_vs_band(_pctile(lb.get("loved"), v),
                                       "fewer keepers than usual", "typical",
                                       "unusually many keepers"),
                      hint="share of stops you saved or put on repeat"))
    v, n = _mean(mood["skip_rate"], path)
    pct = _pctile(lb.get("skip_rate"), v)
    axes.append(_axis("skip_rate", "history", "Historical skip rate", v, n=n,
                      band=_band(lb.get("skip_rate")), pct=pct, fmt="share",
                      # The one axis with an unambiguous bad end: being fed tracks
                      # you have historically bailed out of is never what you want.
                      tone="warn" if (pct is not None and pct > 90) else None,
                      verdict=_vs_band(pct, "you finish these", "typical",
                                       "you have skipped these"),
                      hint="mean rate at which you ABANDONED these tracks in the "
                           "past — high means it is feeding you songs you bail on"))

    # --- 4. era & genre movement -------------------------------------------
    v, n = _mean(mood["year"], path)
    sv, _ = _mean(mood["year"], seed_idx)
    axes.append(_axis("year", "era", "Mean year", v, n=n, seed=sv,
                      band=_band(lb.get("year")), fmt="year",
                      pct=_pctile(lb.get("year"), v),
                      verdict=_vs_seed(v, sv, _band(lb.get("year")),
                                       "newer", "older"),
                      hint="mean release year of the journey vs the seed's"))
    v, n = _step_jump(mood["year"], full)
    pct = _pctile(jb.get("year"), v)
    axes.append(_axis("year_jump", "era", "Year jump / step", v, n=n,
                      band=_band(jb.get("year")), pct=pct, fmt="years",
                      verdict=_vs_band(pct, "tighter era than you listen in",
                                       "natural", "time-travels"),
                      hint="mean |Δrelease year| between consecutive stops"))
    g = genre_ids[full]
    churn = 10.0 * float(np.mean(g[1:] != g[:-1])) if g.size > 1 else None
    pct = _pctile(bands.get("churn"), churn)
    axes.append(_axis("genre_churn", "era", "Genre switches / 10", churn,
                      n=int(full.size), band=_band(bands.get("churn")), pct=pct,
                      fmt="count",
                      verdict=_vs_band(pct, "stays in one lane",
                                       "natural", "channel-hops"),
                      hint="genre changes per 10 stops, against your own rate"))

    cov = int(np.sum(mood["_acoustic"][path]))
    return {
        "axes": axes,
        "coverage": {"acoustic": cov, "stops": int(path.size),
                     "share": round(cov / max(int(path.size), 1), 3)},
        "reference": {"sessions": int(bands.get("n_sessions", 0))},
        "groups": [
            {"key": "mood", "label": "Acoustic mood",
             "note": "where the journey sits on the axes people describe mood "
                     "with. The tick is your seed."},
            {"key": "flow", "label": "Continuity",
             "note": "transition statistics — the actual measure of whether an "
                     "algorithm conserves a vibe. Inside your band is the goal, "
                     "not zero."},
            {"key": "history", "label": "Your history",
             "note": "what you already did with these exact tracks."},
            {"key": "era", "label": "Era & genre",
             "note": "movement in time and style."},
        ],
    }


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
def phase(key: str, msg: str, **extra) -> None:
    """Announce a generation PHASE on stdout, which the server fans out over SSE
    to the lab (see `api::extend_events`).

    A journey takes ~8s and the bulk of it is invisible setup — loading the baked
    artifact, pulling 19k sonic vectors, calibrating bands against real sessions.
    A spinner for that is a lie of omission: it says "working" when the honest
    statement is "reading your listening history to calibrate the bands". Every
    phase below is emitted BEFORE the work it names, so the client always shows
    the step actually in flight rather than the last one finished."""
    emit({"kind": "phase", "phase": key, "msg": msg, **extra})


def extend(model_dir: Path, predictor: str, prm: dict) -> dict:
    """Generate a journey and return the full lab payload (stops + diagnostics)."""
    phase("artifact", "loading the model's baked vocabulary")
    art = load_artifact(model_dir)
    latents = art.item_latents
    artist_ids = _relevance_ids(art, "artist")
    genre_ids = _relevance_ids(art, "genre")
    genre_names = {int(k): (v.get("genre") or "—") for k, v in art.items.items()
                   if int(k) < art.n_items}
    phase("music", f"fetching sonic vectors for {art.n_items} tracks",
          items=art.n_items)
    music_vecs, music_mask = _load_music_vectors(art)
    phase("mood", "reading mood features (valence / energy / tempo / era)")
    mood = load_mood_table(art)
    phase("bands", "calibrating against your own listening sessions")
    bands = natural_bands(art, mood, music_vecs, music_mask, genre_ids)

    # Expand playlists/albums BEFORE resolution — resolve_prefix only knows
    # single tracks, so a pasted playlist URL would resolve to nothing.
    # Named separately from `seed`: expansion is the one phase that can stall on
    # something outside this machine (the Spotify Web API), so when a journey
    # hangs here the readout should say so rather than blame the model.
    if any(parse_container(t) for t in prm["seed"]):
        phase("expand", "expanding the seed playlist / album via Spotify")
    tokens, seed_info = expand_seed(list(prm["seed"]))
    phase("seed", "resolving the seed against the model's vocabulary")
    seed_idx, unknown = resolve_prefix(art, tokens)
    if seed_idx.size == 0:
        raise SystemExit(
            f"extend: none of the {len(tokens)} seed track(s) are in this "
            f"model's vocabulary ({art.n_items} items). The vocab is the "
            f"library the model was trained on; seed with tracks from it, or "
            f"with a playlist that overlaps it.")

    phase("scorer", f"loading the {predictor} weights")
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

    stops = rollout(art, score_fn, predict_latent_fn, seed_idx, core_idx, use_anchor,
                    anchor, prm, artist_ids, genre_ids, genre_names, latents,
                    music_vecs, music_mask, mood, mood_ref)

    phase("scoring", "scoring the journey against your bands")
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
        # The listener-facing read: same journey, described in terms that can be
        # confirmed or refuted by ear, each against this listener's own bands.
        "mood_report": mood_report(stops, seed_idx, art, mood, bands,
                                   music_vecs, music_mask, genre_ids),
    }


def title_keys(art, artist_ids: np.ndarray) -> np.ndarray:
    """Title identity for the dedupe constraint: the same recording appears in the
    vocab under multiple releases (single + album + edit), each its own item index,
    so index-level next-distinct does not stop a replay. Key on
    (normalized title, artist id)."""
    keys = np.full(art.n_items, -1, dtype=np.int64)
    vocab: dict[tuple, int] = {}
    for key, meta in art.items.items():
        i = int(key)
        if not (0 <= i < art.n_items):
            continue
        name = (meta.get("name") or "").strip().lower()
        if not name:
            continue
        # Strip the usual release-variant suffixes so "X" and "X - edit" collide.
        base = re.split(r"\s*[-–(\[]\s*", name)[0].strip() or name
        keys[i] = vocab.setdefault((base, int(artist_ids[i])), len(vocab))
    return keys


def rollout(art, score_fn, predict_latent_fn, seed_idx, core_idx, use_anchor,
            anchor, prm, artist_ids, genre_ids, genre_names, latents,
            music_vecs, music_mask, mood, mood_ref, narrate: bool = True) -> list:
    """THE autoregressive generation loop + the shared retrieval policy.

    Factored out of `extend` so that the offline rollout EVALUATION
    (`seq_continuation_eval.py rollout`) drives the identical code path instead of
    reimplementing the policy. A second implementation is the specific failure this
    module exists to prevent — see the module docstring: if the evaluator's policy
    drifted from the lab's, an offline rollout score would not describe the
    journeys anyone actually hears, and comparing two models would silently compare
    two policies.

    `narrate=False` silences the per-step SSE events for batch evaluation, where
    thousands of steps of progress chatter is noise rather than a readout."""
    rng = np.random.default_rng(int(prm["seed_rng"]))
    # Generation runs from the CORE when anchoring (the app's `begin()` rule), but
    # every loaded seed counts as used so nothing is replayed.
    seq = (core_idx if use_anchor else seed_idx).astype(np.int64).tolist()
    used = set(seed_idx.tolist())
    used_artists: list[int] = [int(artist_ids[i]) for i in seed_idx.tolist()]

    title_key = title_keys(art, artist_ids)
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
            # What it picked, in listener vocabulary, and what it turned down.
            "mood": {k: _finite(mood[k][choice])
                     for k, _label, _fmt in STOP_MOOD},
            "passed": passed_over(scores, pool_idx, choice, art, mood),
        }
        # Δ from the track this one is heard AFTER — the audible quantity. The
        # reference for step 1 is the last seed track, so the opening transition
        # is measured like every other one.
        prev = seq[-1] if seq else None
        if prev is not None:
            stop["delta"] = {}
            for k, _label, _fmt in STOP_MOOD:
                a, b = mood[k][prev], mood[k][choice]
                stop["delta"][k] = (round(float(b - a), 4)
                                    if np.isfinite(a) and np.isfinite(b) else None)
        # Per-stop grounding: how close this pick is to the seed's mood centroid.
        if mood_ref is not None and music_mask is not None and music_mask[choice]:
            stop["mood_sim"] = round(float(music_vecs[choice] @ mood_ref), 4)
        stops.append(stop)
        # The generation loop is the longest phase and used to be silent. Naming
        # each pick as it lands turns the wait into the thing being waited for —
        # you watch the journey get written, and a model that locks onto one
        # artist is visible while it happens instead of 8 seconds later.
        if narrate:
            emit({"kind": "step", "step": step + 1, "of": int(prm["steps"]),
                  "name": stop["name"], "artist": stop["artist"],
                  "genre": stop["genre"], "energy": _finite(mood["energy"][choice]),
                  "year": _finite(mood["year"][choice])})

        seq.append(choice)
        used.add(choice)
        used_artists.append(int(artist_ids[choice]))
        if title_key[choice] >= 0:
            played_titles.add(int(title_key[choice]))

    return stops


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
