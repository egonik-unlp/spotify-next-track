#!/usr/bin/env python3
"""Next-track SEQUENCE meta-learner — the XGBoost stacker (the "X leg").

Where seq_blend.py fuses the base legs with a fixed z-normalized linear blend,
this module LEARNS the combination: a gradient-boosted tree (xgboost,
binary:logistic) that reranks candidate items from the base-leg scores plus a
handful of cheap candidate features. It is the ensemble's meta-learner.

TRAINING DATA (learning-to-rerank, pointwise). For every TRAIN session:
  * the true next item is one POSITIVE row (label 1),
  * N negative candidate items are sampled by popularity (label 0),
and each (session, candidate) row carries the feature vector

    [ markov_score(cand),          # M leg full-vocab score at cand  (if in legs)
      gru_cosine(cand),            # R leg full-vocab score at cand  (if in legs)
      ann_cosine(cand),            # A leg full-vocab score at cand  (if in legs)
      log(1 + play_count(cand)),   # popularity prior
      recency_of_cand_in_prefix,   # (last position of cand in prefix + 1)/len, else 0
      artist_match_to_prefix,      # fraction of prefix items sharing cand's artist
      genre_match_to_prefix ]      # fraction of prefix items sharing cand's genre

The leg-score columns are gated by `legs` (a subset of markov / gru / ann, in
that canonical order); the four candidate features are always present. The leg
score vectors come from the SAME per-session full-vocab scorers the rest of the
harness uses:
  * seq_baselines.markov_scorer   -> M
  * seq_blend.gru_score_fn(GRU)   -> R   (the trained seq_nexttrack GRU)
  * seq_ann.ann_score_fn(ANN)     -> A   (the trained seq_ann feed-forward twin)

EVAL. For a TEST session, the trained booster scores ALL vocab items (one row
per item) -> a full-vocab score vector -> seq_common.eval_from_scores applies
the identical prefix-exclusion (next-distinct) ranking + metrics as every other
run. So `stacker_score_fn` is drop-in wherever the harness expects a
`score_fn(prefix) -> length-n_items vector`.

Configurable ensemble: `legs=["markov","gru"]` = RNN+XGB, `["markov","gru","ann"]`
= ANN+RNN+XGB, etc. The base GRU/ANN are trained here from the same artifact
(reusing seq_nexttrack.fit / seq_ann.fit verbatim) unless pre-built scorers are
passed in.

CLI (mirrors the other sequence predictors; registry passes a `--hyperparams`
FILE, inline JSON also accepted). `legs` accepts a list or a "+"/","-joined
string (so the registry enum "markov+gru+ann" works):

  seq_stacker.py train --dataset <dir> --output <run_dir> --hyperparams <json|path>
  seq_stacker.py fixture --output <dir>          # write a synthetic artifact

Run under the shared predictor venv (torch 2.12.0+cpu / xgboost 3.2 / numpy)."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from seq_common import (
    emit,
    eval_from_scores,
    load_artifact,
    make_fixture,
    predict_ranking,
    write_outputs,
)
from seq_baselines import markov_scorer, train_play_counts
from seq_nexttrack import (
    DEFAULTS as GRU_DEFAULTS, SEED, SeqNextLatent, fit as gru_fit,
)
from seq_blend import gru_score_fn
from seq_ann import (
    DEFAULTS as ANN_DEFAULTS, AnnNextLatent, ann_score_fn, fit as ann_fit,
)

# Canonical leg order -> its feature-column name. Only legs listed in `legs`
# contribute a column, but always in THIS order so the feature matrix is stable.
CANONICAL_LEGS = ["markov", "gru", "ann"]
LEG_COLUMN = {"markov": "markov_score", "gru": "gru_cosine", "ann": "ann_cosine"}
# The four candidate features appended after the leg columns (always present).
CAND_FEATURES = ["log1p_play_count", "recency_in_prefix",
                 "artist_match", "genre_match"]

DEFAULTS = {
    "legs": ["markov", "gru", "ann"],   # subset of CANONICAL_LEGS
    "n_negatives": 50,                  # popularity-sampled negatives per session
    "xgb_rounds": 200,                  # boosting rounds (n_estimators)
    "xgb_max_depth": 6,
    "xgb_lr": 0.1,                      # xgboost eta
    "xgb_subsample": 0.8,
    "xgb_colsample": 0.8,
    "xgb_min_child_weight": 1.0,
    # Base-learner (GRU / ANN) training knobs. These override the base modules'
    # own DEFAULTS for the shared axes; the rest (arch, num_layers, tau, ...)
    # stay at the base module defaults.
    "base_hidden": 128,
    "base_epochs": 30,
    "base_lr": 1e-3,
    "base_loss": "infonce",
    "seed": SEED,                       # shared seed (base learners + xgboost)
    "k": 10,                            # Recall@k reported in metrics.recall_at_k
}


# --------------------------------------------------------------------------- #
# Hyperparams                                                                 #
# --------------------------------------------------------------------------- #
def parse_legs(value) -> list[str]:
    """Accept a JSON list (["markov","gru"]) or a "+"/","-joined string
    ("markov+gru+ann"), validate against CANONICAL_LEGS, dedupe into canonical
    order."""
    if isinstance(value, str):
        raw = [t for t in re.split(r"[+,\s]+", value.strip()) if t]
    else:
        raw = list(value)
    raw = [t.strip().lower() for t in raw]
    bad = [t for t in raw if t not in CANONICAL_LEGS]
    assert not bad, f"unknown legs {bad}; choose from {CANONICAL_LEGS}"
    legs = [leg for leg in CANONICAL_LEGS if leg in set(raw)]
    assert legs, "at least one leg is required"
    return legs


def load_hp(spec: str) -> dict:
    """Path to a JSON file (registry convention) or an inline JSON string.
    Merges over DEFAULTS, normalizes `legs`, validates."""
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    user = json.loads(raw)
    hp = {**DEFAULTS, **user}
    hp["legs"] = parse_legs(hp["legs"])
    hp["seed"] = int(hp["seed"])
    hp["n_negatives"] = int(hp["n_negatives"])
    hp["xgb_rounds"] = int(hp["xgb_rounds"])
    hp["xgb_max_depth"] = int(hp["xgb_max_depth"])
    assert hp["n_negatives"] >= 1 and hp["xgb_rounds"] >= 1
    return hp


def feature_names(legs: list[str]) -> list[str]:
    """The stacker's feature columns for a leg set (leg columns then the four
    candidate features), in the fixed canonical order."""
    return [LEG_COLUMN[leg] for leg in CANONICAL_LEGS if leg in legs] + list(CAND_FEATURES)


# --------------------------------------------------------------------------- #
# Item-level feature tables                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class ItemArrays:
    n_items: int
    play_count: np.ndarray        # (n_items,) float64
    artist_code: np.ndarray       # (n_items,) int64
    genre_code: np.ndarray        # (n_items,) int64
    n_artists: int
    n_genres: int


def build_item_arrays(art) -> ItemArrays:
    """Vectorizable per-item metadata: play_count and integer artist/genre codes
    (so artist/genre match is a bincount + gather rather than per-item dict
    lookups)."""
    n = art.n_items
    play_count = np.zeros(n, dtype=np.float64)
    artist_code = np.empty(n, dtype=np.int64)
    genre_code = np.empty(n, dtype=np.int64)
    artist_ids: dict[str, int] = {}
    genre_ids: dict[str, int] = {}
    for i in range(n):
        meta = art.items.get(str(i), {})
        play_count[i] = float(meta.get("play_count", 0) or 0)
        a = meta.get("artist", "?")
        g = meta.get("genre", "unknown")
        artist_code[i] = artist_ids.setdefault(a, len(artist_ids))
        genre_code[i] = genre_ids.setdefault(g, len(genre_ids))
    return ItemArrays(n, play_count, artist_code, genre_code,
                      len(artist_ids), len(genre_ids))


# --------------------------------------------------------------------------- #
# Feature builder                                                             #
# --------------------------------------------------------------------------- #
def build_feature_rows(prefix: np.ndarray, leg_vecs: dict[str, np.ndarray],
                       cand: np.ndarray, legs: list[str],
                       ia: ItemArrays) -> np.ndarray:
    """Feature matrix (len(cand), n_features) for the candidates of one prefix.

    * leg columns  : leg_vecs[leg][cand] for each leg in canonical order,
    * log1p_play_count,
    * recency_in_prefix : (last position of cand in prefix + 1)/len, else 0,
    * artist_match / genre_match : fraction of prefix items sharing cand's
      artist / genre.
    Columns follow feature_names(legs) exactly."""
    L = max(int(prefix.shape[0]), 1)

    # Recency: most-recent occurrence of each item in the prefix (later = higher).
    recency_full = np.zeros(ia.n_items, dtype=np.float64)
    for pos, item in enumerate(prefix):
        recency_full[int(item)] = (pos + 1) / L  # later positions overwrite -> last wins

    # Artist / genre presence fraction over the prefix, gathered by item code.
    artist_frac = np.bincount(ia.artist_code[prefix], minlength=ia.n_artists).astype(np.float64) / L
    genre_frac = np.bincount(ia.genre_code[prefix], minlength=ia.n_genres).astype(np.float64) / L

    cols: list[np.ndarray] = []
    for leg in CANONICAL_LEGS:
        if leg in legs:
            cols.append(leg_vecs[leg][cand])
    cols.append(np.log1p(ia.play_count[cand]))
    cols.append(recency_full[cand])
    cols.append(artist_frac[ia.artist_code[cand]])
    cols.append(genre_frac[ia.genre_code[cand]])
    return np.column_stack(cols).astype(np.float32)


# --------------------------------------------------------------------------- #
# Base legs                                                                   #
# --------------------------------------------------------------------------- #
def _gru_hp(hp: dict) -> dict:
    d = dict(GRU_DEFAULTS)
    d.update(hidden=hp["base_hidden"], epochs=hp["base_epochs"],
             lr=hp["base_lr"], loss=hp["base_loss"])
    return d


def _ann_hp(hp: dict) -> dict:
    d = dict(ANN_DEFAULTS)
    d.update(hidden=hp["base_hidden"], epochs=hp["base_epochs"],
             lr=hp["base_lr"], loss=hp["base_loss"], seed=hp["seed"])
    return d


def build_leg_scorers(art, legs: list[str], hp: dict,
                      ckpt_dir: Path | None = None, save: bool = False) -> dict:
    """Train / build the requested base legs and return {leg: score_fn(prefix)
    -> full-vocab vector}. Reuses seq_nexttrack.fit and seq_ann.fit verbatim, so
    the base learners are byte-identical to their standalone runs at the same
    hyperparameters + seed.

    Neural base legs (GRU/ANN) are snapshotted so a promoted stacker serves
    without retraining: with `save=True` their weights are written under
    `ckpt_dir` (`base_gru.pt` / `base_ann.pt`); at predict, if those checkpoints
    exist under `ckpt_dir` they are LOADED instead of retrained. The Markov leg
    is a cheap train-only stat, always rebuilt from the artifact."""
    scorers: dict = {}
    if "markov" in legs:
        scorers["markov"] = markov_scorer(art)
        emit({"kind": "log", "msg": "built train-only Markov scorer (M leg)"})
    if "gru" in legs:
        ghp = _gru_hp(hp)
        gp = (ckpt_dir / "base_gru.pt") if ckpt_dir else None
        if gp and gp.exists():
            gru_model = SeqNextLatent(art.latent_dim, ghp["hidden"], ghp["arch"],
                                      ghp["num_layers"], ghp["dropout"],
                                      bidirectional=ghp["bidirectional"],
                                      residual=ghp["residual"])
            gru_model.load_state_dict(torch.load(gp, map_location="cpu"))
            gru_model.eval()
            emit({"kind": "log", "msg": "loaded GRU base leg (R) from checkpoint"})
        else:
            emit({"kind": "log", "msg": "training GRU base leg (R)"})
            gru_model = gru_fit(art, ghp, seed=hp["seed"])
            if save and gp:
                torch.save(gru_model.state_dict(), gp)
        scorers["gru"] = gru_score_fn(gru_model, art.item_latents)
    if "ann" in legs:
        ahp = _ann_hp(hp)
        ap = (ckpt_dir / "base_ann.pt") if ckpt_dir else None
        if ap and ap.exists():
            ann_model = AnnNextLatent(art.latent_dim, ahp["hidden"], ahp["dropout"])
            ann_model.load_state_dict(torch.load(ap, map_location="cpu"))
            ann_model.eval()
            emit({"kind": "log", "msg": "loaded ANN base leg (A) from checkpoint"})
        else:
            emit({"kind": "log", "msg": "training ANN base leg (A)"})
            ann_model = ann_fit(art, ahp, seed=hp["seed"])
            if save and ap:
                torch.save(ann_model.state_dict(), ap)
        scorers["ann"] = ann_score_fn(ann_model, art.item_latents)
    return scorers


def _leg_vec_getter(legs: list[str], leg_scorers: dict):
    """Cache the (expensive) per-prefix full-vocab leg vectors by prefix bytes,
    exactly as seq_blend_eval caches the GRU/Markov vectors."""
    cache: dict[bytes, dict[str, np.ndarray]] = {}

    def get(prefix: np.ndarray) -> dict[str, np.ndarray]:
        key = prefix.tobytes()
        vecs = cache.get(key)
        if vecs is None:
            vecs = {leg: np.asarray(leg_scorers[leg](prefix), dtype=np.float64)
                    for leg in legs}
            cache[key] = vecs
        return vecs
    return get


def _sample_negatives(prob: np.ndarray, n: int, true: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Sample up to `n` negative item indices by popularity, excluding `true`.
    Sampled without replacement; if the pool is smaller than `n`, take it all."""
    pool = np.delete(np.arange(prob.shape[0]), true)
    p = prob[pool]
    tot = p.sum()
    if tot <= 0:
        p = None
    else:
        p = p / tot
    if pool.shape[0] <= n:
        return pool
    return rng.choice(pool, size=n, replace=False, p=p)


# --------------------------------------------------------------------------- #
# Stacker train / score                                                       #
# --------------------------------------------------------------------------- #
def build_training_matrix(art, legs: list[str], leg_scorers: dict, hp: dict,
                          ia: ItemArrays, get_legvec) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the pointwise learning-to-rerank matrix (X, y) over TRAIN
    sessions: 1 positive (true next) + N popularity-sampled negatives each."""
    rng = np.random.default_rng(hp["seed"])
    counts = train_play_counts(art)
    prob = counts / counts.sum() if counts.sum() > 0 else np.ones(art.n_items) / art.n_items
    n_neg = hp["n_negatives"]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_sessions = 0
    for s in art.train_sessions:
        seq = art.session(int(s))
        if seq.shape[0] < 2:
            continue
        prefix = seq[:-1].astype(np.int64)
        true = int(seq[-1])
        legvec = get_legvec(prefix)
        negs = _sample_negatives(prob, n_neg, true, rng)
        cand = np.concatenate([[true], negs]).astype(np.int64)
        X = build_feature_rows(prefix, legvec, cand, legs, ia)
        y = np.zeros(cand.shape[0], dtype=np.float32)
        y[0] = 1.0
        xs.append(X)
        ys.append(y)
        n_sessions += 1
    assert xs, "no usable train sessions (need length >= 2)"
    emit({"kind": "log", "msg":
          f"stacker training matrix: {n_sessions} sessions, "
          f"{sum(x.shape[0] for x in xs)} rows, {xs[0].shape[1]} features "
          f"(1 pos + up to {n_neg} neg / session)"})
    return np.vstack(xs), np.concatenate(ys)


def train_stacker(art, legs: list[str], leg_scorers: dict, hp: dict,
                  ia: ItemArrays | None = None) -> xgb.Booster:
    """Train the xgboost binary:logistic reranker over the base-leg scores +
    candidate features. Returns the trained Booster."""
    if ia is None:
        ia = build_item_arrays(art)
    fnames = feature_names(legs)
    get_legvec = _leg_vec_getter(legs, leg_scorers)
    X, y = build_training_matrix(art, legs, leg_scorers, hp, ia, get_legvec)

    dtrain = xgb.DMatrix(X, label=y, feature_names=fnames)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": hp["xgb_max_depth"],
        "eta": hp["xgb_lr"],
        "subsample": hp["xgb_subsample"],
        "colsample_bytree": hp["xgb_colsample"],
        "min_child_weight": hp["xgb_min_child_weight"],
        "seed": hp["seed"],
    }
    booster = xgb.train(params, dtrain, num_boost_round=hp["xgb_rounds"])
    emit({"kind": "log", "msg":
          f"trained xgboost stacker: {hp['xgb_rounds']} rounds, depth "
          f"{hp['xgb_max_depth']}, legs {'+'.join(legs)}, features {fnames}"})
    return booster


def stacker_score_fn(booster: xgb.Booster, art, legs: list[str],
                     leg_scorers: dict, ia: ItemArrays | None = None):
    """Return score_fn(prefix) -> full-vocab booster score vector — the "X leg"
    the ensemble harness feeds to eval_from_scores. Scores EVERY vocab item for
    the prefix; prefix exclusion (next-distinct) is applied downstream by
    eval_from_scores, so no exclusion is done here."""
    if ia is None:
        ia = build_item_arrays(art)
    fnames = feature_names(legs)
    all_idx = np.arange(art.n_items, dtype=np.int64)
    get_legvec = _leg_vec_getter(legs, leg_scorers)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        legvec = get_legvec(prefix)
        X = build_feature_rows(prefix, legvec, all_idx, legs, ia)
        return booster.predict(xgb.DMatrix(X, feature_names=fnames))
    return score_fn


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def train(dataset: Path, run_dir: Path, hp_spec: str) -> None:
    hp = load_hp(hp_spec)
    art = load_artifact(dataset)
    legs = hp["legs"]

    emit({"kind": "log", "msg":
          f"dataset {art.manifest['dataset_id']}: {art.n_items} items, "
          f"{int(art.test_sessions.size)} test sessions; stacker legs "
          f"{'+'.join(legs)} + candidate features"})

    run_dir.mkdir(parents=True, exist_ok=True)
    ia = build_item_arrays(art)
    # Snapshot the neural base legs into the run dir so a promoted stacker
    # serves without retraining them.
    leg_scorers = build_leg_scorers(art, legs, hp, ckpt_dir=run_dir, save=True)
    booster = train_stacker(art, legs, leg_scorers, hp, ia)
    score_fn = stacker_score_fn(booster, art, legs, leg_scorers, ia)

    metrics, predictions = eval_from_scores(art, score_fn, k=hp["k"])
    metrics["legs"] = legs
    metrics["features"] = feature_names(legs)

    write_outputs(run_dir, metrics, predictions)
    booster.save_model(str(run_dir / "stacker.json"))
    (run_dir / "hyperparams.json").write_text(json.dumps(hp))

    emit({"kind": "log", "msg":
          f"stacker ({'+'.join(legs)})  Recall@{hp['k']} {metrics['recall_at_k']:.3f}  "
          f"Recall@20 {metrics['recall_at_20']:.3f}  "
          f"MRR {metrics['mrr']:.3f}  hit@10 {metrics['hit_rate']:.3f}  "
          f"(n_test {metrics['n_test']})"})
    emit({"kind": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Serving-time ranking: reload the XGB booster + the snapshotted base legs
    and rank the vocab for the caller's session prefix (see
    seq_common.predict_ranking). Base-leg checkpoints (base_gru.pt/base_ann.pt)
    baked at promote mean no retraining here."""
    hp = load_hp(str(model_dir / "hyperparams.json"))
    art = load_artifact(model_dir)
    legs = hp["legs"]
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "stacker.json"))
    leg_scorers = build_leg_scorers(art, legs, hp, ckpt_dir=model_dir)
    score_fn = stacker_score_fn(booster, art, legs, leg_scorers)
    predict_ranking(art, score_fn, input_dir, output, k=hp["k"])
    emit({"kind": "done"})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument("--output", required=True, type=Path)
    tr.add_argument("--hyperparams", required=True)  # file path or inline JSON
    pr = sub.add_parser("predict", help="rank the next track for a session prefix")
    pr.add_argument("--model", required=True, type=Path)
    pr.add_argument("--input", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    fx = sub.add_parser("fixture", help="write a synthetic SEQUENCE artifact")
    fx.add_argument("--output", required=True, type=Path)
    fx.add_argument("--n-items", type=int, default=20)
    fx.add_argument("--n-sessions", type=int, default=40)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    if args.cmd == "train":
        train(args.dataset, args.output, args.hyperparams)
    elif args.cmd == "predict":
        predict(args.model, args.input, args.output)
    else:
        make_fixture(args.output, n_items=args.n_items, n_sessions=args.n_sessions)


if __name__ == "__main__":
    main()
