#!/usr/bin/env python3
"""Multi-step CONTINUATION eval for the next-track sequence models.

The primary leaderboard eval (seq_common.eval_from_scores) is leave-LAST-out:
prefix = seq[:-1], truth = the single last item, recall@10 = "did the one
literal next track land in top-10". That metric structurally rewards eager,
bigram-like prediction (predicting the rest of the album). This harness measures
the complementary "holisticness" signal: how well the model predicts the whole
CONTINUATION of a session, not just the literal next track.

It is deliberately an ADDITIVE, on-demand harness (modelled on seq_partb_eval.py)
rather than a change to eval_from_scores, so the primary path and every run's
recall@10 stay byte-identical and comparable to all historical runs. It reuses
seq_common.load_artifact and the exact next-distinct ranking rules, and reuses
the model's own score_fn (GRU checkpoint or a baseline scorer) — no retrain.

Metrics (both higher-better), for a hold-out of the last m items:
  prefix' = seq[:-m]  (score FROM here)   S = seq[-m:]  (true continuation set)
  suffix_recall@k = |S_distinct ∩ topk| / |S_distinct|
  cont_prec@k     = |topk ∩ S| / k
where S_distinct = suffix items NOT already in prefix' (a replayed track is
unrecoverable under the next-distinct exclusion, so it is dropped from the
denominator rather than unfairly capping the metric). We exclude ONLY prefix'
from the candidate pool — never S — so the answers stay rankable. No leakage:
splits are by session and teacher forcing draws only from TRAIN sessions, so
holding out a TEST session's suffix exposes no training data.

Config (CLI flags override env; env defaults suit the local instance):
  LENSING_CONTINUATION_M   number of trailing items to hold out (default 3)
Run under the shared predictor venv (torch only needed for --model)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seq_common import load_artifact, SeqArtifact  # noqa: E402
from seq_baselines import SCORERS  # noqa: E402


def rollout_eval(model_dir: Path, predictor: str, art: SeqArtifact, bands_path,
                 sessions: int, prefix_len: int, steps: int, seed: int = 1337):
    """AUTOREGRESSIVE rollout eval — the regime the product actually runs in.

    Every other metric in this project (including `continuation_eval` below)
    scores ONE ranked list produced from a REAL prefix. This one hands the model
    its own output: prefix -> pick -> append -> pick again, `steps` times, exactly
    as the playlist lab does, then measures whether the generated session HOLDS
    its mood or decays across its own halves (`hold@N`, tools/rollout_hold.py).

    Why a separate on-demand harness rather than a new column in
    `eval_from_scores`: a rollout costs `steps` full-vocab scoring passes per
    session against the primary path's ONE, so folding it in would multiply every
    run's eval cost by ~20x. Sampled and on-demand keeps it affordable AND keeps
    every historical run's recall@10 byte-comparable — the same reasoning that
    made `continuation_eval` additive.

    The generation loop is IMPORTED from seq_extend, never reimplemented: the
    retrieval policy (artist penalty/cooldown, mood anchor, dedupe, temperature)
    has to be the one the lab actually serves, or an offline score would describe
    journeys nobody hears.
    """
    sys.path.insert(0, str(HERE.parent / "tools"))
    import seq_extend as se
    from rollout_hold import score as hold_score
    from seq_common import _load_music_vectors, _relevance_ids

    artist_ids = _relevance_ids(art, "artist")
    genre_ids = _relevance_ids(art, "genre")
    genre_names = {int(k): (v.get("genre") or "—") for k, v in art.items.items()
                   if int(k) < art.n_items}
    music_vecs, music_mask = _load_music_vectors(art)
    mood = se.load_mood_table(art)
    if not bands_path:
        raise SystemExit(
            "rollout eval needs FADE bands: run `tools/rollout_hold.py calibrate "
            "--model <dir> --out bands.json` first, then pass --bands")
    bands = json.loads(Path(bands_path).read_text())
    # The LEVEL/jump bands come from the artifact (mood_report computes them); only
    # the FADE bands are pre-calibrated, since they are the expensive pass.
    live = se.natural_bands(art, mood, music_vecs, music_mask, genre_ids)
    live["fade"] = bands["fade"]

    score_fn, predict_latent_fn, _info = se.build_scorer(predictor, model_dir, art)
    prm = dict(se.DEFAULTS)
    prm["steps"] = steps

    rng = np.random.default_rng(seed)
    pool = [int(s) for s in art.test_sessions.tolist()
            if art.session(int(s)).size >= prefix_len + 2]
    if not pool:
        raise SystemExit(f"no test session has >= {prefix_len + 2} items")
    pick = rng.choice(len(pool), size=min(sessions, len(pool)), replace=False)

    rows = []
    for n, pi in enumerate(sorted(pick), 1):
        s = pool[pi]
        prefix = art.session(s).astype(np.int64)[:prefix_len]
        core_idx, anchor = se.dominant_core(prefix, music_vecs)
        use_anchor = (anchor is not None and float(prm["anchor_lambda"]) > 0
                      and core_idx.size >= int(prm["anchor_min_core"]))
        mood_ref = None
        if music_vecs is not None and music_mask is not None:
            cov = prefix[music_mask[prefix]]
            if cov.size:
                c = music_vecs[cov].mean(axis=0)
                nrm = float(np.linalg.norm(c))
                if nrm > 0:
                    mood_ref = c / nrm
        stops = se.rollout(art, score_fn, predict_latent_fn, prefix, core_idx,
                           use_anchor, anchor, prm, artist_ids, genre_ids,
                           genre_names, art.item_latents, music_vecs, music_mask,
                           mood, mood_ref, narrate=False)
        report = se.mood_report(stops, prefix, art, mood, live, music_vecs,
                                music_mask, genre_ids)
        r = hold_score({"stops": stops, "mood_report": report}, live)
        rows.append({"session": s, "anchored": bool(use_anchor),
                     "core": int(core_idx.size),
                     **{k: r[k] for k in ("hold", "fade", "stasis", "level_dev")}})
        if n % 10 == 0:
            print(f"  {n}/{len(pick)} rollouts", flush=True)

    ok = [r for r in rows if r["hold"] is not None]
    agg = {"n_rollouts": len(rows), "n_scored": len(ok), "prefix_len": prefix_len,
           "steps": steps,
           "anchored_share": round(sum(r["anchored"] for r in rows) / max(len(rows), 1), 3)}
    for key in ("hold", "fade", "stasis", "level_dev"):
        vals = [r[key] for r in ok if r[key] is not None]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 5) if vals else None
    return agg, rows


def continuation_eval(art: SeqArtifact, score_fn, k: int = 10, m: int = 3):
    """Leave-last-m-out continuation eval. Returns (metrics, per_session) where
    metrics carries suffix_recall_at_k / cont_prec_at_k (omitted if no session
    qualified) plus n_continuation / continuation_m for provenance."""
    suffix_recalls: list[float] = []
    cont_precs: list[float] = []
    n_used = 0
    for s in art.test_sessions:
        s = int(s)
        seq = art.session(s).astype(np.int64)
        if seq.shape[0] < m + 1:
            continue
        prefix = seq[:-m]
        suffix = seq[-m:]

        scores = np.asarray(score_fn(prefix), dtype=np.float64).copy()
        assert scores.shape[0] == art.n_items, "score_fn must cover the full vocab"
        # Exclude ONLY the (shortened) prefix — the continuation stays rankable.
        scores[prefix] = -np.inf
        order = np.argsort(-scores, kind="stable")
        topk = [int(i) for i in order[:k]]

        prefix_set = set(int(x) for x in prefix.tolist())
        topk_set = set(topk)
        # De-duplicated suffix items that are actually recoverable (not replays
        # of something already in the prefix, which the exclusion removed).
        s_distinct = [x for x in dict.fromkeys(int(v) for v in suffix.tolist())
                      if x not in prefix_set]
        if not s_distinct:
            continue
        n_used += 1

        hit = sum(1 for x in s_distinct if x in topk_set)
        suffix_recalls.append(hit / len(s_distinct))
        suffix_set = set(int(v) for v in suffix.tolist())
        cont_precs.append(sum(1 for i in topk if i in suffix_set) / k)

    metrics: dict = {"continuation_m": m, "n_continuation": n_used}
    if suffix_recalls:
        metrics["suffix_recall_at_k"] = sum(suffix_recalls) / len(suffix_recalls)
    if cont_precs:
        metrics["cont_prec_at_k"] = sum(cont_precs) / len(cont_precs)
    return metrics, {"suffix_recall": suffix_recalls, "cont_prec": cont_precs}


def build_gru_score_fn(art: SeqArtifact, ckpt_dir: Path):
    """A cached cosine-to-vocab score_fn from a trained SeqNextLatent checkpoint
    (hyperparams.json + model.pt), identical to seq_partb_eval's builder."""
    import torch
    from seq_nexttrack import SeqNextLatent  # noqa: E402

    hp = json.loads((ckpt_dir / "hyperparams.json").read_text())
    model = SeqNextLatent(art.latent_dim, hp["hidden"], hp.get("arch", "gru"),
                          hp.get("num_layers", 1), hp.get("dropout", 0.0),
                          bidirectional=hp.get("bidirectional", False),
                          residual=hp.get("residual", False))
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    model.eval()
    latents = art.item_latents
    item_norm = torch.nn.functional.normalize(torch.from_numpy(latents), dim=1)
    cache: dict[bytes, np.ndarray] = {}

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        key = prefix.tobytes()
        v = cache.get(key)
        if v is None:
            x = torch.from_numpy(latents[prefix][None, :, :])
            with torch.no_grad():
                pred = torch.nn.functional.normalize(model.predict_next(x), dim=1)
                v = (pred @ item_norm.t()).squeeze(0).numpy().astype(np.float64)
            cache[key] = v
        return v
    return score_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path,
                    help="SEQUENCE artifact directory (sequence-manifest.json + binaries)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", type=Path,
                     help="trained SeqNextLatent checkpoint dir (hyperparams.json + model.pt)")
    src.add_argument("--baseline", choices=sorted(SCORERS.keys()),
                     help="use a baseline scorer instead of a checkpoint")
    ap.add_argument("-k", type=int, default=10, help="ranking cutoff (default 10)")
    ap.add_argument("-m", "--continuation-m", type=int,
                    default=int(os.environ.get("LENSING_CONTINUATION_M", "3")),
                    help="trailing items to hold out (default env LENSING_CONTINUATION_M or 3)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the metrics JSON here (default: print only)")
    # --- autoregressive rollout mode (see rollout_eval) ---
    ap.add_argument("--rollout", action="store_true",
                    help="autoregressive rollout eval (hold@N) instead of the "
                         "leave-last-m-out continuation eval")
    ap.add_argument("--predictor", default=None,
                    help="registry predictor name (required by --rollout; a model "
                         "dir does not record which predictor produced it)")
    ap.add_argument("--bands", type=Path, default=None,
                    help="fade bands from tools/rollout_hold.py calibrate")
    ap.add_argument("--rollout-sessions", type=int, default=100)
    ap.add_argument("--rollout-prefix", type=int, default=30)
    ap.add_argument("--rollout-steps", type=int, default=20)
    args = ap.parse_args()

    art = load_artifact(args.dataset)

    if args.rollout:
        if args.model is None or not args.predictor:
            raise SystemExit("--rollout needs --model <promoted dir> and --predictor")
        agg, rows = rollout_eval(args.model, args.predictor, art, args.bands,
                                 args.rollout_sessions, args.rollout_prefix,
                                 args.rollout_steps)
        agg["source"] = f"model:{args.model.name}"
        agg["predictor"] = args.predictor
        print(f"\nrollout eval — {agg['source']} ({args.predictor})")
        print(f"  sessions        {agg['n_scored']}/{agg['n_rollouts']} scored"
              f"  (prefix {agg['prefix_len']}, steps {agg['steps']}, "
              f"anchored {agg['anchored_share']:.0%})")
        for key in ("hold", "fade", "stasis", "level_dev"):
            v = agg[f"{key}_mean"]
            print(f"  {key + '@N':<14}  {'—' if v is None else f'{v:.5f}'}")
        if args.out is not None:
            args.out.write_text(json.dumps({**agg, "per_session": rows}, indent=1))
            print(f"wrote {args.out}")
        return

    if args.model is not None:
        score_fn = build_gru_score_fn(art, args.model)
        source = f"model:{args.model.name}"
    else:
        score_fn = SCORERS[args.baseline](art)
        source = f"baseline:{args.baseline}"

    metrics, _ = continuation_eval(art, score_fn, k=args.k, m=args.continuation_m)
    metrics["source"] = source
    print(f"dataset: {art.n_items} items, {int(art.test_sessions.size)} test sessions; "
          f"{source}, m={args.continuation_m}, k={args.k}", flush=True)
    print(f"  n_continuation   {metrics['n_continuation']}", flush=True)
    print(f"  suffix_recall@{args.k}  {metrics.get('suffix_recall_at_k', float('nan')):.4f}",
          flush=True)
    print(f"  cont_prec@{args.k}      {metrics.get('cont_prec_at_k', float('nan')):.4f}",
          flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
