#!/usr/bin/env python3
"""New candidate-scoring base learners for the next-track SEQUENCE task.

These are reusable *score-fn factories*: each `*_scorer(art, ...)` returns a
`score_fn(prefix: np.ndarray[int]) -> np.ndarray[float, n_items]` that plugs
directly into `seq_common.eval_from_scores` and the blend / stacker drivers
(same contract as `seq_baselines.markov_scorer`). Higher score = more likely
next item; the shared eval excludes prefix items and ranks the rest.

All learned legs are TRAIN-ONLY — statistics/embeddings are built from
`art.train_sessions` exclusively, so they stay leak-free on the chronological
split (mirroring `seq_baselines`). Content-kNN uses only the item latents
(content, never the split) and is leak-free by construction.

Legs
----
  content_knn_scorer(art, agg="max"|"mean")
      Pure content memory: score each candidate by the aggregated cosine
      similarity of its song-AE latent to the prefix items' latents. No
      training. Leak-free (content only).

  markov2_scorer(art, backoff=True)
      Second-order (trigram) Markov: from TRAIN sessions count
      (prev2, prev1) -> next over consecutive triples (self-loops on the
      predicted step dropped, same conventions as the first-order baseline);
      score keys on the prefix's last two items, smoothly backing off to
      first-order Markov (which itself carries an artist/genre + popularity
      tail) when the trigram context is unseen. Train-only.

  item2vec_scorer(art, dim=64, ...)
      Dense co-listening embedding (skip-gram w/ negative sampling) trained ON
      TRAIN SESSIONS ONLY (each train session's ordered item list is one
      "sentence"). Candidate score = cosine of its embedding to the pooled
      (mean) prefix embedding. Uses gensim.Word2Vec when importable, otherwise
      a compact torch SGNS implementation.

      NOTE (leakage): we deliberately do NOT reuse the corpus-wide
      pipeline/corpus/out/track_vectors.npy embedding — that is fit over the
      FULL play history and would leak the chronological test split into the
      candidate scorer. Everything here is refit from art.train_sessions only.

Run under the shared predictor venv (torch 2.12.0+cpu / numpy). gensim is
optional; absence just switches item2vec to the torch fallback.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from seq_common import (
    SeqArtifact,
    emit,
    eval_from_scores,
    load_artifact,
    make_fixture,
    write_outputs,
)
from seq_baselines import BACKOFF_K, markov_scorer, train_play_counts


# --------------------------------------------------------------------------- #
# 1. Content-kNN (no training; content only)                                  #
# --------------------------------------------------------------------------- #
def content_knn_scorer(art: SeqArtifact, agg: str = "max"):
    """Score each candidate by the cosine similarity of its song-AE latent to
    the prefix items' latents, aggregated (max or mean) over the prefix.

    Pure content memory — no train statistics are read, so this is leak-free
    (it never touches the train/test split). `agg="max"` = nearest-neighbour
    recall ("closest thing I've heard this session"); `agg="mean"` = centroid
    similarity ("fits the average of this session")."""
    assert agg in ("max", "mean"), "agg must be 'max' or 'mean'"
    # Row-normalize once so cosine is a plain dot product.
    latents = art.item_latents.astype(np.float64)
    norm = np.linalg.norm(latents, axis=1, keepdims=True)
    unit = latents / (norm + 1e-12)  # (n_items, D)

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        if prefix.size == 0:
            return np.zeros(unit.shape[0], dtype=np.float64)
        sub = unit[prefix]                       # (P, D)
        sims = unit @ sub.T                       # (n_items, P) cosine to each prefix item
        if agg == "max":
            return sims.max(axis=1)
        return sims.mean(axis=1)
    return score_fn


# --------------------------------------------------------------------------- #
# 2. Second-order Markov (trigram) with first-order back-off                  #
# --------------------------------------------------------------------------- #
def build_train_trigrams(art: SeqArtifact):
    """Count (prev2, prev1) -> next over consecutive triples in TRAIN sessions.

    Same conventions as the first-order baseline: consecutive (non-skip)
    positions only, and the immediate self-loop on the predicted step (c == b)
    is dropped since it carries no next-distinct routing signal. Returns, per
    (prev2, prev1) key, a tuple (probs, support) where probs is the normalized
    P(next | prev2, prev1) over observed targets and support is the raw total
    triple count for that key (used for the back-off trust weight)."""
    tri: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for s in art.train_sessions:
        seq = art.session(int(s))
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
            a, b, c = int(a), int(b), int(c)
            if c == b:
                continue  # self-loop on the predicted step
            tri[(a, b)][c] += 1.0
    cond: dict[tuple[int, int], tuple[dict[int, float], float]] = {}
    for key, tgts in tri.items():
        tot = float(sum(tgts.values()))
        if tot > 0:
            cond[key] = ({v: w / tot for v, w in tgts.items()}, tot)
    return cond


def markov2_scorer(art: SeqArtifact, backoff: bool = True):
    """Trigram scorer keyed on the prefix's last two items.

    With `backoff=True` the trigram distribution is trust-weighted against the
    first-order Markov scorer (seq_baselines.markov_scorer, which already blends
    track/artist/genre and a popularity tail), so the ranking is always total
    and unseen contexts degrade gracefully: trigram -> first-order -> artist/
    genre -> popularity. trust = n_key / (n_key + BACKOFF_K), reusing the
    baseline's back-off constant. With `backoff=False` only the trigram counts
    are used, with a tiny popularity tail purely to keep the order total."""
    tri = build_train_trigrams(art)
    n_items = art.n_items
    counts = train_play_counts(art)
    pop = counts / (counts.max() + 1e-9)         # weak tie-break tail
    base_fn = markov_scorer(art) if backoff else None

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        entry = None
        if prefix.size >= 2:
            key = (int(prefix[-2]), int(prefix[-1]))
            entry = tri.get(key)

        tri_scores = np.zeros(n_items, dtype=np.float64)
        n_key = 0.0
        if entry is not None:
            probs, n_key = entry
            vs = np.fromiter(probs.keys(), dtype=np.int64, count=len(probs))
            ws = np.fromiter(probs.values(), dtype=np.float64, count=len(probs))
            tri_scores[vs] = ws

        if backoff:
            base = base_fn(prefix).astype(np.float64)
            trust = n_key / (n_key + BACKOFF_K)   # 0 when context unseen
            return trust * tri_scores + (1.0 - trust) * base
        # No back-off: trigram counts + tiny popularity tail for a total order.
        return tri_scores + 1e-6 * pop
    return score_fn


# --------------------------------------------------------------------------- #
# 3. Item2vec (co-listening skip-gram; train-only)                            #
# --------------------------------------------------------------------------- #
def _train_item2vec_gensim(sentences, n_items, dim, window, epochs, neg, seed):
    from gensim.models import Word2Vec  # type: ignore

    model = Word2Vec(
        sentences=[[str(i) for i in s] for s in sentences],
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,                 # skip-gram
        negative=neg,
        epochs=epochs,
        workers=1,
        seed=seed,
    )
    emb = np.zeros((n_items, dim), dtype=np.float64)
    for i in range(n_items):
        key = str(i)
        if key in model.wv:
            emb[i] = model.wv[key]
    return emb


def _train_item2vec_torch(sentences, n_items, dim, window, epochs, neg, seed, lr):
    """Compact skip-gram with negative sampling (SGNS) in torch. Sentences are
    train sessions (ordered item lists). Trains input/output embeddings on
    (center, context) positive pairs against a unigram^0.75 noise distribution;
    the input embeddings are returned as the item vectors."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Build positive (center, context) pairs within the window.
    centers: list[int] = []
    contexts: list[int] = []
    unigram = np.zeros(n_items, dtype=np.float64)
    for seq in sentences:
        L = len(seq)
        for i in range(L):
            ci = int(seq[i])
            unigram[ci] += 1.0
            lo = max(0, i - window)
            hi = min(L, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                centers.append(ci)
                contexts.append(int(seq[j]))
    if not centers:
        return np.zeros((n_items, dim), dtype=np.float64)

    centers_t = torch.as_tensor(np.asarray(centers, dtype=np.int64))
    contexts_t = torch.as_tensor(np.asarray(contexts, dtype=np.int64))

    # Unigram^0.75 negative-sampling distribution.
    noise = unigram ** 0.75
    if noise.sum() == 0:
        noise[:] = 1.0
    noise = noise / noise.sum()
    noise_t = torch.as_tensor(noise, dtype=torch.float64)

    emb_in = torch.nn.Embedding(n_items, dim)
    emb_out = torch.nn.Embedding(n_items, dim)
    torch.nn.init.uniform_(emb_in.weight, -0.5 / dim, 0.5 / dim)
    torch.nn.init.zeros_(emb_out.weight)
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=lr)

    n_pairs = centers_t.shape[0]
    batch = 2048
    for _ in range(epochs):
        perm = torch.as_tensor(rng.permutation(n_pairs))
        for start in range(0, n_pairs, batch):
            idx = perm[start:start + batch]
            c = centers_t[idx]
            pos = contexts_t[idx]
            b = c.shape[0]
            negs = torch.multinomial(noise_t, b * neg, replacement=True).view(b, neg)

            v_c = emb_in(c)                              # (b, D)
            v_pos = emb_out(pos)                         # (b, D)
            v_neg = emb_out(negs)                        # (b, neg, D)

            pos_score = (v_c * v_pos).sum(dim=1)         # (b,)
            neg_score = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)  # (b, neg)
            loss = -(torch.nn.functional.logsigmoid(pos_score).mean()
                     + torch.nn.functional.logsigmoid(-neg_score).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()

    return emb_in.weight.detach().numpy().astype(np.float64)


def item2vec_scorer(art: SeqArtifact, dim: int = 64, window: int = 5,
                    epochs: int = 5, neg: int = 5, lr: float = 5e-3,
                    seed: int = 1337):
    """Train a co-listening item2vec on TRAIN SESSIONS ONLY and return a
    score_fn that ranks candidates by cosine to the pooled (mean) prefix
    embedding.

    Prefers gensim.Word2Vec (sg=1) when importable; otherwise falls back to the
    compact torch SGNS above. Never reuses the corpus-wide
    track_vectors.npy (that would leak the test split — see module docstring)."""
    sentences = [art.session(int(s)).astype(np.int64)
                 for s in art.train_sessions
                 if art.session(int(s)).shape[0] >= 2]

    try:
        import gensim  # noqa: F401
        emb = _train_item2vec_gensim(sentences, art.n_items, dim, window,
                                     epochs, neg, seed)
        backend = "gensim"
    except Exception:
        emb = _train_item2vec_torch(sentences, art.n_items, dim, window,
                                    epochs, neg, seed, lr)
        backend = "torch"

    unit = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    emit({"kind": "log", "msg":
          f"item2vec trained ({backend}) dim={dim} on {len(sentences)} "
          f"train sessions"})

    def score_fn(prefix: np.ndarray) -> np.ndarray:
        if prefix.size == 0:
            return np.zeros(unit.shape[0], dtype=np.float64)
        pooled = unit[prefix].mean(axis=0)                 # centroid of prefix
        pooled = pooled / (np.linalg.norm(pooled) + 1e-12)
        return unit @ pooled                                # (n_items,) cosine
    return score_fn


# --------------------------------------------------------------------------- #
# Registry + CLI                                                              #
# --------------------------------------------------------------------------- #
def build_scorer(art: SeqArtifact, mode: str, hp: dict):
    if mode == "content-knn":
        return content_knn_scorer(art, agg=hp.get("agg", "max"))
    if mode == "markov2":
        return markov2_scorer(art, backoff=bool(hp.get("backoff", True)))
    if mode == "item2vec":
        return item2vec_scorer(
            art,
            dim=int(hp.get("dim", 64)),
            window=int(hp.get("window", 5)),
            epochs=int(hp.get("epochs", 5)),
            neg=int(hp.get("neg", 5)),
            lr=float(hp.get("lr", 5e-3)),
            seed=int(hp.get("seed", 1337)),
        )
    raise ValueError(f"unknown mode {mode!r}")


MODES = ("content-knn", "markov2", "item2vec")


def run(dataset: Path, run_dir: Path, mode: str, hp: dict) -> None:
    art = load_artifact(dataset)
    emit({"kind": "log", "msg":
          f"seq-model {mode} on {art.manifest['dataset_id']}: "
          f"{art.n_items} items, test sessions {int(art.test_sessions.size)}"})
    k = int(hp.get("k", 10))
    score_fn = build_scorer(art, mode, hp)
    metrics, predictions = eval_from_scores(art, score_fn, k=k)
    metrics["model"] = mode
    write_outputs(run_dir, metrics, predictions)
    emit({"kind": "log", "msg":
          f"[{mode}] Recall@{k} {metrics['recall_at_k']:.3f}  "
          f"Recall@20 {metrics['recall_at_20']:.3f}  "
          f"MRR {metrics['mrr']:.3f}  hit@10 {metrics['hit_rate']:.3f}  "
          f"(n_test {metrics['n_test']})"})
    emit({"kind": "done"})


# --------------------------------------------------------------------------- #
# Fixture self-test                                                           #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Build each scorer on the synthetic fixture and push it through the
    shared eval; assert full-vocab shape, finiteness and metric ranges."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        ds = Path(td) / "fixture"
        make_fixture(ds, n_items=30, n_sessions=120, latent_dim=32, seed=7)
        art = load_artifact(ds)
        n_items = art.n_items

        cases = {
            "content-knn(max)": content_knn_scorer(art, agg="max"),
            "content-knn(mean)": content_knn_scorer(art, agg="mean"),
            "markov2(backoff)": markov2_scorer(art, backoff=True),
            "markov2(no-backoff)": markov2_scorer(art, backoff=False),
            "item2vec": item2vec_scorer(art, dim=32, epochs=8, seed=7),
        }

        # gensim availability report.
        try:
            import gensim  # noqa: F401
            gensim_avail = True
        except Exception:
            gensim_avail = False

        print(f"\n{'='*60}\nFIXTURE SELF-TEST  (n_items={n_items}, "
              f"n_test={int(art.test_sessions.size)})\n{'='*60}")
        print(f"gensim available: {gensim_avail}  "
              f"(item2vec backend: {'gensim' if gensim_avail else 'torch'})\n")

        # Sanity-check the raw score vectors on a representative prefix.
        probe = art.session(int(art.test_sessions[0]))[:-1].astype(np.int64)
        ok = True
        for name, fn in cases.items():
            v = np.asarray(fn(probe), dtype=np.float64)
            shape_ok = v.shape == (n_items,)
            finite_ok = bool(np.all(np.isfinite(v)))
            m, _ = eval_from_scores(art, fn, k=10)
            r10 = m["recall_at_10"]
            mrr = m["mrr"]
            range_ok = all(0.0 <= m[key] <= 1.0 for key in
                           ("recall_at_k", "recall_at_10", "recall_at_20",
                            "mrr", "hit_rate"))
            passed = shape_ok and finite_ok and range_ok
            ok = ok and passed
            print(f"  {name:22s} R@10={r10:.3f}  MRR={mrr:.3f}  "
                  f"shape={'ok' if shape_ok else 'BAD'} "
                  f"finite={'ok' if finite_ok else 'BAD'} "
                  f"range={'ok' if range_ok else 'BAD'}  "
                  f"{'PASS' if passed else 'FAIL'}")
        print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}\n")
        return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=False)

    tr = sub.add_parser("run", help="score a real SEQUENCE artifact")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument("--output", required=True, type=Path)
    tr.add_argument("--mode", required=True, choices=list(MODES))
    tr.add_argument("--hyperparams", default="{}")

    fx = sub.add_parser("fixture", help="write a synthetic SEQUENCE artifact")
    fx.add_argument("--output", required=True, type=Path)

    sub.add_parser("selftest", help="fixture self-test of every scorer")

    args = ap.parse_args()
    if args.cmd == "run":
        p = Path(args.hyperparams)
        hp = json.loads(p.read_text() if p.exists() else args.hyperparams)
        run(args.dataset, args.output, args.mode, hp)
    elif args.cmd == "fixture":
        make_fixture(args.output)
    else:
        # Default (no subcommand or `selftest`) runs the fixture self-test.
        raise SystemExit(_selftest())


if __name__ == "__main__":
    main()
