# Next-track: graded-relevance metrics + new base learners — 2026-07-14

Two threads, one campaign. **(A)** Make the evaluation credit landing the right
*band / vibe* even when the exact track is wrong — the observation that "the
first choice is usually not the exact track, but the artist is usually right".
**(B)** Try genuinely NEW leak-free base learners (the prior campaign's verdict
was that the ceiling needs new *signal*, not more fusion of the same kind) and
score them on both the exact and the new graded metrics.

## A — graded-relevance metrics (now first-class)

`predictors/seq_common.eval_from_scores` now also emits, over the SAME
prefix-excluded ranking as the exact metric:
- **artist@k** — fraction of test sessions where some top-k candidate shares the
  true next track's artist,
- **genre@k** — likewise on primary genre,
- **artist_mrr** — MRR of the first same-artist candidate over the full ranking.

Wired end-to-end as first-class metrics: `Metrics.artist_recall_at_k /
genre_recall_at_k / artist_mrr` (Rust), `MetricsSpec` name-maps
(`artist@k`/`genre@k`/`artist_mrr`, higher-better), `domain.toml`
`[metrics].columns = [recall@10, artist@10, genre@10, mrr, hit@10]` +
`[sequence].relevance_fields = [artist, genre]`, and the UI metric table
(0..1 ratios). The exact-metric path is byte-identical (purely additive keys);
the champion reproduces R@10 0.1726 / MRR 0.0963 exactly.

**Why it matters:** exact next-track recall is modest by construction (55% of
test targets are cold). The graded metrics show the model is far more useful than
the 17% exact number suggests — the champion lands the right **artist in the
top-10 ~32%** of the time and the right **genre ~40%**, and surfaces a same-artist
track at rank ~3 on average (artist_mrr 0.30). This is the honest read of "how
close" the recommendations are.

## B — new base learners (leak-free), scored on exact + graded

Artifact `seq-20260713-014226` (7,154 sessions / 19,402 items; leak-free
chronological split; cold-item rate 0.55). Leave-last-out next-distinct eval;
bootstrap 95% CI over 1,431 test sessions (2,000 resamples) + paired per-session
Δ vs the champion. GRU leg = the on-record champion checkpoint (no retrain). New
legs from `predictors/seq_models.py`, each a full-vocab `score_fn(prefix)`:
- **C = content-kNN** — cosine of each candidate's 64-dim song-AE latent to the
  prefix latents (max-agg). No training; content only (leak-free).
- **M2 = 2nd-order Markov** — train-only trigram `(prev2,prev1)→next` with
  trust-weighted back-off to the first-order Markov (M).
- **I = item2vec** — skip-gram co-listening embedding trained on TRAIN sessions
  only (torch SGNS; gensim absent), candidate = cosine to mean-pooled prefix.
  Deliberately NOT the full-history `track_vectors.npy` (that leaks the split).

### Standalone — none beats the bar

| model | R@10 [95% CI] | artist@10 | genre@10 | MRR | artist_mrr |
|---|---|---|---|---|---|
| M  first-order Markov (bar) | 0.1069 [0.092, 0.124] | 0.298 | 0.389 | 0.0694 | 0.280 |
| **M2 2nd-order Markov** | 0.1076 [0.092, 0.124] | 0.299 | 0.389 | 0.0724 | 0.282 |
| R  GRU-infonce h256 | 0.0957 [0.081, 0.112] | 0.294 | **0.450** | 0.0411 | 0.218 |
| C  content-kNN (max) | 0.0636 [0.050, 0.077] | 0.272 | 0.405 | 0.0321 | 0.197 |
| C  content-kNN (mean) | 0.0559 [0.044, 0.068] | 0.226 | 0.396 | 0.0295 | 0.167 |
| I  item2vec (train-only) | 0.0517 [0.041, 0.063] | 0.166 | 0.312 | 0.0214 | 0.113 |

- **2nd-order Markov ≈ first-order** (0.1076 vs 0.1069, CIs identical): the
  trigram back-off falls through to the bigram almost always — sessions are too
  short / sparse for a trigram to fire. No new signal.
- **item2vec is the weakest leg** — a co-listening SGNS trained on only 5.7k
  train sessions is too sparse to localize; also worst on artist/genre.
- **content-kNN is weak standalone** on exact recall but its genre@10 (0.405)
  already edges the Markov bar — content captures *vibe* the counts miss.

### Blended into the champion — content is the only additive signal

| blend (z-norm over candidates) | R@10 [95% CI] | artist@10 | genre@10 | MRR | artist_mrr | Δ R@10 vs ★ [95% CI] |
|---|---|---|---|---|---|---|
| **★ R+M (CHAMPION)** | 0.1726 [0.154, 0.192] | 0.3215 | 0.3969 | 0.0963 | 0.3006 | — |
| **R+M+C (+content)** | **0.1740** [0.154, 0.194] | **0.3298** | 0.4004 | **0.1014** | **0.3091** | +0.0014 [−0.010, +0.013] |
| R+M2 (M2 for M) | 0.1733 [0.154, 0.192] | 0.3222 | 0.3976 | 0.0988 | 0.3021 | +0.0007 [+0.000, +0.002] |
| R+M+M2 (+markov2) | 0.1733 [0.154, 0.193] | 0.3208 | 0.3941 | 0.0976 | 0.2996 | +0.0007 [−0.003, +0.004] |
| M+C (markov+content) | 0.1705 [0.151, 0.190] | 0.3229 | 0.3899 | 0.0946 | 0.3007 | −0.0021 [−0.015, +0.011] |
| R+C (gru+content) | 0.1118 [0.096, 0.129] | 0.3277 | **0.4689** | 0.0534 | 0.2655 | −0.0608 [−0.079, −0.044] |
| R+M+I (+item2vec) | 0.1349 [0.118, 0.152] | 0.3270 | 0.3990 | 0.0870 | 0.2959 | −0.0377 [−0.053, −0.022] |
| R+M+C+I (all four) | 0.1502 [0.133, 0.168] | 0.3263 | 0.4011 | 0.0945 | 0.3054 | −0.0224 [−0.036, −0.008] |

- **R+M+C is the best all-around leg set.** It does NOT significantly beat the
  champion on exact recall (paired Δ +0.0014, CI straddles 0 → a tie), but it is
  the top blend on **MRR (0.0963→0.1014), artist@10 (0.3215→0.3298),
  artist_mrr (0.3006→0.3091)** and nudges genre up too — a small, consistent lift
  on exactly the "right band/vibe" axis this campaign set out to measure. Being a
  fixed-GRU + deterministic-content + deterministic-Markov blend, it has no seed
  variance to confirm.
- **content is the only genuinely additive signal**, exactly as the prior
  campaign predicted. It ranks *different* correct/near-correct tracks (latent
  vibe similarity) than the GRU (session recurrence) and Markov (co-occurrence).
- **item2vec DRAGS the blend** (R+M+I 0.135 ≪ champion): a noisy leg with a low
  ceiling pulls the z-blend down. Drop it.
- **2nd-order Markov is neutral** — swapping/adding it moves nothing (Δ CI at 0).
- **R+C has the best genre@10 (0.469)** of any config — content + GRU are both
  vibe-driven — but collapses on exact recall; Markov's count signal is essential
  to the exact win.

## Verdict

**Champion unchanged for the exact-recall board: R+M z-blend (0.1726).** No new
base learner beats it on exact recall (every paired-Δ CI straddles or sits below
0). **But the graded metrics are now the honest scorecard**, and on them the
model is much stronger than 17% exact suggests (artist@10 ~0.32, genre@10 ~0.40).
**Content-kNN is the one new signal worth keeping**: as a third leg (R+M+C) it
gives a small, consistent lift on MRR + artist metrics at no exact-recall cost —
promote it to a graded-preferred variant / follow-up, not a new exact champion.
Co-listening item2vec and trigram Markov add nothing on this data.

## Follow-ups
- Register **R+M+C** as a 3-leg blend variant (needs `seq_blend` generalized past
  2 legs) and 3-seed it if the GRU leg is retrained; report it as the
  graded-preferred model.
- Content signal has headroom: try acoustic (af_*) features per candidate and a
  learned content projection (not raw song-AE cosine) as the C leg.
- Still open from prior campaigns: longer/cross-session context; a `predict`
  subcommand to make ranking models best-models-promotable + app-exportable.

## Provenance
Off-server driver `predictors/seq_partb_eval.py` (reuses `eval_from_scores`, the
champion GRU checkpoint, `seq_baselines.markov_scorer`, `seq_models`
content-kNN/markov2/item2vec). Results JSON in the session scratchpad.
Graded metrics landed in `seq_common.py` + Rust `Metrics`/`MetricsSpec` +
`domain.toml` + UI; `:8100` restarted to surface the columns. gensim absent →
item2vec used the torch SGNS fallback. No new champion; `seq-blend` stands.
