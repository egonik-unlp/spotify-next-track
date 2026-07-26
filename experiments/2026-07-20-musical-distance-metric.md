# Musical-distance metric (`music@10`) + offline backfill comparison

**Date:** 2026-07-20
**Scope:** introduce a symmetric "musical distance" between tracks, promote it as
a leaderboard metric (`music@10`) alongside recall@10 / artist@10 / genre@10 /
mrr / hit@10, and back-fill it across the existing next-track runs.

## 1. What the metric is

A proper distance `d(a,b)` over INTRINSIC musical content — the sonic counterpart
to the behavioral co-listening space in `spotify_tracks`. Built as a
block-weighted, partial-whitened, L2-normalized fusion of the 64-d Song-AE latent
(whitened), the 384-d name/tag text embedding, 5 metadata numerics, and a
per-genre AE prototype; Euclidean on the stored vectors is a true metric
(`= sqrt(2 - 2·cos)`, so Qdrant Cosine ranks identically).

Materialized by `pipeline/build_content_metric.py` into the Qdrant collection
`spotify_tracks_content_metric` (23,529 pts) with three named vectors that form a
tightness↔discovery **dial**:

- **`balanced`** (517-d, DEFAULT) — coherent, keeps cross-artist variety.
- **`tight`** (517-d) — maximizes genre/artist purity (leans same-artist).
- **`sonic`** (64-d, AE-only) — most cross-artist "sounds-alike" discovery.

Design + validation background: the intrinsic space is nearly orthogonal to
behavioral co-listening (pairwise Spearman ≈ 0.04); the whitened Song-AE latent
is the workhorse (genre@10 kNN purity 0.40→0.66 with whitening) and raw ReccoBeats
acoustics alone are near-noise for musical similarity (genre@10 ≈ 0.06, the MIR
"glass ceiling"). The v1 fusion reaches genre@10 ≈ 0.85 / artist@10 ≈ 0.61 kNN
purity; `balanced` trades some purity for cross-artist variety.

## 2. `music@10` as a run metric

Continuous "sounds-alike" graded relevance:

> **music@10** = mean over test cases of the best (max) cosine, in the `balanced`
> musical-distance space, between the true next track and the top-10 retrieved
> candidates.

A top-10 that contains the exact truth scores 1.0 (self-similarity), so `music@10`
is a **smooth superset of recall@10**: it credits landing something that *sounds
like* the truth even on an exact/artist/genre miss. 0..1, higher-better.
DISPLAY-ONLY — the primary metric stays recall@10.

Touch points (integrated, verified, live 2026-07-20):
`predictors/seq_common.py` (`_load_music_vectors` + computation in
`eval_from_scores`; best-effort — omitted, never fails, if the index is absent),
`crates/lensing-core/src/run.rs` (`Metrics.music_at_k`),
`crates/lensing-core/src/domain.rs` (`extract`/`extract_name_known` `music` arm),
`domain.toml` `[metrics].columns`, `ui/src/lib/metrics.ts` + `ui/src/api/types.ts`.
A companion `GET /api/pathfinder/similar?id=&uri=&dial=&k=` serves "songs like
this" over the same index.

## 3. Offline backfill methodology

`music@10` is computed at eval time, so the 53 pre-existing runs had it blank. It
was recomputed OFFLINE and NON-DESTRUCTIVELY (nothing written under `data/`) from
each run's stored `predictions.json` (`actual` truth index + `top_k_ids`), the
run's dataset `items.json` (item index → uri → sha256 point id), and the
`balanced` vectors — identical formula to the live metric. All 53 runs share the
same 1431-case test split, so the column is directly comparable.

Sanity: **corr(recall@10, music@10) = 0.609** across the 53 runs — correlated
(a better ranker also lands closer misses) but far from redundant, i.e. it carries
distinct signal.

## 4. Per-family summary

| predictor family | runs | best recall@10 | best music@10 | mean music@10 |
|---|--:|--:|--:|--:|
| seq-blend | 23 | 0.223 | 0.461 | 0.441 |
| seq-nexttrack | 24 | 0.159 | **0.482** | **0.448** |
| seq-markov | 1 | 0.107 | 0.407 | 0.407 |
| seq-ann | 1 | 0.090 | 0.426 | 0.426 |
| seq-stacker | 3 | 0.087 | 0.418 | 0.411 |
| seq-popularity | 1 | 0.001 | 0.185 | 0.185 |

**Headline finding:** the recall@10 champion is NOT the music@10 champion.
`seq-blend` wins exact-hit recall (0.223), but `seq-nexttrack` — which regresses
the next track's content latent directly and retrieves by cosine — lands the
**musically closest** predictions (best music@10 = 0.482 at `d3d4a`, and the
highest family mean). When a next-track model misses the exact track, it still
tends to surface something that *sounds* right. The `seq-popularity` baseline is
correctly worst on music@10 (0.185), confirming the metric is not trivially
saturated.

## 5. Full leaderboard (all 53 runs, one shared 1431-case test split)

| # | run | predictor | recall@10 | artist@10 | genre@10 | **music@10** | mrr | n |
|--:|---|---|--:|--:|--:|--:|--:|--:|
| 1 | a781c-seq-blend | seq-blend | 0.223 | 0.342 | 0.403 | **0.461** | 0.127 | 1431 |
| 2 | 55314-seq-blend | seq-blend | 0.217 | 0.340 | 0.403 | **0.460** | 0.119 | 1431 |
| 3 | 4fd6c-seq-blend | seq-blend | 0.216 | 0.340 | 0.400 | **0.455** | 0.126 | 1431 |
| 4 | 1b89c-seq-blend | seq-blend | 0.212 | 0.333 | 0.401 | **0.454** | 0.122 | 1431 |
| 5 | d593e-seq-blend | seq-blend | 0.212 | 0.333 | 0.401 | **0.454** | 0.122 | 1431 |
| 6 | fb487-seq-blend | seq-blend | 0.209 | 0.331 | 0.398 | **0.452** | 0.119 | 1431 |
| 7 | f455d-seq-blend | seq-blend | 0.208 | 0.337 | 0.400 | **0.454** | 0.121 | 1431 |
| 8 | 66c96-seq-blend | seq-blend | 0.205 | 0.333 | 0.399 | **0.452** | 0.117 | 1431 |
| 9 | 08f1c-seq-blend | seq-blend | 0.205 | 0.338 | 0.404 | **0.455** | 0.118 | 1431 |
| 10 | 2c5e1-seq-blend | seq-blend | 0.200 | 0.347 | 0.414 | **0.458** | 0.117 | 1431 |
| 11 | 1a295-seq-blend | seq-blend | 0.195 | 0.331 | 0.397 | **0.448** | 0.111 | 1431 |
| 12 | 4520d-seq-blend | seq-blend | 0.194 | 0.331 | 0.396 | **0.447** | 0.112 | 1431 |
| 13 | 57432-seq-blend | seq-blend | 0.193 | 0.335 | 0.400 | **0.447** | 0.111 | 1431 |
| 14 | 38725-seq-blend | seq-blend | 0.186 | 0.331 | 0.394 | **0.449** | 0.102 | 1431 |
| 15 | a73ac-seq-blend | seq-blend | 0.173 | — | — | **0.441** | 0.096 | 1431 |
| 16 | 645ae-seq-blend | seq-blend | 0.173 | 0.321 | 0.397 | **0.441** | 0.096 | 1431 |
| 17 | 96f53-seq-blend | seq-blend | 0.171 | 0.329 | 0.398 | **0.443** | 0.095 | 1431 |
| 18 | 50e1d-seq-blend | seq-blend | 0.171 | 0.299 | 0.342 | **0.396** | 0.105 | 1431 |
| 19 | 36dce-seq-blend | seq-blend | 0.167 | 0.329 | 0.398 | **0.444** | 0.092 | 1431 |
| 20 | 61137-seq-nexttrack | seq-nexttrack | 0.159 | 0.329 | 0.449 | **0.465** | 0.073 | 1431 |
| 21 | 4cc06-seq-blend | seq-blend | 0.158 | 0.319 | 0.391 | **0.434** | 0.094 | 1431 |
| 22 | d3d4a-seq-nexttrack | seq-nexttrack | 0.157 | 0.336 | 0.471 | **0.482** | 0.070 | 1431 |
| 23 | c1b8e-seq-nexttrack | seq-nexttrack | 0.157 | 0.327 | 0.446 | **0.462** | 0.064 | 1431 |
| 24 | 23b6c-seq-nexttrack | seq-nexttrack | 0.153 | 0.334 | 0.456 | **0.475** | 0.069 | 1431 |
| 25 | d070f-seq-blend | seq-blend | 0.152 | 0.319 | 0.391 | **0.429** | 0.091 | 1431 |
| 26 | 212f7-seq-nexttrack | seq-nexttrack | 0.149 | 0.328 | 0.472 | **0.472** | 0.062 | 1431 |
| 27 | 9a37f-seq-nexttrack | seq-nexttrack | 0.148 | 0.331 | 0.451 | **0.450** | 0.067 | 1431 |
| 28 | 141a6-seq-nexttrack | seq-nexttrack | 0.138 | 0.314 | 0.418 | **0.458** | 0.065 | 1431 |
| 29 | cf1dd-seq-blend | seq-blend | 0.128 | 0.296 | 0.341 | **0.383** | 0.082 | 1431 |
| 30 | 546d0-seq-nexttrack | seq-nexttrack | 0.127 | 0.335 | 0.458 | **0.451** | 0.050 | 1431 |
| 31 | 1bbdb-seq-nexttrack | seq-nexttrack | 0.123 | 0.338 | 0.450 | **0.456** | 0.053 | 1431 |
| 32 | 89d6a-seq-nexttrack | seq-nexttrack | 0.123 | 0.338 | 0.450 | **0.456** | 0.053 | 1431 |
| 33 | 6c67b-seq-nexttrack | seq-nexttrack | 0.123 | 0.338 | 0.450 | **0.456** | 0.053 | 1431 |
| 34 | aefd5-seq-nexttrack | seq-nexttrack | 0.123 | 0.338 | 0.450 | **0.456** | 0.053 | 1431 |
| 35 | b5244-seq-nexttrack | seq-nexttrack | 0.123 | 0.338 | 0.450 | **0.456** | 0.053 | 1431 |
| 36 | 49721-seq-nexttrack | seq-nexttrack | 0.123 | 0.338 | 0.450 | **0.456** | 0.053 | 1431 |
| 37 | 31702-seq-nexttrack | seq-nexttrack | 0.122 | 0.302 | 0.440 | **0.450** | 0.048 | 1431 |
| 38 | 93bad-seq-blend | seq-blend | 0.117 | 0.284 | 0.337 | **0.377** | 0.079 | 1431 |
| 39 | b4d6e-seq-nexttrack | seq-nexttrack | 0.117 | 0.338 | 0.458 | **0.456** | 0.050 | 1431 |
| 40 | a1d9d-seq-nexttrack | seq-nexttrack | 0.115 | 0.333 | 0.452 | **0.457** | 0.049 | 1431 |
| 41 | 5d967-seq-markov | seq-markov | 0.107 | — | — | **0.407** | 0.069 | 1431 |
| 42 | 8851d-seq-nexttrack | seq-nexttrack | 0.104 | 0.312 | 0.460 | **0.437** | 0.042 | 1431 |
| 43 | 7fdb2-seq-nexttrack | seq-nexttrack | 0.098 | 0.305 | 0.458 | **0.436** | 0.040 | 1431 |
| 44 | 9dfb2-seq-nexttrack | seq-nexttrack | 0.096 | — | — | **0.432** | 0.041 | 1431 |
| 45 | dd261-seq-nexttrack | seq-nexttrack | 0.096 | 0.294 | 0.450 | **0.432** | 0.041 | 1431 |
| 46 | e96ae-seq-nexttrack | seq-nexttrack | 0.093 | 0.303 | 0.464 | **0.436** | 0.040 | 1431 |
| 47 | 7e528-seq-ann | seq-ann | 0.090 | 0.299 | 0.435 | **0.426** | 0.039 | 1431 |
| 48 | 371b1-seq-stacker | seq-stacker | 0.087 | 0.310 | 0.377 | **0.418** | 0.049 | 1431 |
| 49 | f111d-seq-stacker | seq-stacker | 0.085 | 0.300 | 0.375 | **0.408** | 0.049 | 1431 |
| 50 | d46cf-seq-stacker | seq-stacker | 0.080 | 0.310 | 0.377 | **0.409** | 0.042 | 1431 |
| 51 | d3f9e-seq-nexttrack | seq-nexttrack | 0.055 | 0.238 | 0.396 | **0.368** | 0.027 | 1431 |
| 52 | b1242-seq-nexttrack | seq-nexttrack | 0.041 | 0.219 | 0.382 | **0.392** | 0.022 | 1431 |
| 53 | 63765-seq-popularity | seq-popularity | 0.001 | — | — | **0.185** | 0.001 | 1431 |

## 6. Caveats

- **Backfill vs native:** these 53 music@10 values are computed offline and shown
  in this report only — the runs' stored `metrics.json` / DB rows are NOT edited
  (hand-editing `data/` is a non-negotiable). New runs get `music@10` natively in
  the leaderboard; to give a champion a native value, re-run it through the API.
- **Metric-vs-retrieval decoupling:** the retrieval latent (song-AE, various dims
  per dataset) is distinct from the `balanced` metric space, so music@10 does not
  merely re-score a model in its own space.
- **`music@10` is display-only.** It should not silently become a selection
  objective; promotion still ranks by recall@10 unless `[metrics].primary` is
  deliberately changed.
- The metric inherits the `balanced` dial's mild circularity (text block embeds
  the artist name; genre block is a genre prototype), so it partly rewards
  same-artist/genre proximity — intended for "sounds/feels alike," not a perceptual
  ground truth.

## 7. Follow-ups (not done)

- Fold this report into the LaTeX docs via the **report-curator** agent (add a
  music@10 column to the leaderboard table + a recall-vs-music scatter figure).
- Reconcile `experiments/PROJECT-FACTS.md` with the music@10 field-guide note.
- Optional: wire `d` as the pathfinder edge cost (changes path behaviour — needs
  its own before/after validation); add a "similar songs" UI view.
