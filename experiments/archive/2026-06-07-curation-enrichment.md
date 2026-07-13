# Curation / metadata feature enrichment A/B (xgboost + ridge)

Date: 2026-06-07
Family: xgboost (champion), ridge (linear reference)

## Goal

Test whether adding seven curation/metadata fields to the leakage-free p128
recipe lowers habit-fit MAE, and — critically — audit whether any gain comes
from soft leakage rather than legitimate taste signal.

**Baseline (control).** xgboost on `ds-20260607-172307-p128-s42` (the
leakage-free p128 recipe; play_count / completion_ratio / skip_rate OFF),
promoted as `habit-fit`. Held hyperparams: `learning_rate=0.05, max_depth=6,
n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0,
reg_lambda=1.0, val_fraction=0.15, early_stopping_rounds=0, weight_gamma=0.0`;
data split seed 42 (n_train 2294 / n_test 573). Control xgb 10-seed mean
**MAE 5.129**, R² 0.654. Control ridge (auto_alpha, deterministic) **MAE
6.853**, R² 0.429. (Control runs reused from the noise-band campaign,
`2026-06-07-noise-band-study.md`.)

**Datasets.**
- Control: `ds-20260607-172307-p128-s42` (234 cols).
- Treatment: `ds-20260607-203256-p128-s42` — identical recipe and split, plus
  7 new fields ON: `is_saved`, `in_playlist_count`, `on_repeat_count`,
  `artist_followers` (log), `release_year`, `artist_count`, `album_type`
  (one-hot). 246 cols (+12: 6 numerics + `is_saved_missing` indicator + 5
  `album_type` one-hots). Identical split (n_rows 2867, n_train 2294 / n_test
  573, n_excluded 0); cum EVR@128 0.9999997.

**Configs.** Treatment = 10 xgboost training seeds (same 10 as the noise-band
pool) + 1 ridge run, all on the treatment dataset, hyperparams held at the
baseline. Control = the 10 xgboost seeds + ridge already run on the control
dataset. 11 new runs; 0 failed, 0 explosions, 0 spurious stops.

**Decision rule.** A treatment-minus-control xgb mean-MAE improvement counts as
CONFIRMED only if it exceeds the xgb 2σ band (0.359, from the noise-band study)
*and* survives the soft-leakage audit (the play_count proxy `on_repeat_count`
must not dominate gain importance).

## Outcome (one line)

**CONFIRMED non-leakage win:** the curation features cut xgboost mean MAE by
0.561 (5.129 → 4.568, Δ ≫ 2σ band of 0.359) with R² 0.654 → 0.712, and the
leakage audit clears it — the deliberate-curation signal `in_playlist_count`
dominates (rank 1, 23.5% gain) while the play_count proxy `on_repeat_count` is
minor (rank 4, 3.5%). The gain is xgboost-specific; ridge does not move.

## Results — per-seed MAE / R² (control vs treatment)

MAE in raw habit-fit units. **Bold** = best cell per metric. Winner row
(treatment xgb mean) bolded.

### xgboost (10 training seeds, split s42 fixed)

| seed | control run | ctrl MAE | ctrl R² | treatment run | treat MAE | treat R² |
|---|---|---|---|---|---|---|
| 42    | run-20260607-193801-98be4-xgboost | 5.169 | 0.635 | run-20260607-213852-45ece-xgboost | 4.605 | 0.710 |
| 84    | run-20260607-193801-4fd96-xgboost | 5.088 | 0.656 | run-20260607-213852-b37c7-xgboost | 4.662 | 0.707 |
| 99    | run-20260607-193801-cc394-xgboost | 5.110 | 0.654 | run-20260607-213852-23705-xgboost | 4.493 | 0.722 |
| 123   | run-20260607-193801-2e660-xgboost | 5.235 | 0.644 | run-20260607-213852-f8571-xgboost | **4.483** | 0.716 |
| 256   | run-20260607-193801-42d81-xgboost | 4.988 | 0.666 | run-20260607-213852-c0f31-xgboost | 4.483 | 0.720 |
| 512   | run-20260607-193801-5a919-xgboost | 5.049 | 0.668 | run-20260607-213852-87444-xgboost | 4.565 | **0.721** |
| 1000  | run-20260607-193801-3ce21-xgboost | 5.257 | 0.648 | run-20260607-213852-9d9da-xgboost | 4.571 | 0.715 |
| 2024  | run-20260607-193801-3d595-xgboost | **4.968** | **0.681** | run-20260607-213852-20646-xgboost | 4.529 | 0.701 |
| 31337 | run-20260607-193801-745bd-xgboost | 5.237 | 0.636 | run-20260607-213852-f1f1c-xgboost | 4.709 | 0.700 |
| 7     | run-20260607-193801-768b2-xgboost | 5.188 | 0.649 | run-20260607-213852-72693-xgboost | 4.578 | 0.709 |
| **mean** | | **5.129** | **0.654** | | **4.568** | **0.712** |
| stdev (population) | | 0.099 | | | 0.072 | |

**Δ mean MAE = 0.561** (control − treatment). xgb 2σ band = 0.359 (combined
band 0.187, see noise-band study; 2σ rounded the campaign uses is 0.375 — the
margin clears both). Per-seed: treatment beats control on 10/10 seeds; the
pools do not overlap (worst treatment 4.709 ≈ best control 4.968 still
separated). **Significant — CONFIRMED.**

### ridge (deterministic, single run each)

| dataset | run | MAE | R² |
|---|---|---|---|
| control   | run-20260607-193801-abb89-ridge | 6.853 | 0.429 |
| treatment | run-20260607-213853-da2c1-ridge | 6.714 | 0.349 |

**Δ ridge MAE = 0.139**, inside the ridge 2σ band (0.317; combined 0.191).
**Not significant for ridge.** Note also the ridge treatment R² is *lower*
(0.429 → 0.349) even as MAE drops slightly — the extra mostly-sparse columns
add variance the linear model does not exploit. The curation gain is
**xgboost-specific**: trees can carve the non-linear `in_playlist_count`
signal that ridge cannot.

## Soft-leakage audit (gain importances, treatment seed-42 run)

Source: `run-20260607-213852-45ece-xgboost/model.ubj`, gain importance via
xgboost 3.2.0; feature indices mapped through the treatment dataset manifest
(246 cols). Total gain 148.13 across 188 features that received splits.

### Top 15 features by gain share

| rank | feature | gain | share | splits |
|---|---|---|---|---|
| 1  | **in_playlist_count** (new) | 34.8 | **23.52%** | 143 |
| 2  | artist=Laika Perra Rusa | 6.8 | 4.62% | 8 |
| 3  | artist=Rammstein | 5.8 | 3.89% | 18 |
| 4  | **on_repeat_count** (new, play_count proxy) | 5.2 | 3.54% | 52 |
| 5  | track_popularity_missing | 4.8 | 3.25% | 3 |
| 6  | pca_108 | 3.0 | 2.03% | 146 |
| 7  | genre_primary=alternative metal | 2.9 | 1.93% | 18 |
| 8  | **is_saved** (new) | 2.8 | 1.89% | 20 |
| 9  | artist=Peces Raros | 2.7 | 1.84% | 7 |
| 10 | pca_54 | 2.2 | 1.50% | 157 |
| 11 | pca_76 | 1.8 | 1.22% | 133 |
| 12 | genre_primary=argentine indie | 1.7 | 1.14% | 10 |
| 13 | pca_61 | 1.5 | 1.01% | 154 |
| 14 | pca_112 | 1.4 | 0.94% | 136 |
| 15 | artist=__other__ | 1.4 | 0.93% | 41 |

### Rank / gain share of every new feature

| feature | rank | gain | share | splits |
|---|---|---|---|---|
| in_playlist_count    | 1   | 34.8 | 23.52% | 143 |
| on_repeat_count      | 4   | 5.2  | 3.54%  | 52 |
| is_saved             | 8   | 2.8  | 1.89%  | 20 |
| is_saved_missing     | 30  | 1.0  | 0.69%  | 3 |
| artist_count         | 67  | 0.5  | 0.33%  | 13 |
| release_year         | 80  | 0.4  | 0.25%  | 136 |
| album_type=album     | 89  | 0.3  | 0.24%  | 15 |
| artist_followers_log | 113 | 0.2  | 0.17%  | 98 |
| album_type=compilation | 130 | 0.2 | 0.13% | 32 |
| album_type=single    | 176 | 0.0  | 0.03%  | 3 |
| album_type=unknown   | —   | 0.0  | 0.00%  | 0 (no split) |
| album_type=__other__ | —   | 0.0  | 0.00%  | 0 (zero-variance) |

Aggregates: curation trio (`is_saved`+`in_playlist_count`+`on_repeat_count`)
= 28.95% of gain; all 7 new metadata fields = 30.80%; all pca_* embedding
dims combined = 43.39%.

### Audit verdict — LEGITIMATE

The audit's tempering trigger was "if `on_repeat_count` dominates, temper the
verdict." It does not. `on_repeat_count` (the play_count proxy and the only
soft-leakage suspect, since the target is built from play_count) sits at rank 4
with 3.54% gain — minor. The dominant new feature is `in_playlist_count`
(23.52%), a deliberate-curation act that is upstream of and distinct from
listening counts: a user puts a track in playlists because they value it, which
is exactly the taste signal we want to capture. `is_saved` (1.89%) is likewise
deliberate-curation. The modest R² lift (0.654 → 0.712, not toward 0.8+)
corroborates: a leak of the target would have pushed R² much higher. **The win
is legitimate taste signal, not leakage.**

Residual watch: `on_repeat_count` is non-zero (3.54%, 52 splits) and is a
play_count proxy. The win does not depend on it, but it is not nothing — see
follow-up 1 for the recommended (not built) ablation.

## Findings

- **Curation beats embedding nuance.** A single deliberate-curation count
  (`in_playlist_count`) carries nearly a quarter of the model's gain — more
  than the entire artist/genre one-hot block individually and rivaling the
  combined 128 PCA embedding dims (43%). The embedding captures *what a track
  sounds like*; `in_playlist_count` captures *what the user did about it*. They
  are complementary, and the behavioral signal is dense and cheap.
- **xgboost-specific.** Ridge gains nothing significant and loses R². The
  curation signal is non-linear / interaction-heavy (e.g. high playlist count
  matters more conditional on genre), which trees exploit and a linear model
  cannot. Confirms the standing "trees want richer feature sets than linear"
  note from the pca-dim-scan campaign.
- **`album_type` was a dead axis.** Four of five one-hots are near-zero gain;
  `album_type=__other__` is zero-variance (the 4 known types cover the corpus)
  and `=unknown` never split. The one-hot block cost 5 columns for ~0.4% gain.
  Harmless but prunable.
- **Failure modes:** none. All 11 runs succeeded, no explosions, no spurious
  stops (consistent with the noise-band campaign's clean record for both
  families).
- **Comparison-basis caveat.** This A/B is a 10-train-seed pool on the single
  s42 split, whereas the parallel pca-dim-scan reports 6-split-seed means. Both
  improve over the same s42 control. The Δ here is robust to training-seed luck
  (the relevant noise source for a fixed split + fixed recipe), but a
  multi-split rebuild of the curation recipe would make it directly comparable
  to the dim-scan numbers (follow-up 2).

## Best on record after this work

xgboost on the curation recipe is the new best-on-record by MAE. Champion
(`habit-fit`) is **not** auto-promoted (see promotion recommendation below). A
parallel candidate, the p160 recipe from `2026-06-07-pca-dim-scan.md` (6-split
mean 4.800 MAE / R² 0.691, also unpromoted), is on a different feature axis and
a different measurement basis — the two are not yet head-to-head.

| model | dataset | MAE | RMSE | R² | source |
|---|---|---|---|---|---|
| xgboost (curation recipe, 10-seed mean, s42 split) | ds-20260607-203256-p128-s42 | **4.57** | ~9.95 | **0.71** | this report |
| xgboost p160 recipe (6-split mean, candidate, unpromoted) | ds-…-172307 recipe @ pca=160 | 4.80 | — | 0.69 | 2026-06-07-pca-dim-scan.md |
| xgboost (promoted as `habit-fit`) | ds-20260607-172307-p128-s42 | 5.17 | 11.19 | 0.63 | bootstrap run |
| ridge | ds-20260607-172307-p128-s42 | 6.85 | 13.98 | 0.43 | bootstrap run |
| baseline-median | ds-20260607-172307-p128-s42 | 10.08 | 19.54 | −0.11 | bootstrap run |

## Follow-ups (ranked)

1. **`on_repeat`-ablation confirmation (recommended, not built).** Rebuild the
   curation recipe with `on_repeat_count` OFF (the only play_count proxy) and
   rerun the 10-seed xgb pool. Expected: MAE rises by < ~0.1 (its gain share is
   3.5%); if the win survives without it, the legitimacy verdict is airtight and
   `on_repeat_count` can be dropped to eliminate the soft-leakage surface
   entirely. Needs a new dataset build — design via dataset-architect.
2. **Multi-split confirmation of the curation recipe.** Rebuild the curation
   recipe at split seeds {7, 84, 123, 256, 999} (matching the noise-band /
   dim-scan split pool) and run xgb, to express the win as a 6-split mean
   directly comparable to the p160 candidate. Since σ_split > σ_train, this is
   the more decisive confirmation.
3. **Combine the axes: curation features at pca=160.** The two unpromoted
   improvements are on orthogonal axes (features vs dims). A curation + p160
   recipe could stack the gains; run it head-to-head against both candidates
   before any promotion.
4. **Prune the dead `album_type` one-hot block** (and consider dropping
   `artist_followers_log`, rank 113) when finalizing the production recipe —
   ~5 columns for negligible gain.
