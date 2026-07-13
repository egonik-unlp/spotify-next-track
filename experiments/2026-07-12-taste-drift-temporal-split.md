# Taste drift: chronological vs random split (temporal generalization) — 2026-07-12

Goal: answer "does the model take into account *when* a song was listened to" in
the most informative form — **is the user's taste stationary?** Every AUC on
record is measured on a RANDOM train/test split, which lets the model see tracks
discovered in the SAME period on both sides of the split. This asks the honest
forward-looking question instead: train on what was discovered EARLIER, test on
what was discovered LATER. If AUC collapses, taste drifts and the leaderboard
overstates real forward-prediction skill.

## Method

The lensing build API originally produced only a RANDOM split (`test_ratio` +
`seed`). **This campaign ADDED a first-class chronological-split capability to the
builder** (`split_order_field` on `BuildConfig` / the `POST /api/datasets` body):
when set, rows are ordered ascending by that payload field (earliest → train,
latest → test) with PCA, vocabulary and imputation all fit on the early-train rows
only (leak-free); the manifest self-describes it via `split.strategy =
"chronological"` + `order_field`. Backward-compatible (serde-defaults to
`"random"`), unit-tested, `cargo check` clean. See the Provenance section.

Two independent measurements, in agreement:

1. **Definitive — framework chronological builds.** Four datasets built through the
   new capability from the active `spotify_tracks_song_ae` collection: full/
   intrinsic × random/chronological. The champion + a model sweep are evaluated on
   each dataset's own native split. The full-random build reproduces the on-record
   champion to 5 digits (0.73490), confirming the fresh builds are correct.
2. **Cross-check — off-framework re-split** of the pre-existing `ds-20260615-163042`
   (random-built) features, re-split by `first_played` in Python with the champion
   config. Agrees with (1) to ~0.007 AUC.

- **Datasets (definitive, all `spotify_tracks_song_ae`, 23,529 rows, seed 42, target
  `rotation`/none):** full-random `ds-20260712-183615` (1,377 cols), full-chrono
  `ds-20260712-183629`, intrinsic-random `ds-20260712-183630` (71 cols = 64 PCA +
  7 numeric, identity one-hots OFF), intrinsic-chrono `ds-20260712-183631`.
- **Model:** champion `xgboost-classifier` config (n_est 500, depth 6, lr 0.05,
  subsample/colsample 0.8, seed 42), xgboost 3.2.0, `tree_method=hist`; sweep adds
  catboost (iter 500, depth 6, lr 0.05) and logistic (C=1, standardized).

### Cross-check dataset (off-framework arm)

- **Dataset:** `ds-20260615-163042-p64-s42` (the most-updated robust build; 23,529
  rows, 1,377 cols = 64 song-AE PCA + 9 numeric + identity one-hots @ artist
  1000 / genre 300; target `rotation`, transform none). Exported via
  `GET /api/datasets/<id>/archive` → `features.f32` (X), `target.f32` (y, clean
  0/1), `row_ids.u64` joined to `items.json` for `first_played`, plus the
  framework's own `train_idx`/`test_idx` for the baseline.
- **Model:** champion `xgboost-classifier` config (n_est 500, depth 6, lr 0.05,
  subsample/colsample 0.8, seed 42), xgboost 3.2.0, `tree_method=hist`.
- **`first_played` span:** 2014-05 → 2026-06 (12 years; every year 2016–2025
  carries 900–3,600 tracks — ample for a temporal split).
- **Noise band:** AUC 2σ = 0.0126.

## Results

### Definitive: model sweep on framework chronological builds

All datasets `spotify_tracks_song_ae`; each model evaluated on its dataset's native
split. AUC:

| model | full · random | full · chrono | drift | intrinsic · random | intrinsic · chrono |
|---|---|---|---|---|---|
| **xgboost** (champion) | **0.73490** | **0.61894** | **+0.116** | 0.71020 | 0.59858 |
| catboost | 0.72889 | 0.61294 | +0.116 | 0.70172 | 0.59556 |
| logistic | 0.72141 | 0.61234 | +0.109 | 0.64836 | 0.59016 |

- **Drift is ~0.11–0.12 AUC (≈9 bands) for EVERY model family** — it is the
  data/taste, not a model quirk. Full-random reproduces the on-record champion
  exactly (0.73490), validating the builds.
- **Identity one-hots barely help forward prediction.** Dropping them costs only
  ~0.02 chrono (0.619→0.599) vs ~0.025 random (0.735→0.710): they mostly memorize
  same-era artists. The transferable taste signal is the intrinsic ~0.60 floor.
- **Time-aware (recency-weighted xgboost, chrono):** best 0.62153 (half-life 0.15,
  full) vs 0.61894 unweighted — **+0.003, sub-band**. Upweighting recent
  discoveries does NOT recover the drift gap; drift is not a simple recency effect.

### Cross-check: off-framework re-split (agrees)

Re-splitting the pre-existing random-built `ds-20260615-163042` by `first_played`:

| protocol | AUC | train rot | test rot | n_test |
|---|---|---|---|---|
| Random split (native) | 0.73490 | 0.475 | 0.465 | 4,706 |
| Chronological (oldest 80% → newest 20%) | 0.62579 | 0.504 | 0.350 | 4,706 |
| Chronological, censoring-controlled (≥180 d replay opportunity) | 0.62319 | 0.504 | 0.391 | 3,136 |

The off-framework chrono (0.626) sits ~0.007 above the leak-free framework build
(0.619) — because re-splitting a random-built dataset lets its PCA/vocabulary see
future rows. Same story either way: **drift ≈ 0.11.** Chrono test period 2024-08 →
2026-06.

### Walk-forward (train on `first_played < Y`, test on year Y)

| test year | n_train | n_test | AUC | test rot |
|---|---|---|---|---|
| 2019 | 5,638 | 972 | 0.6001 | 0.526 |
| 2020 | 6,610 | 898 | 0.6272 | 0.508 |
| 2021 | 7,508 | 2,071 | 0.6548 | 0.380 |
| 2022 | 9,579 | 3,006 | 0.6577 | 0.571 |
| 2023 | 12,585 | 3,629 | 0.6697 | 0.557 |
| 2024 | 16,214 | 3,245 | 0.5895 | 0.506 |
| 2025 | 19,459 | 2,713 | 0.6318 | 0.356 |
| 2026 | 22,172 | 1,357 | 0.6213 | 0.266 |

Predicting any single future year lands at **0.59–0.67 — never near the 0.735
random baseline**, and more training history does not close the gap.

## Verdict

**Taste is NOT stationary; the leaderboard AUC is optimistic for forward
prediction by ~0.11.** Three converging reads:

1. **The random-vs-chronological gap is huge and out-of-band** (0.109 ≈ 8.7
   bands). The champion's 0.735 is substantially an artifact of the random split
   letting same-era artist/genre identity leak across train/test.
2. **It is real drift, not right-censoring.** Recent tracks are mechanically
   less-rotated (less elapsed replay opportunity: chrono test rot 0.350 vs random
   0.465). Controlling for it — restricting the chrono test to tracks with ≥180 d
   of exposure — barely moves AUC (0.62579 → 0.62319). The penalty is genuine
   non-stationarity, not the base-rate shift.
3. **Mechanism = identity churn.** The convergent project finding is that
   rotation is driven by artist/genre **identity** one-hots. In a chronological
   split the test period is full of newly discovered artists that were unseen in
   training (they fall into `__other__`), so the model must fall back on the
   weaker intrinsic acoustic/metadata signal — which only reaches ~0.62. The SAE
   study's "linear decodability ~0.14, no clean taste atom" is the same story:
   remove identity memorization and little generalizable taste signal remains.

## Implications

- **Read every on-record AUC as an upper bound on same-era ranking, not forward
  skill.** Real "will I replay a song I discover next" performance is ~0.62–0.63
  on this corpus, ~0.11 below the random-split leaderboard.
- The lever for forward performance is **generalizable intrinsic signal** (better
  acoustics / cross-artist structure), not more identity resolution — higher
  vocab caps and bigger nets (both spent levers) only sharpen same-era
  memorization, which does not transfer.

## Follow-ups

- **Chronological-split capability — DONE this campaign** (`split_order_field` in
  the builder; framework-built chronological datasets are now promotable and
  repeatable). Remaining: an `upstream-contribute` of the generalized change.
- **Intrinsic-only forward baseline — DONE** (chrono 0.599; identity adds only
  ~0.02 forward). See the sweep table.
- **Time-aware models — DONE (weak lever)** — recency-weighted xgboost gives only a
  sub-band gain; drift is not a simple recency effect.
- **Retrain cadence (open):** because taste drifts, a deployed model needs periodic
  retraining; quantify the half-life (AUC vs months-since-train) via the
  walk-forward, now cheap with the chronological-split option.
- **Better transferable signal (open):** the forward ceiling is the intrinsic ~0.60
  floor; richer acoustics / cross-artist structure is the only lever that would
  move forward AUC (identity and capacity are spent, same-era-only).

## Provenance

Framework capability added to the builder (`crates/lensing-pipeline/src/build.rs`
`chronological_split` + `BuildConfig.split_order_field`; `crates/lensing-core`
`SplitInfo.strategy`/`order_field`; `crates/lensing-server` `BuildRequest`;
CLI unchanged/random) — `cargo check` clean, 2 unit tests pass, demonstrated
end-to-end on a transient scratch server (production 8096 never restarted). The
new binary is compiled (`target/release/lensing-server`); expose the option on
8096 by restarting it when convenient. Definitive datasets (all
`spotify_tracks_song_ae`): full-random `ds-20260712-183615`, full-chrono
`ds-20260712-183629`, intrinsic-random `ds-20260712-183630`, intrinsic-chrono
`ds-20260712-183631`. Framework champion run on `183629` = AUC 0.61894
(= off-framework, reproduced to the digit). Sweep + cross-check scripts:
`scratchpad/sweep2.py`, `scratchpad/drift.py`. No model promoted.

NOTE — three datasets built early in this campaign from the WRONG collection
(`spotify_tracks`, the 200-dim behavioral co-listening embedding, which soft-leaks
replay and inflated chrono AUC to ~0.71) are stale artifacts and should be ignored/
deleted: `ds-20260712-182109`, `ds-20260712-182836`, `ds-20260712-182837`. The
build defaults to `domain.toml`'s `spotify_tracks`; rotation work must pass
`collection: "spotify_tracks_song_ae"`.
