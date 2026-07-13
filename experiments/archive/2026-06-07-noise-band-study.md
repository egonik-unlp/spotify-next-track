# MAE noise-band study — training-seed + split-seed reproducibility

**Date:** 2026-06-07. Runner: experiment-runner agent.
Supersedes `2026-06-07-INTERIM-noise-band-study.md` (now removed).

## Goal

Measure the empirical noise floor of the habit-fit MAE metric so future
single-split margins can be judged against a real band instead of a guess.
Two noise sources, two predictor families:

- **Axis 1 — training stochasticity** (data split fixed at s42): vary the
  predictor's internal seed. 10 xgboost seeds; 3 ridge reruns (ridge is
  deterministic, expected σ ≈ 0 — a control on the harness itself).
- **Axis 2 — split luck** (recipe fixed, data-split seed varied): rebuild the
  p128 dataset under 5 fresh split seeds (s7/s84/s123/s256/s999) plus the s42
  baseline; train each family once per split. xgboost uses seed=42; the s42
  xgb point reuses the Axis-1 control, and one ridge determinism run is the
  s42 ridge point.

**Baseline / champion (held constant):**
- Dataset `ds-20260607-172307-p128-s42` (pca=128, leakage-free; n_train 2294 /
  n_test 573, n_cols 234, n_excluded 0).
- Promoted `habit-fit` = xgboost, MAE **5.17**, RMSE 11.19, R² 0.63.
- Ridge MAE **6.85**, RMSE 13.98, R² 0.43 → nominal **1.68-MAE** gap under test.
- Definitions used: `noiseband-xgb` (xgboost), `noiseband-ridge` (ridge).

**Decision rule (verbatim):** combined band = √(σ_train² + σ_split²), reported
as MAE ± 2σ. The 1.68 gap is CONFIRMED if 1.68 > 2·combined(wider-variance
model); NOT-established if ≤ 2·combined (then recommend multi-split/k-fold
default); borderline (1–2σ) → confirmed-but-fragile. No promotion.

## Outcome in one line

The MAE noise floor is small (combined 2σ ≈ ±0.37 xgb / ±0.38 ridge MAE) and
the 1.68-MAE xgboost-over-ridge gap is **CONFIRMED at ~8.8σ** — not noise; no
champion change (this study only measures noise).

## Control gate (decision-rule step 1)

seed=42 xgboost on `ds-20260607-172307-p128-s42` →
`run-20260607-193801-98be4-xgboost` MAE = 5.169137679811519 = champion
5.169 ± 0.001 → **GATE PASSED**. Also matches the pre-campaign
`run-20260607-172548-8dc12-xgboost` (5.169137679811519) bit-for-bit.

## Results

### Axis 1 — training stochasticity (split fixed s42)

xgboost, 10 seeds:

| run_id | seed | MAE |
|---|---|---|
| run-20260607-193801-3d595-xgboost | 2024 | **4.968** |
| run-20260607-193801-42d81-xgboost | 256 | 4.988 |
| run-20260607-193801-5a919-xgboost | 512 | 5.049 |
| run-20260607-193801-4fd96-xgboost | 84 | 5.088 |
| run-20260607-193801-cc394-xgboost | 99 | 5.110 |
| run-20260607-193801-98be4-xgboost (CONTROL) | 42 | 5.169 |
| run-20260607-193801-768b2-xgboost | 7 | 5.188 |
| run-20260607-193801-2e660-xgboost | 123 | 5.235 |
| run-20260607-193801-745bd-xgboost | 31337 | 5.237 |
| run-20260607-193801-3ce21-xgboost | 1000 | 5.257 |

ridge determinism reruns (3×, all s42):

| run_id | MAE |
|---|---|
| run-20260607-193801-98382-ridge | 6.853271046350934 |
| run-20260607-193801-cde9e-ridge | 6.853271046350934 |
| run-20260607-193801-abb89-ridge | 6.853271046350934 |

All three ridge reruns are identical to 15 sig-figs → σ_train(ridge) = 0.

### Axis 2 — split luck (recipe fixed, split seed varied)

xgboost (seed=42), 6 splits:

| run_id | split | MAE |
|---|---|---|
| run-20260607-193810-ab092-xgboost | s84 | **4.812** |
| run-20260607-193811-8eec5-xgboost | s999 | 4.964 |
| run-20260607-193811-79a76-xgboost | s256 | 5.036 |
| run-20260607-193811-6e0b5-xgboost | s123 | 5.061 |
| run-20260607-193801-98be4-xgboost (reuse control) | s42 | 5.169 |
| run-20260607-193810-10ed4-xgboost | s7 | 5.255 |

ridge, 6 splits:

| run_id | split | MAE |
|---|---|---|
| run-20260607-193811-a4da2-ridge | s84 | **6.454** |
| run-20260607-193811-675b4-ridge | s256 | 6.623 |
| run-20260607-193811-1aa6f-ridge | s123 | 6.769 |
| run-20260607-193811-12f02-ridge | s7 | 6.843 |
| run-20260607-193801-98382-ridge (reuse det run) | s42 | 6.853 |
| run-20260607-193811-1bfb7-ridge | s999 | 6.992 |

(MAE in raw habit-fit units; values < 6 digits shown to 3 decimals.)

### Measured noise statistics

| quantity | value (MAE units) |
|---|---|
| σ_train(xgb) — stdev of 10 seeds | **0.1047** |
| σ_train(ridge) — stdev of 3 reruns | **0.0000** |
| σ_split(xgb) — stdev of 6 splits | **0.1554** |
| σ_split(ridge) — stdev of 6 splits | **0.1908** |
| combined band xgb = √(σt²+σs²) | 0.1874 → **2σ = ±0.375** |
| combined band ridge = √(σt²+σs²) | 0.1908 → **2σ = ±0.382** |

Mean MAEs: xgb ≈ 5.05–5.13 (split / seed pools), ridge ≈ 6.76–6.85.

## Findings

- **Verdict on the 1.68 gap: CONFIRMED.** The wider-variance model is ridge
  (combined band 0.1908; 2σ = 0.382). 1.68 > 0.382, so 1.68 > 2·combined(wider)
  by a wide margin — the gap is ~8.8σ of the wider band. This is far past the
  "confirmed" threshold; it is not a near-miss or borderline case. The pools
  do not even touch: worst xgboost MAE anywhere (5.257, seed=1000) is well
  below the best ridge MAE anywhere (6.454, s84). xgboost's superiority over
  ridge on this dataset is real and robust to both seed and split luck.

- **Split luck dominates training stochasticity for xgboost** (σ_split 0.155 >
  σ_train 0.105). Both are small, but a single-split result carries more risk
  from *which rows landed in the test set* than from the model's RNG. This is
  the more important number for future significance calls.

- **Ridge is bit-for-bit deterministic** (σ_train = 0 across 3 reruns) — a
  clean control confirming the training harness introduces no hidden run-to-run
  noise; all observed ridge variance is split luck.

- **The s42 baseline split is roughly median-typical, not lucky.** s42 xgb
  (5.169) sits in the upper-middle of the split pool; s42 ridge (6.853) is near
  the top (worst) end. The promoted champion's headline number is not flattered
  by an easy split.

- **No failures, no explosions, no spurious stops** across the 23 campaign runs
  (26 total incl. 3 pre-campaign baseline runs). Bounded-family prior held:
  neither tree (xgboost) nor linear (ridge, auto_alpha) blew up through the
  log-target inverse.

## Best on record after this work

Unchanged — this study measures noise only and promotes nothing.

| model | dataset | MAE | RMSE | R² | source |
|---|---|---|---|---|---|
| xgboost (promoted `habit-fit`) | ds-20260607-172307-p128-s42 | 5.17 | 11.19 | 0.63 | bootstrap run |
| ridge | ds-20260607-172307-p128-s42 | 6.85 | 13.98 | 0.43 | bootstrap run |
| baseline-median | ds-20260607-172307-p128-s42 | 10.08 | 19.54 | -0.11 | bootstrap run |

## Follow-ups

1. **Adopt the significance rule** (now in PROJECT-FACTS): a single-split MAE
   improvement counts as real only if it exceeds ~0.38 MAE (2σ of the wider
   band, ~0.4 to be safe); margins under that need a 3-seed × 3-split repeat.
   Since σ_split > σ_train, prefer multi-split confirmation over multi-seed.
2. Re-measure the band at other pca dims / feature sets before assuming 0.38
   transfers; bands are dataset-specific.
3. Optionally fold k-fold CV into the default training harness so split luck is
   averaged out at the source.
