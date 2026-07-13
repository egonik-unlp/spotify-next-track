# SVM kernel scan on rotation, then conditional SVM×tree blend

**Date:** 2026-06-09
**Predictor families:** svm (kernel scan), autoencoder-xgboost (control / noise),
blend (stage 2) · lightgbm (stage-2 member)
**Dataset family:** metadata-only rotation, pca floor (`ds-20260609-141622-p1-s42`
+ seed 17 / 101 siblings); control anchor on `ds-20260608-204402-p64-s42`.

## Goal

Test whether any SVM kernel (rbf / linear / poly / sigmoid) produces a rotation
predictor competitive with the metadata-only XGBoost champion, and — only if one
passes — whether it adds orthogonal signal to a tree in a grid-weighted blend.

**Baseline (held).** Champion = metadata-only XGBoost, **MAE 0.3879 / R² 0.191 /
RMSE 0.4475** on `ds-20260608-204402-p64-s42` (12,256 rows, seed-42 split
9,805 train / 2,451 test; target `metadata.rotation`, transform `none`). It
exists only as the `mae_xgb` branch inside the autoencoder-xgboost runs (the AE
branch tunes to weight 0), not as a saved standalone definition; reproduced here
as a control via `autoencoder-xgboost` kNN k=10 (byte-identical, blend_weight=0).
SVM held params (frozen except the scanned kernel): C=1.0, epsilon=0.1,
gamma=0.0 (=scale), degree=3, coef0=0.0, standardization on. The native
`xgboost` predictor trips the binary-target NaN-MAPE failure, so
autoencoder-xgboost is the control/member throughout (it guards the zero-actual
case and succeeds). Rotation noise band was UNMEASURED entering this campaign;
it is measured here.

## Outcome (one line)

**SVM is refuted standalone and as a blend member on rotation — every kernel
collapses to a worse-than-mean predictor (R² −0.13 to −8.5) whose only "win" is a
deceptively low MAE artifact of the ε-tube snapping a binary target to {0.1, 0.9};
the grid-weighted blend, optimizing that broken MAE, mis-selected pure SVM (weight
[0, 1]) over the tree — nothing registered, nothing promoted. First measured
rotation noise band: σ_split = 0.0043, band (2σ) = 0.0086 MAE.**

## Results

Stage 1 (8 runs) — all on the metadata-only `none`-transform feature matrix
except ctrl-xgb-178 (the 178-col anchor):

| run id | config | predictor | dataset | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|---|---|---|
| run-20260609-141717-9a8ba-autoencoder-xgboost | ctrl-xgb-178 | autoencoder-xgboost k10 (w=0) | ds-20260608-204402-p64-s42 | 0.3879 | **0.4475** | **0.191** | 43.9% |
| run-20260609-141717-28b3f-autoencoder-xgboost | ctrl-xgb-meta | autoencoder-xgboost k10 (w=0) | ds-20260609-141622-p1-s42 | 0.3879 | **0.4475** | **0.191** | 43.9% |
| run-20260609-141717-89c5f-svm | svm-rbf | svm (kernel=rbf) | ds-20260609-141622-p1-s42 | 0.3847 | 0.5287 | −0.130 | 57.6% |
| run-20260609-141717-b980e-svm | svm-linear | svm (kernel=linear) | ds-20260609-141622-p1-s42 | 0.3901 | 0.5443 | −0.198 | 58.5% |
| run-20260609-141717-73727-svm | svm-poly | svm (kernel=poly, deg 3) | ds-20260609-141622-p1-s42 | **0.3814** | 0.5303 | −0.137 | 57.7% |
| run-20260609-141717-fd50a-svm | svm-sigmoid | svm (kernel=sigmoid) | ds-20260609-141622-p1-s42 | 0.6240 | 1.5313 | −8.479 | 70.6% |
| run-20260609-141717-aac3d-autoencoder-xgboost | noise-s17 | autoencoder-xgboost k10 (w=0) | ds-20260609-141641-p1-s17 | 0.3931 | 0.4487 | 0.181 | 44.9% |
| run-20260609-141717-fcbbe-autoencoder-xgboost | noise-s101 | autoencoder-xgboost k10 (w=0) | ds-20260609-141641-p1-s101 | 0.3964 | 0.4513 | 0.178 | 45.2% |

Stage 2 (2 runs, conditional — ran because rbf/linear/poly cleared the MAE-only
leg of gate 3; see Findings on why that leg is degenerate). Both on
`ds-20260609-141622-p1-s42`, mean rule, grid weights, val_fraction 0.15:

| run id | config | members | fitted weights | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| run-20260609-143114-528ef-blend | blend-svm-xgb | autoencoder-xgboost k10 (w=0) + svm(poly) | **[0.0, 1.0]** | 0.3810 | 0.5297 | −0.134 |
| run-20260609-143114-fa66a-blend | blend-svm-lgbm | lightgbm + svm(poly) | **[0.0, 1.0]** | 0.3810 | 0.5297 | −0.134 |

Member solo metrics inside the blends (on-disk `blend.json`): autoencoder-xgboost
solo MAE 0.3870 / R² 0.184; lightgbm solo MAE 0.3885 / R² 0.174; svm(poly) solo
MAE 0.3810 / R² −0.134. The grid put weight 1.0 on the SVM in both blends — i.e.
each blend collapsed to **pure SVM**, discarding the tree.

Best cell per metric is bolded; the champion-equivalent controls hold the best
RMSE and R². Best raw MAE (0.3810, blend≡svm-poly) is the degenerate artifact, not
a win — see Findings.

## Findings

**1. Control gate PASSED; the AE-branch zeroes out on both feature matrices.**
ctrl-xgb-178 reproduced the champion to **MAE 0.387891 vs 0.3879 (|Δ| 0.000009),
R² 0.191, RMSE 0.4475** — well inside the measured band. ctrl-xgb-meta on the
metadata-only matrix is **byte-identical** (0.387891 / 0.191 / 0.4475). On-disk
`blend_weight = 0.0` in every autoencoder-xgboost run, with `mae_xgb =
0.3878913565318093` reproduced exactly. So dropping the 64 AE-latent columns to
the pca-floor changes nothing for the tree — the fair anchor is identical to the
178-col champion, and the batch is trustworthy.

**2. First measured rotation noise band: σ_split = 0.00430, band (2σ) = 0.00859.**
Across the three seeded metadata-only XGBoost runs — s42 0.387891, s17 0.393072,
s101 0.396417 (mean 0.392460) — sample σ_split = 0.004296, so the 2σ
significance band is **0.0086 MAE** on the [0,1] rotation scale. This retroactively
settles the AE-blend campaign's open question: the 0.3879 (metadata-only) vs
0.3965 (latent+metadata) edge is **Δ 0.0086 ≈ exactly 1·band** — at the very edge
of 2σ, i.e. "likely better, not confidently". The 0.3879 vs 0.4098 (AE-latent-only)
gap is ~2.5·band — that one is real.

**3. The SVM "MAE wins" are a binary-target ε-tube artifact, not skill — the
R²/RMSE collapse is the honest read.** rbf/linear/poly land at MAE 0.381–0.390,
inside ctrl+band, so they trip the MAE-only leg of gate 3. But their **R² is
negative (−0.13 to −0.20) and RMSE is 0.53–0.54 vs the tree's 0.45** — they are
*worse than predicting the mean*. The prediction-spread diagnostic (read from
on-disk `predictions.json`) explains it: with epsilon=0.1 on a {0,1} target, the
SVR snaps its outputs to the tube edges — rbf/linear/poly all show
**p50 ≈ 0.10, p90 ≈ 0.90**, a hard two-level step function, not a calibrated
probability. On a binary target a two-level predictor pinned near the right
classes scores a low MAE (each row is ~0.1 off its 0/1 label) while a calibrated
[0,1] regressor (the tree, spread 0.16/0.41/0.81) carries larger average absolute
error but vastly better squared error and ranking. MAE is degenerate here; the
metric that matters for Pathfinder's `fit_norm` (ranking / calibration ⇒ R², RMSE)
says SVM loses decisively. This is the SAME framework limitation already logged for
binary rotation targets (NaN-MAPE / MAE-degeneracy family), now seen from the MAE
side.

**4. sigmoid SVR blew up (documented degeneracy, captured as a data point).**
svm-sigmoid: MAE 0.624, RMSE 1.531, R² −8.48, with predictions ranging to **24.7**
on a [0,1] target — the classic unbounded sigmoid-SVR divergence. Not a blocker;
recorded as the worst arm exactly as the design anticipated.

**5. The grid-weighted blend did the OPPOSITE of self-disabling — it mis-selected
pure SVM.** Both stage-2 blends fitted weights **[0.0, 1.0]**, zeroing the tree and
keeping the SVM. This is *not* "SVM adds orthogonal signal"; it is the binary-MAE
degeneracy propagating into the grid objective. The grid optimizes the blend's
validation **MAE**, and svm(poly)'s solo MAE (0.3810) edges the tree's (xgb 0.3870
/ lgbm 0.3885) for the artifact reason in finding 3 — so the MAE-driven grid picks
the worse-R² model. The blends therefore collapsed to pure SVM (MAE 0.3810, R²
−0.134), strictly worse than either tree member on R²/RMSE. The AE-blend
precedent's "grid weight → 0 = refutation" reasoning holds, but with a sharp
caveat: **on a binary target, grid-by-MAE is itself unsafe — it can promote a
worse-than-mean member.** A blend grid for rotation should optimize R²/logloss,
not MAE.

**6. Decision rule outcome — nothing saved, nothing promoted.** Per gate 3's
strict-beat branch (register + 3-seed only if a kernel beats champion by > band on
MAE **AND R²**): no kernel beats on R² (all negative vs +0.191), so no
registration and no seed-robustness study. Per the stage-2 save gate (promote only
if the blend beats ctrl-xgb-meta by > band on MAE **and R²**, 3-seed confirmed):
neither blend beats on R² (both −0.134), so nothing is promoted or tagged. The
champion (metadata-only XGBoost, still the `mae_xgb` branch, still not a saved
standalone definition) is unchanged. **SVM is the 5th dense-regime family to lose
to one-hot identity on this corpus**, now on rotation, with the new twist that it
even fools an MAE-driven blend grid.

## Best on record after this work — ROTATION target (unchanged)

| model | dataset | MAE | RMSE | R² | source |
|---|---|---|---|---|---|
| **metadata-only XGBoost** (= the `mae_xgb` branch, AE-latent dropped; not a saved standalone definition) | ds-20260608-204402-p64-s42 | **0.3879** | **0.4475** | **0.191** | experiments/2026-06-09-ae-xgboost-blend.md (confirmed here) |
| xgboost on latent+metadata combined | ds-20260608-204402-p64-s42 | 0.3965 | — | 0.166 | rotation pivot seed run |
| xgboost on AE-latent only (64 cols) | ds-20260608-205802-p64-s42 | 0.4098 | — | 0.132 | rotation pivot seed run |
| svm best (poly, ε-tube artifact MAE) | ds-20260609-141622-p1-s42 | 0.3814 | 0.5303 | −0.137 | THIS report — REFUTED (R² < 0) |

The metadata-only XGBoost champion stands. The SVM row is listed for completeness
and is explicitly refuted: its lower MAE is an artifact, its R² is negative.

## Follow-ups

1. **The metadata-only XGBoost champion is still not a saved definition.** Register
   it first-class (it is reproducible byte-for-byte via autoencoder-xgboost kNN
   k=10, w=0, on the 178-col or the metadata-only matrix) so it can be pinned in
   the best-models group and served to Pathfinder — and run the 3-seed
   confirmation (the noise data is already half-built: s17 0.3931, s101 0.3964,
   s42 0.3879 are exactly that study for the champion config). The band is now
   measured, so significance is decidable.
2. **Fix the blend grid objective for binary targets.** Grid-by-MAE just
   mis-selected a worse-than-mean member; an R²/logloss objective (or AUC) is the
   right criterion for rotation. This is a framework decision (upstream), the same
   family as the NaN-MAPE binary-metric limitation — worth a single coherent
   "binary-target metric pipeline" fix.
3. **Identity RESOLUTION remains the only open lever** (raise artist/genre vocab
   caps so the `__other__` tail gets real columns) — feature modalities and now a
   5th model family have all failed; the ceiling is set by categorical identity.

## Artifacts

- Datasets built: `ds-20260609-141622-p1-s42` (DS-META, metadata-only, pca floor,
  transform none, 115 cols = 1 pca + 7 numeric + 107 one-hot, 9,805/2,451 @ seed
  42), `ds-20260609-141641-p1-s17`, `ds-20260609-141641-p1-s101` (noise siblings,
  same recipe, seeds 17/101). One throwaway dataset `ds-20260609-141540-p1-s42`
  was built with the default `log_target=true` (transform log1p) before I set
  `log_target=false`; it is unused by any run (the canonical rotation recipe is
  transform `none`).
- Runs: 8 stage-1 + 2 stage-2, all `succeeded` (run ids in the tables above).
- Diagnostics read from disk (API does not surface them): `blend_weight`,
  `mae_ae`, `mae_xgb` from `data/runs/<id>/metrics.json`; `weights` + member
  `solo_metrics` from `data/runs/<id>/blend.json`; prediction spread from
  `data/runs/<id>/predictions.json`.
