# 2026-06-09 — Autoencoder-latent + XGBoost blend ensemble (first campaign on the rotation target)

## Goal

First experiment on the pivoted target. As of 2026-06-08 the active
`[target]` is **`rotation` = (play_count ≥ 2)**, a binary [0,1] taste-fit
signal (transform `none`, units P(rotation)) that feeds the Pathfinder
`W_FIT` term. Engagement is demoted to a documented non-target field.

**Hypothesis:** a precomputed 64-dim song-autoencoder latent (Qdrant
collection `spotify_tracks_song_ae`) carries the track's *identity*. Blending
an AE-latent regressor with a metadata-XGBoost model
(`yhat = w·yhat_ae + (1−w)·yhat_xgb`, `w` tuned on an internal train holdout)
might beat metadata/xgboost alone.

**Baselines on record (rotation target, same family of datasets):**

- xgboost on latent+metadata **combined** (this same dataset,
  ds-20260608-204402-p64-s42): MAE **0.3965**, R² 0.166 — PRIMARY baseline to beat.
- xgboost on AE-latent **only** (ds-20260608-205802-p64-s42, 64 cols): MAE 0.4098, R² 0.132.
- metadata-only xgboost: captured directly as the `mae_xgb` diagnostic in
  every blend run, ~0.388.

Held constant across all runs: dataset ds-20260608-204402-p64-s42 (64 AE
latent + 114 metadata = 178 cols, seed 42, 9805 train / 2451 test, 12256 rows,
no exclusions — rotation==0 is the valid negative class); XGB branch at
defaults (500 trees, depth 6, lr 0.05, subsample/colsample 0.8); blend weight
auto-tuned on a `val_fraction` holdout carved from train (test untouched).

**Predictor:** `autoencoder-xgboost` (new this session,
`predictors/autoencoder-xgboost/train.py`). Splits dataset columns by manifest
kind — latent = `kind.type=="pca"` (the 64 AE-latent cols), metadata = the
rest. AE branch = kNN (Euclidean on raw latent) or StandardScaler+RidgeCV on
latent cols; XGB branch on metadata cols; blend weight tuned on the holdout.
Diagnostics `mae_ae` / `mae_xgb` / `blend_weight` are written into the on-disk
`data/runs/<id>/metrics.json` only (the API metrics object carries only
mae/rmse/r2/mape/medape/n_test).

## Outcome in one line

The blend adds **nothing** — the holdout tuner drives `w → 0.0` in all 6 arms because the AE-latent standalone signal (best kNN mae_ae 0.426, ridge 0.462) is uniformly weaker than metadata-XGBoost (0.3879), so every run collapses to identical metadata-only XGBoost (**MAE 0.3879, R² 0.191**), which edges the 0.3965 latent+metadata-combined baseline but by less than an UNMEASURED noise band — no significant win, nothing promoted.

## Results

All 6 runs **succeeded**. Because `w → 0` everywhere, the blended test metrics
are byte-identical across all runs (same metadata-XGB branch, same seed); the
only axis that moves is the AE-branch standalone error `mae_ae`. mape/medape
shown as % of the [0,1] target.

| run id | AE branch | blended MAE | mae_ae | mae_xgb | w | R² | RMSE | mape | medape |
|---|---|---|---|---|---|---|---|---|---|
| run-20260609-131545-e34e4 | kNN k=10  | **0.38789** | **0.4259** | 0.38789 | 0.0 | **0.1906** | **0.4475** | **43.9%** | **47.4%** |
| run-20260609-131545-29775 | kNN k=25  | **0.38789** | 0.4403 | 0.38789 | 0.0 | **0.1906** | **0.4475** | **43.9%** | **47.4%** |
| run-20260609-131545-5eec5 | kNN k=50  | **0.38789** | 0.4510 | 0.38789 | 0.0 | **0.1906** | **0.4475** | **43.9%** | **47.4%** |
| run-20260609-131545-2f638 | kNN k=100 | **0.38789** | 0.4609 | 0.38789 | 0.0 | **0.1906** | **0.4475** | **43.9%** | **47.4%** |
| run-20260609-131545-b3b88 | kNN k=200 | **0.38789** | 0.4702 | 0.38789 | 0.0 | **0.1906** | **0.4475** | **43.9%** | **47.4%** |
| run-20260609-131545-e84cf | ridge auto | **0.38789** | 0.4623 | 0.38789 | 0.0 | **0.1906** | **0.4475** | **43.9%** | **47.4%** |

All rows are co-winners on the blended metric (they are the same model); the
distinguishing column is `mae_ae`.

### The `mae_ae(k)` curve (kNN arm, AE-latent standalone)

| k | mae_ae |
|---|---|
| 10  | 0.4259 |
| 25  | 0.4403 |
| 50  | 0.4510 |
| 100 | 0.4609 |
| 200 | 0.4702 |

Monotonically **worse** as k grows: the AE latent's rotation signal is
strongest in tight local neighborhoods (k=10) and washes out toward the base
rate as the neighborhood widens. The ridge-on-latent arm (mae_ae 0.4623) lands
between k=100 and k=200 — a global linear read of the latent is no better than
a ~150-neighbor average. Crucially, **even the best AE standalone (0.4259) is
worse than metadata-only XGBoost (0.3879) and worse than AE-latent-only
*xgboost* (0.4098 on record)** — kNN/ridge underuse the latent relative to
trees, and the latent underperforms metadata regardless.

## Findings

- **The blend collapses to metadata-only XGBoost, deterministically.** The
  weight tuner is doing its job: given an AE branch that is strictly worse than
  the XGB branch on the holdout, the optimal convex weight is `w=0`. This is
  not a tuning artifact to fix — it is the correct answer to the question
  asked. The AE latent carries no rotation signal that metadata does not
  already carry better.
- **Metadata-only XGBoost (0.38789) beats the latent+metadata-*combined*
  xgboost baseline (0.3965).** Same dataset, same seed, same trees — the only
  difference is whether the 64 AE-latent columns are fed to a single unified
  XGBoost. Feeding them in makes it *worse* by ~0.0086 MAE. So the AE latent
  doesn't just fail to help as a separate branch — it actively **dilutes** a
  unified xgboost (64 low-signal dims the trees spend splits on). The
  blend-architecture's one real benefit here is that it *isolates* the latent
  into a branch the tuner can then zero out, recovering the clean
  metadata-only result.
- **Significance is NOT established.** The 0.38789 vs 0.3965 gap is 0.0086 MAE
  on a [0,1] target. The noise band on the rotation target is **UNMEASURED** —
  the historical "<0.2 MAE = noise" band was for the engagement scale
  (MAE ~2.4) and does NOT transfer to this ~0.39 / [0,1] scale. A delta this
  small could easily be split-seed noise. Per the decision rule we report the
  number but do **not** assert a significant win, and nothing is promoted/saved
  from this campaign.
- **Failure mode interpreted, not papered over.** This is a clean refutation of
  the hypothesis: the song-AE latent's *identity* signal is fully subsumed by
  (and weaker than) the artist/genre/album one-hot metadata for the rotation
  decision. This is the same convergent lesson the engagement campaigns found
  (identity one-hots beat dense vectors — co-listening, content-text, acoustic);
  the song-AE latent is now a fourth dense modality that loses to one-hots,
  this time on the rotation target.

## Best on record after this work (rotation target)

| model | dataset | MAE | R² | source |
|---|---|---|---|---|
| **metadata-only xgboost** (= the `mae_xgb` branch in every blend run; the AE-latent cols dropped) | ds-20260608-204402-p64-s42 | **0.3879** | **0.191** | this campaign |
| xgboost on latent+metadata combined | ds-20260608-204402-p64-s42 | 0.3965 | 0.166 | rotation pivot seed run (run-20260608-204415-6eeda) |
| xgboost on AE-latent only | ds-20260608-205802-p64-s42 | 0.4098 | 0.132 | rotation pivot seed run |
| autoencoder-xgboost blend (kNN or ridge AE branch) | ds-20260608-204402-p64-s42 | 0.3879 (w→0, ≡ metadata-only xgboost) | 0.191 | this campaign |

Caveat: all single-split, single-seed; deltas below an as-yet-unmeasured noise
band. The 0.3879 leader is the metadata-only branch *inside* the blend runs —
it is not a saved standalone definition (none was created; the decision rule
did not authorize a promotion).

## Follow-ups

1. **Measure the rotation noise band (immediate, blocks every claim here).**
   Repeat the headline config — a plain metadata XGBoost on the
   ds-2026...-p64-s42 recipe — across 3–5 data-split seeds, plus xgboost
   internal-seed spread at the fixed split. Until σ_split is known we cannot say
   whether 0.3879 vs 0.3965 (the metadata-only edge over latent+metadata
   combined) is real. This is the single most valuable next run.
2. **Confirm the dilution finding cheaply.** The implied recommendation is "drop
   the 64 AE-latent cols and run plain metadata XGBoost as the rotation
   champion." That's exactly the `mae_xgb` branch; worth registering it as a
   first-class definition *after* the noise band shows the edge is real.
3. **Do NOT pursue a non-blend latent+metadata xgboost variant** as an
   improvement axis — we already have it (the 0.3965 baseline) and it is worse
   than dropping the latent. The latent is a dead modality for rotation, matching
   the convergent identity-beats-dense-vectors finding.
4. **best-model-selector** should be spawned to review the group: the rotation
   champion landscape has shifted (metadata-only xgboost now leads on record),
   and the group should be checked for family diversity and for any stale
   engagement-target models that no longer reflect the active target.

## References

- Prior identity-vs-dense-vector findings: see PROJECT-FACTS.md "Known pitfalls
  per feature / dataset axis" (content-text REJECTED, acoustic REJECTED,
  convergent-finding note) — all on the historical engagement target.
- Rotation pivot record: PROJECT-FACTS.md "Taste-fit reframe for Pathfinder
  (2026-06-08)".
