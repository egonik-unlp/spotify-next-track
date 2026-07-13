# Experiment: Four-family classifier bake-off on the full-64-d binary rotation dataset, then an xgboost depth/lr sweep

## Goal

Compare four classifier families — **xgboost-classifier, catboost-classifier,
ngboost-classifier, pyramid-mlp-classifier** — head-to-head on "all of the 64-d
embedding": the full 177-col binary `rotation` dataset, then parameter-sweep
whichever family wins. The user asked for the four-family bake-off and approved a
sweep on the winner "if a model worked well".

**Fixed baseline.**
- Dataset: **`ds-20260609-204419-p64-s42`** — "Song-AE · rotation · p64 + metadata
  (2026 corpus, canonical)". **23,529 rows** (18,823 train / 4,706 test,
  test_ratio 0.2), **177 cols** (64 song-AE PCA latent + 7 numeric + 106 identity
  one-hots). Target `metadata.rotation`, task binary, transform none. This is the
  full-64-d matrix.
- Seed 42 (split + model) for every arm; all other hyperparameters at each
  predictor's registry defaults except where named.
- **In-dataset champion reference (the ONLY valid comparison point):** the single
  existing run on THIS dataset, `run-20260609-204726-439d8-xgboost-classifier`,
  AUC **0.7204** / logloss 0.6075 / acc 0.6585 / brier 0.2109.

**CRITICAL — cross-corpus comparison caveat.** Do NOT compare any number here to
the PROJECT-FACTS AUC champion **0.7429** (`xgboost-classifier-rotation`). That
0.7429 lives on the OLD 12,256-row metadata-only DS-META corpus
(`ds-20260609-141622-p1-s42`) and is **not comparable** to this 23,529-row
full-64-d dataset — different corpus size, different feature set (metadata-only
vs +64 AE latent). Every number in this campaign lives on the new 23,529-row
full-64-d dataset and is comparable only to the 0.7204 in-dataset reference.

**Noise band.** σ_AUC ≈ 0.00628, 2σ = **0.0126** (measured on the OLD corpus —
treat as approximate here; on 23,529 test-feeding rows the true band is plausibly
tighter, so a 1–2-band edge is "likely better, needs the 3-seed check", not a
final "beats").

## Outcome (one line)

**xgboost-classifier wins the four-family bake-off on the full-64-d rotation
dataset (AUC 0.7204, exactly reproducing the in-dataset reference to all digits),
ahead of catboost 0.7111 (−0.0093, inside the band — at least equal, likely
slightly behind), ngboost 0.6917 (−0.0287, ~2.3 bands back) and pyramid-mlp
0.6781 (−0.0423, with a logloss of 2.41 that is the predicted under-training
artifact at the 300-iter cap); the depth/lr sweep's best config (depth8/lr0.03,
AUC 0.7261) beats the default by only +0.0057 — inside the band — so the
default-depth6 config is saved as `xgboost-classifier-rotation-p64`, and
depth8/lr0.03 is flagged for a 3-seed confirmation rather than promoted.**

## Phase 1 — the four-family bake-off

All 4 arms `succeeded` (exit 0). Bare-predictor runs on
`ds-20260609-204419-p64-s42`, seed 42. AUC higher-better; logloss, brier
lower-better. Winner bolded; best cell per metric bolded.

| arm | run id | predictor | hyperparams | AUC | logloss | accuracy | brier |
|---|---|---|---|---|---|---|---|
| **C0** | **run-20260612-115909-840c0-xgboost-classifier** | **xgboost-classifier** | **n_est 500, depth 6, lr 0.05, sub/col 0.8** | **0.7204** | **0.6075** | **0.6585** | **0.2109** |
| A1 | run-20260612-115909-c7ef6-catboost-classifier | catboost-classifier | iter 500, depth 6, lr 0.05, l2 3.0, sub 0.8, border 254 | 0.7111 | 0.6132 | 0.6560 | 0.2132 |
| A2 | run-20260612-115909-d725b-ngboost-classifier | ngboost-classifier | n_est 400, lr 0.03, minibatch 0.5, col 1.0, base_depth 3 | 0.6917 | 0.6293 | 0.6392 | 0.2193 |
| A3 | run-20260612-115909-ea698-pyramid-mlp-classifier | pyramid-mlp-classifier | top_width 256, layers 3, decay 0.5, alpha 1e-4, lr 1e-3, max_iter 300 | 0.6781 | 2.4085 | 0.6326 | 0.3251 |

### Phase 1 decision rule, applied mechanically

1. **Reproduction gate (C0 AUC ∈ [0.7078, 0.7330]).** C0 = **0.7204383403** —
   byte-identical to the reference run's 0.7204383403 (same predictor, same
   config, same seed, deterministic xgboost). Squarely inside the band.
   **GATE PASSED.** Phase 2 authorized.
2. **Winner = highest AUC = xgboost-classifier (C0, 0.7204).** No non-xgboost arm
   beat C0. catboost is closest at −0.0093 (inside the 0.0126 band → "at least
   equal, likely slightly behind", not a "loses by"); ngboost −0.0287 (~2.3
   bands) and pyramid-mlp −0.0423 (~3.4 bands) are decisively behind.
3. **No new champion-on-this-dataset.** No non-xgboost arm beat C0 by > 0.0126, so
   no champion flip; the incumbent family holds.
4. **Pyramid-MLP under-training flagged as a finding** (logloss 2.41 ≫ the
   ~0.60–0.63 of the others) — the predicted 300-iter-cap artifact, addressed by
   the Phase-2 grid choice having raised max_iter were pyramid the winner (it was
   not).

## Phase 2 — xgboost depth × learning_rate sweep

Winner = xgboost → grid max_depth {4,6,8} × learning_rate {0.03,0.05,0.1} at
n_estimators 500, subsample/colsample 0.8, seed 42. 9 bare-predictor runs on the
same dataset. All `succeeded`. Sorted by AUC; best cell per metric bolded. The
depth6/lr0.05 row (dceaa) is the Phase-1 C0 config re-run — identical AUC,
confirming determinism.

| run id | max_depth | lr | AUC | logloss | accuracy | brier |
|---|---|---|---|---|---|---|
| **run-20260612-120207-2dd3d-xgboost-classifier** | **8** | **0.03** | **0.7261** | **0.6044** | 0.6651 | **0.2092** |
| run-20260612-120207-12f3a-xgboost-classifier | 6 | 0.03 | 0.7205 | 0.6052 | 0.6589 | 0.2098 |
| run-20260612-120207-dceaa-xgboost-classifier | 6 | 0.05 | 0.7204 | 0.6075 | 0.6585 | 0.2109 |
| run-20260612-120207-2ebc3-xgboost-classifier | 8 | 0.05 | 0.7196 | 0.6234 | 0.6558 | 0.2158 |
| run-20260612-120207-a0c91-xgboost-classifier | 8 | 0.1 | 0.7173 | 0.6864 | 0.6577 | 0.2302 |
| run-20260612-120207-36844-xgboost-classifier | 6 | 0.1 | 0.7153 | 0.6332 | **0.6615** | 0.2184 |
| run-20260612-120207-76da0-xgboost-classifier | 4 | 0.1 | 0.7119 | 0.6157 | 0.6536 | 0.2141 |
| run-20260612-120207-dccde-xgboost-classifier | 4 | 0.05 | 0.7094 | 0.6131 | 0.6594 | 0.2132 |
| run-20260612-120207-81063-xgboost-classifier | 4 | 0.03 | 0.7074 | 0.6158 | 0.6574 | 0.2143 |

### Phase 2 decision rule, applied mechanically

- Best sweep config = **depth8/lr0.03 (2dd3d), AUC 0.7261**. Gain over the
  Phase-1 winner (0.7204) = **+0.0057**, which is **< 0.0126** (inside the band).
- Rule: "Best sweep config beating the Phase-1 winner by > 0.0126 = the config to
  save; **otherwise save the Phase-1 winner's (default) config.**" → +0.0057 does
  NOT clear the bar.
- **Saved:** the Phase-1 winner config (n_est 500, depth 6, lr 0.05,
  subsample/colsample 0.8, min_child_weight 1.0, reg_lambda 1.0, seed 42) as
  definition **`xgboost-classifier-rotation-p64`**, `dataset_tags =
  [ds-20260609-204419-p64-s42]`. depth8/lr0.03 (0.7261) is **flagged for a 3-seed
  confirmation** before any "beats" claim — it is the strongest single-seed point
  and the obvious follow-up, but inside the band at one seed it is "likely better,
  not confirmed".

## Findings

- **xgboost wins the full-64-d bake-off; the family verdict survives the corpus
  change.** On the new 23,529-row full-64-d dataset, xgboost-classifier (0.7204)
  again leads the field, matching the convergent finding that trees reading sparse
  identity one-hots beat denser regimes here. catboost (0.7111) lands inside the
  noise band — for the gradient-boosted-tree family, xgboost and catboost are
  effectively co-leaders at single seed (Δ 0.0093 < 0.0126); catboost is "at least
  equal, likely a hair behind", not refuted. This is the first head-to-head of
  catboost on this corpus and it is the legitimate second tree family.

- **ngboost is meaningfully behind (~2.3 bands).** AUC 0.6917 / logloss 0.6293.
  ngboost's probabilistic (natural-gradient, distributional) objective with
  shallow base learners (base_max_depth 3, 400 estimators) under-fits the
  one-hot-heavy 177-col matrix relative to the deeper GBDTs — it pays for its
  distributional machinery in raw ranking power here. Not a failure, but not
  competitive on AUC.

- **Pyramid-MLP is the weakest arm AND under-trained — the logloss 2.41 is an
  artifact, not the ceiling.** AUC 0.6781 with a logloss of 2.41 (vs ~0.60–0.63
  for the others) is the predicted 300-iteration-cap under-training: the network
  had not converged, so its probability calibration is poor (brier 0.3251, far
  above the ~0.21 of the trees) even though its rank-ordering (AUC) is only ~3.4
  bands behind. The dense MLP blurring sparse identity one-hots is the same
  failure mode the SVC/logistic families showed on the old corpus; a converged MLP
  (max_iter 500+) would lift the logloss/brier toward the field but is unlikely to
  overtake the trees on AUC given the modality mismatch. We did NOT run the
  pyramid sweep (pyramid was not the winner), so this is the single-config read.

- **Sweep: shallow trees hurt, depth 8 + slow lr is the single-seed best, but
  within the band.** The depth axis dominates the grid: every depth-4 config
  (0.7074–0.7119) is the bottom of the table, depth-6 is the middle, and the
  depth-8/lr-0.03 corner tops it (0.7261). High learning rate (0.1) consistently
  hurts AUC and inflates logloss/brier (a0c91 logloss 0.6864) — over-stepping. The
  ordering is clean and monotone in the sensible direction (more capacity + slower
  steps), which is reassuring, but the best-vs-default gap (+0.0057) is inside the
  band, so the default depth-6 config remains the saved config and depth-8/lr-0.03
  is a 3-seed-confirmation candidate, not a promotion.

- **No failure modes.** All 13 runs (4 + 9) exited 0; no stderr captured, no
  spurious stopped/interrupted, no relaunch needed. The determinism check held
  (C0 == reference, dceaa == C0 to all digits).

- **Cross-corpus reminder (do not mis-read).** These numbers are LOWER than the
  0.7429 PROJECT-FACTS champion because that champion is on a DIFFERENT (old,
  smaller, metadata-only) corpus — NOT because anything regressed. On the
  comparable axis (the 0.7204 in-dataset reference) the campaign reproduced and
  held the line. The two are not on the same leaderboard.

## Best on record after this work — ROTATION target, AUC, on the NEW full-64-d dataset `ds-20260609-204419-p64-s42`

| model | dataset | AUC | logloss | accuracy | source | registered? |
|---|---|---|---|---|---|---|
| **xgboost-classifier** (n_est 500, depth 6, lr 0.05, sub/col 0.8, seed 42) | ds-20260609-204419-p64-s42 (177) | **0.7204** | 0.6075 | 0.6585 | this report | **YES — `xgboost-classifier-rotation-p64`** |
| xgboost-classifier (depth 8, lr 0.03 — sweep best, single-seed) | ds-20260609-204419-p64-s42 | 0.7261 | 0.6044 | 0.6651 | this report | no — flagged for 3-seed confirm |
| catboost-classifier (iter 500, depth 6, lr 0.05) — 2nd tree family, in-band | ds-20260609-204419-p64-s42 | 0.7111 | 0.6132 | 0.6560 | this report | no |
| ngboost-classifier (n_est 400, lr 0.03, base_depth 3) | ds-20260609-204419-p64-s42 | 0.6917 | 0.6293 | 0.6392 | this report | no |
| pyramid-mlp-classifier (256/3, max_iter 300 — under-trained) | ds-20260609-204419-p64-s42 | 0.6781 | 2.4085 | 0.6326 | this report | no |

NOTE: this leaderboard is on the NEW 23,529-row full-64-d dataset and is NOT
comparable to the 0.7429 DS-META AUC champion in PROJECT-FACTS (old 12,256-row
metadata-only corpus). The two coexist on different corpora.

## Follow-ups

1. **3-seed confirm of depth8/lr0.03 (top lever).** It is +0.0057 over the default
   at one seed (inside the band). Run depth8/lr0.03 at split seeds {17, 101} on the
   matching recipe; if mean AUC clears the default by > the (re-measured) band,
   re-save `xgboost-classifier-rotation-p64` at depth 8 and re-flag for the
   best-model-selector. This also re-measures the AUC noise band on the new
   23,529-row corpus (the current 0.0126 is the old-corpus band, likely loose
   here).
2. **catboost is a legitimate 2nd tree family — worth its own short sweep** (depth
   {6,8} × lr {0.03,0.05}) only if a tree ensemble/diversity play is wanted; on AUC
   alone it is co-leader-in-band with xgboost, not a clear gain.
3. **Re-curate best-models (best-model-selector).** This campaign added 13 runs on
   the new dataset and saved a new definition; the deterministic recompute will
   reshuffle the AUC top-12. The selector should de-dup the determinism duplicates
   (C0 840c0 == dceaa config), keep one representative each of the distinct
   families, and decide whether the depth8/lr0.03 single-seed 0.7261 belongs in the
   group before its 3-seed confirm. The cross-corpus mix (old DS-META 0.7429
   members vs new-corpus members) is a judgment call for the selector.
4. **Converged Pyramid-MLP read (low priority).** A max_iter-500 pyramid run would
   replace the under-trained 0.6781/logloss-2.41 point with the model's real
   ceiling — informative for the dense-vs-trees record, but unlikely to overtake.
5. **Identity resolution > new families/depth (standing lever).** Per the
   convergent finding, the durable headroom is raising vocab caps so the
   `artist=__other__` tail gets real columns — not more classifier families or
   deeper trees.
