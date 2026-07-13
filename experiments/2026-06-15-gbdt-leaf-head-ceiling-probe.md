# Experiment: GBDT-leaf-embedding → head ceiling-probe vs the bare xgboost-classifier champion (rotation / AUC)

## Goal

Test whether the classic Facebook GBDT→LR trick, generalized to GBDT→MLP,
beats the bare tree on the full-64-d binary `rotation` dataset. A
gradient-boosted tree **hard-frozen at the champion config** emits per-row leaf
indices; those leaves are one-hot-encoded into the tree's learned axis-aligned
partition; a linear (L2 logistic) or shallow-MLP head trains on top. The
hypothesis was that this would **NOT** beat the bare tree — three convergent
on-record findings predict it: trees are the ceiling here, every dense head caps
below the trees on this dataset, and feeding a learned latent downstream
dilutes. The expected refutation is itself the citable finding; an in-band tie
would trigger a 3-seed spot-check; a clear win (> 0.7330) would overturn the
prior and open a new family.

**Fixed baseline / control.**
- Dataset: **`ds-20260609-204419-p64-s42`** — "Song-AE · rotation · p64 +
  metadata (2026 corpus, canonical)". **23,529 rows** (18,823 train / 4,706 test,
  test_ratio 0.2), **177 cols** (64 song-AE PCA latent + 7 numeric + 106 identity
  one-hots). Target `metadata.rotation`, task binary, transform none. The
  full-64-d matrix.
- Seed 42 (split + model) for every arm; all unspecified head hyperparameters at
  the predictor's registry defaults.
- **Control / champion of record:** the registered `xgboost-classifier-rotation-p64`
  config, reference run `run-20260609-204726-439d8-xgboost-classifier`, AUC
  **0.7204383** / logloss 0.6075 / acc 0.6585 / brier 0.2109. Hyperparams: n_est
  500, depth 6, lr 0.05, subsample/colsample 0.8, min_child_weight 1.0, reg_lambda
  1.0, seed 42. Deterministic — reproduces byte-identical.
- **The leaf-emitting GBDT in every treatment arm is EXACTLY this champion config**
  — it is hard-frozen inside the `gbdt-leaf-head-classifier` predictor, not a
  tunable. Only the head varies.

**Noise band.** σ_AUC ≈ 0.00628, 2σ = **0.0126** (AUC, higher-better), measured
on the champion across split seeds 42/17/101. Refutation threshold = champion −
band = 0.7204383 − 0.0126 = **0.7078**.

**Cross-matrix caveat (do not mis-read).** ALL arms run on
`ds-20260609-204419-p64-s42` ONLY. Do NOT compare to the 0.7429 DS-META champion
(old 12,256-row metadata-only corpus) nor the 0.7299 high-vocab numbers
(497-col matrix) — different corpora / matrices. The only valid comparison is the
in-dataset 0.7204 champion reference.

## Outcome (one line)

**REFUTED, exactly as predicted: all three GBDT→head arms cap below the bare tree
by 1.8–2.5 noise bands (best = GBDT→linear AUC 0.6978, then GBDT→MLP[256,128]
0.6921, GBDT→MLP[128] 0.6897 — every one below the 0.7078 refutation threshold),
the control reproduced the champion to all 16 digits (0.7204383403100314), and
every head is badly mis-calibrated relative to the trees (logloss 0.82–1.31 vs
0.61, brier 0.26–0.30 vs 0.21) — learning a head over one-hot tree leaves
destroys both ranking and calibration; no T4 fired, no definition saved.**

## Results

Base batch = 4 runs (C0 control + T1–T3 treatments). All `succeeded` (exit 0),
no stderr, no spurious stops, no relaunch needed. Single split, seed 42. AUC
higher-better; logloss, brier lower-better. Winner (control) bolded; best cell
per metric bolded.

| arm | run id | predictor | head / config | AUC | logloss | accuracy | brier |
|---|---|---|---|---|---|---|---|
| **C0 (control)** | **run-20260615-151551-f2911-xgboost-classifier** | **xgboost-classifier** | **bare champion: n_est 500, depth 6, lr 0.05, sub/col 0.8, mcw 1.0, λ 1.0, seed 42** | **0.7204** | **0.6075** | **0.6585** | **0.2109** |
| T1 | run-20260615-151551-b4db5-gbdt-leaf-head-classifier | gbdt-leaf-head-classifier | head=linear (L2 logistic on one-hot leaves), l2 1.0 | 0.6978 | 1.3119 | 0.6441 | 0.2944 |
| T3 | run-20260615-151551-44b1d-gbdt-leaf-head-classifier | gbdt-leaf-head-classifier | head=mlp, hidden [256,128], head_lr 1e-3 | 0.6921 | 1.1813 | 0.6396 | 0.2951 |
| T2 | run-20260615-151551-9d655-gbdt-leaf-head-classifier | gbdt-leaf-head-classifier | head=mlp, hidden [128], head_lr 1e-3 | 0.6897 | 0.8162 | 0.6356 | 0.2603 |

(Treatment rows sorted by AUC, descending.)

### Decision rule, applied mechanically

1. **Control gate — C0 must reproduce AUC 0.7204383… to ≥4 decimals.** C0 =
   **0.7204383403100314**, byte-identical to the reference run's
   0.7204383403100314 (all 16 digits — deterministic xgboost, same config, same
   seed). **GATE PASSED.** Treatments are interpretable.
2. **Refuted branch (expected).** Threshold = champion − band = **0.7078**. Every
   treatment scores below it: T1 0.6978 (−0.0227, **~1.8 bands** below the
   champion), T3 0.6921 (−0.0283, **~2.2 bands**), T2 0.6897 (−0.0307, **~2.4
   bands**). The best treatment (0.6978) is itself 0.0100 below the 0.7078
   threshold. **The leaf-embedding stack is REFUTED — it loses by more than one
   band.** → No T4. No definition saved. Record as a finding; add a
   gbdt-leaf-head pitfall entry.
3. In-band-tie branch (would have fired T4): **not triggered** (best treatment
   0.6978 < 0.7078).
4. Win branch (> 0.7330): **not triggered** (best treatment 0.6978).

No prerequisite seed-variant datasets were built (T4 did not fire), as the design
specified.

## Findings

- **The expected refutation is confirmed, decisively.** Re-encoding the frozen
  champion tree's output as one-hot leaf indices and learning a fresh head on top
  *loses* AUC versus simply reading the tree's own leaf-value sums — by 1.8–2.5
  bands across all three heads. The leaf one-hots are a strictly lossier
  representation of the tree than the tree itself: the tree's additive leaf-value
  scoring already optimally weights every leaf jointly across 500 boosting rounds
  with shrinkage; replacing those learned, shrunk leaf weights with a head fit
  *post-hoc* on ~10k one-hot leaf-membership dimensions discards the boosting
  structure and re-estimates it worse. This is the same family as the on-record
  "learned-latent-fed-downstream dilutes" finding (the AE-latent dilution,
  `archive/2026-06-09-ae-xgboost-blend.md`) and the "every dense head caps below
  trees on this dataset" finding (`archive/2026-06-12-burn-deep-topology.md`):
  here the *input* to the dense head is the tree's own partition, and it still
  caps below — the dense head cannot even recover the tree that generated its
  features.

- **The linear head wins among the treatments, the MLPs do not help.** GBDT→linear
  (0.6978) > GBDT→MLP[256,128] (0.6921) > GBDT→MLP[128] (0.6897). Adding MLP
  capacity over the leaf one-hots does not recover the tree — it makes ranking
  slightly *worse* than the linear logistic read (the wider MLP edges the narrow
  one on AUC but both trail the linear head). A linear logistic regression on
  one-hot leaves is the literal Facebook GBDT→LR recipe, and it is the best of the
  three — consistent with the recipe's original framing as a *cheap linear* trick,
  not a deep-head play. None of it beats the tree.

- **CALIBRATION IS BADLY BROKEN ON EVERY HEAD — the over-confidence the smoke test
  flagged held up, and is worse than its AUC deficit.** The bare tree sits in its
  calibrated regime (logloss 0.6075, brier 0.2109). Every head blows past it:
  - **T1 (linear) logloss 1.3119, brier 0.2944** — confirms the design's explicit
    calibration flag (the smoke test saw logloss 1.34 at l2=1.0; the experiment
    run lands 1.31, same regime). L2-logistic on ~10k sparse leaf one-hots with the
    default inverse-C 1.0 is under-regularized: it drives a handful of leaf weights
    to large magnitudes and emits over-confident probabilities, so the logloss is
    ~2.2× the tree's even though the AUC gap is only ~1.8 bands. This is the
    sklearn-pyramid over-confidence signature (logloss far above the trees' ~0.61
    while AUC is only modestly behind), not a leakage signature — a
    **calibration/over-confidence red flag** for best-model-selector if any of
    these runs were ever (wrongly) considered for the group.
  - **T2 (MLP[128]) logloss 0.8162, brier 0.2603** — the best-calibrated head, but
    still markedly worse than the tree and the worst on AUC.
  - **T3 (MLP[256,128]) logloss 1.1813, brier 0.2951** — the wider MLP is nearly
    as over-confident as the linear head; extra capacity over leaf one-hots buys
    over-confidence, not calibration.
  None of these reaches the ~0.60–0.63 logloss / ~0.21 brier regime the trees
  hold; the leaf-head family is mis-calibrated by construction at default
  regularization.

- **No `failed`/NaN-MAPE trap, no failures.** The plugin reuses the
  classification metric block (auc/logloss/accuracy/brier — no MAPE), so the
  binary-target NaN-MAPE trap was correctly dodged: all 4 runs landed `succeeded`,
  exit 0, no stderr captured. No spurious `stopped`/`interrupted`, no relaunch.

- **Reproducibility.** C0 reproduced the champion to all 16 digits (deterministic
  xgboost). The MLP heads use a seeded sklearn MLP with adam + early stopping (10%
  val, n_iter_no_change 10), which is near- but not bit-deterministic; the linear
  head is the cleaner deterministic control. Single-seed reads for the heads —
  but the margins (1.8–2.5 bands below the tree) are far outside the band, so the
  refutation does not hinge on seed jitter (which is why the in-band T4 branch did
  not, and need not, fire).

- **Cross-matrix reminder.** These numbers are on the 177-col full-64-d matrix
  and are comparable only to the 0.7204 in-dataset champion. They are NOT
  comparable to the 0.7429 DS-META champion (old metadata-only corpus) nor the
  0.7299 high-vocab xgboost (497-col matrix). Nothing regressed; the tree held the
  line and the head stack lost on its own matrix.

## Best on record after this work — ROTATION target, AUC, on the full-64-d dataset `ds-20260609-204419-p64-s42`

The champion is unchanged. The leaf-head arms are recorded below the line for
the record; none is registered.

| model | dataset | AUC | logloss | accuracy | source | registered? |
|---|---|---|---|---|---|---|
| **xgboost-classifier** (n_est 500, depth 6, lr 0.05, sub/col 0.8, seed 42) | ds-20260609-204419-p64-s42 (177) | **0.7204** | 0.6075 | 0.6585 | archive/2026-06-12-classifier-bakeoff-p64.md (reproduced here) | **YES — `xgboost-classifier-rotation-p64`** |
| xgboost-classifier (depth 8, lr 0.03 — sweep best, single-seed) | ds-20260609-204419-p64-s42 | 0.7261 | 0.6044 | 0.6651 | archive/2026-06-12-classifier-bakeoff-p64.md | no — flagged for 3-seed confirm |
| catboost-classifier (iter 500, depth 6, lr 0.05) — 2nd tree family, in-band | ds-20260609-204419-p64-s42 | 0.7111 | 0.6132 | 0.6560 | archive/2026-06-12-classifier-bakeoff-p64.md | no |
| burn-deep-classifier (topology mlp, lr 0.0005, [128,64], raw one-hots) — best ANN | ds-20260609-204419-p64-s42 | 0.7038 | 0.6221 | 0.6521 | archive/2026-06-12-burn-deep-topology.md | no |
| gbdt-leaf-head-classifier (head=linear, l2 1.0) — best leaf-head, REFUTED | ds-20260609-204419-p64-s42 | 0.6978 | 1.3119 | 0.6441 | this report | no |
| ngboost-classifier (n_est 400, lr 0.03, base_depth 3) | ds-20260609-204419-p64-s42 | 0.6917 | 0.6293 | 0.6392 | archive/2026-06-12-classifier-bakeoff-p64.md | no |
| gbdt-leaf-head-classifier (head=mlp, [256,128], head_lr 1e-3) — REFUTED | ds-20260609-204419-p64-s42 | 0.6921 | 1.1813 | 0.6396 | this report | no |
| gbdt-leaf-head-classifier (head=mlp, [128], head_lr 1e-3) — REFUTED | ds-20260609-204419-p64-s42 | 0.6897 | 0.8162 | 0.6356 | this report | no |
| pyramid-mlp-classifier (256/3, max_iter 300) | ds-20260609-204419-p64-s42 | 0.6781 | 2.4085 | 0.6326 | archive/2026-06-12-classifier-bakeoff-p64.md | no |

NOTE: this leaderboard is on the 23,529-row full-64-d dataset and is NOT
comparable to the 0.7429 DS-META champion nor the 0.7299 high-vocab xgboost.

## Follow-ups

1. **None on this axis — the leaf-head family is closed for rotation.** All three
   heads lose by >1 band with broken calibration; the prior is confirmed, not
   contested. Re-tuning the head regularization (raising l2 to tame T1's
   over-confidence) might fix *calibration* but cannot close a ~1.8-band *AUC*
   deficit — the leaf one-hots are a lossy view of a tree that already scores
   better directly. Not worth the machinery.
2. **best-model-selector — no champion change, but exclude the leaf-head runs if
   they auto-seat.** The deterministic top-12 recompute will have considered the 4
   new runs by AUC; C0 (0.7204) is a determinism duplicate of the existing
   champion config and the three leaf-head arms (0.6897–0.6978) are mid-pack with
   the over-confidence calibration flag (T1 logloss 1.31). Recommend a curation
   pass to de-dup C0 and exclude the leaf-head arms from the group (a refuted,
   mis-calibrated family — same disposition as the under-trained pyramid-mlp).
3. **Standing lever unchanged: identity resolution > new heads/families.** Per the
   convergent finding, the durable headroom is raising vocab caps (the high-vocab
   matrix already lifted xgboost to 0.7299), not new representations layered on the
   tree.
