# Experiment: burn-deep-classifier topology bake-off on the full-64-d binary rotation dataset

## Goal

Replace the failed sklearn pyramid-MLP (a diagnosed representation/optimization
pathology — raw one-hots standardized into outlier spikes, logloss ~2.0 =
confidently miscalibrated) with a NEW Rust/burn predictor
**`burn-deep-classifier`** that the user asked for as "a change of topology" in
"full rust". The predictor splits the matrix into a standardized continuous
block (64 PCA + 7 numeric) and **RAW** (un-standardized) one-hot identity groups
(artist 61, genre_primary 41, album_type 4), and exposes a `topology` enum
{mlp, embeddings, wide_deep}. Phase 1 compares the three topologies; Phase 2
tunes the winner. Two questions: (i) how much does the Rust reimplementation +
raw one-hots + proper training recover vs the sklearn pyramid, and (ii) does a
representation-aware ANN reach the tree tier on this one-hot-heavy data.

**Fixed baseline.**
- Dataset: **`ds-20260609-204419-p64-s42`** — "Song-AE · rotation · p64 +
  metadata (2026 corpus, canonical)". **23,529 rows** (18,823 train / 4,706 test
  @ seed 42), **177 cols** (64 song-AE PCA latent + 7 numeric + 106 identity
  one-hots). Target `metadata.rotation`, task binary, transform none. The
  full-64-d matrix, same dataset as all prior full-64-d work.
- The burn predictor's internal split seed is fixed (1337) in the binary; there
  is no `seed` hyperparam. Runs are *near*-deterministic given hyperparams (see
  Findings — a sub-band residual non-determinism was observed).
- **Held hyperparameters (all Phase-1 arms):** epochs 250, patience 20,
  val_fraction 0.15 (early-stop on validation logloss, best-epoch weights
  restored; test split untouched), hidden [128,64], embed_dim 8, dropout 0.1,
  lr 0.001, batch_size 256, activation relu. The ONLY Phase-1 axis is `topology`.
- **In-dataset references (same dataset — do NOT compare cross-corpus to the old
  0.7429 DS-META champion):** xgboost-classifier **0.7204** (champion), catboost
  0.7111, ngboost 0.6917, best converged sklearn pyramid-mlp 0.6792 / 0.6781
  under-trained. See `experiments/archive/2026-06-12-classifier-bakeoff-p64.md`
  and `experiments/archive/2026-06-12-pyramid-mlp-arch-scan.md`.
- **Noise band.** σ_AUC ≈ 0.00628, 2σ = **0.0126** (measured on the OLD corpus —
  approximate here; on 4,706 test rows the true band is plausibly tighter).

## Outcome (one line)

**The full-rust `burn-deep-classifier` obliterates the sklearn pyramid's
calibration blowup — every topology lands at logloss ~0.62 (vs the pyramid's
2.0–2.4) at AUC ~0.69–0.70, recovering the calibration in full and lifting AUC to
the pyramid's converged ceiling; the plain `mlp` topology over RAW one-hots wins
Phase 1 (AUC 0.7001, logloss 0.6211), narrowly ahead of `embeddings` (0.6934) and
`wide_deep` (0.6918) — embeddings beats wide_deep but both are inside one band of
mlp and of each other; the Phase-2 mlp tune (lr 0.0005, hidden [128,64]) tops out
at AUC 0.7038 — +0.0037 over the Phase-1 winner, INSIDE the band, so the Phase-1
config stands and, at 0.7038 < ~0.71, the net lands ~1 band below catboost
(0.7111) and ~1.3 bands below xgboost (0.7204): representation-aware and now
calibration-fixed, but still short of the tree tier, so NO definition is saved.**

## Phase 1 — topology bake-off (axis = topology)

All 3 arms `succeeded` (exit 0). Bare-predictor runs on
`ds-20260609-204419-p64-s42`. Held params as above (epochs 250, patience 20,
val_fraction 0.15, hidden [128,64], embed_dim 8, dropout 0.1, lr 0.001, batch 256,
relu). AUC higher-better; logloss, brier lower-better. Winner bolded; best cell
per metric bolded.

| arm | run id | topology | AUC | logloss | accuracy | brier |
|---|---|---|---|---|---|---|
| **T-mlp** | **run-20260612-140116-41fc9-burn-deep-classifier** | **mlp** | **0.7001** | **0.6211** | **0.6470** | **0.2169** |
| T-emb | run-20260612-140116-0eda4-burn-deep-classifier | embeddings | 0.6934 | 0.6251 | 0.6436 | 0.2187 |
| T-wd | run-20260612-140116-e85be-burn-deep-classifier | wide_deep | 0.6918 | 0.6281 | 0.6407 | 0.2198 |

### Phase 1 decision rule, applied mechanically

1. **Winner = highest test AUC = `mlp` (T-mlp, AUC 0.7001).** embeddings −0.0067
   and wide_deep −0.0083 both sit inside the 0.0126 band of mlp — "at least
   equal, likely slightly behind", not refuted.
2. **embeddings vs wide_deep (the user's "1 vs 2"):** embeddings 0.6934 beats
   wide_deep 0.6918 by +0.0016 — sub-band, effectively a tie, embeddings a hair
   ahead. Of the user's two proposed topologies, embeddings is the (marginally)
   better one, but the *plain* mlp control beats both.
3. **Phase-2 gate (Phase-1 best AUC ≥ 0.700):** mlp = 0.7000586 ≥ 0.700.
   **GATE PASSED (barely, by 0.00006).** Phase 2 authorized on `mlp`.

## Phase 2 — tune the winning topology (mlp)

Grid on `mlp`: lr {0.0005, 0.001, 0.002} × hidden {[256,128], [128,64]} = 6 runs,
holding epochs 250 / patience 20 / val_fraction 0.15 / batch 256 / dropout 0.1 /
embed_dim 8 / relu. (The embed_dim extra run applies only if the winner is
`embeddings`; the winner was `mlp`, so it was not run.) All 6 `succeeded`. Sorted
by AUC; winner bolded, best cell per metric bolded. The lr-0.001/[128,64] row
(18c9e) is the Phase-1 T-mlp config re-run.

| run id | lr | hidden | AUC | logloss | accuracy | brier |
|---|---|---|---|---|---|---|
| **run-20260612-140211-6575a-burn-deep-classifier** | **0.0005** | **[128,64]** | **0.7038** | 0.6221 | 0.6521 | 0.2172 |
| run-20260612-140211-21ad5-burn-deep-classifier | 0.002 | [128,64] | 0.6997 | 0.6227 | **0.6532** | 0.2174 |
| run-20260612-140211-18c9e-burn-deep-classifier | 0.001 | [128,64] | 0.6996 | **0.6213** | 0.6447 | **0.2170** |
| run-20260612-140211-fecb0-burn-deep-classifier | 0.0005 | [256,128] | 0.6966 | 0.6232 | 0.6424 | 0.2179 |
| run-20260612-140211-4c359-burn-deep-classifier | 0.002 | [256,128] | 0.6963 | 0.6245 | 0.6460 | 0.2181 |
| run-20260612-140211-0f4e1-burn-deep-classifier | 0.001 | [256,128] | 0.6957 | 0.6252 | 0.6424 | 0.2187 |

### Phase 2 decision rule, applied mechanically

- Best tuned config = **lr 0.0005 / [128,64] (6575a), AUC 0.7038**. Gain over the
  Phase-1 winner (0.7001) = **+0.0037**, which is **< 0.0126** (inside the band).
- Rule: "Best tuned config beating the Phase-1 winner by > 0.0126 = the config to
  save; **else save the Phase-1 winner config.**" → +0.0037 does NOT clear the
  bar → the Phase-1 winner config is the nominal config of record.
- Save-a-definition rule: "persist a definition IF the best AUC is competitive —
  within ~1 band (≈0.013) of or above the catboost/xgboost tier (AUC ≥ ~0.71)."
  Campaign best = **0.7038 < 0.71** (−0.0073 vs catboost 0.7111, −0.0166 vs
  xgboost 0.7204). **NOT competitive at the tree tier → NO definition saved, no
  dataset tag, nothing flagged for the best-model-selector to enter the tree
  tier.** Reported as the topology comparison + the (now calibration-fixed)
  ceiling, per the rule.

## Findings

- **(i) The Rust/burn reimplementation FIXED the calibration blowup completely —
  logloss 2.0 → ~0.62.** The single most important result. The sklearn pyramid's
  signature pathology was logloss 2.0–2.4 / brier 0.33 (confidently miscalibrated)
  — diagnosed as raw one-hots being standardized into outlier spikes. Every
  `burn-deep-classifier` arm posts logloss **0.621–0.628** and brier **0.217–0.220**
  — squarely in the trees' ~0.60–0.63 / ~0.21 regime. The fix is exactly the
  representation change the design hypothesized: keeping the identity one-hots
  **raw** (the predictor standardizes only the 64 PCA + 7 numeric continuous block)
  removes the outlier-spike pathology that wrecked the sklearn pyramid's
  calibration. This is a clean confirmation that the pyramid's logloss-2.0 was a
  *representation* artifact (fixable), not an intrinsic property of MLPs on this
  data. The PROJECT-FACTS pyramid entry's "broken calibration" verdict is now
  qualified: it was broken *for the sklearn-pyramid representation*, and a
  representation-aware net repairs it.

- **(i, cont.) AUC also recovered to (and slightly past) the pyramid's converged
  ceiling.** Best burn AUC 0.7038 vs the best *converged* sklearn pyramid 0.6792
  (arch scan) / 0.6781 (under-trained bake-off) = **+0.025 / +0.026**, ~2 bands of
  genuine ranking-power gain. So the topology change bought both a calibration fix
  (the headline) AND a modest but real AUC lift over the entire pyramid family. The
  prior facts conclusion that "the MLP family is closed for AUC at ceiling ≈0.68"
  is **revised**: a raw-one-hot rust net reaches ~0.70.

- **(ii) embeddings vs wide_deep vs plain mlp — plain mlp wins, all within one
  band.** Phase-1: mlp 0.7001 > embeddings 0.6934 > wide_deep 0.6918, a 0.0083-wide
  strip entirely inside the 0.0126 band. The user's "option 1 vs option 2" question:
  **embeddings (option 1) edges wide_deep (option 2) by +0.0016 (sub-band, a tie)**,
  but the plain dense `mlp` control over the concatenated raw one-hots beats both.
  Interpretation: with only three identity groups of modest cardinality (artist 61,
  genre 41, album_type 4), learned per-group embeddings and a wide-linear branch
  add machinery without adding signal a plain MLP over the raw one-hots doesn't
  already capture — the same "the identity is already in the one-hots" lesson the
  tree campaigns taught, now within the ANN family. Embeddings would likely matter
  more at higher cardinality (raise the vocab caps and re-test — see follow-ups).

- **(iii) Does a representation-aware ANN reach the tree tier? Not quite — ~1 band
  short, but the gap closed dramatically.** Campaign-best 0.7038 is −0.0073 vs
  catboost (0.7111, ~0.6 band) and −0.0166 vs xgboost (0.7204, ~1.3 bands). The net
  is now within ~1 band of the *second* tree family (catboost) and ~1.3 bands of
  the champion — a far cry from the sklearn pyramid's ~3.3-band deficit. So the
  representation fix nearly closes the dense-vs-trees gap that the convergent
  finding predicted, but trees still read the sparse identity one-hots a hair more
  effectively than even a raw-one-hot MLP. The save bar (≥~0.71) is not met, so no
  definition is persisted; the net is a strong, now-honest dense baseline, not a
  new champion.

- **mlp tuning is flat and prefers the SMALLER net + slower lr.** All 6 Phase-2
  cells sit in a 0.0081-wide AUC strip (0.6957–0.7038), entirely inside the band —
  no config is significantly better than another. The only visible structure:
  every [128,64] cell (0.6996–0.7038) beats every [256,128] cell (0.6957–0.6966),
  i.e. the **narrower hidden geometry is uniformly better** (more capacity slightly
  hurts on this modality — the same "depth/width hurts" signature the pyramid arch
  scan showed), and within [128,64] the **slowest lr (0.0005) is best**. Clean,
  sensible directions, but all sub-band.

- **Residual sub-band non-determinism (minor, noted).** The design stated runs are
  deterministic given hyperparams (fixed internal split seed 1337). The Phase-1
  T-mlp config (lr 0.001, [128,64]) scored AUC **0.700059**; its Phase-2 re-run
  (18c9e, identical config) scored **0.699616** — Δ 0.00044, ~0.07 of a band.
  Negligible for any decision here, but it means the burn binary is *near*- not
  *bit*-deterministic (likely float-reduction/thread-scheduling order in the burn
  training loop). Worth a fixed-config repeat if a future campaign needs a tight
  determinism guarantee from this predictor.

- **No failure modes.** All 9 runs (3 + 6) exited 0; no stderr captured, no
  spurious `stopped`/`interrupted`, no relaunch needed, the predictor's graceful
  early-stop fired cleanly on every arm. The new predictor is well-behaved.

## Best on record after this work — ROTATION target, AUC, on the NEW full-64-d dataset `ds-20260609-204419-p64-s42`

| model | dataset | AUC | logloss | accuracy | source | registered? |
|---|---|---|---|---|---|---|
| **xgboost-classifier** (n_est 500, depth 6, lr 0.05, sub/col 0.8, seed 42) | ds-20260609-204419-p64-s42 (177) | **0.7204** | 0.6075 | 0.6585 | `archive/2026-06-12-classifier-bakeoff-p64.md` | **YES — `xgboost-one` (run 439d8)** |
| xgboost-classifier (depth 8, lr 0.03 — sweep best, single-seed) | ds-20260609-204419-p64-s42 | 0.7261 | 0.6044 | 0.6651 | same | no — flagged for 3-seed confirm |
| catboost-classifier (iter 500, depth 6, lr 0.05) — 2nd tree family, in-band | ds-20260609-204419-p64-s42 | 0.7111 | 0.6132 | 0.6560 | same | no |
| **burn-deep-classifier (mlp, lr 0.0005, hidden [128,64]) — best ANN, calibration fixed** | ds-20260609-204419-p64-s42 | **0.7038** | 0.6221 | 0.6521 | **this report** | **no (below the ~0.71 save bar)** |
| burn-deep-classifier (mlp, lr 0.001, hidden [128,64] — Phase-1 winner config) | ds-20260609-204419-p64-s42 | 0.7001 | 0.6211 | 0.6470 | this report | no |
| burn-deep-classifier (embeddings, defaults) | ds-20260609-204419-p64-s42 | 0.6934 | 0.6251 | 0.6436 | this report | no |
| ngboost-classifier (n_est 400, lr 0.03, base_depth 3) | ds-20260609-204419-p64-s42 | 0.6917 | 0.6293 | 0.6392 | same as xgboost | no |
| burn-deep-classifier (wide_deep, defaults) | ds-20260609-204419-p64-s42 | 0.6918 | 0.6281 | 0.6407 | this report | no |
| pyramid-mlp-classifier (sklearn, best converged [512,256]) — SUPERSEDED by burn-deep | ds-20260609-204419-p64-s42 | 0.6792 | 2.0584 | 0.6326 | `archive/2026-06-12-pyramid-mlp-arch-scan.md` | no |

NOTE: this leaderboard is on the NEW 23,529-row full-64-d dataset and is NOT
comparable to the 0.7429 DS-META AUC champion in PROJECT-FACTS (old 12,256-row
metadata-only corpus). The burn-deep ANN slots between the tree tier and the
ngboost/old-pyramid tier — the dense family's new, calibration-fixed ceiling on
this corpus.

## Follow-ups

1. **Identity resolution > topology (the standing lever, now also for the ANN).**
   The convergent finding: the headroom is raising the artist/genre vocab caps so
   the `__other__` tail gets real columns — NOT more topology. This is ALSO the
   most likely way to make `embeddings` finally pay off: per-entity embeddings have
   little to learn over 61 artist + 41 genre one-hots, but at (say) 256 artist
   columns the embedding topology should start beating the plain mlp. Re-run the
   Phase-1 bake-off on a higher-cardinality rebuild (consult the dataset-architect)
   before concluding embeddings are useless here.
2. **A converged-longer / regularization sweep on mlp (low priority).** AUC is flat
   across the grid and capped ~0.70; a dropout {0.0, 0.2, 0.3} or weight-decay axis,
   or longer patience, might add a fraction — but everything so far is sub-band, so
   this is unlikely to clear the ~0.71 save bar. Skip unless the ANN family is
   wanted for ensemble diversity.
3. **burn-deep-classifier as an ensemble member (diversity, not AUC).** At 0.7038
   with genuinely *different* (gradient-descent, dense) inductive bias from the
   GBDTs, it is a candidate for a consensus/blend even though it loses on solo AUC —
   its errors should decorrelate from the trees. A blend (xgboost + catboost +
   burn-deep-mlp) on a logloss/AUC objective is the natural next ensemble probe.
4. **Confirm determinism if needed.** The sub-band Phase-1↔Phase-2 re-run drift
   (Δ 0.00044) means the binary is near- not bit-deterministic; a fixed-config
   triple-run would quantify the predictor's own run-to-run σ if a tight guarantee
   is ever required.
5. **Pin the calibration win in the family record.** This report supersedes the
   "MLP family is closed / calibration is a fixed ceiling" reading for the rust
   predictor — the next ANN campaign should start from `burn-deep-classifier (mlp,
   raw one-hots)`, not the sklearn pyramid.
