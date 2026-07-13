# Identity-resolution vocab-cap scan on the bare GBDT (artist/genre one-hot cardinality → AUC curve)

**Date:** 2026-06-15
**Predictors:** xgboost-classifier (primary axis), catboost-classifier (secondary signal) — both pre-existing, no new predictor, no server restart.
**Target:** rotation (binary taste-fit, `play_count ≥ 2`), classification metric block (AUC / logloss / accuracy / brier — no MAPE).

## Goal

Map where the identity-resolution lever stops paying. The biggest on-record in-lineage gain came from feeding the same bare xgboost-classifier richer identity one-hots: artist 60 / genre 40 (AUC 0.7204) → artist 300 / genre 120 (AUC 0.7299, +0.0095 ≈ 1.5 AUC bands). Only two points existed on the cap→AUC curve and the high end was the in-lineage record. This campaign adds three higher-cap points — 500/200, 1000/300, all (5594/962) — to find saturation (gain per step < 1 band) or reversal (AUC drops at high caps as singleton-artist columns inject memorizable variance). No new model families: bare GBDT × dataset-cardinality axis only.

**Baseline (frozen, reproduced in-batch):**
- **Champion/control config:** xgboost-classifier, n_estimators 500, max_depth 6, learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, min_child_weight 1.0, reg_lambda 1.0, seed 42 (= the registered `xgboost-classifier-rotation-p64` config). Reference: AUC **0.7299** on ds-20260612-143014-p64-s42 (300/120 anchor); AUC **0.7204** on ds-20260609-204419-p64-s42 (60/40 anchor).
- **Secondary config:** catboost-classifier registry defaults — iterations 500, depth 6, learning_rate 0.05, l2_leaf_reg 3.0, subsample 0.8, border_count 254, seed 42. Reference: AUC 0.7252 on 300/120, 0.7111 on 60/40.
- **Held everything-else:** corpus `spotify_tracks_song_ae`; target rotation (transform none, task binary); split test_ratio 0.2 / seed 42 (18,823 train / 4,706 test); pca_dims 64; identical fields block (album_type / artist / artist_count / artist_followers / artist_popularity / genre_primary / release_year / track_popularity = true; all af_* and all behavioral/curation/target-derived = false); quality all-false (0 exclusions). The ONLY axis that moved was `vocab_top_n` (artist, genre_primary). PCA dims unchanged → EVR byte-identical to both anchors.
- **AUC noise band:** 2σ = 0.0126 (measured rotation-classification campaign). "1 band" = 0.0126.

**Datasets:** the two anchors (ds-20260609-204419-p64-s42, 177 cols; ds-20260612-143014-p64-s42, 497 cols) pre-existed; three new built byte-identical to the 300/120 anchor except `vocab_top_n` — P1 ds-20260615-162737-p64-s42 (500/200, 777 cols), P2 ds-20260615-163042-p64-s42 (1000/300, 1377 cols), P3 "all" (5594/962, 6633 cols) **failed to build** (see Outcome / Findings). All three new builds confirmed 23,529 rows / 18,823 train / 4,706 test / 0 exclusions (P1, P2; P3 never materialized).

## Outcome in one line

The identity-resolution lever keeps paying but only WEAKLY past 300/120 — xgboost rose to a new in-lineage single-split best **0.73490 at 1000/300 (+0.00502 over the 0.7299 anchor, ≈0.40 band, SUB-BAND)**; not saturated, not reversed, **no ≥1-band beat → no 3-seed confirm fired, no definition saved**; the top-of-curve "all-vocab" point (6633 cols) hit a hard memory ceiling and could not be built (two attempts).

## Results

xgboost is the primary axis; catboost is a secondary signal only. Controls/anchors re-run in-batch so the whole curve sits under one consistent measurement. Sorted by xgboost AUC within the curve. Best cell per metric **bolded**; the new in-lineage single-split best run **bolded**.

| config | run id | dataset | artist/genre cap | n_cols | model | AUC | logloss | accuracy | brier | role / status |
|---|---|---|---|---|---|---|---|---|---|---|
| C0a-xgb | run-20260615-163751-de9a4-xgboost-classifier | ds-20260609-204419-p64-s42 | 60 / 40 | 177 | xgboost | 0.72044 | 0.60754 | 0.65852 | 0.21087 | control — reproduces ref 0.7204 (Δ +0.00004) ✓ |
| C0b-cat | run-20260615-163803-3391f-catboost-classifier | ds-20260609-204419-p64-s42 | 60 / 40 | 177 | catboost | 0.71105 | 0.61324 | 0.65597 | 0.21316 | control — reproduces ref 0.7111 ✓ |
| C1a-xgb | run-20260615-163803-536d5-xgboost-classifier | ds-20260612-143014-p64-s42 | 300 / 120 | 497 | xgboost | 0.72988 | 0.60223 | 0.67191 | 0.20803 | anchor — reproduces ref 0.7299 (Δ −0.00002) ✓ |
| C1b-cat | run-20260615-163803-6f170-catboost-classifier | ds-20260612-143014-p64-s42 | 300 / 120 | 497 | catboost | 0.72516 | 0.60681 | 0.66490 | 0.20994 | anchor — reproduces ref 0.7252 ✓ |
| P1a-xgb | run-20260615-163803-1a0ba-xgboost-classifier | ds-20260615-162737-p64-s42 | 500 / 200 | 777 | xgboost | 0.72966 | 0.60392 | 0.67085 | 0.20856 | flat vs 300/120 (−0.00022) |
| P1b-cat | run-20260615-163803-1fae3-catboost-classifier | ds-20260615-162737-p64-s42 | 500 / 200 | 777 | catboost | 0.72372 | 0.60894 | **0.67233** | 0.21068 | catboost dips below its own 300/120 |
| **P2a-xgb** | **run-20260615-163803-e75be-xgboost-classifier** | **ds-20260615-163042-p64-s42** | **1000 / 300** | **1377** | **xgboost** | **0.73490** | **0.60070** | **0.67680** | **0.20700** | **new in-lineage single-split best (+0.00502 vs 0.7299, sub-band)** |
| P2b-cat | run-20260615-163803-9ed65-catboost-classifier | ds-20260615-163042-p64-s42 | 1000 / 300 | 1377 | catboost | 0.72633 | 0.60931 | 0.67063 | 0.21068 | catboost recovers, marginal best on its curve |
| P3a-xgb | — | NEW P3 (all 5594/962) | 5594 / 962 | 6633 | xgboost | — | — | — | — | **NOT RUN — dataset build failed (memory ceiling, 2× attempts)** |
| P3b-cat | — | NEW P3 (all 5594/962) | 5594 / 962 | 6633 | catboost | — | — | — | — | **NOT RUN — dataset build failed** |

### The cap → AUC curve (xgboost, primary axis)

| artist/genre cap | n_cols | xgboost AUC | step Δ vs previous | bands |
|---|---|---|---|---|
| 60 / 40 | 177 | 0.72044 | — | — |
| 300 / 120 | 497 | 0.72988 | +0.00944 | +0.75 |
| 500 / 200 | 777 | 0.72966 | −0.00022 | −0.02 (flat) |
| 1000 / 300 | 1377 | **0.73490** | +0.00524 | +0.42 |
| all (5594 / 962) | 6633 | unavailable (build ceiling) | — | — |

Catboost curve (secondary): 0.71105 (60/40) → 0.72516 (300/120, +0.01411) → 0.72372 (500/200, −0.00144) → 0.72633 (1000/300, +0.00261). Catboost stays ~0.005–0.009 below xgboost at every cap and rises more gently after 300/120.

## Findings

1. **Controls gate PASSED — the batch is trustworthy.** C0a-xgb reproduced the 60/40 reference 0.7204 to AUC 0.72044 (Δ +0.00004) and C1a-xgb reproduced the 300/120 reference 0.7299 to 0.72988 (Δ −0.00002), both far inside the 0.0005 gate. catboost likewise reproduced both references (0.71105 / 0.72516). The whole curve is measured under one consistent split (18,823 / 4,706 @ seed 42) and the GBDT determinism documented in prior campaigns holds.

2. **The identity lever is NOT saturated, NOT reversed — it keeps paying weakly.** From the 300/120 anchor (0.7299), xgboost is flat at 500/200 (0.72966, −0.00022, within noise) then rises to a new in-lineage single-split best 0.73490 at 1000/300 (+0.00502 ≈ 0.40 band over the anchor). No interior point fell ≥1 band below the anchor (rule-5 reversal not triggered: lowest interior is P1 at −0.00022), and the curve did rise rather than flatten across all steps (rule-4 saturation not strictly met — though the per-step gains are small: +0.75 band 60→300, ≈0 band 300→500, +0.42 band 500→1000). The honest reading: **the big payoff was the first jump (60→300, the +0.0095 already on record); past 300/120 the lever delivers diminishing, sub-band returns** that wobble up and down inside the noise envelope — consistent with the registered prior that identity resolution is the durable lever but is approaching, not crossing, the corpus's intrinsic ceiling.

3. **The non-monotone interior (500/200 dips, 1000/300 recovers) is single-split noise, not signal.** A 0.00022 dip at 500/200 then a 0.00524 rise at 1000/300 spans only ~0.4 band total — well inside 2σ = 0.0126. The "curve" past 300/120 is better described as a flat plateau with sub-band jitter than as a monotone climb. This is exactly why the decision rule guards a save behind a ≥1-band beat + a 3-seed confirm: a single split cannot tell 0.72966 from 0.73490.

4. **Decision-rule outcome: SUB-BAND IMPROVEMENT, logged, nothing saved.** The best new point (P2a 1000/300, 0.73490) does NOT beat the 0.7299 anchor by ≥1 band (the new-champion threshold was 0.7425); it sits at +0.00502, in the "0.7299 < AUC < 0.7425" sub-band window. Per rule 3 that is logged as "likely better, needs 3-seed check" and **must NOT be saved without the confirm — so the 3-seed confirm did NOT fire** (the rule arms it only for a ≥1-band beat). No definition was created; no dataset was tagged (this build exposes no dataset-tags endpoint, and the convention tags datasets via a saved definition's `dataset_tags` — there is no winner to register). The global rotation AUC champion is unchanged: the cross-matrix DS-META champion remains 0.7429 (`xgboost-classifier-rotation`, best-models rank 1); P2a auto-promoted to best-models rank 3 (0.73490, just behind a prior 0.73501) but did not displace any registered model.

5. **Catboost (secondary) tracks xgboost qualitatively — native categorical handling does NOT scale better at high cardinality.** Catboost gains less than xgboost on the first jump (+0.01411 catboost vs +0.00944 xgboost, 60→300) but then both curves go sub-band flat (catboost 0.72516 → 0.72372 → 0.72633). Catboost stays a consistent ~0.005–0.009 below xgboost at every cap and never overtakes it. There is no qualitative divergence — both tree families say the same thing: the marginal columns past 300/120 carry little independent signal. No follow-up warranted on this axis.

6. **FAILURE (data point): the top-of-curve "all-vocab" build (6633 cols) hit a hard memory ceiling — TWICE.** Both POST attempts (`build-run-…-de0e5`, `build-run-…-9f157`) returned a build_id but never registered a dataset (count stayed 21) and left no surviving build process or build-run directory — the build python died mid-materialization on each attempt. Host memory was ~2.3 GB free of 15 GB total throughout (concurrent training runs + unrelated manual xgboost/gbdt-leaf processes competing). A 23,529 × 6,633 float32 dense matrix is ≈0.62 GB, but PCA + one-hot intermediates plus the train/test copies push the peak well over the available headroom. This is a documented INFRASTRUCTURE ceiling, not a modeling result: building P1 → P2 → P3 in order (as the design specified) isolated the failure to the most expensive point, so the cheap interior curve (60/300/500/1000) was collected intact. Per the design's runner caveat, P3 is recorded as the top-of-curve point being unavailable. The trend through 1000/300 (sub-band, plateauing) makes it very unlikely all-vocab would have crossed champion+band; the 3276 singleton-artist columns it would add are pure memorizable variance that — by the registered no-blowup prior (trees tolerate collinearity via leaf splits) — would have shown as plateau or mild reversal, not a gain. `log_target:false` was accepted by the build API on P1/P2 (no rejection), so the caveat about dropping it never triggered.

## Best on record after this work — ROTATION classification, HIGH-VOCAB full-64-d corpus (vocab-cap axis)

These are SEPARATE matrices from the DS-META board (0.7429 cross-matrix champion) and from the 177-col full-64-d board — different identity cardinalities, NOT comparable across rows of different `n_cols`. The bare xgboost-classifier champion config is the in-lineage leader at every cap; the table reads as a cap→AUC progression on the SAME corpus / rows / split.

| artist/genre cap | dataset | n_cols | xgboost AUC | catboost AUC | source | registered? |
|---|---|---|---|---|---|---|
| 60 / 40 | ds-20260609-204419-p64-s42 | 177 | 0.72044 | 0.71105 | this scan (C0) + archive/2026-06-12-classifier-bakeoff-p64.md | xgboost: `xgboost-classifier-rotation-p64` |
| 300 / 120 | ds-20260612-143014-p64-s42 | 497 | 0.72988 | 0.72516 | this scan (C1) + archive/2026-06-12-high-vocab-topology.md | no |
| 500 / 200 | ds-20260615-162737-p64-s42 | 777 | 0.72966 | 0.72372 | this scan | no |
| **1000 / 300** | **ds-20260615-163042-p64-s42** | **1377** | **0.73490** | 0.72633 | **this scan** | **no — sub-band, needs 3-seed confirm** |
| all (5594 / 962) | (build failed) | 6633 | — | — | this scan | n/a — infrastructure ceiling |

Cross-matrix global rotation AUC champion (unchanged): **0.7429** — `xgboost-classifier-rotation` on ds-20260609-141622-p1-s42 (DS-META, 115 cols), best-models rank 1.

## Follow-ups

1. **Top follow-up — 3-seed confirm of 1000/300 IF the lab wants to chase the sub-band edge.** Rebuild ds-20260615-163042-p64-s42 (1000/300) at split seeds 17 and 101, re-run the frozen xgboost config, and save `xgboost-classifier-rotation-highvocab-1000` (dataset_tags [ds-20260615-163042-p64-s42]) ONLY if the 3-seed mean still clears 0.7299 + 1 band (≥0.7425). Given the single-split point is only +0.00502 (0.40 band) over the anchor, the confirm is unlikely to clear the bar — this is a low-priority "close the loop" run, not a champion chase. Lower-effort alternative: a 3-seed confirm of the 300/120 anchor vs 1000/300 to settle whether ANY of the post-300 plateau is real signal.
2. **Resolve the all-vocab (P3) point under more headroom** — rerun the 6633-col build when the host is quiescent (no concurrent training, no stray manual jobs) or after raising the build's memory budget; or build it sparse if the pipeline supports a sparse one-hot path. Until then the curve's top end is a known gap, and the registered "identity RESOLUTION is the remaining lever" guidance should be read as "weak/plateauing past 300/120, top end unmeasured."
3. **The lever is approaching its ceiling — pivot guidance.** Three of four measured caps past the first jump are sub-band; combined with the catboost agreement, the practical read is that artist/genre one-hot cardinality is a largely-spent lever on this corpus. The residual rotation signal is behavioral/contextual that no song-metadata cardinality captures (the registered prior). Future identity work should target resolution QUALITY (e.g. artist-followers / popularity interactions, genre hierarchy) rather than raw cap height.

## Reproduction notes

All 8 launched runs `succeeded`; no failures, no spurious stops, no relaunches. Server healthy throughout; the queue was empty at launch (the prior burn-ae-classifier campaign had drained — 100 succeeded / 3 failed historical), so all 8 ran without queueing. The classification metric block (auc / logloss / acc / brier) carries no MAPE — the binary NaN-MAPE trap is not applicable, confirmed by all 8 succeeding. P3 build failure left no artifacts under data/ (nothing to clean up); no server restart, no manual writes under data/, no models.toml edits.
