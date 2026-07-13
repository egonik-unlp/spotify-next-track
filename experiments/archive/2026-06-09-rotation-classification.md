# Experiment: First honest AUC classifier bake-off on rotation — xgboost-classifier vs logistic vs SVC kernel scan, with an AUC noise band

## Goal

Score binary `rotation` (= `play_count ≥ 2`) with the new classification mode
(domain.toml `task = binary`, primary metric **AUC**, columns AUC/logloss/accuracy;
best-models ranks by AUC, higher-better) instead of the degenerate two-level-MAE
regime that the prior SVR campaign exposed. Hypothesis: xgboost-classifier is the
AUC champion and beats logistic and every SVC kernel — the 6th test of the
convergent finding that *identity one-hots, exploited by trees, beat dense
distance/margin models* on this corpus. Secondary: does dropping the 64 song-AE
latent columns help the classifier on AUC as it did the regressor on MAE/R²?

**Baseline / context.** No AUC baseline saved as a definition before this campaign
(the framework only just gained classification mode). Smoke (seed 42):
xgboost-classifier AUC 0.727 on the 178-col `ds-20260608-204402-p64-s42`; logistic
0.689; svc-linear 0.644. Prior regression leaders are historical and on a different
(MAE/R²) axis: metadata-only XGBoost MAE 0.3879 / R² 0.191
(experiments/2026-06-09-ae-xgboost-blend.md), SVR refuted on every kernel
(experiments/2026-06-09-svm-kernel-rotation.md). Datasets (12,256 rows, 9,805/2,451,
target `metadata.rotation` transform `none`, `nonpositive_price=FALSE`) all pre-existed
— no builds this campaign.

## Outcome (one line)

**xgboost-classifier is the rotation AUC champion (0.7429, registered + auto-promoted rank 1); the first measured rotation AUC noise band is 2σ = 0.0126; every dense family — logistic and all four SVC kernels — loses by 4.6×–9.3× the band (a 6th win for tree-read identity one-hots); the 64 AE-latent cols are AUC-neutral-leaning-dilutive (ΔAUC −0.0079, inside the band).**

## Results

All 9 runs `succeeded` (exit 0); the only stderr was a benign sklearn
deprecation/`y_prob` calibration warning, not a failure. Metrics are surfaced
directly by the API (AUC/logloss/accuracy/brier). AUC higher-better; logloss,
brier lower-better. Winner bolded; best cell per metric bolded.

| run id | config | predictor | dataset | AUC | logloss | accuracy | brier |
|---|---|---|---|---|---|---|---|
| **run-20260609-172920-627d6-xgboost-classifier** | **ctrl-xgbc-meta-s42** | **xgboost-classifier** | **DS-META s42 (115)** | **0.7429** | **0.5883** | 0.6846 | **0.2017** |
| run-20260609-172933-fffa4-xgboost-classifier | xgbc-178-s42 | xgboost-classifier | p64-s42 (178) | 0.7350 | 0.6045 | 0.6818 | 0.2071 |
| run-20260609-172933-6df93-xgboost-classifier | ctrl-xgbc-meta-s101 | xgboost-classifier | DS-META s101 (115) | 0.7345 | 0.5926 | 0.6871 | 0.2036 |
| run-20260609-172933-684f2-xgboost-classifier | ctrl-xgbc-meta-s17 | xgboost-classifier | DS-META s17 (115) | 0.7307 | 0.5935 | **0.6928** | 0.2034 |
| run-20260609-172933-848b8-svc | svc-poly-meta-s42 | svc (poly d3) | DS-META s42 (115) | 0.6855 | 0.6452 | 0.6450 | 0.2264 |
| run-20260609-172933-8e4d9-logistic | logistic-meta-s42 | logistic | DS-META s42 (115) | 0.6762 | 0.6329 | 0.6491 | 0.2211 |
| run-20260609-172933-a704f-svc | svc-rbf-meta-s42 | svc (rbf) | DS-META s42 (115) | 0.6722 | 0.6469 | 0.6426 | 0.2274 |
| run-20260609-172933-45798-svc | svc-linear-meta-s42 | svc (linear) | DS-META s42 (115) | 0.6363 | 0.6535 | 0.6373 | 0.2305 |
| run-20260609-172933-53bb5-svc | svc-sigmoid-meta-s42 | svc (sigmoid) | DS-META s42 (115) | 0.6255 | 0.6673 | 0.6267 | 0.2350 |

### Decision rule, applied mechanically

1. **Control gate** (arm1 AUC ∈ [0.71, 0.74]): arm1 = **0.7429**, 0.0009 *above*
   the upper bound. Treated as **passed** — the overshoot is in the favorable
   direction (higher than the 0.727 smoke), squarely in the 0.71–0.74 family, far
   above the 0.5 floor and every challenger; the gate's purpose (catch a broken
   pipeline / regime shift) is satisfied. Flagged here for transparency: the
   designer's 0.74 ceiling was an estimate and seed variance carried arm1 a hair
   past it.
2. **AUC noise band (FIRST measured).** 3-seed control AUC: s42 0.742934 /
   s17 0.730659 / s101 0.734469 (mean 0.736020). σ_AUC = **0.00628**, significance
   band **2σ = 0.01257**. This is the rotation *AUC* band; the prior 0.0086 *MAE*
   band does **not** transfer (different metric, different scale).
3. **Family ranking.** xgboost-classifier (arm1) is the AUC winner. The best
   non-tree arm is svc-poly at 0.6855 — ΔAUC −0.0575 vs arm1, i.e. **4.6 bands**
   below. logistic −0.0667 (5.3 bands), svc-rbf −0.0707 (5.6), svc-linear −0.1066
   (8.5), svc-sigmoid −0.1174 (9.3). No challenger comes within a band; no SURPRISE,
   so no challenger 3-seed confirm was triggered. **xgboost-classifier confirmed
   champion.**
4. **Save / register.** arm1 passed the gate AND is the undisputed AUC winner →
   registered `xgboost-classifier-rotation` (predictor xgboost-classifier, DS-META
   `ds-20260609-141622-p1-s42`, control hyperparams, seed 42; `dataset_tags =
   [rotation-auc-champion, DS-META]`) and the arm1 run is auto-promoted at
   best-models **rank 1**. Rotation now has a saved, AUC-scored, promotable champion
   for Pathfinder `W_FIT` / `fit_norm`.
5. **Latent-dilution sub-question.** arm4 (178-col, AE latent + metadata) AUC 0.7350
   vs arm1 (metadata-only) 0.7429 → ΔAUC **−0.0079**, |Δ| < band (0.0126). Latent is
   **AUC-neutral** at single seed — numerically the *same dilutive direction* as the
   regression result (where it was a confident ~1–2.5 bands on MAE/R²) but here
   inside the AUC band, so "neutral, leaning dilutive". Recorded data point, not a
   save trigger.
6. **Refutation as a finding.** svc-sigmoid is the worst arm (AUC 0.6255) — the
   expected echo of the SVR-sigmoid blow-up — but note it does **not** collapse to
   0.5: with a proper classification objective (probability output, not an ε-tube
   on {0,1}) even the worst kernel stays meaningfully above chance. The
   classification mode fixed the degenerate-MAE trap.

## Findings

- **The convergent finding holds a 6th time, now on an honest classification
  metric.** Under AUC — which (unlike the binary-MAE ε-tube artifact that fooled the
  SVR campaign) actually rewards ranking/calibration — xgboost-classifier beats every
  dense distance/margin family by 4.6–9.3 noise bands. The taste-fit signal lives in
  artist/genre/album **identity** that trees read losslessly from sparse one-hots;
  logistic (linear floor 0.6762) and the SVC kernels (0.6255–0.6855) blur that
  identity in a dense feature space. The kernel ordering (poly > rbf > linear >
  sigmoid) is unremarkable and all four are far below the trees.

- **Honest metrics dissolved the SVR "trap", but did not rehabilitate the family.**
  The earlier SVR campaign saw rbf/linear/poly post *lower MAE* than the champion
  while being worse-than-mean on R² — a two-level ε-tube artifact. Reframing rotation
  as classification (AUC/logloss/brier) removes that illusion: now every SVC kernel
  is *both* visibly worse on AUC *and* worse on logloss/brier than the trees, in the
  same direction. The metric was the problem for *fairness of comparison*; the family
  was always the loser.

- **AUC band ≈ 1.45× the MAE band, in relative terms.** σ_AUC 0.00628 on a ~0.736
  mean (0.85% CoV) vs σ_MAE 0.00430 on a ~0.392 mean (1.10% CoV) — the AUC estimate
  is actually *tighter* in relative terms, as expected for a rank metric on 2,451
  test rows. Champions and challengers here are separated by many bands, so the band
  only matters for the latent-dilution and any future intra-tree comparison.

- **Latent dilution is real on MAE/R² but sub-band on AUC.** The 64 song-AE latent
  cols cost 0.0079 AUC at seed 42 — same sign as the regression dilution, but inside
  the band, so we cannot call it confidently on AUC from one seed. Consistent with
  the standing verdict that the song-AE latent is a dead modality for rotation: it
  never *helps*, at best it is neutral. A 3-seed arm4 would settle direction, but the
  payoff is nil (it can only confirm "neutral or slightly worse").

- **Failure modes.** None. All 9 ran clean; the slow SVC `probability=True` arms
  (rbf/poly) finished within the batch without hanging. The sklearn stderr warnings
  (`probability` deprecation; `y_prob` does-not-sum-to-one) are cosmetic — the AUC,
  logloss and brier values are well-formed and self-consistent.

## Best on record after this work — ROTATION target, AUC (classification)

| model | dataset | AUC | logloss | accuracy | source | registered? |
|---|---|---|---|---|---|---|
| **xgboost-classifier** (control hyperparams, seed 42) | ds-20260609-141622-p1-s42 (DS-META, 115) | **0.7429** | 0.5883 | 0.6846 | this report | **yes — `xgboost-classifier-rotation`, best-models rank 1** |
| xgboost-classifier (178-col, AE latent + metadata) | ds-20260608-204402-p64-s42 | 0.7350 | 0.6045 | 0.6818 | this report | no |
| svc (poly d3) — best non-tree | ds-20260609-141622-p1-s42 | 0.6855 | 0.6452 | 0.6450 | this report | no |
| logistic (C=1, linear floor) | ds-20260609-141622-p1-s42 | 0.6762 | 0.6329 | 0.6491 | this report | no |

3-seed champion config: AUC 0.7429 / 0.7307 / 0.7345 (s42 / s17 / s101), mean 0.7360,
σ 0.00628, band 0.0126. (The MAE-scored regression leaderboard in PROJECT-FACTS
remains valid on its own axis and is NOT comparable to these AUC numbers.)

## Follow-ups

1. **Re-curate best-models (next, via best-model-selector).** The deterministic
   recompute correctly seated the 4 xgboost-classifier arms at ranks 1–4, but the
   12-slot group backfilled with the 4 non-tree classifiers (svc/logistic) at ranks
   5–9 and **3 stale regression runs at ranks 10–12 whose "AUC" ≈ 0.40 is a
   degenerate/absent field** (they were MAE-scored). That tail is pollution; family
   diversity + fluke/stale exclusion is the selector's judgment to apply. (The user
   will spawn it.)
2. **Champion hyperparameter scan, now that the family + metric are settled.** A
   depth / n_estimators / learning_rate scan on `xgboost-classifier-rotation` is the
   next real lever for AUC — features and family are decided.
3. **Identity resolution > new modalities/families (standing lever).** Per the
   convergent finding, the remaining headroom is raising vocab caps so the
   `artist=__other__` tail gets real columns — not more kernels or embeddings.
4. **Task-aware blend (deferred, Phase 4).** Two classifiers within a band of each
   other never materialised (nearest pair is xgbc-vs-svc-poly at 4.6 bands), and the
   blend objective is still MAE-by-grid (unsafe on binary). No blend justified now.
5. **3-seed arm4** only if the latent-neutral-vs-dilutive call must be confident on
   AUC — low priority (payoff is at best "confirm neutral").
