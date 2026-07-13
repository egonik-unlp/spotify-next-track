# PROJECT FACTS — Spotify Engagement from Metadata

Living, agent-maintained roll-up of empirical knowledge: the leaderboard,
noise bands, dataset lineage, per-family field guide and hard-won pitfalls.
The campaign reports in experiments/*.md are PRIMARY; this file is their
index. When they disagree, the newest report wins — and this file is stale
and must be updated. Writers: the experiment-runner agent and the
model-definitions experiment workflow reconcile this file as part of every
campaign report.

Last updated: 2026-07-12 — after experiments/2026-07-12-taste-drift-temporal-split.md, a TEMPORAL-GENERALIZATION / TASTE-DRIFT study that also ADDED a first-class chronological-split build option (`split_order_field`). VERDICT: **taste is NOT stationary — every on-record AUC is a random split and OVERSTATES forward prediction by ~0.11–0.12.** On framework-built `spotify_tracks_song_ae` datasets, the champion xgboost scores 0.73490 random vs **0.61894 chronological** (train oldest-80% by `first_played`, leak-free early-train fit) — a 0.116 drop ≈ 9 bands, UNIVERSAL across model families (catboost/logistic identical ~0.11), robust to a ≥180-day censoring control and walk-forward (0.59–0.67 every year). Identity one-hots barely help forward (intrinsic-only chrono 0.599, −0.02); time-aware recency-weighting is a weak lever (+0.003 sub-band). Mechanism: rotation is identity-driven, future artists are unseen one-hots → falls to the intrinsic ~0.60 floor (matching the SAE's ~0.14 decodability read). Read the leaderboard as same-era ranking, not forward skill. New capability + a wrong-collection pitfall (must build rotation from `spotify_tracks_song_ae`, not the default `spotify_tracks`) recorded under "Noise bands & significance thresholds". Framework champion run on the chrono dataset reproduced 0.61894 to the digit; off-framework cross-check agreed (0.626). Prior update — after experiments/2026-07-12-high-vocab-capacity-scan.md, a CAPACITY SCAN of burn-deep-classifier ANNs on the HIGH-VOCAB rotation dataset `ds-20260612-143014-p64-s42` (497 cols, 23,529 rows, 18,823/4,706 @ seed 42). 6 runs = two topology ladders (embeddings ed8/lr0.0005 [PRIMARY]; mlp lr0.001 [SECONDARY]) × geometry {[128,64] control / [256,128] wider / [256,128,64] deeper}; embed_dim held 8, epochs 250 / patience 20. VERDICT: **CEILING — capacity is saturated for BOTH topologies at high vocab.** Neither widening nor deepening beats either sub-family's [128,64] control by >1 band; the ONLY out-of-band move is deepening EMBEDDINGS to 3 layers, which HURTS (B3 [256,128,64] 0.70750, −0.0152 ≈ 1.2 bands vs the emb control). Results by AUC: emb [128,64] **0.72274** (B1, WINNER — reproduces the on-record emb best 0.7228 to 4 digits, gate PASS) > mlp [256,128,64] 0.71800 ≈ emb [256,128] 0.71798 > mlp [256,128] 0.71576 > mlp [128,64] 0.71229 (B4 — reproduces on-record mlp 0.7129, gate PASS) > emb [256,128,64] 0.70750. Both reproduction gates PASSED; NO overfit (max logloss 0.617 ≪ 0.8 — patience 20 held; the deeper-net-overfit pitfall did NOT fire at the 250-epoch budget); calibration healthy (logloss 0.610–0.617 / brier 0.211–0.214, trees' regime). No arm cleared the 0.7318 save bar → NO material win, NO definition saved, incumbent `burn-deep-embeddings-highvocab` (0.7192) UNCHANGED, no 3-seed confirm fired. CROSS-FAMILY read (Step 5): best mlp 0.71800 ≥ best emb 0.72274 − 0.0126 → **capacity, not the embeddings mechanism, was the lever** — the two topologies are tied within a band at their best geometries (Δ +0.0047; control-to-control +0.0105, also sub-band), so the earlier 'embeddings wins at high vocab' is an in-band tie, not a mechanism win. Best ANN stays ~0.2 band below catboost (0.7252) and ~0.6 below xgboost (0.7299) — capacity does NOT cross the tree gap. One model PROMOTED for the record: `emb-128-64-ed8-highvocab` (run-20260712-171008-065bd, B1, carrying an auto-queued per-model SAE); displaced no registered model. All 6 runs succeeded, no relaunches. Prior update — after experiments/2026-07-12-pyramid-vs-rect-sae.md, a FRAMEWORK + representation campaign. (1) **Ported the per-model SAE (`POST /api/interp/model-sae`) into `predictor-burn-deep-clf`** — it previously existed only for `burn-mlp` (a regressor) and was undeclared in registry.toml, so NO classification MLP could be probed. New `model_sae.rs` (reconstructs arch from the promoted dir's layout/scaler/hyperparams/contract, assembles the split cont_std+one-hot input, captures trunk activations, output-reference = P(class==1)) + `forward_capture`/`capture_stage_activations` in model.rs + a `model-sae` subcommand, reusing the shared `lensing-sae`/`lensing-interp` engines unchanged; `model_sae_args` declared for `burn-deep-classifier`. `supports_model_sae` now holds for the family and **promotion auto-queues a per-model SAE**. (2) Used it to compare a PYRAMID burn-deep mlp [256,128,64] vs a UNIFORM mlp [128,128,128] on the canonical full-64-d dataset `ds-20260609-204419-p64-s42` (topology mlp, lr 5e-4, early-stop). VERDICT: **the two are AUC-TIED (0.69857 vs 0.69937, Δ 0.0008 ≪ 0.0126 band) and both sit in the on-record burn-deep mlp tier (~0.70), but the SAE shows different concept ALLOCATION** — the pyramid consolidates interpretable concepts into its 64-d bottleneck (L2→L3 concepts 72→42) while the uniform net keeps a broader set to the end (66→54); both fully utilize atoms (0 dead/0 rare at 2× width), give ~103–105/105 identity segments a dedicated separating atom, reconstruct >96% variance, and add almost NO linear decodability over the raw input (~0.111→~0.12 by depth, AUC-tied) — i.e. same signal captured, different layout. NEW pitfall: **deeper burn-deep-clf nets OVERFIT at the full 100-epoch budget with patience 0** (logloss blows to 0.96–1.22, AUC −0.015, ranking survives / calibration destroyed) — early stopping (patience 10) restores the calibrated 0.699/0.625 regime; a budget artifact, not a ceiling. Two models PROMOTED (`pyramid-mlp-256-128-64`, `rect-mlp-128-128-128`, both carrying auto-queued SAE analyses); NO definition saved (sub-champion, tied tier, below the ~0.71 bar); champion unchanged (xgboost 0.7204 this-corpus / 0.7429 DS-META). Note: `GET /api/interp/models` still lists empty for this family — that list gates on `probe_args` (layer-probe, burn-mlp-only), separate from `model_sae_args`; the SAE works regardless via the auto-queue / direct endpoint. Prior update — after experiments/2026-06-15-vocab-cap-scan-gbdt.md, an identity-resolution VOCAB-CAP SCAN on the bare GBDT (xgboost-classifier champion config + catboost defaults) mapping the artist/genre one-hot cardinality → AUC curve on the HIGH-VOCAB rotation lineage. 8 runs = 5 caps × 2 trees (P3 = 2 runs NOT RUN). Tests the registered "remaining lever is identity RESOLUTION — raise vocab caps" hypothesis. VERDICT: **the lever keeps paying but only WEAKLY past 300/120, and is approaching its ceiling.** xgboost curve: 60/40 0.72044 → 300/120 0.72988 (+0.00944, +0.75 band — the big jump, already on record) → 500/200 0.72966 (−0.00022, FLAT) → 1000/300 **0.73490** (+0.00524 vs 500, +0.00502 vs the 300/120 anchor ≈ 0.40 band — a new in-lineage SINGLE-SPLIT best but SUB-BAND). NOT saturated by the strict rule (curve rose), NOT reversed (no interior point ≥1 band below anchor), but three of four post-jump caps are sub-band → effectively a plateau with noise jitter. No ≥1-band beat (champion+band threshold was 0.7425) → the 3-seed confirm did NOT fire, **no definition saved**, no dataset tagged (this build exposes no dataset-tags endpoint; tagging rides a saved definition and there is no winner). Global rotation AUC champion unchanged (DS-META 0.7429, `xgboost-classifier-rotation`, rank 1); P2a 0.73490 auto-promoted to best-models rank 3 but displaced nothing registered. Catboost (secondary) tracks xgboost qualitatively — native categorical handling does NOT scale better at high cardinality (0.71105 → 0.72516 → 0.72372 → 0.72633, always ~0.005–0.009 below xgboost, same sub-band plateau). CONTROLS reproduced both anchors to all digits (C0a 0.72044 vs 0.7204 Δ +0.00004; C1a 0.72988 vs 0.7299 Δ −0.00002) — batch trustworthy. FAILURE/data-point: the top-of-curve "all-vocab" build (5594/962, 6633 cols) hit a HARD MEMORY CEILING — TWO POST attempts each returned a build_id but never materialized a dataset (host ~2.3 GB free of 15 GB with concurrent training); a documented INFRASTRUCTURE ceiling, not a modeling result — building P1→P2→P3 in order isolated it and the cheap interior curve was collected intact. NEW datasets: P1 ds-20260615-162737-p64-s42 (500/200, 777 cols) and P2 ds-20260615-163042-p64-s42 (1000/300, 1377 cols), both 23,529 rows / 0 exclusions, byte-identical to the 300/120 anchor except vocab_top_n. All 8 runs succeeded (classification metric block, no MAPE trap). Prior update — after experiments/2026-06-15-burn-ae-classifier-capacity-scan.md, the first characterization of the NEW `burn-ae-classifier` predictor (two-phase Burn: MSE-autoencoder pretrain → transplant encoder into a softmax classifier, fine-tuned with cross-entropy) on the canonical full-64-d rotation dataset `ds-20260609-204419-p64-s42`. 13 ad-hoc runs scanning latent dim, classifier-head depth/width, freeze vs fine-tune, and activation. VERDICT: **capacity is fully saturated / the gap did NOT close** — the 12 fine-tuned configs span just 0.69175–0.69719 (0.0054, entirely INSIDE the 0.0126 band): a flat tie across latent size (32/64/128 indistinguishable), head depth/width (linear probe 0.69175 only −0.0054 below the deepest head), and activation (relu > gelu > silu, all sub-band — smooth activations slightly HURT). Best = cfg3 enc[128,64]/clf[128,64]/fine-tune/relu **0.69719**, +0.00494 over the 0.69225 baseline (sub-band, NOT a material beat), ~1.8 bands below the xgboost champion (0.7204) and statistically EQUAL to the plain burn-deep mlp (0.7038, −0.0066 sub-band) — the unsupervised AE pretrain buys a stable init but NO AUC over a from-scratch raw-one-hot mlp. The ONE real effect: FREEZING the encoder under a DEEP head COLLAPSES the model (cfg10 0.68210, −0.0151 ≈ 1.2 bands, the only out-of-band row) while a frozen SHALLOW head stays in-band (cfg11 0.69530) — freeze_encoder=true is a documented trap with deep heads. Calibration healthy (logloss ~0.62 / brier ~0.22, the trees' regime). NO definition saved; family closed for AUC as configured (next lever is the pretraining OBJECTIVE, not size). Determinism: near- not bit- (the baseline config re-run scored 0.69645 vs 0.69225, +0.0042 ≈ 0.33 band). Prior update — after experiments/2026-06-15-gbdt-leaf-head-ceiling-probe.md, a GBDT-leaf-embedding→head ceiling-probe on the canonical full-64-d rotation dataset `ds-20260609-204419-p64-s42`. The Facebook GBDT→LR trick (a tree hard-frozen at the champion config emits leaf indices, one-hot-encoded, with a linear or shallow-MLP head trained on top) vs the bare xgboost-classifier champion (0.7204). VERDICT: **REFUTED as predicted** — all three heads cap BELOW the bare tree by 1.8–2.5 bands (linear 0.6978, mlp[256,128] 0.6921, mlp[128] 0.6897; refutation threshold 0.7078) AND are badly mis-calibrated (logloss 0.82–1.31 / brier 0.26–0.30 vs the tree's 0.61 / 0.21). The control reproduced the champion to all 16 digits (0.7204383403100314). No T4 fired (best treatment below the in-band branch), no definition saved. A FIFTH on-record confirmation that learning a head over a learned tree representation dilutes — the dense head cannot even recover the tree that generated its features; the leaf-head family is closed for rotation. Prior update — after experiments/archive/2026-06-12-high-vocab-topology.md, the HIGH-VOCAB paired re-run of the burn-deep-classifier topology bake-off on `ds-20260612-143014-p64-s42` (artist cap 300 / genre 120; identity cardinality the only axis vs the canonical full-64-d dataset). VERDICT: the "embeddings starved at low cardinality" hypothesis from experiments/archive/2026-06-12-burn-deep-topology.md is **RESOLVED-CONFIRMED (directionally)** — the topology ranking REVERSED: `embeddings` (0.7192) now beats `mlp` (0.7129, +0.0063) and `wide_deep` (0.7061), undoing the old +0.0067 mlp-over-embeddings margin (a +0.0130 swing across the band). Both trees ALSO moved up on the wider matrix (xgboost 0.7204→**0.7299**, catboost 0.7111→**0.7252**) — high vocab helps every family; the in-dataset tree tier is 0.7252–0.7299. The best embeddings net reaches the BOTTOM of that tree tier (Phase-2 ed8/lr0.0005 0.7228, −0.0024 vs catboost, in-band) and the saved Phase-1-winner config (embeddings, ed16, lr0.001, 0.7192) is the FIRST ANN to clear the save bar on rotation: registered as `burn-deep-embeddings-highvocab` (dataset_tags [ds-20260612-143014-p64-s42]), flagged for best-model-selector. embed_dim is already saturated at 8 (ed8≥ed16≥ed32, sub-band); the gain came from enabling embeddings + lr 0.0005, not from width. Determinism confirmed (ed16/lr0.001 reproduced to 4 digits). Prior update — built `ds-20260612-143014-p64-s42`, a HIGH-VOCAB variant of the canonical full-64-d rotation dataset (artist one-hot cap 60->300, genre_primary 40->120; everything else byte-identical) to feed this bake-off. Prior update after experiments/archive/2026-06-12-burn-deep-topology.md —
a burn-deep-classifier (NEW Rust/burn predictor) topology bake-off {mlp, embeddings,
wide_deep} + a 6-cell mlp tune on the full-64-d rotation dataset. Verdict: the full-rust
net with RAW (un-standardized) one-hots **FIXES the sklearn pyramid's calibration blowup
in full** (logloss 2.0–2.4 → **~0.62**, brier 0.33 → ~0.22, the trees' regime) and lifts
AUC to **0.7038** (best, mlp/lr0.0005/[128,64]) — ~2 bands over the converged sklearn
pyramid (0.6792). Plain `mlp` over raw one-hots wins Phase 1 (0.7001) over embeddings
(0.6934 ≈ wide_deep 0.6918 — both within one band). Best 0.7038 is INSIDE the band of the
Phase-1 winner and **below the ~0.71 save bar** (−0.0073 vs catboost, −0.0166 vs xgboost):
NO definition saved. This REVISES the prior "MLP family closed / calibration a fixed
ceiling" verdict — that was a sklearn-representation artifact; a raw-one-hot rust net
reaches ~0.70 and the tree gap shrinks from ~3.3 to ~1 band.

Prior campaign — experiments/archive/2026-06-12-pyramid-mlp-arch-scan.md —
a 6-arm pyramid-MLP architecture scan (max_iter raised to 800) on the full-64-d
rotation dataset. Verdict: the bake-off's "300-iter-cap under-training" framing is
**REFUTED** — every geometry stops itself at 73–100 iters (well below 300) by the
optimizer's tolerance, so budget was never binding; the family's AUC ceiling is ≈0.68
(best arch wide-shallow [512,256] = 0.6792, +0.0011 = zero over the default) and the
high logloss is a real calibration ceiling for the SKLEARN pyramid (see the
burn-deep revision above — fixable via raw one-hots). No definition saved (below the
0.683 save bar); the SKLEARN MLP family is closed for AUC, ~3.3 bands below xgboost.

Prior campaign — experiments/archive/2026-06-12-classifier-bakeoff-p64.md —
the four-family classifier bake-off on the NEW full-64-d corpus. Verdicts: (a)
on the NEW 23,529-row full-64-d dataset `ds-20260609-204419-p64-s42` (177 cols),
**xgboost-classifier wins the four-family bake-off at AUC 0.7204** (reproducing the
in-dataset reference to all digits), now REGISTERED as `xgboost-classifier-rotation-p64`.
catboost-classifier 0.7111 is co-leader IN-BAND (−0.0093 < 0.0126, at least equal,
likely a hair behind — the legitimate 2nd tree family); ngboost 0.6917 is ~2.3 bands
back; pyramid-mlp 0.6781 is weakest (the AUC is genuine; the high logloss is a
calibration ceiling, NOT the 300-iter artifact it was first read as — see the arch
scan above). (b) An xgboost depth×lr sweep found depth8/lr0.03 best
(AUC 0.7261) but only +0.0057 over the default — INSIDE the band — so the default
depth-6 config is the saved one; depth8/lr0.03 is flagged for a 3-seed confirm.
(c) **CROSS-CORPUS CAVEAT (do not mis-read):** these numbers are LOWER than the 0.7429
DS-META champion below because that champion is on the OLD 12,256-row metadata-only
corpus (`ds-20260609-141622-p1-s42`) — NOT comparable; nothing regressed. The two AUC
leaderboards (old DS-META, new full-64-d) coexist on different corpora. PRIOR verdicts
from 2026-06-09 stand on their own corpus/axis (the 0.7429 DS-META champion, the dense-family
refutations, the MAE/R² regression results) and are unchanged.

## ACTIVE TARGET: rotation (binary taste-fit), pivoted 2026-06-08

The active `[target]` is **`rotation` = (play_count ≥ 2)**, a binary [0,1]
taste-fit signal (transform `none`, units `P(rotation)`) that feeds the
Pathfinder `W_FIT` term. It is more learnable (xgboost AUC ~0.73–0.77) than
engagement magnitude (R² ~0.3) and naturally spread for `fit_norm`. Engagement
(play_count × completion_ratio, log1p) is **demoted to a documented non-target
field** — its findings below are HISTORICAL (different target, different scale,
MAE ~2.4). Build rotation datasets with `quality.nonpositive_price = FALSE`
(rotation==0 is the valid negative class, not an outlier). Full pivot narrative
+ the binary-metric framework limitation: see "Taste-fit reframe for Pathfinder
(2026-06-08)" below.

## Best on record (leaderboard) — ROTATION target

| model | dataset | MAE | RMSE | R² | source |
|---|---|---|---|---|---|
| **metadata-only xgboost** (AE-latent cols dropped; = the `mae_xgb` branch inside every blend run — not yet a saved standalone definition) | ds-20260608-204402-p64-s42 | **0.3879** | 0.4475 | **0.191** | experiments/archive/2026-06-09-ae-xgboost-blend.md |
| xgboost on latent+metadata combined | ds-20260608-204402-p64-s42 | 0.3965 | — | 0.166 | rotation pivot seed run (run-20260608-204415-6eeda-xgboost) |
| xgboost on AE-latent only (64 cols) | ds-20260608-205802-p64-s42 | 0.4098 | — | 0.132 | rotation pivot seed run |
| autoencoder-xgboost blend (kNN or ridge AE branch, w→0) | ds-20260608-204402-p64-s42 | 0.3879 (≡ metadata-only xgboost) | 0.4475 | 0.191 | experiments/archive/2026-06-09-ae-xgboost-blend.md |

Read: on rotation, **dropping the 64 AE-latent columns and running plain
metadata xgboost is the best result on record** — feeding the latent into a
unified xgboost (0.3965) or blending an AE-latent branch (collapses to w=0)
both do worse or equal. The latent DILUTES rotation prediction. All numbers are
single-split / single-seed and the 0.3879↔0.3965 edge is below an unmeasured
noise band — the multi-seed study (below) must confirm before this leader is
trusted or registered as a definition. NOTE: rotation R²/MAE are on the [0,1]
binary scale and are NOT comparable to the engagement numbers below.

## Best on record (AUC leaderboard) — ROTATION classification target

Classification mode (domain.toml `task = binary`, primary metric **AUC**,
higher-better; best-models ranks by AUC). Source: experiments/archive/2026-06-09-rotation-classification.md.
These AUC numbers are NOT comparable to the MAE/R² regression leaderboard above —
different metric, different (calibrated probability vs ε-tube) regime.

| model | dataset | AUC | logloss | accuracy | source | registered? |
|---|---|---|---|---|---|---|
| **xgboost-classifier** (control hyperparams: n_est 500, depth 6, lr 0.05, subsample/colsample 0.8, seed 42) | ds-20260609-141622-p1-s42 (DS-META, 115) | **0.7429** | 0.5883 | 0.6846 | experiments/archive/2026-06-09-rotation-classification.md | **YES — `xgboost-classifier-rotation`, best-models rank 1, dataset_tags [rotation-auc-champion, DS-META]** |
| xgboost-classifier (178-col, AE latent + metadata) | ds-20260608-204402-p64-s42 | 0.7350 | 0.6045 | 0.6818 | same | no |
| svc (poly d3) — best non-tree | ds-20260609-141622-p1-s42 | 0.6855 | 0.6452 | 0.6450 | same | no |
| logistic (C=1, linear floor) | ds-20260609-141622-p1-s42 | 0.6762 | 0.6329 | 0.6491 | same | no |
| svc (rbf) | ds-20260609-141622-p1-s42 | 0.6722 | 0.6469 | 0.6426 | same | no |
| svc (linear) | ds-20260609-141622-p1-s42 | 0.6363 | 0.6535 | 0.6373 | same | no |
| svc (sigmoid) — worst, expected refutation | ds-20260609-141622-p1-s42 | 0.6255 | 0.6673 | 0.6267 | same | no |

Read: on the honest classification metric, **xgboost-classifier is the rotation
champion (AUC 0.7429)** — now the registered, auto-promoted standalone the regression
work never produced. Every dense distance/margin family (logistic + 4 SVC kernels)
loses by 4.6–9.3 AUC bands; reframing rotation as classification dissolved the
binary-MAE ε-tube illusion that made SVR *look* competitive, and the verdict is the
same convergent finding (identity one-hots + trees win), just honest now. 3-seed
champion config: AUC 0.7429 / 0.7307 / 0.7345 (s42 / s17 / s101). Latent-dilution
sub-result: arm4 (178-col) −0.0079 AUC vs metadata-only, inside the band —
neutral-leaning-dilutive (same direction as the regression dilution, sub-band on AUC).

## Best on record (AUC leaderboard) — ROTATION classification, NEW full-64-d corpus

Classification mode, AUC higher-better. Source:
experiments/archive/2026-06-12-classifier-bakeoff-p64.md. **This is a SEPARATE corpus from the
DS-META leaderboard above** — dataset `ds-20260609-204419-p64-s42` is the 2026 corpus,
**23,529 rows** (18,823/4,706 @ seed 42), **177 cols** (64 song-AE PCA latent + 7 numeric
+ 106 identity one-hots), full-64-d. These AUC numbers are NOT comparable to the 0.7429
DS-META champion (old 12,256-row, metadata-only) — different corpus size AND feature set.
The only valid in-corpus comparison point is the pre-existing reference run
`run-20260609-204726-439d8-xgboost-classifier` (AUC 0.7204), which the bake-off reproduced
to all digits.

| model | dataset | AUC | logloss | accuracy | source | registered? |
|---|---|---|---|---|---|---|
| **xgboost-classifier** (n_est 500, depth 6, lr 0.05, sub/col 0.8, seed 42) | ds-20260609-204419-p64-s42 (177) | **0.7204** | 0.6075 | 0.6585 | this corpus' bake-off | **YES — `xgboost-classifier-rotation-p64`, dataset_tags [ds-20260609-204419-p64-s42]** |
| xgboost-classifier (depth 8, lr 0.03 — sweep best, single-seed) | ds-20260609-204419-p64-s42 | 0.7261 | 0.6044 | 0.6651 | same | no — flagged for 3-seed confirm |
| catboost-classifier (iter 500, depth 6, lr 0.05) — 2nd tree family, in-band | ds-20260609-204419-p64-s42 | 0.7111 | 0.6132 | 0.6560 | same | no |
| gbdt-leaf-head-classifier (head=linear, l2 1.0) — Facebook GBDT→LR trick, REFUTED (best leaf-head, −1.8 bands vs tree, over-confident) | ds-20260609-204419-p64-s42 | 0.6978 | 1.3119 | 0.6441 | `2026-06-15-gbdt-leaf-head-ceiling-probe.md` | no |
| ngboost-classifier (n_est 400, lr 0.03, base_depth 3) | ds-20260609-204419-p64-s42 | 0.6917 | 0.6293 | 0.6392 | same | no |
| gbdt-leaf-head-classifier (head=mlp, [256,128], head_lr 1e-3) — REFUTED, mis-calibrated | ds-20260609-204419-p64-s42 | 0.6921 | 1.1813 | 0.6396 | `2026-06-15-gbdt-leaf-head-ceiling-probe.md` | no |
| gbdt-leaf-head-classifier (head=mlp, [128], head_lr 1e-3) — REFUTED, worst leaf-head | ds-20260609-204419-p64-s42 | 0.6897 | 0.8162 | 0.6356 | `2026-06-15-gbdt-leaf-head-ceiling-probe.md` | no |
| pyramid-mlp-classifier (512/2/0.5 wide-shallow, max_iter 800 — best converged MLP arch) | ds-20260609-204419-p64-s42 | 0.6792 | 2.0584 | 0.6326 | `2026-06-12-pyramid-mlp-arch-scan.md` | no |
| pyramid-mlp-classifier (256/3, max_iter 300 — first read; AUC genuine, logloss = calibration ceiling) | ds-20260609-204419-p64-s42 | 0.6781 | 2.4085 | 0.6326 | bake-off | no |
| **burn-deep-classifier** (Rust/burn; topology mlp, lr 0.0005, hidden [128,64], raw one-hots) — best ANN, calibration FIXED | ds-20260609-204419-p64-s42 | 0.7038 | 0.6221 | 0.6521 | `2026-06-12-burn-deep-topology.md` | no (below ~0.71 save bar) |
| burn-deep-classifier (mlp, UNIFORM [128,128,128], lr5e-4, early-stop) — SAE rep-study, broader concept set to the end (66→54) | ds-20260609-204419-p64-s42 | 0.69937 | 0.6244 | 0.6490 | `2026-07-12-pyramid-vs-rect-sae.md` | promoted `rect-mlp-128-128-128` (auto SAE analysis); not a definition |
| burn-deep-classifier (mlp, PYRAMID [256,128,64], lr5e-4, early-stop) — SAE rep-study, concepts consolidate into 64-d neck (72→42) | ds-20260609-204419-p64-s42 | 0.69857 | 0.6254 | 0.6405 | `2026-07-12-pyramid-vs-rect-sae.md` | promoted `pyramid-mlp-256-128-64` (auto SAE analysis); not a definition |
| burn-deep-classifier (topology mlp, lr 0.001, [128,64] — Phase-1 winner config) | ds-20260609-204419-p64-s42 | 0.7001 | 0.6211 | 0.6470 | `2026-06-12-burn-deep-topology.md` | no |
| burn-deep-classifier (topology embeddings, defaults) | ds-20260609-204419-p64-s42 | 0.6934 | 0.6251 | 0.6436 | `2026-06-12-burn-deep-topology.md` | no |
| burn-deep-classifier (topology wide_deep, defaults) | ds-20260609-204419-p64-s42 | 0.6918 | 0.6281 | 0.6407 | `2026-06-12-burn-deep-topology.md` | no |
| **burn-ae-classifier** (AE-transplant, enc[128,64]/clf[128,64], fine-tune, relu) — best AE-transplant config | ds-20260609-204419-p64-s42 | 0.69719 | 0.6216 | 0.6496 | `2026-06-15-burn-ae-classifier-capacity-scan.md` | no (sub-band over baseline; ~1.8 bands below champion) |
| burn-ae-classifier (enc[128,64]/clf[64], fine-tune — baseline config) | ds-20260609-204419-p64-s42 | 0.69225 | — | — | `2026-06-15-burn-ae-classifier-capacity-scan.md` | no |
| burn-ae-classifier (enc[128,64]/clf[128,64], FROZEN encoder) — DEEP-HEAD-FROZEN COLLAPSE, worst row (−1.2 bands) | ds-20260609-204419-p64-s42 | 0.68210 | 0.6378 | 0.6373 | `2026-06-15-burn-ae-classifier-capacity-scan.md` | no |

Read: xgboost-classifier wins this corpus' four-family bake-off (AUC 0.7204);
catboost is co-leader IN-BAND (−0.0093 < 0.0126) and is the legitimate 2nd tree
family; ngboost (~2.3 bands back) under-fits with shallow base learners; pyramid-mlp
is weakest — the 6-arm arch scan (max_iter 800) confirmed AUC ceiling ≈0.68 (best
0.6792, +0.0011 over the default) and that the high logloss is a calibration ceiling
(every arm self-stops at 73–100 iters, NOT the 300 cap — budget was never binding).
The depth×lr sweep is monotone in the sensible direction (depth 8 + slow lr best, lr 0.1
hurts) but the best-vs-default gap (+0.0057) is sub-band, so the default config is saved
and depth8/lr0.03 awaits a 3-seed confirm. **burn-deep-classifier** (NEW Rust/burn ANN, raw one-hots) lands AUC 0.7038 (best,
mlp) — between the tree tier and ngboost; its logloss ~0.62 / brier ~0.22 FIX the sklearn
pyramid's calibration blowup (2.0–2.4 / 0.33), shrinking the dense-vs-trees gap from
~3.3 bands to ~1 band, though it still does not clear the ~0.71 save bar.


## Best on record (AUC leaderboard) — ROTATION classification, HIGH-VOCAB full-64-d corpus

Classification mode, AUC higher-better. Sources:
experiments/archive/2026-06-12-high-vocab-topology.md (the 300/120 anchor + burn-deep topology) and
experiments/2026-06-15-vocab-cap-scan-gbdt.md (the bare-GBDT VOCAB-CAP SCAN: caps 60/300/500/1000 × {xgboost, catboost}).
This board now spans an identity-cardinality AXIS — rows at different `n_cols` are SEPARATE matrices and are NOT
comparable to each other (or to the 177-col / DS-META boards); read it as a cap→AUC progression on the SAME corpus /
23,529 rows / 18,823/4,706 @ seed 42. **SEPARATE matrix from every board above** —
dataset `ds-20260612-143014-p64-s42` is the SAME 2026 corpus / 23,529 rows / 18,823/4,706
@ seed 42 as the full-64-d board, but **497 cols** (64 song-AE PCA latent + 9 numeric +
426 identity one-hots: artist 301, genre_primary 121, album_type 4) — identity caps raised
to artist 300 / genre 120. AUC here is NOT comparable to the 177-col full-64-d board nor to
DS-META; the only valid in-dataset references are the Phase-0 trees re-run on THIS matrix
(below). Built specifically to test the "embeddings starved" hypothesis — see header.

| model | dataset | AUC | logloss | accuracy | source | registered? |
|---|---|---|---|---|---|---|
| **xgboost-classifier** (n_est 500, depth 6, lr 0.05, sub/col 0.8, seed 42) — in-dataset leader @ 300/120 | ds-20260612-143014-p64-s42 (497) | **0.7299** | 0.6022 | 0.6719 | this dataset's bake-off; reproduced 0.72988 in the 2026-06-15 vocab scan (C1a, Δ −0.00002) | no |
| catboost-classifier (registry defaults) — 2nd tree family, in-band @ 300/120 | ds-20260612-143014-p64-s42 | 0.7252 | 0.6068 | 0.6649 | same; reproduced 0.72516 in the vocab scan (C1b) | no |
| **xgboost-classifier** (champion config) @ 1000/300 — NEW in-lineage SINGLE-SPLIT best, SUB-BAND | ds-20260615-163042-p64-s42 (1377) | **0.73490** | 0.6007 | 0.6768 | 2026-06-15-vocab-cap-scan-gbdt.md (P2a) | **no — +0.00502 vs 0.7299 (0.40 band, sub-band); 3-seed confirm NOT fired; best-models rank 3** |
| catboost-classifier (defaults) @ 1000/300 | ds-20260615-163042-p64-s42 (1377) | 0.72633 | 0.6093 | 0.6706 | same (P2b) | no |
| xgboost-classifier (champion config) @ 500/200 — FLAT vs 300/120 | ds-20260615-162737-p64-s42 (777) | 0.72966 | 0.6039 | 0.6709 | same (P1a) | no |
| catboost-classifier (defaults) @ 500/200 | ds-20260615-162737-p64-s42 (777) | 0.72372 | 0.6089 | 0.6723 | same (P1b) | no |
| xgboost-classifier (champion config) @ 60/40 — low-cap anchor | ds-20260609-204419-p64-s42 (177) | 0.72044 | 0.6075 | 0.6585 | same (C0a; = the 177-col board) | `xgboost-classifier-rotation-p64` |
| catboost-classifier (defaults) @ 60/40 | ds-20260609-204419-p64-s42 (177) | 0.71105 | 0.6132 | 0.6560 | same (C0b) | no |
| xgboost @ all-vocab (5594/962, 6633 cols) | (build FAILED — memory ceiling, 2× attempts) | — | — | — | same (P3a/P3b NOT RUN) | n/a |
| burn-deep-classifier (topology embeddings, ed8, lr0.0005, [128,64]) — Phase-2 best, in tree tier | ds-20260612-143014-p64-s42 | 0.7228 | 0.6105 | 0.6645 | same | no |
| **burn-deep-classifier** (topology embeddings, ed16, lr0.001, [128,64]) — Phase-1 winner, SAVED | ds-20260612-143014-p64-s42 | 0.7192 | 0.6087 | 0.6694 | same | **YES — `burn-deep-embeddings-highvocab`, dataset_tags [ds-20260612-143014-p64-s42]; flagged for best-model-selector** |
| burn-deep-classifier (topology mlp, lr0.001, [128,64]) | ds-20260612-143014-p64-s42 | 0.7129 | 0.6166 | 0.6617 | same | no |
| burn-deep-classifier (topology wide_deep, lr0.001, [128,64]) | ds-20260612-143014-p64-s42 | 0.7061 | 0.6256 | 0.6526 | same | no |
| burn-deep-classifier (embeddings, ed8, lr0.0005, [128,64]) — capacity-scan B1, WINNER; reproduces the Phase-2 emb best 0.7228 (Δ −0.0001) | ds-20260612-143014-p64-s42 | **0.72274** | 0.6104 | 0.6643 | `2026-07-12-high-vocab-capacity-scan.md` | promoted `emb-128-64-ed8-highvocab` (+auto SAE); not a definition |
| burn-deep-classifier (mlp, lr0.001, [256,128,64]) — capacity-scan B6, best mlp geometry (+0.0057 vs B4, sub-band) | ds-20260612-143014-p64-s42 | 0.71800 | 0.6127 | 0.6606 | `2026-07-12-high-vocab-capacity-scan.md` | no |
| burn-deep-classifier (embeddings, ed8, lr0.0005, [256,128]) — capacity-scan B2, WIDER hurts emb (−0.0048 vs B1) | ds-20260612-143014-p64-s42 | 0.71798 | 0.6132 | 0.6651 | `2026-07-12-high-vocab-capacity-scan.md` | no |
| burn-deep-classifier (mlp, lr0.001, [256,128]) — capacity-scan B5 | ds-20260612-143014-p64-s42 | 0.71576 | 0.6122 | 0.6664 | `2026-07-12-high-vocab-capacity-scan.md` | no |
| burn-deep-classifier (mlp, lr0.001, [128,64]) — capacity-scan B4 control; reproduces on-record mlp 0.7129 (Δ −0.0006) | ds-20260612-143014-p64-s42 | 0.71229 | 0.6169 | 0.6585 | `2026-07-12-high-vocab-capacity-scan.md` | no |
| burn-deep-classifier (embeddings, ed8, lr0.0005, [256,128,64]) — capacity-scan B3, DEEPER hurts emb (−0.0152 ≈ 1.2 bands, worst arm) | ds-20260612-143014-p64-s42 | 0.70750 | 0.6167 | 0.6600 | `2026-07-12-high-vocab-capacity-scan.md` | no |

Read: raising identity cardinality (artist 60→300, genre 40→120) **FLIPS the burn-deep
topology ranking** — `embeddings` rose from last-ish (below mlp by 0.0067 on the 60/40
matrix) to FIRST (above mlp by 0.0063 here): the "embeddings starved" hypothesis is
RESOLVED-CONFIRMED directionally (the per-arm margin is sub-band → 3-split confirm pending,
but the ranking reordered as predicted). The trees ALSO gained on the wider matrix
(xgboost +0.0095, catboost +0.0141) — high vocab is better signal for everyone, not an
embeddings-only effect — so xgboost remains the in-dataset champion (0.7299) and the
embeddings net reaches only the BOTTOM of the tree tier (best 0.7228, −0.0024 vs catboost,
in-band). The saved Phase-1-winner embeddings config (0.7192) is the FIRST ANN to clear the
save bar on rotation (within ~1 band of the tree tier). embed_dim is saturated at 8
(ed8≥ed16≥ed32, all sub-band); the win came from enabling embeddings + lr 0.0005, not from a
wider table. Logloss ~0.61 / brier ~0.21 stay in the trees' calibrated regime. Collinearity
note: at high vocab the manifest flags 21 near-perfect genre↔artist one-hot pairs (a rare
genre carried by a single newly-admitted artist) — absorbed by all families, no pathology.

VOCAB-CAP SCAN read (2026-06-15, `2026-06-15-vocab-cap-scan-gbdt.md`): mapping the bare-GBDT cap→AUC curve shows the identity-resolution lever is **approaching its ceiling**. xgboost: 60/40 0.72044 → 300/120 0.72988 (**+0.00944, +0.75 band — the big jump**) → 500/200 0.72966 (−0.00022, FLAT) → 1000/300 **0.73490** (+0.00502 vs the anchor, ≈0.40 band — a new in-lineage SINGLE-SPLIT best but SUB-BAND). Three of four post-jump caps are sub-band → a PLATEAU with noise jitter, not a climb; the curve is NOT saturated by the strict per-step rule (it rose) and NOT reversed (no interior point ≥1 band below the anchor), but no point crosses champion+band (0.7425) → **no 3-seed confirm, no definition saved.** Catboost tracks xgboost qualitatively (0.71105→0.72516→0.72372→0.72633, always ~0.005–0.009 below, same plateau) — native categorical handling does NOT scale better at high cardinality. The top-of-curve **all-vocab (5594/962, 6633 cols) point is UNMEASURED** — its dataset build hit a hard memory ceiling twice (see the dataset-axis pitfalls / lineage), so the curve's high end is a known gap; the trend through 1000/300 makes it unlikely to have crossed the bar (the 3276 singleton-artist cols it would add are memorizable variance, expected to plateau/mildly-reverse not gain). Practical read: artist/genre one-hot CAP HEIGHT is a largely-spent lever on this corpus — future identity work should target resolution QUALITY (followers/popularity interactions, genre hierarchy), not bigger caps.

CAPACITY SCAN read (2026-07-12, `2026-07-12-high-vocab-capacity-scan.md`): the burn-deep-classifier ANN family is ALSO capacity-saturated at high vocab. Two geometry ladders (embeddings ed8/lr0.0005; mlp lr0.001; [128,64]→[256,128]→[256,128,64], embed_dim held 8) leave the [128,64] control best for BOTH topologies (emb B1 0.72274 = on-record 0.7228 reproduced; mlp B4 0.71229 = on-record 0.7129 reproduced); widening is a sub-band wash and DEEPENING EMBEDDINGS to 3 layers HURTS by 1.2 bands (B3 0.70750). Best mlp (B6 0.71800) ties best emb (B1 0.72274) within a band → the 'embeddings wins at high vocab' edge is an in-band tie, not a mechanism win; capacity, not topology, is the (spent) lever. No arm crosses the tree tier (catboost 0.7252 / xgboost 0.7299) or the 0.7318 save bar → no definition saved, incumbent `burn-deep-embeddings-highvocab` unchanged; B1 promoted for the record (`emb-128-64-ed8-highvocab`, +auto SAE).

## HISTORICAL leaderboard — engagement target (demoted, not the active target)

| model | dataset | MAE | RMSE | R² | source |
|---|---|---|---|---|---|
| xgboost (metadata-only; auto-promoted `best-xgboost-20260608-161931-33428`) | ds-20260608-161709-p1-s42 | 2.424 | 7.764 | 0.437 | bootstrap first-run |
| baseline-median (group-median floor) | ds-20260608-161709-p1-s42 | 3.024 | 10.625 | −0.054 | bootstrap first-run |
| ridge (auto_alpha) | ds-20260608-161709-p1-s42 | 3.074 | 10.100 | 0.047 | bootstrap first-run |

Read (historical): xgboost beats the median floor by ~20% MAE (R² 0.437) —
intrinsic song metadata carries real engagement signal; ridge barely clears the
floor, so the metadata→engagement signal is non-linear / interaction-heavy.
These remain valid for the engagement *field* but it is no longer the model's
delivered target.

## Noise bands & significance thresholds

- **TEMPORAL GENERALIZATION / TASTE DRIFT — every on-record AUC is a RANDOM split
  and OVERSTATES forward prediction by ~0.11–0.12; taste is NON-stationary
  (MEASURED 2026-07-12, `2026-07-12-taste-drift-temporal-split.md`).** Definitive,
  via a NEW first-class chronological-split build option (see the capability note
  below): on framework-built `spotify_tracks_song_ae` datasets, the champion
  xgboost scores **0.73490 random vs 0.61894 CHRONOLOGICAL** (train = oldest-80% by
  `first_played`, PCA/vocab fit on early-train only; test = newest-20%) — a
  **0.116 drop ≈ 9 bands**. The drift is **UNIVERSAL across model families**
  (catboost 0.72889→0.61294; logistic 0.72141→0.61234) → it is the data/taste, not
  a model quirk. It is REAL drift, not right-censoring (≥180-d-exposure control
  barely moves the off-framework cross-check: 0.62579→0.62319; walk-forward lands
  0.59–0.67 every year 2019–2026). Mechanism = identity churn: rotation is
  identity-driven, future-discovered artists are unseen (`__other__`), so the model
  falls to the intrinsic floor (~0.60). **Identity one-hots barely help FORWARD**
  (dropping them costs only ~0.02 chrono: full 0.61894 vs intrinsic 0.59858; vs
  ~0.025 random) — they mostly memorize same-era artists. **Time-aware
  (recency-weighted xgboost) is a WEAK lever** (+0.003 sub-band; drift is not a
  simple recency effect). CONSEQUENCE: read the whole AUC leaderboard as same-era
  ranking, NOT forward skill — real "will I replay a song I discover next"
  performance is ~0.60–0.62 on this corpus; the only forward lever is generalizable
  intrinsic signal (better acoustics / cross-artist structure), NOT identity
  resolution or capacity (both spent, same-era-only). Definitive datasets:
  full-random `ds-20260712-183615`, full-chrono `ds-20260712-183629`,
  intrinsic-random `ds-20260712-183630`, intrinsic-chrono `ds-20260712-183631`. An
  off-framework re-split cross-check agreed to ~0.007 (0.626); framework champion
  run on `183629` = 0.61894 to the digit.
- **NEW BUILD CAPABILITY: chronological train/test split (`split_order_field`) —
  added 2026-07-12.** `POST /api/datasets` / `BuildConfig` now accepts
  `split_order_field` (e.g. `"first_played"`): rows split by ascending value of
  that payload field (earliest→train, latest→test), PCA/vocab/imputation fit on the
  early-train rows only (leak-free); manifest self-describes via `split.strategy`/
  `order_field`. Default remains random seeded shuffle; backward-compatible
  (serde-defaults `"random"`), 2 unit tests in `crates/lensing-pipeline/src/build.rs`.
  Code: `chronological_split` + `BuildConfig.split_order_field` (build.rs),
  `SplitInfo.strategy`/`order_field` (lensing-core manifest.rs), `BuildRequest`
  (lensing-server api.rs). The new binary is compiled but PRODUCTION 8096 still runs
  the old one — restart 8096 to expose the option there (demonstrated on a transient
  scratch server; 8096 never restarted). Upstream-contribute candidate. **PITFALL:
  the build defaults to `domain.toml`'s `spotify_tracks` (200-dim behavioral
  co-listening embedding, which SOFT-LEAKS replay — inflates rotation AUC to ~0.71
  chrono / higher random); rotation work MUST pass `collection:
  "spotify_tracks_song_ae"`.** Stale wrong-collection artifacts from this campaign to
  ignore/delete: `ds-20260712-182109`, `-182836`, `-182837`.
- **ROTATION target (active): MEASURED 2026-06-09 (svm-kernel campaign).**
  σ_split = **0.00430**, significance band 2σ = **0.0086 MAE** on the [0,1] scale.
  Measured as the sample stdev of metadata-only XGBoost MAE across three split
  seeds on the canonical recipe: s42 0.387891, s17 0.393072, s101 0.396417 (mean
  0.392460). Consequences: the 0.3879 (metadata-only) vs 0.3965 (latent+metadata)
  edge is Δ 0.0086 ≈ **1·band** → "likely better, not confident"; the 0.3879 vs
  0.4098 (AE-latent-only) gap is ~2.5·band → **real**. NOTE the binary-target MAE
  caveat: on rotation, MAE is a degenerate skill metric (a two-level ε-tube
  predictor scores low MAE while being worse-than-mean on R²/RMSE — see SVM
  pitfall) — judge rotation models on R²/RMSE/AUC, use MAE only for same-family
  band significance. The 3-seed champion-confirmation is the immediate next
  campaign (the s17/s101 points above already are it for the champion config).
- **ROTATION AUC band (active classification metric): MEASURED 2026-06-09
  (rotation-classification campaign).** σ_AUC = **0.00628**, significance band 2σ =
  **0.0126** (AUC, higher-better). Sample stdev of the registered xgboost-classifier
  champion's AUC across three split seeds on DS-META: s42 0.742934, s17 0.730659, s101
  0.734469 (mean 0.736020). This is the band for any "beats" claim on the AUC axis; it
  does NOT replace the 0.0086 *MAE* band (different metric/scale/regime — MAE band for
  regression families, AUC band for classification). Consequence: the champion beats
  every dense family by 4.6–9.3 bands (decisive); the AE-latent dilution (−0.0079) sits
  inside the band (neutral call, one seed).
- **ENGAGEMENT target (historical): not measured either.** The old guidance
  (treat MAE differences below ~0.2 as noise by analogy) stands for that scale.

## Dataset lineage

| dataset (recipe) | feeds | notes |
|---|---|---|
| ds-20260608-204402-p64-s42 (pca=64 over the `spotify_tracks_song_ae` AE-latent collection + 114 metadata cols = 178; target rotation/none; `nonpositive_price=FALSE`) | rotation pivot seed run (xgboost latent+metadata, run-20260608-204415-6eeda); the entire 2026-06-09 autoencoder-xgboost blend campaign | 12,256 rows (NO exclusions — rotation==0 is the negative class), 9,805 train / 2,451 test @ seed 42. The 64 "pca" cols ARE the precomputed 64-dim song-autoencoder latent (kind.type=="pca" in the manifest), NOT a behavioral co-listening PCA. The autoencoder-xgboost predictor splits columns by this manifest kind: latent branch = the 64; metadata branch = the other 114. |
| ds-20260609-204419-p64-s42 (pca=64 song-AE latent + 7 numeric + 106 identity one-hots = 177; target rotation/none; `nonpositive_price=FALSE`) — the **NEW 2026-corpus canonical full-64-d** rotation dataset | the 2026-06-12 four-family classifier bake-off (xgboost/catboost/ngboost/pyramid-mlp + the xgboost depth×lr sweep, 13 runs); the registered `xgboost-classifier-rotation-p64` lives here; pre-existing reference run run-20260609-204726-439d8-xgboost-classifier (AUC 0.7204); the 2026-06-12 burn-deep-classifier topology bake-off + mlp tune (9 runs, `2026-06-12-burn-deep-topology.md`, no definition saved); the 2026-06-15 gbdt-leaf-head-classifier ceiling-probe (4 runs, `2026-06-15-gbdt-leaf-head-ceiling-probe.md`, REFUTED, no definition saved); the 2026-06-15 burn-ae-classifier capacity + head-architecture scan (13 ad-hoc runs, `2026-06-15-burn-ae-classifier-capacity-scan.md`, family closed for AUC, no definition saved) | **23,529 rows** (18,823 train / 4,706 test @ seed 42), 177 cols. The NEW (larger, 2026) corpus — NOT the 12,256-row DS-META corpus; AUC numbers on this dataset are NOT comparable to the 0.7429 DS-META champion. xgboost reproduces 0.7204 deterministically (byte-identical across the reference run, C0, and the depth6/lr0.05 sweep cell). |
| ds-20260612-143014-p64-s42 (HIGH-VOCAB rotation: identical to ds-20260609-204419-p64-s42 EXCEPT vocab caps artist 60->**300**, genre_primary 40->**120**; pca=64 song-AE latent + 7 numeric + **426** identity one-hots = **497**; target rotation/none; seed 42 / test_ratio 0.2; `nonpositive_price=FALSE`) | a planned re-run of the burn-deep-classifier topology bake-off ({mlp, embeddings, wide_deep}) to test whether entity embeddings pay off once identity cardinality is large enough -- the `2026-06-12-burn-deep-topology.md` finding was that `embeddings` LOST to plain `mlp` because cardinality was starved at the canonical caps | **23,529 rows** (18,823 train / 4,706 test @ seed 42), 497 cols. SAME corpus/rows/split as the canonical; differs ONLY in identity cardinality. One-hot groups: artist **301** (top-300 + __other__, 50.4% of artist mass, boundary min freq 14), genre_primary **121** (top-120 + __other__, 78.8% of genre mass, boundary min freq 31), album_type 4 (unchanged). Canonical caps captured only artist 25.1% / genre 60.8%. PCA dims unchanged => EVR identical to ds-20260609-204419-p64-s42. |
| ds-20260615-162737-p64-s42 (VOCAB-SCAN P1: identical to ds-20260612-143014-p64-s42 EXCEPT vocab caps artist 300->**500**, genre_primary 120->**200**; pca=64 song-AE latent + 9 numeric + identity one-hots = **777** cols; target rotation/none; seed 42 / test_ratio 0.2; `nonpositive_price=FALSE`) | the 2026-06-15 GBDT vocab-cap scan (P1a xgboost 0.72966, P1b catboost 0.72372; `2026-06-15-vocab-cap-scan-gbdt.md`) — the 500/200 point on the cap→AUC curve | **23,529 rows** (18,823 train / 4,706 test @ seed 42), 777 cols. SAME corpus/rows/split as the canonical & 300/120 anchor; differs ONLY in identity cardinality. xgboost FLAT vs 300/120 (−0.00022, within noise). PCA dims unchanged => EVR identical to the 177/497-col siblings. No definition saved (sub-band), no tags (no winner). |
| ds-20260615-163042-p64-s42 (VOCAB-SCAN P2: identical to the above EXCEPT vocab caps artist 500->**1000**, genre_primary 200->**300**; pca=64 song-AE latent + 9 numeric + identity one-hots = **1377** cols; target rotation/none; seed 42 / test_ratio 0.2; `nonpositive_price=FALSE`) | the 2026-06-15 GBDT vocab-cap scan (P2a xgboost **0.73490** — the NEW in-lineage single-split best, SUB-BAND; P2b catboost 0.72633; `2026-06-15-vocab-cap-scan-gbdt.md`) — the 1000/300 point on the cap→AUC curve | **23,529 rows** (18,823 train / 4,706 test @ seed 42), 1377 cols. P2a auto-promoted to best-models rank 3 (0.73490) but displaced nothing registered; +0.00502 vs the 0.7299 anchor (0.40 band, sub-band) → no 3-seed confirm fired, **no definition saved**. Candidate dataset for a future 3-seed confirm of the 1000/300 edge. PCA dims unchanged => EVR identical to siblings. |
| ds (NOT BUILT) — VOCAB-SCAN P3 "all-vocab" (artist 5594 / genre 962 = full uncapped, ~6633 cols) | NOTHING — build FAILED twice (memory ceiling) in the 2026-06-15 vocab-cap scan | The top-of-curve point. Two `POST /api/datasets` attempts each returned a build_id but never materialized a dataset (host ~2.3 GB free of 15 GB with concurrent training); a 23,529 × 6,633 float32 dense matrix (~0.62 GB) + PCA/one-hot intermediates + train/test copies exceeded headroom. Recorded as a documented INFRASTRUCTURE ceiling, not a modeling result. Rebuild when the host is quiescent (or with a sparse one-hot path) to close the curve's high end. |
| ds-20260608-205802-p64-s42 (pca=64 AE-latent ONLY, no metadata cols) | xgboost on AE-latent only (rotation), the AE-latent-only baseline | 64 cols. Used to measure the latent's standalone rotation signal (MAE 0.4098, R² 0.132 — weaker than metadata). |
| ds-20260609-141622-p1-s42 (METADATA-ONLY rotation: pca floor=1, NO AE latent; same recipe as p64-s42 with AE cols dropped; target rotation/`none`; `nonpositive_price=FALSE`) | the 2026-06-09 svm-kernel scan (all SVM arms + fair-anchor control + both stage-2 blends) AND the 2026-06-09 rotation-classification bake-off (registered AUC champion `xgboost-classifier-rotation` lives on this recipe) | 115 cols = 1 pca floor + 7 numeric + 107 one-hot; 12,256 rows, 9,805/2,451 @ seed 42. The fair SVM anchor: metadata-only xgboost on this matrix = byte-identical 0.387891 to the 178-col champion (AE branch w=0), so dropping the 64 latent cols is a no-op for the tree. Built with `log_target=false` to get transform `none` (the canonical rotation recipe). |
| ds-20260609-141641-p1-s17, ds-20260609-141641-p1-s101 (METADATA-ONLY rotation, same recipe, split seeds 17 / 101) | the rotation noise-band measurement (σ_split) | 115 cols each. metadata-only xgboost MAE s17 0.393072 / s101 0.396417 — with s42 0.387891 these three define σ_split = 0.0043 (band 0.0086). Also serve as 2 of the 3 seeds for the pending champion 3-seed confirmation. |
| ds-20260609-141540-p1-s42 (THROWAWAY: same metadata-only recipe but built with default `log_target=true` ⇒ transform log1p) | nothing — unused | Built by mistake before `log_target=false` was set; the canonical rotation recipe is transform `none`. Left in place, fed no run. |
| ds-20260608-161709-p1-s42 (pca=1, metadata-only over `spotify_tracks`; nonpositive-engagement excluded) — HISTORICAL (engagement) | bootstrap baseline trio (baseline-median / ridge / xgboost) | 12,106 rows (12,256 corpus − 150 engagement≤0); 9,685 train / 2,421 test @ seed 42; 115 cols = 1 PCA embedding PC + 8 metadata numerics/indicators + genre(41) + artist(61) + album_type(5); target engagement (log1p). pca_dims=1 is the framework FLOOR (see pitfalls). |

### Corpus collections

- **`spotify_tracks_song_ae`** (active rotation collection) — carries a
  precomputed 64-dim song-autoencoder latent per track plus the `rotation`
  (play_count≥2) label in payloads; the server is repointed here so Pathfinder's
  graph is the rich song-character manifold. The latent encodes song *identity*.
- `spotify_tracks` (historical engagement collection) — 200-dim co-listening
  behavioral embedding + metadata.
- `spotify_tracks_content` (rejected) — 384-dim content-text embedding +
  ReccoBeats audio features.

## Known pitfalls per predictor family

Generic priors (replace with measured facts as campaigns land):

- **autoencoder-xgboost (weighted-blend ensemble) — the blend self-disables when
  the AE branch is weak (MEASURED 2026-06-09, rotation).** `yhat = w·yhat_ae +
  (1−w)·yhat_xgb` with `w` tuned on a train holdout. On rotation the song-AE
  latent's standalone error (kNN best 0.4259 @ k=10, monotonically worse to
  0.4702 @ k=200; ridge 0.4623) is uniformly worse than the metadata-XGB branch
  (0.3879), so the tuner drives **w → 0.0 in every arm** and the blend collapses
  to identical metadata-only xgboost. This is correct behavior, not a bug — but
  it means the blend can only ever *match* the stronger branch here, never beat
  it. The architecture's one real value was isolating the latent so the tuner
  could discard it, recovering a cleaner result than the unified
  latent+metadata xgboost (0.3879 vs 0.3965). The informative axis in the kNN
  k-scan is the `mae_ae(k)` curve (read from on-disk metrics.json: `mae_ae`,
  `mae_xgb`, `blend_weight` — the API does NOT surface these), not the blended
  metric (constant across the scan). Verdict: not worth pursuing further for
  rotation; latent is a dead modality.
- **kernel SVM (SVR) REFUTED on rotation — and its MAE is a binary-target trap
  (MEASURED 2026-06-09).** All four kernels on the metadata-only matrix
  (ds-20260609-141622-p1-s42): rbf MAE 0.3847 / R² −0.130, linear 0.3901 / −0.198,
  poly(deg3) 0.3814 / −0.137, sigmoid 0.6240 / R² −8.48 (predictions up to 24.7 —
  classic unbounded sigmoid-SVR blow-up). Note rbf/linear/poly post LOWER MAE than
  the 0.3879 champion yet are WORSE-THAN-MEAN on R²/RMSE (champion R² 0.191, RMSE
  0.4475 vs SVM RMSE 0.53–0.54). Cause (from on-disk predictions.json): with
  epsilon=0.1 on a {0,1} target the SVR snaps outputs to the tube edges — p50≈0.10,
  p90≈0.90, a hard two-level step — which scores a deceptively low MAE (each row
  ~0.1 off its 0/1 label) while destroying calibration/ranking. On a binary target
  MAE is degenerate; the honest verdict (R²/RMSE) is that SVM loses decisively.
  Verdict: do NOT pursue SVM for rotation. This is the 5th dense-regime family to
  lose to one-hot identity here (see convergent finding).
- **xgboost-classifier is the rotation AUC CHAMPION; logistic + every SVC kernel are
  REFUTED on AUC (MEASURED 2026-06-09, classification mode).** On DS-META
  (ds-20260609-141622-p1-s42, metadata-only, AUC-scored): xgboost-classifier AUC
  **0.7429** (logloss 0.5883, acc 0.685) — registered as `xgboost-classifier-rotation`,
  best-models rank 1. The dense families lose by 4.6–9.3 AUC bands (svc-poly 0.6855,
  logistic 0.6762, svc-rbf 0.6722, svc-linear 0.6363, svc-sigmoid 0.6255). Unlike the
  SVR campaign — where the binary-MAE ε-tube made rbf/linear/poly *look* competitive on
  MAE while worse-than-mean on R² — the classification metric is honest: every SVC
  kernel is *both* worse on AUC *and* worse on logloss/brier, same direction. Notably
  svc-sigmoid does NOT collapse to 0.5 (stays 0.6255) — a proper probability objective
  removes the unbounded-sigmoid blow-up that wrecked the SVR-sigmoid run. xgboost-classifier
  control hyperparams: n_est 500, depth 6, lr 0.05, subsample/colsample 0.8, seed 42.
  Verdict: xgboost-classifier is the family of record for rotation classification; do not
  pursue logistic/SVC. The 64 AE-latent cols cost the classifier 0.0079 AUC (178-col 0.7350
  vs metadata-only 0.7429) — inside the AUC band, so neutral-leaning-dilutive (same dilutive
  direction as the regressor, sub-band on AUC). Next lever is a champion hyperparameter scan
  + identity-resolution (vocab caps), NOT new families.
- **catboost-classifier is a legitimate 2nd tree family — co-leader IN-BAND with
  xgboost on the new full-64-d corpus (MEASURED 2026-06-12).** On
  `ds-20260609-204419-p64-s42` (23,529 rows, 177 cols), catboost-classifier
  (iter 500, depth 6, lr 0.05, l2 3.0, subsample 0.8, border 254, seed 42) AUC
  **0.7111** / logloss 0.6132 vs xgboost-classifier 0.7204 — Δ −0.0093, INSIDE the
  0.0126 band, so "at least equal, likely a hair behind", NOT refuted. First
  head-to-head of catboost here; it is the one non-xgboost family that stays in-band
  on the GBDT regime. Use as a tree-diversity / second-opinion member; do not expect
  it to beat xgboost on AUC alone.
- **ngboost-classifier under-fits the one-hot-heavy matrix (~2.3 bands back,
  MEASURED 2026-06-12).** On the same dataset, ngboost-classifier (n_est 400,
  lr 0.03, minibatch 0.5, col 1.0, base_max_depth 3, seed 42) AUC **0.6917** /
  logloss 0.6293 — −0.0287 (~2.3 bands) below xgboost. Its natural-gradient
  distributional objective with SHALLOW base learners (depth 3) pays for its
  probabilistic machinery in raw ranking power on 177 sparse one-hot cols. Not a
  failure, just not competitive on AUC; reach for it only if calibrated predictive
  distributions are needed, not for the AUC leaderboard.
- **pyramid-mlp-classifier is the weakest family — AUC ceiling ≈ 0.68 AND the
  high logloss is a CALIBRATION ceiling, NOT a fixable training-budget artifact
  (MEASURED 2026-06-12, both the bake-off AND the arch scan
  `2026-06-12-pyramid-mlp-arch-scan.md`).**
  On `ds-20260609-204419-p64-s42`, pyramid-mlp (256/3/0.5, alpha 1e-4, lr_init 1e-3,
  seed 42) AUC **0.6781**, logloss **2.4085**, brier **0.3251** (vs ~0.60–0.63 logloss
  / ~0.21 brier for the trees). The earlier "300-iter-cap under-training artifact"
  framing is **REFUTED**: a 6-arm architecture scan with max_iter raised to **800**
  found *no arm hit the cap* — every geometry stops itself at **73–100 iterations**
  via sklearn's `n_iter_no_change` tolerance, well below even 300. The 256/3/0.5 arch
  at max_iter 800 reproduced the AUC 0.6781 to all digits (identical 73 iters), so the
  budget was NEVER binding. The blown logloss is the optimizer's chosen plateau at this
  alpha/lr — a real calibration ceiling, not a truncated trajectory. Architecture
  barely moves it: all 6 geometries land in a 0.0080-wide AUC strip 0.6712–0.6792
  (entirely inside the ~0.0126 band), best = wide-shallow [512,256] at **0.6792**
  (+0.0011 over the default = zero), worst = wide-deep — adding depth/capacity hurts
  slightly. A dense MLP blurring 106 sparse identity one-hots is the same
  failure mode as the SVC/logistic dense families (the convergent finding); the family
  sits ~3.3 bands below xgboost 0.7204 and is **closed for AUC** — do not re-scan
  geometry or budget. Raising max_iter does NOT help; if calibrated MLP probabilities
  are ever needed, wrap the head in Platt/isotonic, or sweep lr/alpha, not iterations. **REVISED 2026-06-12 (`2026-06-12-burn-deep-topology.md`):** the "calibration is a
  fixed ceiling for MLPs" reading applies to the SKLEARN pyramid's REPRESENTATION
  (raw one-hots standardized into outlier spikes), NOT to MLPs in general. A
  full-rust net that keeps the identity one-hots RAW (`burn-deep-classifier`,
  standardizing only the 64 PCA + 7 numeric block) drops logloss 2.0–2.4 → ~0.62 and
  brier 0.33 → ~0.22 (the trees' regime) AND lifts AUC to 0.7038 (~2 bands over the
  best converged sklearn pyramid 0.6792). So the sklearn pyramid is superseded by
  burn-deep for any future ANN work; the sklearn family stays closed (its
  representation can't be fixed without re-engineering the predictor) but "ANNs are
  intrinsically miscalibrated / capped at 0.68 here" is FALSE — see the burn-deep
  entry below.
- **burn-deep-classifier (NEW Rust/burn ANN, raw one-hots) — the calibration-FIXED
  dense family; ~1 band below trees, plain `mlp` topology wins (MEASURED 2026-06-12,
  `2026-06-12-burn-deep-topology.md`).** On `ds-20260609-204419-p64-s42` (177 cols), this
  predictor splits the matrix into a STANDARDIZED continuous block (64 PCA + 7 numeric)
  and RAW one-hot identity groups (artist 61, genre 41, album_type 4) — the
  representation fix that cures the sklearn pyramid's logloss-2.0 blowup. Topology
  bake-off {mlp, embeddings, wide_deep} (epochs 250, patience 20, val_frac 0.15,
  hidden [128,64], embed_dim 8, dropout 0.1, lr 0.001, batch 256, relu): **mlp AUC 0.7001
  / logloss 0.6211** wins, embeddings 0.6934, wide_deep 0.6918 — a 0.0083-wide strip,
  ALL inside the 0.0126 band (the user's "embeddings vs wide_deep" = embeddings +0.0016,
  sub-band tie; plain mlp beats both). A 6-cell mlp tune (lr {0.0005,0.001,0.002} ×
  hidden {[256,128],[128,64]}) is FLAT (0.6957–0.7038) and prefers the SMALLER net
  ([128,64] > [256,128] uniformly) + slowest lr (0.0005); best = **0.7038** (+0.0037 over
  Phase-1, inside band). Verdict: a representation-aware net nearly closes the
  dense-vs-trees gap (now ~0.6 band below catboost 0.7111, ~1.3 below xgboost 0.7204 — vs
  the sklearn pyramid's ~3.3) but does NOT clear the ~0.71 save bar, so NO definition
  saved. Use it as the dense baseline of record (supersedes the sklearn pyramid) and as an
  ensemble-DIVERSITY member (different inductive bias from the GBDTs), NOT as an AUC
  contender on its own. Embeddings are likely starved by the LOW identity cardinality
  (61/41/4) — re-test the topology bake-off after raising vocab caps. NOTE: the burn binary
  is NEAR- not bit-deterministic (the lr0.001/[128,64] config scored 0.700059 in Phase 1
  vs 0.699616 re-run in Phase 2, Δ 0.00044 ≈ 0.07 band) despite the fixed internal split
  seed 1337 — sub-band, but run a fixed-config triple if a tight guarantee is needed.
- **burn-deep-classifier is CAPACITY-SATURATED at high vocab too — the [128,64] control is best for BOTH topologies and deeper EMBEDDINGS actively HURT (MEASURED 2026-07-12, `2026-07-12-high-vocab-capacity-scan.md`).** On the HIGH-VOCAB matrix `ds-20260612-143014-p64-s42` (497 cols), two geometry ladders (held: ed8, epochs 250, patience 20, dropout 0.1, batch 256, val_frac 0.15; lr at each topology's on-record best — emb 0.0005 / mlp 0.001) show geometry is a flat-to-negative axis. Embeddings: [128,64] **0.72274** (reproduces the on-record 0.7228 to 4 digits, gate PASS) > [256,128] 0.71798 (−0.0048) > [256,128,64] **0.70750** (−0.0152 ≈ 1.2 bands, the ONLY out-of-band move — a 3rd layer over a learned entity-embedding table dilutes ranking). MLP: [128,64] 0.71229 (reproduces the on-record 0.7129, gate PASS) < [256,128] 0.71576 < [256,128,64] 0.71800 (both sub-band gains; neither beats the control by >1 band). No overfit signature (max logloss 0.617 ≪ 0.8; patience 20 held — the patience-0 deeper-net-overfit pitfall did NOT fire at the 250-epoch budget); calibration healthy (logloss 0.610–0.617 / brier 0.211–0.214). Cross-family (Step 5): best mlp 0.71800 is within a band of best embeddings 0.72274 → the two topologies are TIED at their best geometries; the 'embeddings wins at high vocab' ordering (`archive/2026-06-12-high-vocab-topology.md`) is an in-band tie, not a mechanism win — capacity, not the embeddings mechanism, is the lever, and it is spent. Best ANN stays ~0.2 band below catboost (0.7252), ~0.6 below xgboost (0.7299) — capacity does NOT cross the tree gap; no arm cleared the 0.7318 save bar → NO definition saved, incumbent `burn-deep-embeddings-highvocab` (0.7192) unchanged. Best arm promoted for the record: `emb-128-64-ed8-highvocab` (run-20260712-171008-065bd, +auto SAE). Verdict: the geometry axis is CLOSED for burn-deep-clf at BOTH cardinalities (the canonical 60/40 matrix also preferred [128,64]) — the remaining lever is identity-resolution QUALITY, not model or vocab size. Near- not bit-deterministic (B1 0.72274 vs on-record 0.7228, Δ 0.00006 sub-band).

- **SAE capture (2026-07-12, high-vocab `ds-20260612-143014-p64-s42`, `2026-07-12-high-vocab-capacity-scan.md` §Information capture):** AUC-tied MLP[256,128,64] (0.71800) and emb[128,64] (0.72274) hit the same rotation decodability ceiling (peak linear R² ~0.14–0.15, 245/245 identity segments each get a dedicated separating atom) but ALLOCATE differently — the MLP proliferates ~200 concepts peaking mid-trunk (55→113→32) then collapses through its 64-d neck, the winner consolidates 108 monotonically (33→75); deepening embeddings to [256,128,64] (0.70750) HURTS by DILUTION not starvation (0 dead atoms, 0 dropped embedding concepts, but a weaker learned-embedding input — input probe R² 0.123 vs 0.133 — and redundant mid-trunk concepts it never recovers from: decodability dips 0.132→0.127→0.135). Reproduces the `2026-07-12-pyramid-vs-rect-sae.md` consolidation-vs-proliferation pattern across families — invisible to AUC. The dataset-SAE **dropped-signal null holds** (0 target-correlated atoms above the 0.3 concept bar on all three — no clean monosemantic taste-fit atom at this data scale, so `compare_embedding` is inconclusive by construction here). Analyses `interp-run-20260712-171243-f65f7-msae` (emb winner), `...171915-c52e4-msae` (mlp), `...171915-fe7db-msae` (emb deep); B6/B3 promoted as `mlp-256-128-64-highvocab` / `emb-256-128-64-highvocab` to enable the read.

- **burn-ae-classifier (NEW Rust/burn two-phase AE-transplant) — capacity-SATURATED, statistically EQUAL to plain burn-deep mlp, ~1.8 bands below trees; freeze+deep-head is a COLLAPSE trap (MEASURED 2026-06-15, `2026-06-15-burn-ae-classifier-capacity-scan.md`).** Phase 1 pretrains an autoencoder to reconstruct the assembled feature vector `[standardized 64 PCA + 7 numeric | raw one-hots]` (D=177) by MSE; Phase 2 transplants the encoder into encoder→latent→clf-head→softmax and fine-tunes with cross-entropy (`freeze_encoder` toggles fixed-extractor vs end-to-end). On `ds-20260609-204419-p64-s42` (177 cols), a 13-config scan (held: ae_epochs 80, ae_lr 0.001, epochs 200, lr 0.001, batch 256, dropout 0.1, patience 25, val_frac 0.15) is FLAT: the 12 fine-tuned configs span 0.69175–0.69719 (**0.0054, entirely inside the 0.0126 band**). Latent dim 32≈64≈128 (cfg9/cfg1/cfg5, 0.00067 spread); head depth/width sub-band and the **linear probe (clf [], 0.69175) is only −0.0054 below the deepest head** — a single linear layer on the fine-tuned latent captures all the family's skill, capacity is saturated at the SMALLEST net; deeper encoder ([256,128,64]) and large ([384,192]/[192,96]) add nothing. Activation: relu 0.69719 > gelu 0.69340 > silu 0.69242 (sub-band, but smooth activations are a hair WORSE). Best = **cfg3 enc[128,64]/clf[128,64]/fine-tune/relu, AUC 0.69719** — +0.00494 over the 0.69225 baseline (sub-band, NOT a material beat), ~1.84 bands below xgboost (0.7204), ~1.1 below catboost (0.7111), and **−0.0066 (sub-band) vs the comparable burn-deep mlp (0.7038): the unsupervised AE pretrain buys a stable init but NO AUC over a from-scratch raw-one-hot mlp.** Calibration healthy (logloss ~0.62 / brier ~0.22, the trees' regime — far better than the sklearn pyramid's 2.0+ or the gbdt-leaf-head's 0.82–1.31). THE ONE REAL EFFECT — freeze_encoder is a TRAP for deep heads: frozen + deep head (cfg10 [128,64], **0.68210**) collapses −0.0151 ≈ 1.2 bands vs its fine-tuned twin (the ONLY out-of-band row in the campaign), while frozen + shallow head (cfg11 [128], 0.69530) stays in-band — a deep head stacked on a frozen MSE-suboptimal extractor just underfits the label. **Use freeze_encoder=true only with a (near-)linear head; the registry default freeze_encoder=false (end-to-end fine-tune) is correct.** Verdict: family CLOSED for AUC on rotation as configured — capacity is exhausted, the next lever is the pretraining OBJECTIVE (supervised/contrastive), not size; NO definition saved. Same dilution family as the AE-latent / burn-deep-head / gbdt-leaf-head findings (a learned-representation→head that does not beat the direct model). Determinism: NEAR- not bit- (the baseline config re-run cfg1 scored 0.69645 vs the original 0.69225, Δ 0.0042 ≈ 0.33 band) — fixed internal split seed, sub-band jitter. All 13 runs `succeeded` (metric block auc/logloss/acc/brier, no MAPE — binary NaN-MAPE trap dodged). Per-run early-stop epoch is NOT exposed via the API (empty stderr_tail on success).
- **gbdt-leaf-head-classifier REFUTED on rotation — a learned head over one-hot tree
  leaves LOSES to the bare tree by 1.8–2.5 bands AND is mis-calibrated (MEASURED
  2026-06-15, `2026-06-15-gbdt-leaf-head-ceiling-probe.md`).** The Facebook GBDT→LR
  trick generalized: a tree hard-frozen at the champion config (n_est 500 / depth 6 /
  lr 0.05 / sub-col 0.8 / mcw 1.0 / λ 1.0 / seed 42) emits per-row leaf indices,
  one-hot-encoded (~10k leaf-membership dims), and a head trains on top. On
  `ds-20260609-204419-p64-s42` (177 cols), every head caps BELOW the bare champion
  (AUC 0.7204): **linear (L2 logistic, l2 1.0) AUC 0.6978** (best leaf-head, −0.0227 ≈
  1.8 bands), mlp[256,128] 0.6921 (−2.2 bands), mlp[128] 0.6897 (−2.4 bands). The
  control C0 reproduced 0.7204383403100314 to all 16 digits, so the gap is real, not
  drift. CALIBRATION is broken on every head: logloss 1.31 / 1.18 / 0.82 and brier
  0.29 / 0.30 / 0.26 vs the tree's 0.61 / 0.21 — the linear head at the default
  inverse-C 1.0 is under-regularized and over-confident (the sklearn-over-confidence
  signature: logloss far above the trees while AUC is only modestly behind — NOT a
  leakage signature). The leaf one-hots are a strictly lossier view of the tree than
  the tree's own additive leaf-value scoring (which already jointly weights leaves
  across 500 shrunk boosting rounds); a head re-estimates that worse — it cannot even
  recover the tree that generated its features. This is the SAME family as the
  AE-latent dilution and the burn-deep dense-head cap — a FIFTH confirmation that a
  learned head fed a learned representation dilutes here. Verdict: family CLOSED for
  rotation; raising l2 might fix calibration but cannot close a ~1.8-band AUC deficit.
  No definition saved; no T4 (best treatment below the in-band branch). All 4 runs
  `succeeded` (the plugin reuses the classification metric block — auc/logloss/acc/brier,
  no MAPE — so the binary NaN-MAPE trap is dodged). The mlp head is seeded but adam +
  early-stopping is near- not bit-deterministic; the linear head is the clean
  deterministic control.
- **blend grid-by-MAE is UNSAFE on binary targets (MEASURED 2026-06-09).** The
  `blend` meta-predictor with weight_fit=grid optimizes validation MAE. On
  rotation, both stage-2 blends (xgb+svm, lgbm+svm) fitted weights **[0.0, 1.0]** —
  zeroing the better tree and keeping pure SVM — because svm(poly)'s artifact-low
  MAE (0.3810) edges the trees' (0.387/0.388). The grid thus PROMOTED a
  worse-than-mean member (the opposite of the AE-blend "w→0 self-disable"). Lesson:
  for binary/rotation blends the grid objective must be R²/logloss/AUC, not MAE.
  Until that framework fix lands, do not trust a grid-weighted rotation blend's
  member selection.
- Bounded-by-construction families (trees: leaf averages; rbf kernels:
  revert-to-intercept) cannot blow up through a log-target inverse;
  unbounded ones (linear models, MLPs) can — ship MLPs with
  `clamp_output: true` until proven safe.
- Features usually beat architecture: scan dataset-level axes before deep
  hyperparameter scans.

## Known pitfalls per feature / dataset axis

- **Song-AE latent REJECTED for rotation (2026-06-09).** The precomputed 64-dim
  song-autoencoder latent (`spotify_tracks_song_ae`) carries no rotation signal
  that the artist/genre/album one-hot metadata does not already carry better.
  Standalone it is weak (kNN best 0.426, ridge 0.462, AE-only xgboost 0.410 vs
  metadata-only xgboost 0.388); fed into a unified xgboost it DILUTES (0.3965 vs
  0.3879); blended it gets zeroed out (w→0). This is a FOURTH dense modality to
  lose to categorical identity one-hots on this corpus — see the convergent
  finding below — now confirmed on the rotation target, not just engagement.
- **(HISTORICAL, engagement target) This project predicts engagement from
  INTRINSIC song metadata only.** The canonical engagement build keeps the
  `playback` group (play_count, completion_ratio, skip_rate, shuffle_rate,
  distinct_sessions) and the `curation` group (is_saved, in_playlist_count,
  on_repeat_count) OFF — the first two playback fields compose the target, and
  curation is behaviorally entangled. Turning any on is a soft-leakage control
  build. NOTE: rotation==play_count≥2 is ALSO derived from playback, so
  play_count/completion_ratio likewise stay OFF as features for rotation; the
  demoted `engagement` field stays OFF too (target-derived ⇒ leakage).
- **Content-text embedding tested and REJECTED (2026-06-08, engagement).** Built
  `spotify_tracks_content` — a multilingual sentence embedding
  (paraphrase-multilingual-MiniLM-L12-v2, 384-dim) of each track's
  intrinsic-metadata doc — to replace the behavioral co-listening vector. It did
  NOT help; a pca-dims scan got monotonically WORSE (xgboost content+one-hots MAE
  2.752 @ p16 → 2.830 @ p128 vs the **2.424 one-hot baseline**). Content
  embedding ALONE (p128) = MAE 2.964 / R² 0.144, barely above the median floor.
  Verdict: engagement is driven by artist/genre **identity** (top-60/top-40
  one-hots encode it losslessly), not semantic text similarity; the embedding
  blurs identity and adds noisy dims trees overfit.
- **Acoustic audio features tested and REJECTED (2026-06-08, engagement).** 11
  Spotify-style audio features from ReccoBeats, coverage 64.8%. Controlled test
  (same collection + seed-42 split, pca=1): audio OFF MAE **2.684** / R² 0.298
  vs audio ON MAE **2.775** / R² 0.260 — slightly WORSE; they diluted the
  artist/genre one-hots. Cross-collection numbers are NOT comparable (different
  scroll order ⇒ different split); the within-collection control is 2.684.
- **CONVERGENT FINDING (4 feature modalities + now a 5th MODEL family): the
  taste-fit signal is predicted by artist/genre IDENTITY, captured by one-hots and
  exploited by trees — not by descriptive attributes nor by distance/margin
  models.** Feature modalities that failed: co-listening embedding (~neutral,
  engagement), content-text embedding (hurt, engagement), acoustic features (hurt,
  engagement), song-AE latent (dilutes, ROTATION). And now a MODEL family fails the
  same way: kernel SVM (rbf/linear/poly/sigmoid) on rotation is worse-than-mean
  (R² < 0) — the dense-distance/large-margin regime blurs the sparse categorical
  identity that trees read losslessly, exactly the prior. `artist=__other__` is
  consistently the #1 feature on engagement. Implication: the ceiling is set by
  identity; the residual is behavioral/contextual that no song metadata captures.
  The remaining lever is identity RESOLUTION — raise vocab caps so the `__other__`
  tail gets real columns — NOT more feature modalities and NOT more model families.
  (Rotation noise band now MEASURED at 0.0086 MAE / 0.0126 AUC — see Noise bands.) 6th CONFIRMATION (2026-06-09, AUC classification): xgboost-classifier beats logistic + all 4 SVC kernels by 4.6–9.3 AUC bands on the honest classification metric — the identity-one-hots-and-trees verdict survives the metric change that exposed SVR's MAE artifact. 7th CONFIRMATION (2026-06-15, `2026-06-15-gbdt-leaf-head-ceiling-probe.md`): a learned head over one-hot tree leaves (the Facebook GBDT→LR trick, leaf-emitting tree frozen at the champion config) caps BELOW the bare tree by 1.8–2.5 bands (best leaf-head, linear, 0.6978 vs 0.7204) and is mis-calibrated — the dense head cannot recover the tree that generated its features. Same as the AE-latent dilution and the burn-deep dense-head cap: learning a head over a learned representation dilutes here. Trees scoring their own leaves directly remain the ceiling. 8th OBSERVATION (2026-06-15, `2026-06-15-vocab-cap-scan-gbdt.md`) — this is the FIRST campaign to actually WALK the "remaining lever is identity RESOLUTION — raise vocab caps" prescription on the bare GBDT, and it finds the lever **WEAK and approaching its ceiling past the first jump**: xgboost 60/40 0.72044 → 300/120 0.72988 (+0.75 band, the big jump, already on record) → 500/200 0.72966 (FLAT) → 1000/300 0.73490 (+0.40 band over the anchor, SUB-BAND new single-split best); three of four post-jump caps are sub-band and catboost confirms the same plateau. So the prior "identity resolution is the durable/remaining lever" still holds DIRECTIONALLY (more caps never HURT, the curve trends up) but is now QUALIFIED: raw CAP HEIGHT is a largely-spent lever on this corpus — the residual is behavioral/contextual that no song-metadata cardinality captures, and the next identity gains (if any) must come from resolution QUALITY, not bigger caps. No definition saved (sub-band, no 3-seed beat); the top-of-curve all-vocab point is UNMEASURED (build memory ceiling — see dataset-axis pitfalls). NUANCE (2026-06-12, `2026-06-12-burn-deep-topology.md`): a dense ANN that keeps the
  one-hots RAW (`burn-deep-classifier`) gets to AUC 0.7038 — only ~1 band below the trees,
  NOT the ~3.3 the sklearn pyramid showed. So part of the dense-family deficit on THIS
  corpus was a REPRESENTATION artifact (standardizing sparse one-hots into outlier spikes),
  not purely the dense-regime-blurs-identity prior. The prior still holds directionally
  (trees still win, embeddings don't beat raw one-hots at this cardinality) but the gap is
  representation-sensitive; the durable lever (identity resolution / vocab caps) is also the
  thing most likely to finally make the embedding topology pay off.
- The 200-dim co-listening embedding (`spotify_tracks`) is a BEHAVIORAL vector,
  not song metadata — the payload-based predictors (xgboost, lightgbm,
  random-forest, ridge, svm) are the canonical models there; embedding-based
  predictors would reintroduce behavioral signal.
- **Framework floor: every dataset carries ≥1 PCA embedding column.** The build
  rejects `pca_dims < 1` (server `api.rs`; `pca.rs` requires `k>=1`) and all
  predictors read the combined `features.f32` — there is no native
  zero-embedding / items.json-only feature path. The engagement canonical build
  uses `pca_dims=1` to reduce the behavioral channel to one PC. A truly
  metadata-only matrix would require a framework change; do NOT improvise it in
  the crates.
  - **MEASURED (bootstrap, engagement):** that 1 PC holds 13.8% of the 200-dim
    embedding's variance, yet xgboost gave `pca_0` only **0.98% of total gain,
    rank 27 of 111** — carried by `artist` and `genre_primary` one-hots. One PC
    at one split; a pca-dims scan (1→16→64→128, metadata fixed) is the definitive
    test.
- **DATASET BUILD MEMORY CEILING: a full-uncapped identity one-hot matrix (~6,600+ dense cols) FAILS to build on this host (MEASURED 2026-06-15, `2026-06-15-vocab-cap-scan-gbdt.md`).** The vocab-cap scan's top-of-curve point — artist 5594 / genre 962 (full uncapped vocab → ~6,633 cols) on the 23,529-row rotation corpus — failed to materialize on TWO `POST /api/datasets` attempts: each returned a build_id but the dataset never registered, no build-run dir survived, and the build python died mid-materialization. Root cause is host memory, not the recipe: a 23,529 × 6,633 float32 DENSE matrix is ~0.62 GB, but PCA + one-hot intermediates plus the train/test copies push the PEAK well over the ~2.3 GB free (of 15 GB total) available under concurrent training load. The build path is DENSE (`features.f32`) — there is no sparse one-hot option — so col count × rows is a hard wall. Practical guidance: cap artist one-hots at ≤1000 / genre ≤300 (1377 cols builds fine) under normal load; build very-wide matrices only on a quiescent host, and if a build_id returns but no dataset appears within ~10 min and no build process is alive, treat it as an OOM failure, not a slow build. Building cheap→expensive (P1→P2→P3) isolates the failure to the widest point and protects the interior of any cardinality scan.

## Interpretability config — `[interp] segment_field` is UNDETERMINED (2026-07-12)

The SAE / interpretability capability synced from lensing `7dc5860` (the
`information-capture-analyst` agent + `information-capture` skill, backed by
`POST /api/interp/model-sae`) segments its per-segment representation analysis
by a `[interp] segment_field` key in `domain.toml`. **That key is deliberately
NOT set yet** — `domain.toml` has no `[interp]` section, so the server falls
back to the first `categorical`-role field automatically. This is intentional,
not an oversight.

- **DIRECTIVE for agents:** treat `[interp] segment_field` as *to be
  determined*. Do not silently assume the fallback field is the right one, and
  do not invent a value. When an interpretability run needs a segment axis,
  surface that the field is undetermined and apply the criterion below (or ask).
- **CRITERION for choosing it:** the segment field should be **the field that
  drives errors the most** — i.e. the categorical axis along which the model's
  error concentrates / that best explains residual variance, not just any
  categorical field. Determine it empirically (e.g. per-segment error breakdown
  on a promoted model) before writing it into `domain.toml`.
- Given the on-record CONVERGENT FINDING that taste-fit here is driven by
  artist/genre **identity** one-hots, an identity field (artist or
  genre_primary) is the leading a-priori candidate — but this must be
  *confirmed* against the drives-errors-most criterion, not assumed.
- To set it once determined: add to `domain.toml`
  `\n[interp]\nsegment_field = "<field>"` (optional key, `schema_version`
  stays 1). Until then the automatic first-categorical fallback stands.

## Taste-fit reframe for Pathfinder (2026-06-08)

The model's ultimate consumer is the **Pathfinder** A* search, where it supplies
the `W_FIT` term: `cost(u→v) += W_FIT·(1 − fit_norm(v))` — a per-track [0,1]
desirability that biases the playlist path. `pathfinder/score.py` calls
`POST /api/models/<MODEL_NAME>/predict {point_ids}` and min-max normalizes.

**Reframe:** model **"enters rotation" = play_count ≥ 2** (binary, 44.5% base
rate) instead of engagement magnitude. On the one-hot feature set, xgboost gets
**AUC 0.765 / acc 0.715** — a genuinely useful taste-fit ranker, vs engagement
regression's R²≈0.3. The predicted probability is naturally spread
(p10/p50/p90 = 0.16/0.41/0.81), a cleaner Pathfinder `fit_norm` with no
log-hack. (Engagement stays the documented scientific field; rotation is the
actionable signal that feeds the path search.)

**Pivot executed (2026-06-08):** domain `[target]` = `rotation` (transform none),
engagement demoted to a documented non-target field; `rotation` (play_count>=2)
written to `spotify_tracks_song_ae` payloads; server repointed to that AE
collection (so Pathfinder's graph = the rich song-character manifold). Rotation
xgboost on AE-collection features: **AUC 0.733, acc 0.690, spread 0.16/0.41/0.80**
(predictions saved; AUC computed offline). Build with
`quality.nonpositive_price = FALSE` (rotation==0 is the negative class, not an
outlier). Artifacts: ds-20260608-204402-p64-s42 / run-20260608-204415-6eeda-xgboost.

**FRAMEWORK LIMITATION SURFACED — binary targets break the metric pipeline.**
The predictors + lensing-core compute MAPE/medAPE = |err/actual|, undefined when
actual==0 (half the rotation labels). The run TRAINS fine (exit 0, good
predictions) but is marked `failed` on the NaN metric, blocking auto-promote →
`/api/models/<name>/predict` → Pathfinder's `score.py`. Native classification
isn't supported. Per the ground rules (crates/predictors shouldn't be hacked),
resolution is a DECISION, not an improvisation:
  (a) minimal upstream fix: make the % metrics zero-safe (skip actual==0) so
      binary runs succeed + promote normally — contribute upstream; OR
  (b) self-contained Pathfinder scorer: load the rotation model.ubj, score corpus
      point_ids, feed `score.py` directly (bypass the metric/promotion path).
  NOTE (2026-06-09): the autoencoder-xgboost campaign's 6 runs all reported
  finite MAPE/medAPE (~43.9%/47.4%) and were marked `succeeded` — that predictor
  evidently guards the zero-actual case, so the limitation is per-predictor, not
  universal. The native `xgboost` predictor still trips it (the seed run failed
  on NaN).

## Best-models group curation (best-model-selector agent)

Additive log of judgment-pass overrides applied via `PUT /api/best-models`
(the deterministic top-12-by-MAE recompute can't apply these). Newest first.

### 2026-06-15 (latest) — seventh judgment pass after the GBDT-leaf-head ceiling-probe campaign (`experiments/2026-06-15-gbdt-leaf-head-ceiling-probe.md`)

State on entry: group full (12/12), 64 prior exclusions intact, no pins. The recompute was
fresh (selected_at 2026-06-15T16:00:00 — no manual recompute needed). The REFUTED leaf-head
campaign added 4 runs on the canonical full-64-d matrix `ds-20260609-204419-p64-s42` (seed 42)
and the deterministic AUC top-12 had seated TWO of them: the C0 control `f2911` (AUC 0.7204383403100314)
at rank 7 — byte-identical to the already-seated new-corpus champion `439d8` at rank 6 (a
determinism duplicate of the same config, one model in two slots) — and the best leaf-head arm
`b4db5` (linear head, AUC 0.6978, logloss 1.31) at rank 11. Verified all 4 runs against
`GET /api/runs` (auc/logloss/n_test 4,706) and the campaign config table before excluding. No
suspicious metrics (no leakage signature; the high leaf-head logloss is the documented sklearn
over-confidence signature — AUC only modestly behind while logloss far above the trees — NOT a
near-perfect-AUC / tiny-n_test red flag).

**Excluded (4 new run ids):**

1. **C0 determinism-duplicate of the seated champion → de-dup (`f2911`).**
   `run-20260615-151551-f2911-xgboost-classifier` (AUC 0.7204383403100314) is byte-identical to
   the seated new-corpus champion `run-20260609-204726-439d8-xgboost-classifier` (rank 6, backs
   the `xgboost-one` definition) — same config (n_est 500/depth 6/lr 0.05/sub-col 0.8/seed 42),
   same matrix, all 16 digits identical. One model, not two slots; excluded the duplicate, kept
   439d8 as the canonical new-corpus xgboost representative.
2. **3 REFUTED, mis-calibrated leaf-head arms (closed family).** The campaign verdict is REFUTED:
   every head caps 1.8-2.5 bands BELOW the bare xgboost champion (0.7204) AND is over-confident
   (logloss 0.82-1.31 / brier 0.26-0.30 vs the tree's 0.61 / 0.21). Excluded all three — same
   disposition as the under-trained/closed pyramid-mlp and burn-deep siblings: `b4db5` (linear
   head, l2 1.0, 0.6978 — best leaf-head, was at rank 11), `44b1d` (mlp[256,128], 0.6921),
   `9d655` (mlp[128], 0.6897). The leaf one-hots are a strictly lossier view of the tree than the
   tree's own additive leaf-value scoring; a head re-estimates it worse (the 7th convergent
   confirmation: a learned head over a learned representation dilutes here). Family CLOSED for
   rotation; no definition was saved. Revisit trigger: only if a re-regularized (higher-l2)
   head ever closes the ~1.8-band AUC deficit — calibration alone cannot.

**IN-FLIGHT campaign left untouched (out of scope, flagged for audit).** A separate
`burn-ae-classifier` campaign is RUNNING concurrently on `ds-20260609-204419-p64-s42` (4 runs
`running` at 16:01, ~13 already `succeeded`, AUC ~0.692-0.697). Removing the f2911 + b4db5 slots
let the deterministic recompute backfill ranks 10-12 with three succeeded burn-ae runs (`21f11`
0.6972, `05bce` 0.6971, `e5a75` 0.6966). I did NOT curate the burn-ae runs: the campaign has not
concluded, the runs are legitimately `succeeded`, and curating an in-flight campaign's members
(de-dup / family-monoculture / save-bar judgments) before its report lands would be premature —
they get their own selector pass when that campaign finishes. Likewise the burn-deep high-vocab
follow-up runs now at ranks 4 (`a33b0`, 0.7281) and 8 (`751a2`, 0.7179) postdate the sixth pass
and are a separate burn-deep follow-up — left in place, out of scope for this leaf-head pass.

**Kept / left alone:** all 64 prior exclusions intact (verified the new exclusion list = 68 ids);
no pins (the leaf-head family saved no promoted definition; the champion 439d8 is the auto-seated
representative); no unexcludes/unpins. The 0.7204 new-corpus xgboost champion holds rank 6, the
cross-matrix DS-META 0.7429 champion holds rank 1, and the tree + burn-deep diversity is intact.

**Result: group = 12 entries, 68 total exclusions, no pins.** Predictor families seated:
xgboost-classifier (×3, cross-matrix), burn-deep-classifier (×3, high-vocab), catboost-classifier
(×2), burn-ae-classifier (×3, in-flight backfill).

| rank | model | predictor | AUC | matrix | note |
|---|---|---|---|---|---|
| 1 | best-xgboost-classifier-…-627d6 | xgboost-classifier | 0.7429 | OLD DS-META | registered DS-META champion `xgboost-classifier-rotation` (s42); cross-matrix |
| 2 | best-xgboost-classifier-…-fffa4 | xgboost-classifier | 0.7350 | OLD p64 (178) | AE-latent+metadata dilution arm; cross-matrix |
| 3 | best-xgboost-classifier-…-02fda | xgboost-classifier | 0.7299 | HIGH-VOCAB (497) | high-vocab in-dataset champion |
| 4 | best-burn-deep-classifier-…-a33b0 | burn-deep-classifier | 0.7281 | HIGH-VOCAB (497) | burn-deep high-vocab follow-up (out of scope this pass) |
| 5 | best-catboost-classifier-…-8f15e | catboost-classifier | 0.7252 | HIGH-VOCAB (497) | 2nd tree family, high-vocab |
| 6 | best-xgboost-classifier-…-439d8 | xgboost-classifier | 0.7204 | 60/40 full-64-d (177) | new-corpus champion; backs `xgboost-one`; KEPT over the de-duped C0 twin f2911 |
| 7 | best-burn-deep-classifier-…-e8db6 | burn-deep-classifier | 0.7192 | HIGH-VOCAB (497) | SAVED high-vocab ANN (`burn-deep-embeddings-highvocab`) |
| 8 | best-burn-deep-classifier-…-751a2 | burn-deep-classifier | 0.7179 | HIGH-VOCAB (497) | burn-deep high-vocab follow-up (out of scope this pass) |
| 9 | best-catboost-classifier-…-c7ef6 | catboost-classifier | 0.7111 | 60/40 full-64-d (177) | 2nd tree family on 60/40 matrix |
| 10 | best-burn-ae-classifier-…-21f11 | burn-ae-classifier | 0.6972 | 60/40 full-64-d (177) | IN-FLIGHT campaign backfill — left alone, gets its own pass |
| 11 | best-burn-ae-classifier-…-05bce | burn-ae-classifier | 0.6971 | 60/40 full-64-d (177) | IN-FLIGHT campaign backfill — left alone |
| 12 | best-burn-ae-classifier-…-e5a75 | burn-ae-classifier | 0.6966 | 60/40 full-64-d (177) | IN-FLIGHT campaign backfill — left alone |


### 2026-06-12 (latest) — sixth judgment pass after the HIGH-VOCAB topology campaign (`ds-20260612-143014-p64-s42`)

State on entry: group full (12/12), 55 prior exclusions intact, no pins. Recompute was
fresh (selected_at 2026-06-12T14:35:24 matched the last high-vocab run finish — no manual
recompute needed). The high-vocab topology campaign (`experiments/archive/2026-06-12-high-vocab-topology.md`)
landed 11 runs on a THIRD, non-comparable feature matrix `ds-20260612-143014-p64-s42`
(same 23,529 rows / seed 42 / PCA64 + numerics as the 60/40 full-64-d matrix; identity
vocab raised to artist 300 / genre 120 → 426 one-hot cols). The deterministic AUC top-12
seated NINE of the 11 high-vocab runs (xgboost 02fda 0.7299 at rank 3, catboost 8f15e
0.7252 at rank 4, then SEVEN burn-deep embeddings cells across ranks 5,6,8,9,10,11,12),
crowding the new-corpus xgboost `xgboost-one`/439d8 down to rank 7 and pushing ngboost +
the entire OLD-DS-META dense tail (svc ×4 + logistic) OUT of the group — a burn-deep
near-monoculture (7/12 slots) of one family on one matrix, mostly embed_dim×lr tune cells
spanning a sub-band 0.7183–0.7228, plus the e8db6/b717e determinism twin. No suspicious
metrics (n_test 4,706, logloss ~0.61 in the trees' calibrated regime, no leakage
signature). The group spans THREE non-comparable matrices (OLD DS-META, 60/40 full-64-d,
HIGH-VOCAB full-64-d); a single AUC sort mixes them — curation can de-dup and restore
family/matrix-variant diversity but cannot make the cross-matrix ranking like-for-like.

**Excluded (9 new run ids) — verified each against `GET /api/runs` auc/logloss + the
campaign config table before excluding:**

1. **High-vocab burn-deep family de-dup → kept ONE representative (e8db6).** All 9 high-vocab
   burn-deep runs are the same family on the same matrix; 7 are embeddings embed_dim×lr tune
   cells (sub-band 0.7183–0.7228) and the burn binary is near- not bit-deterministic
   (e8db6 0.71923 vs its twin b717e 0.71919 = ed16/lr0.001, Δ 0.00004 ≈ 0.006 band — NOT a
   distinct better model). Kept `run-20260612-143411-e8db6-burn-deep-classifier` (embeddings,
   ed16, lr0.001, **AUC 0.7192**) — the campaign's SAVED Phase-1-winner config
   (`burn-deep-embeddings-highvocab`, the FIRST ANN to clear the rotation save bar) per the
   runner's decision rule — as the single high-vocab burn-deep representative. Excluded the
   other 8: the determinism twin `b717e` (ed16/lr0.001, 0.71919), the Phase-2 best `3310f`
   (ed8/lr0.0005, 0.7228) and the rest of the embed_dim×lr grid `d9bf7` (ed16/lr0.0005, 0.7210),
   `75804` (ed32/lr0.0005, 0.7200), `0b77d` (ed8/lr0.001, 0.7193), `5f0f2` (ed32/lr0.001, 0.7183),
   plus the two non-embeddings topologies `01d7b` (mlp, 0.7129) and `7d117` (wide_deep, 0.7061).
   NOTE: I kept the SAVED config (e8db6) over the marginally higher Phase-2 cell 3310f
   (0.7228, +0.0036, in-band) — the saved-definition config is the canonical representative;
   the +0.0036 edge is sub-band and the per-arm topology margins are 3-split-confirm-pending.
   e8db6's model record is NOT flagged `promoted`, so it cannot be PINNED — excluding its 8
   siblings keeps it auto-seated without a pin (same mechanism as the prior burn-deep pass).
2. **60/40 burn-deep representative superseded (`6575a`).** `run-20260612-140211-6575a-burn-deep-classifier`
   (the prior pass's single burn-deep rep: mlp/lr0.0005/[128,64] on the 60/40 matrix
   `ds-20260609-204419-p64-s42`, **AUC 0.7038**) is the weaker matrix-variant of a family whose
   better representative (e8db6, 0.7192, high-vocab) is now seated. Keeping both would put the
   same family twice across two near-identical matrices (same corpus/rows/split, vocab the only
   difference) and — per the projection — would re-seat 6575a at rank 7, displacing a distinct
   dense-tail family. Excluded so burn-deep holds exactly one (best-matrix) slot. Revisit
   trigger: unexclude 6575a only if the 60/40 matrix becomes the active comparison axis again.

**De-dup judgment that did NOT trigger an exclusion (kept, stated for audit):** the group now
holds catboost on BOTH the high-vocab matrix (`8f15e`, 0.7252, rank 4) and the 60/40 matrix
(`c7ef6`, 0.7111, rank 7). Unlike burn-deep, catboost is only a 2-cell family (no monoculture)
and excluding c7ef6 would free a slot that only the documented-worst svc-sigmoid (0.6255) could
backfill — a strictly worse outcome. Both catboost cells are distinct, non-redundant tree
members on distinct matrices, so both are KEPT for cross-matrix tree representation.

**Crowded-out distinct families restored (no action needed beyond the burn-deep thinning):**
removing the 8-cell burn-deep flood let the new-corpus xgboost champion `xgboost-one`/439d8
(0.7204) rise to rank 5, ngboost `d725b` (0.6917) re-seat at rank 8, and the OLD-DS-META dense
tail (svc-poly 848b8 0.6855, logistic 8e4d9 0.6762, svc-rbf a704f 0.6722, svc-linear 45798 0.6363)
re-fill ranks 9–12 — restoring the 7-family diversity the deterministic sort had collapsed.
The documented-worst svc-sigmoid (53bb5, 0.6255) now falls to candidate #13 and naturally
drops OUT (the group is full at 12 with strictly better-diversified members) — no longer needs
a standing "leave alone" note. All 55 prior exclusions left intact; no pins, no unexcludes.

**Result: group = 12 entries, 7 distinct predictor families across 3 non-comparable matrices,
64 total exclusions, no pins.** One representative per (family, matrix-variant) except catboost
(kept on both matrices, justified above). The high-vocab board's in-dataset leader is the
high-vocab xgboost 02fda (0.7299, rank 3); the saved high-vocab ANN e8db6 (0.7192) holds the
burn-deep/ANN-diversity slot at rank 6.

| rank | model | predictor | AUC | matrix | note |
|---|---|---|---|---|---|
| 1 | best-xgboost-classifier-…-627d6 | xgboost-classifier | 0.7429 | OLD DS-META | registered DS-META champion `xgboost-classifier-rotation` (s42); cross-matrix |
| 2 | best-xgboost-classifier-…-fffa4 | xgboost-classifier | 0.7350 | OLD p64 (178) | AE-latent+metadata dilution arm; cross-matrix |
| 3 | best-xgboost-classifier-…-02fda | xgboost-classifier | 0.7299 | HIGH-VOCAB (497) | high-vocab in-dataset champion |
| 4 | best-catboost-classifier-…-8f15e | catboost-classifier | 0.7252 | HIGH-VOCAB (497) | 2nd tree family, high-vocab (−0.0047 vs xgboost, in-band) |
| 5 | best-xgboost-classifier-…-439d8 | xgboost-classifier | 0.7204 | 60/40 full-64-d (177) | new-corpus champion; backs the `xgboost-one` definition |
| 6 | best-burn-deep-classifier-…-e8db6 | burn-deep-classifier | 0.7192 | HIGH-VOCAB (497) | SAVED high-vocab ANN config (`burn-deep-embeddings-highvocab`); ANN-diversity slot; first ANN over the save bar |
| 7 | best-catboost-classifier-…-c7ef6 | catboost-classifier | 0.7111 | 60/40 full-64-d (177) | 2nd tree family on 60/40 matrix; distinct, kept |
| 8 | best-ngboost-classifier-…-d725b | ngboost-classifier | 0.6917 | 60/40 full-64-d (177) | 3rd-tier family, ~2.3 bands back |
| 9 | best-svc-…-848b8 | svc (poly d3) | 0.6855 | OLD DS-META | dense-tail member; cross-matrix |
| 10 | best-logistic-…-8e4d9 | logistic | 0.6762 | OLD DS-META | dense-tail member; cross-matrix |
| 11 | best-svc-…-a704f | svc (rbf) | 0.6722 | OLD DS-META | dense-tail member |
| 12 | best-svc-…-45798 | svc (linear) | 0.6363 | OLD DS-META | dense-tail member |

### 2026-06-12 (later) — fifth judgment pass after the burn-deep-classifier topology campaign (full-64-d, new corpus)

State on entry: group full (12/12), 41 prior exclusions intact, no pins. Recompute was
fresh (updated_at 2026-06-12T14:02:27 matched the last burn-deep run finish — no manual
recompute needed). The burn-deep-classifier topology bake-off + mlp tune
(`2026-06-12-burn-deep-topology.md`) added 9 runs of a NEW Rust/burn predictor family on
the NEW 23,529-row full-64-d dataset `ds-20260609-204419-p64-s42`, and the deterministic
AUC top-12 seated EIGHT of them into ranks 5-12 — a near-monoculture by one new family,
most of the eight being lr×hidden tune cells differing only in hyperparameters (the
correlated-vote / family-monoculture risk). No champion changed (xgboost-classifier 0.7204
remains the new-corpus rotation leader; the cross-corpus 0.7429 DS-META champion stays
rank 1). No suspicious metrics (n_test 4,706, logloss ~0.62 in the trees' regime, no
leakage signature).

**Excluded (14 new run ids) — two groups, each verified against `GET /api/runs` auc/logloss
fields and the campaign report config table before excluding:**

1. **Burn-deep family de-dup → kept ONE representative (6575a).** All 9 burn-deep runs are
   the same family on the same corpus; 6 of the Phase-2 cells differ from each other only
   in lr/hidden, and the burn binary is near- but not bit-deterministic (the lr0.001/[128,64]
   config differs ~0.0004 AUC across runs — sub-band, NOT a distinct better model). Kept
   `run-20260612-140211-6575a-burn-deep-classifier` (mlp, lr0.0005/[128,64], **AUC 0.7038**,
   logloss 0.6221 — the campaign best) as the single burn-deep representative; excluded the
   other 8: the Phase-1 winner `41fc9` (mlp lr0.001/[128,64], 0.7001 — a sub-band near-dup of
   6575a, Δ 0.0037 < 0.0126 band), Phase-1 `0eda4` (embeddings, 0.6934) and `e85be` (wide_deep,
   0.6918), and the 5 remaining Phase-2 tune cells `21ad5` (0.6997), `18c9e` (0.6996),
   `fecb0` (0.6966), `4c359` (0.6963), `0f4e1` (0.6957). One model per family slot, not eight.
2. **6 broken-calibration sklearn pyramid-mlp arch-scan cells (closed family).** Removing the
   burn-deep flood exposed a pre-existing same-family crowding: the 6-arm pyramid-mlp
   architecture scan (`2026-06-12-pyramid-mlp-arch-scan.md`, ts 123155) backfilled into ranks
   8-12, all AUC 0.6712-0.6792 with logloss **2.06-2.78** (the documented broken-calibration
   ceiling, NOT a budget artifact — every arm self-stops at 73-100 iters). The prior pass had
   excluded only the bake-off pyramid cell `ea698`; these 6 arch-scan cells were never reached
   because the xgboost monoculture had filled the slots. With burn-deep now the
   calibration-FIXED dense/ANN family of record (logloss ~0.62, which SUPERSEDES the sklearn
   pyramid per the family-pitfall revision), keeping 4-6 broken-calibration sklearn cells in the
   group is a redundant, badly-calibrated same-family monoculture that only adds correlated
   near-floor votes to a consensus predict. Excluded all 6: `0ace5` (0.6792), `8bfab` (0.6781),
   `d3834` (0.6777), `79128` (0.6737), `0df76` (0.6736), `01f07` (0.6712).

**burn-deep earns the ANN/dense diversity slot (rank 5, no pin).** burn-deep-classifier is a
genuinely new family — the ONLY calibrated dense-net on record (logloss ~0.62 / brier ~0.22 vs
the sklearn pyramid's 2.0-2.8 / 0.33). Its best (0.7038) sits ABOVE ngboost (0.6917) and the
sklearn pyramid, ~0.6 band below catboost (0.7111) and ~1.3 below xgboost (0.7204) — a
defensible margin for a distinct-inductive-bias diversity member, NOT an AUC contender. It is
held as the auto-selected rank-5 member (no pin: pins require a promoted model and 6575a backs
no registered definition; excluding its 8 siblings keeps it seated without one). Revisit
trigger: unexclude the embeddings cell and re-run the topology bake-off after raising identity
vocab caps (embeddings are likely starved by the low 61/41/4 cardinality).

**Kept for family diversity / left alone:** ngboost (`d725b`, 0.6917, ~2.3 bands back — 3rd
family) backfilled to rank 6, kept. The OLD-corpus dense tail (svc ×4 + logistic, ranks 7-11,
AUC 0.6255-0.6855) carries the standing prior-pass note (distinct families, genuine AUC-rankable,
cross-corpus DS-META, no duplicates) — left in place; exclude the bottom SVC kernels only if a
consumer's consensus predict drifts toward chance. All 41 prior exclusions left intact.

**Note on a transient mistake (corrected in-pass):** the pyramid-mlp exclusion was first applied
with a wrong timestamp prefix (`115155` instead of the real `123155`), which added 6 nonexistent
ids to the exclusion list with no effect on the group. Those 6 phantom ids were `unexclude`d in
the same corrective PUT that added the correct `123155` ids — the final exclusion list contains
only real run ids.

**Result: group = 11 entries (6 distinct predictor families: xgboost ×3 cross-corpus, catboost,
burn-deep, ngboost, svc, logistic), 55 total exclusions, no pins.** The new-corpus picture is now
one representative per family in the 0.69-0.72 tier (xgboost 0.7204, catboost 0.7111, burn-deep
0.7038, ngboost 0.6917) plus the cross-corpus DS-META leaders and dense tail.

| rank | model | predictor | AUC | corpus | note |
|---|---|---|---|---|---|
| 1 | best-xgboost-classifier-…-627d6 | xgboost-classifier | 0.7429 | OLD DS-META | registered DS-META champion `xgboost-classifier-rotation` (s42); cross-corpus |
| 2 | best-xgboost-classifier-…-fffa4 | xgboost-classifier | 0.7350 | OLD p64 (178) | distinct: AE-latent+metadata dilution arm; cross-corpus |
| 3 | best-xgboost-classifier-…-439d8 | xgboost-classifier | 0.7204 | NEW full-64-d | NEW-corpus champion; backs the `xgboost-one` definition |
| 4 | best-catboost-classifier-…-c7ef6 | catboost-classifier | 0.7111 | NEW full-64-d | 2nd tree family, in-band (−0.0093 < band) |
| 5 | best-burn-deep-classifier-…-6575a | burn-deep-classifier | 0.7038 | NEW full-64-d | NEW family: calibration-fixed ANN representative (logloss 0.62); ~1 band below trees |
| 6 | best-ngboost-classifier-…-d725b | ngboost-classifier | 0.6917 | NEW full-64-d | 3rd-tier family, ~2.3 bands back |
| 7 | best-svc-…-848b8 | svc (poly d3) | 0.6855 | OLD DS-META | prior-pass dense-tail member; cross-corpus |
| 8 | best-logistic-…-8e4d9 | logistic | 0.6762 | OLD DS-META | prior-pass dense-tail member; cross-corpus |
| 9 | best-svc-…-a704f | svc (rbf) | 0.6722 | OLD DS-META | prior-pass dense-tail member |
| 10 | best-svc-…-45798 | svc (linear) | 0.6363 | OLD DS-META | prior-pass dense-tail member |
| 11 | best-svc-…-53bb5 | svc (sigmoid) | 0.6255 | OLD DS-META | prior-pass dense-tail member (documented worst) |


### 2026-06-12 — fourth judgment pass after the four-family classifier bake-off (full-64-d, new corpus)

State on entry: group full (12/12), 30 prior exclusions intact, no pins. Recompute
was fresh (updated_at 2026-06-12T12:17:04 matched the last campaign run finish — no
manual recompute needed). The bake-off + sweep added 13 runs on the NEW 23,529-row
full-64-d dataset `ds-20260609-204419-p64-s42` and saved one named definition. The
deterministic AUC top-12 had filled 11 of 12 slots with xgboost-classifier (a
near-total monoculture): the OLD-corpus 0.7429 / 0.7350 members at ranks 1-2, then
the new-corpus xgboost default config carried THREE times (determinism triplet), the
single-seed unconfirmed sweep best, and six more depth×lr sweep cells — with catboost
clinging to rank 12 and ngboost/pyramid-mlp pushed out entirely.

Registry note (corrects the report): the saved definition is named **`xgboost-one`**
(predictor xgboost-classifier), backed by run **`run-20260609-204726-439d8`**
(depth6/lr0.05, the new-corpus default config, AUC 0.7204) — NOT
`xgboost-classifier-rotation-p64` as the report states. 439d8 is therefore the
canonical new-corpus representative kept in the group. (It is a named definition but
not flagged `promoted`, so it is held as the auto-selected member, not a pin — pin
semantics require a promoted model.)

**Excluded (11 new run ids) — three groups, each verified against `GET /api/runs`
auc fields and the campaign report config table before excluding:**

1. **Determinism triplet → kept ONE representative (439d8).** The new-corpus
   depth6/lr0.05 config was run three times, all AUC 0.7204383403100314 byte-identical:
   the pre-existing reference 439d8, the Phase-1 C0 (840c0), and the Phase-2 cell (dceaa).
   Kept 439d8 (the in-dataset reference AND the run backing the `xgboost-one` definition);
   excluded `run-20260612-115909-840c0-xgboost-classifier` and
   `run-20260612-120207-dceaa-xgboost-classifier` — one model, not three slots.
2. **Redundant same-family/same-corpus depth×lr sweep cells (kept 439d8 as the single
   xgboost new-corpus representative).** Seven sweep cells differ from the kept default
   only in depth/lr — near-identical configs of the same family on the same corpus that
   contribute only correlated votes to a consensus predict (the family-monoculture risk):
   `12f3a` (d6/lr.03, 0.7205), `2ebc3` (d8/lr.05, 0.7196), `a0c91` (d8/lr.1, 0.7173),
   `36844` (d6/lr.1, 0.7153), `76da0` (d4/lr.1, 0.7119), `dccde` (d4/lr.05, 0.7094),
   `81063` (d4/lr.03, 0.7074).
3. **Single-seed, unconfirmed sweep best (1).** `run-20260612-120207-2dd3d-xgboost-classifier`
   (depth8/lr0.03, AUC 0.7261) is +0.0057 over the default — INSIDE the 0.0126 AUC band
   at ONE seed, explicitly flagged by the campaign for a 3-seed confirm. Not suspicious
   (n_test 4,706, no leakage signature) but "likely better, not confirmed"; excluded so it
   does not masquerade as a distinct better model nor crowd a slot before confirmation.
   Revisit trigger: unexclude (and consider re-saving `xgboost-one` at depth 8) if the
   3-seed study clears the re-measured band.
4. **Weak pyramid-mlp — broken calibration (NOT a fixable artifact).** `run-20260612-115909-ea698-pyramid-mlp-classifier`
   AUC 0.6781 with logloss **2.4085** / brier 0.3251. The 6-arm arch scan
   (`2026-06-12-pyramid-mlp-arch-scan.md`, max_iter 800) confirmed this is a genuine
   calibration ceiling, not a 300-iter-cap artifact — every geometry self-stops at
   73–100 iters and the best converged MLP only reaches 0.6792. Its AUC is genuine
   ranking signal but its calibration is broken; **keep excluded** (raising the budget
   does not help). Revisit trigger: only if an lr/alpha-reworked or calibration-wrapped
   pyramid lands materially better — geometry/budget are exhausted.

**Kept for family diversity / left alone:** catboost (`c7ef6`, 0.7111 — the legitimate
2nd tree family, in-band) was rescued from rank 12 to rank 4 simply by removing the
xgboost crowding (no pin needed; pins require a promoted model and catboost is not a
registered definition). ngboost (`d725b`, 0.6917, ~2.3 bands back) backfilled in as a
genuine 3rd family — kept. The OLD-corpus dense tail (svc ×4 + logistic, ranks 6-10,
AUC 0.6255–0.6855) carries the prior pass's standing "leave alone unless consensus
drifts toward chance" note — distinct families, genuine AUC-rankable members, no
duplicates/artifacts — so left in place. The 30 prior exclusions were all left intact.
No pins, no unexcludes, no unpins.

**Cross-corpus caveat (carried, not resolved by curation).** Ranks 1-2 (627d6 0.7429,
fffa4 0.7350) live on the OLD DS-META / 178-col p64 corpora and the dense tail (ranks
6-10) on OLD DS-META; ranks 3-5 (439d8, catboost, ngboost) are the NEW 23,529-row
full-64-d corpus. The deterministic AUC sort ranks the old-corpus 0.7429 above the
new-corpus members, which is NOT a like-for-like ordering (different corpus size AND
feature set — see the new-corpus AUC leaderboard above). Curation cannot fix a
cross-corpus sort; flagged so consumers do not read rank-1 as "best on the current
corpus". The new-corpus champion is rank 3 (439d8 / `xgboost-one`, AUC 0.7204).

**Result: group = 10 entries (4 distinct predictor families), 41 total exclusions, no pins.**

| rank | model | predictor | AUC | corpus | note |
|---|---|---|---|---|---|
| 1 | best-xgboost-classifier-…-627d6 | xgboost-classifier | 0.7429 | OLD DS-META | registered DS-META champion `xgboost-classifier-rotation` (s42); cross-corpus |
| 2 | best-xgboost-classifier-…-fffa4 | xgboost-classifier | 0.7350 | OLD p64 (178) | distinct: AE-latent+metadata dilution arm; cross-corpus |
| 3 | best-xgboost-classifier-…-439d8 | xgboost-classifier | 0.7204 | NEW full-64-d | NEW-corpus champion; backs the `xgboost-one` definition; determinism-triplet representative |
| 4 | best-catboost-classifier-…-c7ef6 | catboost-classifier | 0.7111 | NEW full-64-d | 2nd tree family, in-band (−0.0093 < band) |
| 5 | best-ngboost-classifier-…-d725b | ngboost-classifier | 0.6917 | NEW full-64-d | 3rd family, ~2.3 bands back |
| 6 | best-svc-…-848b8 | svc (poly d3) | 0.6855 | OLD DS-META | prior-pass dense-tail member; cross-corpus |
| 7 | best-logistic-…-8e4d9 | logistic | 0.6762 | OLD DS-META | prior-pass dense-tail member; cross-corpus |
| 8 | best-svc-…-a704f | svc (rbf) | 0.6722 | OLD DS-META | prior-pass dense-tail member |
| 9 | best-svc-…-45798 | svc (linear) | 0.6363 | OLD DS-META | prior-pass dense-tail member |
| 10 | best-svc-…-53bb5 | svc (sigmoid) | 0.6255 | OLD DS-META | prior-pass dense-tail member (documented worst) |

### 2026-06-09 (latest) — third judgment pass after the rotation-classification (AUC) campaign

State on entry: group full (12/12), 25 prior exclusions intact, no pins. Recompute
was fresh (updated_at matched the last run finish — no manual recompute needed). The
framework had just pivoted to classification mode (domain.toml task=binary,
primary=AUC, higher-better). The deterministic AUC recompute correctly seated the 9
genuine AUC-rankable classifier runs at ranks 1-9, but backfilled ranks 10-12 with
stale promoted regression runs whose "AUC" is a degenerate `prev_value` fallback.

**Excluded (5 new run ids) — verified each candidate's on-disk
`data/runs/<id>/metrics.json` (auc field) before excluding, not by name:**

1. **3 stale regression runs polluting ranks 10-12 — NOT AUC-rankable.** Each has
   NO `auc` field; the group's "AUC" value is the run's old MAE carried in via the
   promoted-model `prev_value` fallback on the metric-regime change (regression→AUC):
   - `run-20260608-205802-48044-xgboost` — no auc; "AUC" 0.4098 == its mae (AE-latent-only regressor)
   - `run-20260608-205512-593d5-xgboost` — no auc; "AUC" 0.3965 == its mae (latent+metadata regressor)
   - `run-20260609-141717-fcbbe-autoencoder-xgboost` — no auc; "AUC" 0.3964 == its mae (w=0 champion seed point)
   These are MAE-scored models on a different (regression) axis; they are not comparable
   to AUC and would poison any consensus AUC predict. Excluded.
2. **2 redundant champion control-seed duplicates (kept 627d6/s42 as the single
   representative).** The xgboost-classifier champion config was run on three split
   seeds; ranks 3-4 were the s101 and s17 points of the SAME config (DS-META, identical
   hyperparams), i.e. the AUC noise-band data (mean 0.7360, σ 0.00628, 2σ 0.0126), not
   distinct models:
   - `run-20260609-172933-6df93-xgboost-classifier` — DS-META split s101, AUC 0.7345
   - `run-20260609-172933-684f2-xgboost-classifier` — DS-META split s17, AUC 0.7307
   Kept rank-1 627d6 (s42, AUC 0.7429, the registered `xgboost-classifier-rotation`
   champion) as the single representative. NOTE: rank 2 (fffa4, AUC 0.7350) was KEPT —
   it is a DISTINCT model (178-col AE-latent+metadata, the latent-dilution arm), not a
   seed duplicate of the champion. This extends the redundant-seed-point de-dup rule
   from the prior passes to the classifier campaign.

**The 25 prior exclusions were left in place** — all still correct (engagement-target
runs and the binary-MAE SVR/blend anti-predictors are not AUC-rankable rotation models).
No pins (the champion IS registered as `xgboost-classifier-rotation`, but its best-models
member is already the auto-selected rank-1 entry, so a pin is redundant; no confirmed
model fell out to rescue). No unexcludes/unpins.

**Result: group = 7 entries, all genuine AUC-rankable rotation classifiers, champion
at rank 1. Candidate field is exhausted of AUC-rankable runs (no backfill possible —
only these 9 runs carry a real auc field, and the 2 dropped are champion seed dups).**

| rank | model | predictor | AUC | note |
|---|---|---|---|---|
| 1 | best-xgboost-classifier-...-627d6 | xgboost-classifier | 0.7429 | registered champion `xgboost-classifier-rotation` (DS-META s42) |
| 2 | best-xgboost-classifier-...-ffffa4 | xgboost-classifier | 0.7350 | DISTINCT: 178-col AE-latent+metadata (latent-dilution arm), −0.0079 AUC vs champion, inside band |
| 3 | best-svc-...-848b8 | svc (poly d3) | 0.6855 | best non-tree; −4.6 bands vs champion |
| 4 | best-logistic-...-8e4d9 | logistic | 0.6762 | linear floor; −5.3 bands |
| 5 | best-svc-...-a704f | svc (rbf) | 0.6722 | −5.6 bands |
| 6 | best-svc-...-45798 | svc (linear) | 0.6363 | −8.5 bands |
| 7 | best-svc-...-53bb5 | svc (sigmoid) | 0.6255 | −9.3 bands; the documented worst/refutation echo |

**FLAGGED, LEFT ALONE — family monoculture / weak-vote tail.** Ranks 3-7 are five
dense distance/margin arms (svc + logistic), all 4.6-9.3 AUC bands below the trees and
all sharing the same correlated failure mode (they blur the sparse identity one-hots
that the trees read losslessly — the convergent finding). For a consensus AUC predict
they add little but correlated near-floor votes; svc-sigmoid (0.6255) is essentially a
documented refutation point. I did NOT exclude them: they are genuine, distinct,
AUC-rankable rotation models, and there is NO AUC-rankable backfill — excluding any
would only shrink the group, not improve it. If a consumer's consensus predict starts
tracking toward chance, exclude the bottom SVC kernels (sigmoid first, then linear)
then — but that is a consumer-driven call, not pollution removal. Revisit when the
champion hyperparameter scan (the next campaign per the report follow-ups) lands new
tree arms that would legitimately displace the dense tail.

**FRAMEWORK FOLLOW-UP (re-flagged, NOT fixed here) — `prev_value` is not invalidated
on a metric-regime change.** The 3 stale-regression-run pollution at ranks 10-12 is a
NEW symptom of the same binary-target metric-pipeline gap logged in the prior pass: when
the primary metric changes regime (regression MAE → classification AUC), a promoted
model's stored `prev_value` (its OLD MAE) is carried into the new-metric group as if it
were an AUC, so MAE ≈ 0.40 masquerades as "AUC 0.40" and backfills the tail. The
best-models recompute must invalidate / null a promoted model's `prev_value` when the
served `primary_metric` differs from the metric the value was recorded under (and skip
non-rankable members rather than backfilling them). Same root cause as the
best-models-ranks-by-MAE-on-binary and grid-by-MAE-blend issues — worth folding into the
single "binary-target / metric-regime metric pipeline" upstream follow-up. Out of scope
for this curation pass.


### 2026-06-09 (later) — second judgment pass after the SVM-kernel-rotation campaign

State on entry: group full (12 slots), the 17 prior exclusions all intact. The
recompute was fresh (updated_at matched the last run finish — no manual recompute
needed). The SVM-kernel-rotation campaign (experiments/archive/2026-06-09-svm-kernel-rotation.md)
added 10 new runs (8 stage-1 + 2 stage-2 blends), and the deterministic
top-12-by-MAE recompute PROMOTED THE ANTI-PREDICTORS to the top: ranks 1-4 and 8
were SVM / SVM-blend runs with NEGATIVE R². On the binary `rotation` target MAE is
degenerate — an epsilon-tube SVR snaps predictions to {0.1,0.9} and posts a low MAE
while being worse-than-mean (R² < 0). The genuine champion (autoencoder-xgboost w=0
== metadata-only XGBoost, MAE 0.3879 / R² +0.191) had been pushed down to ranks 5-7.

**Excluded (8 new run ids) — verified each run's R² from on-disk
`data/runs/<id>/metrics.json` before excluding (not by name alone):**

1. **6 negative-R² SVM / SVM-blend anti-predictors.** All are worse-than-mean;
   their low MAE is the binary epsilon-tube artifact, not skill:
   - `run-20260609-143114-528ef-blend` — R² −0.134 (grid mis-selected pure SVM, weights [0,1])
   - `run-20260609-143114-fa66a-blend` — R² −0.134 (same grid pathology, lgbm+svm)
   - `run-20260609-141717-73727-svm` — R² −0.137 (poly, the artifact-lowest MAE 0.3814)
   - `run-20260609-141717-89c5f-svm` — R² −0.130 (rbf)
   - `run-20260609-141717-b980e-svm` — R² −0.198 (linear)
   - `run-20260609-141717-fd50a-svm` — R² −8.48 (sigmoid blow-up, predictions to 24.7); excluded to bar backfill though it was below the cap.
2. **2 redundant w=0 champion duplicates (kept e34e4 as the single representative).**
   The campaign's two control anchors `run-20260609-141717-9a8ba-autoencoder-xgboost`
   (178-col ctrl) and `run-20260609-141717-28b3f-autoencoder-xgboost` (metadata-only
   ctrl) are byte-identical to the existing champion representative
   `run-20260609-131545-e34e4-autoencoder-xgboost` (all blend_weight=0,
   mae_xgb=0.3878913565318093, R² +0.1906, RMSE 0.4475). Same config, same seed-42
   split — the same model counted thrice. Kept e34e4 (consistent with the first pass),
   excluded the two duplicates. This extends the redundant-w=0-blend de-dup rule.

**The 17 prior exclusions were left in place** — all still correct (engagement-target
runs are incomparable to rotation; the earlier redundant w=0 blends are still
redundant). No pins, no unexcludes, no unpins.

**Result: group = 5 entries, all rotation, all positive R². Candidate field is
exhausted of positive-R² rotation runs (no backfill possible).**

| rank | model | predictor | MAE | R² | note |
|---|---|---|---|---|---|
| 1 | best-autoencoder-xgboost-...-e34e4 | autoencoder-xgboost (w=0 == metadata-only xgboost) | 0.3879 | +0.191 | the on-record rotation champion |
| 2 | best-autoencoder-xgboost-...-aac3d | autoencoder-xgboost (w=0) | 0.3931 | +0.181 | SAME champion config, split seed s17 — a noise-band point, NOT a distinct model |
| 3 | best-autoencoder-xgboost-...-fcbbe | autoencoder-xgboost (w=0) | 0.3964 | +0.178 | SAME champion config, split seed s101 — a noise-band point, NOT a distinct model |
| 4 | best-xgboost-...-593d5 | xgboost | 0.3965 | +0.166 | distinct: latent+metadata combined |
| 5 | best-xgboost-...-48044 | xgboost | 0.4098 | +0.132 | distinct: AE-latent-only (64 cols) |

The three DISTINCT rotation models are ranks 1, 4, 5 (champion 0.3879, latent+metadata
0.3965, AE-latent-only 0.4098). Ranks 2-3 (aac3d/fcbbe) are the s17/s101 seed points of
the champion config — left in the group as legitimate positive-R² members (they are the
3-seed noise-band data, σ_split=0.0043), but they are NOT new models and must not be cited
as distinct rotation families. Per the measured band (0.0086), the champion-vs-latent+metadata
edge (Δ 0.0086 ≈ 1·band) is "likely better, not confident"; the champion-vs-AE-latent-only gap
(~2.5·band) is real. No pin made: the champion is still not a saved standalone definition (pins
require a promoted model), and the spread between distinct models is at/below 1·band anyway.

**FRAMEWORK GAP (flagged, NOT fixed here).** This pollution episode is a symptom of a
real framework gap: the best-models group ranks by MAE, but on a binary target MAE is
degenerate (a two-level epsilon-tube predictor scores low MAE while being worse-than-mean).
The deterministic recompute therefore auto-promotes anti-predictors to the TOP, and a
curation pass has to manually exclude them every campaign. This is the same root cause as
the NaN-MAPE binary-metric limitation and the grid-by-MAE blend mis-selection (see pitfalls).
Worth a single coherent "binary-target metric pipeline" follow-up upstream: the best-models
ranking (and the blend grid objective) should use R²/logloss/AUC for binary targets, not MAE.
Out of scope for this curation pass.


### 2026-06-09 — first judgment pass after the AE+XGBoost blend campaign

State on entry: group full (12/12), no prior overrides. Recompute was fresh
(matched the last run finish; no manual recompute needed). The raw top-12-by-MAE
had two pathologies the deterministic sort cannot fix.

**Excluded (17 run ids) — two groups:**

1. **5 redundant w=0 blends (kept 1 representative).** The campaign's 6
   `autoencoder-xgboost` runs (kNN k=10/25/50/100/200 + ridge) all tuned the
   blend weight to w=0, making every run byte-identical metadata-only XGBoost
   (MAE 0.3878913565318093, R² 0.1906, RMSE 0.4475 — identical to the last
   significant digit). The recompute ranked all 6 at the top, so 6 group slots
   (and any consensus-predict vote) were one model counted sixfold. Kept
   `run-20260609-131545-e34e4` (k=10) as the single representative; excluded the
   other five run ids (5eec5, 2f638, b3b88, e84cf, 29775). The choice of which
   to keep is cosmetic (all 6 are numerically identical); k=10 had the best
   AE-branch diagnostic. NOTE: the surviving rank-1 entry is LABELLED
   `autoencoder-xgboost` but is, by w=0, exactly metadata-only XGBoost — the
   honest champion. It is not a saved standalone definition, so it cannot be
   pinned; it stays as the auto-selected representative.
2. **All 12 engagement-target runs (different scale, not comparable).** The
   active target pivoted to `rotation` ([0,1], MAE ~0.39) on 2026-06-08;
   the historical `engagement` runs (MAE ~2.4-3.0, log1p scale) are not
   comparable and would poison any rotation consensus predict. Because the
   rotation candidate field is tiny (only 8 rotation runs exist, 6 of them the
   identical blends), the top-12-by-MAE backfills engagement runs by raw numeric
   MAE no matter how many I remove — so I excluded all of them: the 4 originally
   in the group (xgboost 33428/991d6/89a30/8ce60) plus the 8 that backfilled on
   the next recompute (xgboost d03b0/ef048/ea2a2/f3393/5982c, baseline-median
   92886, ridge f1152/ff724).

**Result: group = 3 entries, all rotation, all dataset-family ds-2026...-p64.**
This is the honest size of the active-target candidate field, not a defect:

| rank | model | predictor | MAE | note |
|---|---|---|---|---|
| 1 | best-autoencoder-xgboost-...-e34e4 | autoencoder-xgboost (w=0 ⇒ metadata-only xgboost) | 0.3879 | the on-record rotation leader |
| 2 | best-xgboost-...-593d5 | xgboost | 0.3965 | latent+metadata combined |
| 3 | best-xgboost-...-48044 | xgboost | 0.4098 | AE-latent-only |

**Family diversity / flukes:** not actionable here. The rotation field is one
real predictor family (xgboost; the blend collapses to it) on one dataset family;
there is no second rotation family to pin in for diversity, and pinning an
off-target engagement model for "diversity" would be wrong (incomparable target).
No suspicious metrics: R² ~0.13-0.19 is modest-and-believable, n_test=2451 is
healthy, no leakage signature. **The 0.3879 vs 0.3965 vs 0.4098 spread is BELOW
the UNMEASURED rotation noise band** (the historical engagement band does not
transfer) — so ranks 1-3 are "equal, likely", not a confirmed ordering. No pin
was made: pins require a promoted model, the metadata-only champion was never
saved as a standalone definition, and the spread is not significant anyway.

**Revisit triggers:** (a) when the rotation noise-band study lands (the immediate
next campaign per the AE-blend follow-ups), re-rank with significance; (b) when
the metadata-only xgboost is registered + promoted as a first-class definition,
consider pinning it and unexcluding the redundant-blend reason becomes moot;
(c) the engagement exclusions are permanent for as long as `rotation` is the
active target — revert (`unexclude`) only if the target pivots back.
