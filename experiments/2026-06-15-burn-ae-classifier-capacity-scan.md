# Experiment: burn-ae-classifier capacity + classifier-head architecture scan

## Goal

First characterization of the brand-new `burn-ae-classifier` predictor (crate
`crates/predictor-burn-ae-clf`, registered THIS session) on rotation. It is a
two-phase Burn model: Phase 1 pretrains an autoencoder to reconstruct the full
assembled feature vector `[standardized PCA+numerics | raw one-hots]` (D=177)
by MSE; Phase 2 transplants the trained encoder into a deep softmax classifier
(encoder → latent → clf head → K logits) and fine-tunes with cross-entropy.
`freeze_encoder` toggles a fixed feature extractor vs end-to-end fine-tune.
The baseline AE-transplant config (enc_hidden [128,64] latent 64, clf_hidden
[64], freeze_encoder=false) had scored **test AUC 0.69225** (run
`run-20260615-151524-4026a-burn-ae-classifier`), below the in-corpus leaders
xgboost-classifier (0.7204) and the calibration-fixed dense ANN
burn-deep-classifier (mlp, 0.7038). Hypothesis: greater encoder/latent/head
capacity (and possibly freeze vs fine-tune, smoother activations) closes some
of the gap. This campaign scans 13 configs to (a) find the best AE-transplant
config, (b) quantify the gap to the leaders, and (c) seed the family's first
PROJECT-FACTS entry. PROJECT-FACTS had NO prior entry for this family.

**Fixed baseline.**
- Dataset: **`ds-20260609-204419-p64-s42`** — the canonical NEW full-64-d
  rotation dataset. **23,529 rows** (18,823 train / 4,706 test @ split seed 42),
  **177 cols** (64 song-AE PCA latent + 7 numeric + 106 identity one-hots).
  Target `metadata.rotation`, task binary, transform none. Used for EVERY run.
- In-corpus references on THIS 177-col board (from prior campaigns): registered
  champion **xgboost-classifier 0.7204** (`xgboost-classifier-rotation-p64`),
  catboost 0.7111, best dense ANN **burn-deep-classifier mlp 0.7038**
  (`2026-06-12-burn-deep-topology.md`), ngboost 0.6917, pyramid-mlp 0.6792.
- **Held hyperparameters (all runs unless an axis varies them):** ae_epochs 80,
  ae_lr 0.001, ae_dropout 0.0, epochs 200 (fine-tune), lr 0.001, batch_size 256,
  dropout 0.1, patience 25, val_fraction 0.15, checkpoint_every 0, activation
  relu, freeze_encoder false.
- All 13 runs were launched AD-HOC (no model definitions created — models.toml
  kept clean).
- **Noise band.** σ_AUC = 0.00628, significance band 2σ = **0.0126** (AUC,
  higher-better; MEASURED 2026-06-09, rotation-classification campaign). Any
  single-split AUC gap below 0.0126 is "at least equal / sub-band tie", not a
  beat.
- The burn binary's internal split seed is fixed; it is NEAR- not
  bit-deterministic. Confirmed here: cfg1 (a re-run of the baseline config at
  full ae_epochs 80) scored 0.69645 vs the prior baseline run's 0.69225, Δ
  0.00420 ≈ 0.33 band — sub-band jitter, do not chase it.
- **Walltime note.** Runs were far slower than the ~15–90 s/run estimate: the
  AE pretrain (80 epochs) + fine-tune (up to 200 epochs early-stopped) on
  18,823×177 rows on CPU, with 9 (Axis-1) then 4 (Axis-2/3) running
  concurrently, took the Axis-1 batch ~37 min wall and the Axis-2/3 batch
  ~25 min wall. Per-run early-stop epoch is NOT exposed via the API
  (`stderr_tail` is empty on success and the metrics block carries only
  auc/logloss/accuracy/brier/n_test), so per-config fine-tune/best-epoch is not
  reported below — it could not be recovered post-hoc without re-instrumenting
  the binary.

## Outcome (one line)

**Capacity does NOT matter for this family on rotation — the 12 fine-tuned
configs span just 0.69175–0.69719 (0.0054, entirely inside the 0.0126 band): a
flat tie across latent size, head depth/width, and activation; the best config
(cfg3, enc[128,64]/clf[128,64], AUC 0.69719) beats the 0.69225 baseline by only
+0.00494 (sub-band, NOT a material beat) and sits ~1.8 bands below the
xgboost leader (0.7204) and ~0.5 band below the comparable burn-deep ANN
(0.7038) — the gap did NOT close; the one real effect is that FREEZING the
encoder under a DEEP head collapses the model (cfg10, 0.68210, −1.2 bands).**

## Results — all 13 configs, sorted by test AUC

mape/medape not applicable (binary classification). Best cell per metric bolded.
Winner row bolded. (auc/acc higher-better; logloss/brier lower-better.)

| rank | config | enc_hidden | clf_hidden | freeze | act | AUC | logloss | brier | accuracy | run id |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **cfg3 — deeper head (best)** | [128,64] | [128,64] | false | relu | **0.69719** | **0.6216** | 0.2169 | 0.6496 | run-20260615-152359-21f11-burn-ae-classifier |
| 2 | cfg5 — wider latent 128 | [256,128] | [128] | false | relu | 0.69712 | 0.6254 | 0.2184 | 0.6485 | run-20260615-152359-05bce-burn-ae-classifier |
| 3 | cfg9 — tight bottleneck 32 | [128,32] | [64] | false | relu | 0.69657 | 0.6246 | 0.2182 | 0.6451 | run-20260615-152359-e5a75-burn-ae-classifier |
| 4 | cfg1 — baseline re-run | [128,64] | [64] | false | relu | 0.69645 | 0.6264 | 0.2189 | **0.6496** | run-20260615-152359-94f8f-burn-ae-classifier |
| 5 | cfg4 — wider head | [128,64] | [256,128] | false | relu | 0.69561 | 0.6235 | 0.2177 | 0.6434 | run-20260615-152359-7a459-burn-ae-classifier |
| 6 | cfg8 — large | [384,192] | [192,96] | false | relu | 0.69541 | 0.6224 | **0.2176** | 0.6426 | run-20260615-152359-cb93b-burn-ae-classifier |
| 7 | cfg11 — wide latent + FREEZE | [256,128] | [128] | true | relu | 0.69530 | 0.6242 | 0.2184 | 0.6409 | run-20260615-160112-aa1db-burn-ae-classifier |
| 8 | cfg12 — deeper head + gelu | [128,64] | [128,64] | false | gelu | 0.69340 | 0.6256 | 0.2186 | 0.6411 | run-20260615-160112-4a764-burn-ae-classifier |
| 9 | cfg7 — deeper encoder | [256,128,64] | [64] | false | relu | 0.69302 | 0.6279 | 0.2196 | 0.6481 | run-20260615-152359-3b006-burn-ae-classifier |
| 10 | cfg13 — deeper head + silu | [128,64] | [128,64] | false | silu | 0.69242 | 0.6230 | 0.2178 | 0.6436 | run-20260615-160113-7567b-burn-ae-classifier |
| 11 | cfg6 — wide everything | [256,128] | [256,128] | false | relu | 0.69240 | 0.6275 | 0.2196 | 0.6392 | run-20260615-152359-86fd5-burn-ae-classifier |
| 12 | cfg2 — linear probe (clf []) | [128,64] | [] | false | relu | 0.69175 | 0.6282 | 0.2199 | 0.6405 | run-20260615-152359-dbb6a-burn-ae-classifier |
| 13 | cfg10 — deep head + FREEZE | [128,64] | [128,64] | true | relu | 0.68210 | 0.6378 | 0.2234 | 0.6373 | run-20260615-160112-17da7-burn-ae-classifier |

Reference baseline (prior run, this session, NOT re-launched here):
enc[128,64]/clf[64]/freeze false, **AUC 0.69225**
(`run-20260615-151524-4026a-burn-ae-classifier`). cfg1 re-runs this config at
ae_epochs 80 and reproduces it to 0.69645 (+0.00420, sub-band — see
determinism note).

All 13 runs `succeeded` (exit 0); none crashed. Note the metrics block carries
auc/logloss/accuracy/brier only (no MAPE) — the binary NaN-MAPE trap is dodged,
same as the gbdt-leaf-head plugin.

## Findings

**(i) Does latent size matter? NO — sub-band.** The latent-dim axis at fixed
shallow-ish head spans cfg9 (latent 32, 0.69657) ≈ cfg5 (latent 128, 0.69712) ≈
cfg1 (latent 64, 0.69645): a 0.00067 spread across a 4× latent range. Even the
"tight bottleneck 32" loses nothing. The AE latent is not the binding
constraint.

**(ii) Does head depth/width matter? NO — sub-band, and the linear probe is the
floor not a cliff.** Deeper head (cfg3 [128,64], 0.69719) is the nominal best;
wider head (cfg4 [256,128], 0.69561) and wide-everything (cfg6 [256,128] enc +
[256,128] head, 0.69240) are lower but in-band; the **linear probe** (cfg2,
clf [], 0.69175) is the worst fine-tuned config but only −0.00544 below cfg3 —
i.e. a single linear layer on the fine-tuned latent already captures essentially
all the family's skill. The deeper encoder (cfg7 [256,128,64], 0.69302) and the
large config (cfg8 [384,192]/[192,96], 0.69541) add nothing. **Capacity is
saturated at the smallest config tested**; the whole fine-tuned set is a 0.0054
strip inside the 0.0126 band — statistically one number.

**(iii) Freeze vs fine-tune — the ONE real effect, and it is a TRAP for deep
heads.** With the encoder FROZEN as a fixed feature extractor: cfg11 (shallow
head [128]) stays in-band at 0.69530 (−0.00189 vs its fine-tuned twin cfg5),
but cfg10 (deep head [128,64]) **collapses to 0.68210** — −0.01509 vs its
fine-tuned twin cfg3, ~1.2 bands, the only out-of-band result in the campaign,
and the worst row overall. Reading: the MSE-pretrained encoder is a decent but
not task-optimal feature extractor; a shallow head can ride it, but a deep head
on TOP of a frozen suboptimal representation just stacks unhelpful capacity and
underfits the label (the gradients that would have re-shaped the encoder toward
the label are blocked). **freeze_encoder=true is safe only with a (near-)linear
head; combined with a deep head it is a documented collapse mode.** End-to-end
fine-tuning is the correct default and is what the registry default
(freeze_encoder=false) does.

**(iv) Activation — relu wins, smooth activations hurt slightly.** Holding the
cfg3 architecture: relu 0.69719 > gelu 0.69340 (−0.00379) > silu 0.69242
(−0.00477). Both deltas are sub-band, but the direction is consistent: the
smooth activations are a hair WORSE, not better. No reason to leave relu.

**(v) The gap to the leaders did NOT close.** Best config 0.69719 vs the
in-corpus xgboost champion 0.7204 = **−0.02321 ≈ 1.84 bands** (decisively
behind). vs catboost 0.7111 = −0.01391 ≈ 1.1 bands. vs the comparable
calibration-fixed dense ANN burn-deep-classifier mlp 0.7038 = −0.00661,
sub-band — so the AE-transplant net is statistically EQUAL to the plain
burn-deep mlp on this matrix, NOT better: the unsupervised AE pretrain buys a
fast/stable init but no AUC over a from-scratch raw-one-hot mlp. Calibration is
healthy (logloss ~0.62, brier ~0.22 — the trees' regime, and far better than
the sklearn pyramid's 2.0+ or the gbdt-leaf-head's 0.82–1.31), so this family is
a legitimate calibrated dense member, just not an AUC contender.

**Decision rule applied.** Primary metric test AUC. Best config (cfg3, 0.69719)
beats the 0.69225 baseline by **+0.00494, < 0.0126 band** → "at least equal,
likely a hair better — NOT a material beat" (the success criterion "materially
beats baseline" is NOT met). The stretch target (reach the ~0.715–0.720 leader
band) is missed by ~1.8 bands. The design instructed NOT to promote/curate
best-models, and no save rule was specified for a sub-band, ~1.8-band-behind
result — **NO definition saved**, models.toml untouched. The server's
deterministic best-models recompute ran on each completion; all 13 sit ~0.69,
below the ~0.72 top-12 tier, so the champion did not change and nothing was
auto-promoted into a serving slot above the existing tree/embeddings leaders.

## Best-on-record context (this 177-col board, after this work)

burn-ae-classifier slots into the existing full-64-d AUC leaderboard between
burn-deep-classifier (mlp, 0.7038) and the gbdt-leaf-head / ngboost cluster
(~0.69). New rows for this family (best + the freeze-collapse pitfall row):

| model | dataset | AUC | logloss | accuracy | brier | run id | registered? |
|---|---|---|---|---|---|---|---|
| xgboost-classifier (champion, ref) | ds-20260609-204419-p64-s42 (177) | **0.7204** | 0.6075 | 0.6585 | — | run-20260609-204726-439d8 | YES — `xgboost-classifier-rotation-p64` |
| burn-deep-classifier (mlp, lr0.0005, [128,64]) — best dense ANN, ref | ds-20260609-204419-p64-s42 | 0.7038 | 0.6221 | 0.6521 | — | `2026-06-12-burn-deep-topology.md` | no |
| **burn-ae-classifier** (enc[128,64]/clf[128,64], fine-tune, relu) — best AE-transplant | ds-20260609-204419-p64-s42 | **0.69719** | 0.6216 | 0.6496 | 0.2169 | run-20260615-152359-21f11 | no (sub-band over baseline; ~1.8 bands below champion) |
| burn-ae-classifier (enc[128,64]/clf[64], fine-tune — baseline config) | ds-20260609-204419-p64-s42 | 0.69225 | — | — | — | run-20260615-151524-4026a | no |
| burn-ae-classifier (enc[128,64]/clf[128,64], FROZEN encoder) — collapse mode | ds-20260609-204419-p64-s42 | 0.68210 | 0.6378 | 0.6373 | 0.2234 | run-20260615-160112-17da7 | no — pitfall |

The registered champions on the other corpora/matrices are unchanged:
`xgboost-classifier-rotation-p64` (0.7204, this 177-col board),
`burn-deep-embeddings-highvocab` (0.7192, the 497-col high-vocab board), and
`xgboost-classifier-rotation` (0.7429, DS-META). Nothing regressed.

## Follow-ups

1. **The family is closed for AUC on rotation as configured.** It is
   statistically equal to the plain burn-deep mlp (−0.0066, sub-band) but ~1.8
   bands behind the tree champion, and capacity is fully saturated at the
   smallest net — no tuning axis explored here moves it. Do NOT spend more
   single-corpus capacity scans on it.
2. **If revisited, change the pretraining objective, not the size.** The AE is
   purely reconstructive (MSE on the assembled vector); the latent it hands the
   classifier is task-agnostic and buys no AUC over a from-scratch mlp. A
   supervised or semi-supervised pretrain (or a contrastive objective on
   rotation-relevant structure) is the only axis with a plausible path to the
   tree tier — geometry, not width.
3. **High-vocab re-test (parallel to burn-deep).** burn-deep's embeddings
   topology only reached the tree tier once identity cardinality was raised
   (artist 300 / genre 120, `ds-20260612-143014-p64-s42`). The AE-transplant net
   ingests the SAME raw one-hots; a single best-config run on the high-vocab
   matrix would test whether the AE reconstruction captures the richer identity
   signal — cheap, one run, and the only "more data" lever left.
4. **freeze_encoder=true needs a head-depth guard.** The deep-frozen collapse
   (cfg10, −1.2 bands) is a sharp, repeatable trap. Anyone using freeze mode
   should pin the head to (near-)linear; consider a registry note or a soft
   warning.
5. **best-model-selector: no action needed from this campaign** — no champion
   change, no suspiciously-good result, nothing reached the top-12 tier.
