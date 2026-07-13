# Pyramid-MLP architecture scan on the full-64-d binary rotation dataset

## Goal

Follow-up to the four-family classifier bake-off
(`experiments/2026-06-12-classifier-bakeoff-p64.md`), where
`pyramid-mlp-classifier` came last at AUC 0.6781 and was flagged as
**under-trained** (logloss 2.4085, brier 0.3251 — read as a 300-iter-cap
artifact, not the model's ceiling). This scan asks: with the iteration budget
raised so the nets converge, does any pyramid architecture lift the MLP family
to a competitive AUC? The axis is the hidden-layer geometry; `max_iter` was
raised to 800 across the board so the comparison would reflect architecture,
not training budget.

### Fixed baseline

- **Dataset:** `ds-20260609-204419-p64-s42` — full-64-d binary rotation;
  23,529 rows (18,823 train / 4,706 test @ seed 42), 177 cols (64 song-AE PCA +
  7 numeric + 106 identity one-hots), target `metadata.rotation`, task binary.
  Same dataset as the bake-off ("all of the 64-d embedding").
- **Held for all arms:** alpha 1e-4, learning_rate_init 1e-3, **max_iter 800**
  (raised from the default 300 so the optimizer could converge), seed 42 (split
  + model).
- **Reference points (same dataset):** pyramid-mlp default/under-trained AUC
  0.6781 (`run-20260612-115909-ea698`, max_iter 300); tree incumbents
  xgboost 0.7204, catboost 0.7111, ngboost 0.6917. Single-seed; AUC noise band
  ≈ 0.0126 (2σ, old-corpus estimate — approximate here).
- **Provenance note:** all six arms were launched in one batch at 12:31:55 on
  2026-06-12; this runner resumed collection on the two long-running
  large-capacity arms (M4, M5, ~60 min each) and assembled the results. No arm
  was re-launched; no definition was created (see decision rule).

## Outcome (one line)

Refuted: raising max_iter did NOT lift the MLP family — every arm stops itself
at 73–100 iterations by the optimizer's own tolerance (well below even 300), so
the budget was never the binding constraint; the best converged pyramid is M1
[512,256] at AUC **0.6792**, a statistical tie with the 0.6781 under-trained
default and ~3.3 bands below xgboost 0.7204. No definition saved.

## Results (sorted by AUC, higher better; logloss/brier lower better)

| arm | realized hidden | AUC | logloss | accuracy | brier | iters (of 800) | run id |
|---|---|---|---|---|---|---|---|
| **M1 wide-shallow** | **[512, 256]** | **0.6792** | **2.0584** | 0.6326 | **0.3216** | 75 | run-20260612-123155-0ace5 |
| M0 re-baseline | [256, 128, 64] | 0.6781 | 2.4085 | 0.6326 | 0.3251 | 73 | run-20260612-123155-8bfab |
| M4 gentle taper | [384, 307, 246] | 0.6777 | 2.7764 | **0.6362** | 0.3292 | 87 | run-20260612-123155-d3834 |
| M2 deep pyramid | [256, 128, 64, 32] | 0.6737 | 2.7631 | 0.6343 | 0.3303 | 100 | run-20260612-123155-79128 |
| M3 steep taper | [512, 169, 56] | 0.6736 | 2.7031 | 0.6258 | 0.3365 | 89 | run-20260612-123155-0df76 |
| M5 wide-deep | [512, 307, 184, 111] | 0.6712 | 2.6457 | 0.6273 | 0.3348 | 84 | run-20260612-123155-01f07 |

All six runs **succeeded** (exit 0); no failures. **No arm hit the max_iter 800
cap** — every one stopped early via sklearn's `n_iter_no_change` tolerance at
73–100 iterations. (M5's realized fourth layer is 111, not the 110 the design
table estimated — float-floor rounding; quoted as realized.)

## Findings

**The premise was wrong: max_iter was never the binding constraint.** The
bake-off flagged the 0.6781/logloss-2.4085 result as a 300-iteration-cap
artifact and prescribed "max_iter ≥ 500 to converge." This scan refutes that
mechanism. With max_iter raised to 800, every arm still halts on its own at
**73–100 iterations** — far below even the old 300 cap. The optimizer reaches
sklearn's loss-improvement tolerance and stops; it is not being truncated by
the cap. The decisive tell: **M0 (the bake-off default arch, now at max_iter
800) reproduced the under-trained run's AUC to all digits — 0.6780926 vs
0.6780926 — because it ran the identical 73 iterations.** Raising the cap from
300 to 800 changed literally nothing for that geometry.

**"Converged?" — yes by the cap criterion, no in any useful sense.** Per the
design's mechanical definition (no max_iter-cap WARNING), all six arms
"converged" — none hit the 800 cap. But they converged to a *poor* optimum: the
logloss (2.06–2.78) and brier (0.32–0.34) remain at the same blown level as the
under-trained default, versus ~0.60–0.63 logloss / ~0.21 brier for the trees.
The early-stopping tolerance triggers on a loss plateau the optimizer cannot
escape at this learning rate — so the high logloss is a genuine *calibration
ceiling at these hyperparameters*, not a half-finished training trajectory. The
prior facts-file hypothesis ("a converged run would lift logloss/brier toward
the field") is therefore refuted at the current alpha/lr.

**Which geometry wins, and why it barely matters.** The whole family lands in a
0.0080-wide AUC strip (0.6712–0.6792), entirely inside the ~0.0126 noise band —
so the geometry ranking is **not statistically resolvable** on a single seed.
Directionally: wider-and-shallower is mildly favored (M1 [512,256] top, M0
[256,128,64] second), and adding depth/capacity hurts slightly (M5 wide-deep
[512,307,184,111] last, M2 deep pyramid second-last). That is consistent with a
dense net struggling to route signal through 106 sparse identity one-hots —
extra layers add capacity to memorize/blur rather than to separate, and the
optimizer plateaus sooner relative to the harder loss surface. Accuracy is
flat (~0.625–0.636) across all arms because the dataset's majority-class floor
dominates a poorly-calibrated head.

**MLP-vs-trees placement.** Best converged pyramid AUC = **0.6792** (M1). That
is:
- **+0.0011 over the 0.6781 under-trained default** — i.e. essentially zero.
  The "under-training" framing overstated the headroom: there was no training
  budget left to recover, only a marginal architecture nudge inside noise.
- **−0.0412 below xgboost 0.7204** (~3.3 bands), and below catboost 0.7111 and
  ngboost 0.6917. The MLP family is confirmed as the weakest family on this
  one-hot-heavy feature set — the same dense-blurs-sparse-one-hots failure mode
  documented for the SVC/logistic families.

**Decision rule applied.** Winner = highest AUC = **M1 [512,256] at 0.6792**.
The save threshold was AUC ≳ 0.683 (within ~3 bands of the catboost/xgboost
tree tier). M1 at 0.6792 is **below 0.683**, and is a statistical tie with the
default rather than the "clearly-best MLP" the rule requires. Therefore
**no definition was persisted** and no dataset tag was set. Verdict recorded:
the MLP family is confirmed below the trees on these features, and the
budget-vs-architecture question is settled — both knobs are exhausted at this
alpha/lr.

**Single-seed caveat.** All numbers are single-seed (42). The within-family
ranking is inside the noise band and would need a 3-seed check to call any
geometry "best"; but the family-level conclusion (MLP ≈ 0.68, trees ≈ 0.72) is
~3 bands wide and robust to seed.

## Best on record after this work — ROTATION target, AUC, on `ds-20260609-204419-p64-s42`

(Unchanged by this scan — no new champion, no new definition.)

| model | dataset | AUC | logloss | accuracy | source | registered? |
|---|---|---|---|---|---|---|
| **xgboost-classifier** (n_est 500, depth 6, lr 0.05, sub/col 0.8, seed 42) | ds-20260609-204419-p64-s42 (177) | **0.7204** | 0.6075 | 0.6585 | `2026-06-12-classifier-bakeoff-p64.md` | **YES — `xgboost-classifier-rotation-p64`** |
| xgboost-classifier (depth 8, lr 0.03 — sweep best, single-seed) | ds-20260609-204419-p64-s42 | 0.7261 | 0.6044 | 0.6651 | `2026-06-12-classifier-bakeoff-p64.md` | no — flagged for 3-seed confirm |
| catboost-classifier (iter 500, depth 6, lr 0.05) — 2nd tree family, in-band | ds-20260609-204419-p64-s42 | 0.7111 | 0.6132 | 0.6560 | `2026-06-12-classifier-bakeoff-p64.md` | no |
| ngboost-classifier (n_est 400, lr 0.03, base_depth 3) | ds-20260609-204419-p64-s42 | 0.6917 | 0.6293 | 0.6392 | `2026-06-12-classifier-bakeoff-p64.md` | no |
| pyramid-mlp-classifier (512/2/0.5 wide-shallow, max_iter 800 — best converged MLP) | ds-20260609-204419-p64-s42 | 0.6792 | 2.0584 | 0.6326 | this report (M1) | no |

## Follow-ups

1. **The MLP family is settled on this feature set — close it for AUC.** Both
   the budget knob (max_iter, this scan) and the architecture knob (six
   geometries, this scan) are exhausted at alpha 1e-4 / lr 1e-3, and the family
   sits ~3 bands below the trees. Do not re-scan geometry. If the MLP is to be
   revisited at all, the only untested lever is the **optimizer/regularization**
   (lower learning_rate_init with a higher iteration floor, or alpha sweep) to
   chase calibration — but the AUC ceiling looks fixed near 0.68, so this is for
   probability quality, not the leaderboard.
2. **The high logloss/brier is now confirmed a real calibration ceiling, not an
   artifact** — if calibrated MLP probabilities are ever needed, wrap the head
   in Platt/isotonic calibration rather than expecting more iterations to fix
   it.
3. Tree tier remains the action: the **depth8/lr0.03 xgboost (0.7261)** still
   awaits its flagged 3-seed confirmation before re-saving `xgboost-...-rotation-p64`
   at depth 8 (see bake-off report follow-ups).
