# Next-track Phase-2: model-family sweep on the PCA spaces + R+M+C on the rich space — 2026-07-15

With the compression axis settled (PCA-128 base, PCA-192 candidate;
`2026-07-15-...`, `2026-07-15b-...`), this sweeps MODEL FAMILY on the chosen
spaces: does a different recurrent cell (LSTM), a non-recurrent pooled MLP
(ANN), or a LEARNED stacker (XGB over Markov+GRU[+ANN]) beat the fixed
GRU×Markov z-blend — and does adding a content-kNN third leg (R+M+C) on the
*rich* PCA space finally beat the champion? Frozen held config across all runs
(loss=infonce, hidden=256, epochs=40, patience=6, lr=0.001, batch=128,
dropout=0.1, k=10; recurrent num_layers=1/bidir=false/residual=false; blend
alpha=0.5/seed=1337; stacker base_hidden=256/base_epochs=40/base_loss=infonce/
n_negatives=50/xgb_rounds=200/max_depth=6/xgb_lr=0.1/seed=1337). Only the model
family and the representation vary. Server serialized (`--max-runs 1`).

**Champion / bar.** Prior champion = R+M z-blend on AE-64, R@10 0.1726 [0.154, 0.192],
MRR 0.0963 (`run-20260714-170928-645ae`). B2 first-order Markov bar = 0.107.
Spaces: PCA-128 `seq-20260715-030509` (base), PCA-192 `seq-20260715-131139`
(candidate). Bootstrap 95% CI = 2,000 resamples over the 1,431 test sessions;
paired per-session Δ vs the same-space GRU anchor (A2 PCA-128 GRU 0.1146 /
A4 PCA-192 GRU 0.123) and vs the champion. All 7 runs passed the sanity gate
(exit 0, n_test=1431).

## Outcome (one line)

**NEW CHAMPION ROW (best on record): R+M+C z-blend on PCA-192 — R@10 0.1859 [0.166, 0.207], MRR 0.1025 — confirmed across 5 GRU-leg reproductions (zero spread; paired-Δ vs the prior champion +0.0133 [+0.0007, +0.0259], entirely >0 every time, no MRR regression); it is train-only + off-server so it is a best-on-record ROW, NOT a promoted model. The learned XGB stacker badly UNDERPERFORMS the fixed z-blend; LSTM edges GRU as the best single model; the Phase-2 base STAYS PCA-128 (GRU-192's edge over PCA-128 still grazes 0).**

## Results — full sweep (ranked by R@10)

| model | run_id / source | R@10 [95% CI] | MRR | artist@10 | genre@10 | paired-Δ vs anchor | paired-Δ vs prior champion |
|---|---|---|---|---|---|---|---|
| **★ R+M+C on PCA-192 (NEW CHAMPION ROW)** | analyze_rmc (A4 GRU+M+C) | **0.1859 [0.1663, 0.2068]** | **0.1025** | 0.331 | 0.394 | vs R+M(192) **+0.0147 [+0.003,+0.026]** | **+0.0133 [+0.0007, +0.0259]** *>0, confirmed* |
| R+M+C on PCA-128 | analyze_rmc (A2 GRU+M+C) | 0.1824 [0.1621, 0.2027] | 0.0993 | 0.332 | 0.396 | vs R+M(128) +0.0154 [+0.004,+0.028] | +0.0098 [−0.004, +0.022] tie |
| — R+M z-blend on AE-64 (prior champion) | run-20260714-170928-645ae | 0.1726 [0.154, 0.192] | 0.0963 | 0.322 | 0.397 | — | — (reference) |
| R+M z-blend on PCA-192 | run-20260715-142440-96f53 | 0.1712 [0.1509, 0.1915] | 0.0952 | 0.329 | 0.398 | vs A4 +0.0482 [+0.033,+0.064] | −0.0014 [−0.013, +0.010] tie |
| R+M z-blend on PCA-128 (A3, reused) | run-20260715-044920-36dce | 0.167 [0.148, 0.187] | 0.092 | 0.329 | 0.398 | — | −0.0056 [−0.018, +0.006] tie |
| **GRU LSTM on PCA-128 (best SINGLE)** | run-20260715-142440-546d0 | **0.1272 [0.1104, 0.1447]** | 0.0495 | 0.335 | 0.458 | vs A2 +0.0126 [+0.0007, +0.024] *>0* | −0.0454 [−0.065, −0.026] |
| GRU on PCA-192 (config #5 = A4 confirm) | run-20260715-142440-89d6a | 0.1230 [0.1062, 0.1398] | 0.0534 | 0.338 | 0.450 | vs A2 +0.0084 [−0.0014, +0.0182] tie | −0.0496 [−0.068, −0.031] |
| GRU on PCA-128 (A2, reused anchor) | run-20260715-042100-a1d9d | 0.1146 [0.099, 0.132] | 0.0489 | 0.333 | 0.452 | — | — |
| ANN (pooled MLP) on PCA-128 | run-20260715-142440-7e528 | 0.0901 [0.0755, 0.1048] | 0.0394 | 0.299 | 0.435 | vs A2 −0.0245 [−0.039, −0.011] | −0.0825 |
| XGB stacker M+G on PCA-128 | run-20260715-142440-371b1 | 0.0867 [0.0727, 0.1013] | 0.0493 | 0.310 | 0.377 | vs A2 −0.0280 [−0.045, −0.011] | −0.0860 |
| XGB stacker M+G+A on PCA-192 | run-20260715-142440-f111d | 0.0853 [0.0706, 0.0992] | 0.0490 | 0.301 | 0.375 | vs A4 −0.0377 [−0.056, −0.020] | −0.0874 |
| XGB stacker M+G+A on PCA-128 | run-20260715-142440-d46cf | 0.0797 [0.0657, 0.0943] | 0.0423 | 0.310 | 0.377 | vs A2 −0.0349 [−0.053, −0.017] | −0.0929 |
| B2 first-order Markov (bar) | on record | 0.107 [0.092, 0.124] | 0.069 | 0.298 | 0.389 | — | — |

Best cell per metric bolded. R+M+C rows are the off-server content-leg driver
(z-blend GRU[from the A2/A4 checkpoint] + Markov + content-kNN max, equal thirds,
byte-identical z-blend to `seq_blend`); all others are API runs.

## Decision-rule application (initial sweep)

1. **Sanity gate:** all 7 runs exit 0, n_test=1431 — PASS, none excluded.
2. **PCA-192 confirmation (config #5):** R@10 0.1230 ∈ [0.107, 0.141] ✓, reproduces
   the 15b PCA-192 GRU (0.123) exactly. Paired-Δ vs the PCA-128 GRU anchor (A2) =
   **+0.0084 [−0.0014, +0.0182]** — grazes 0 → PCA-128 STAYS the base.
3. **Board-win test (non-overlapping marginal CIs on BOTH R@10 and MRR):** no
   model's R@10 CI clears the champion's upper bound (0.192) — the marginal CIs
   overlap. No win under the *strict marginal-CI* reading; R+M+C-192 clears the
   *paired-Δ* statistic (the one that crowned the prior champion) → flagged as a
   board-event candidate for the confirmation below.
4. **Ranking:** best single = **LSTM on PCA-128 (0.1272)**, paired-Δ vs GRU-128
   +0.0126 [+0.0007, +0.024] (>0). Best overall = **R+M+C on PCA-192 (0.1859)**.
5. **Combiner verdict:** the LEARNED XGB stacker (0.080–0.087) badly UNDERPERFORMS
   the fixed z-blend (0.167–0.186) — loses even to its own GRU base leg. Reconfirms
   `2026-07-14-nexttrack-blend-and-ensemble.md`: **the fixed z-blend is strictly
   better than the learned combiner.**
6. **R+M+C:** content is a genuinely ADDITIVE third leg on the rich PCA spaces —
   beats same-space R+M by +0.0154 [+0.004,+0.028] (PCA-128) / +0.0147
   [+0.003,+0.026] (PCA-192), both CI>0 — and on PCA-192 it edges the prior
   champion on the paired-Δ. Stronger than the 2026-07-14 R+M+C-on-AE-64 (which
   only tied) — the richer, higher-EVR space makes the content leg pay off.

## Confirmation (2026-07-15, authorized)

The board-event candidate R+M+C on PCA-192 was confirmed by reproducing the
GRU-192 leg (`seq-nexttrack` has no seed → run-to-run reproduction) **5×** total
(A4 original + config-#5 + 3 fresh reruns `…6c67b`, `…aefd5`, `…b5244`), and
recomputing R+M+C (that checkpoint + train-only Markov + content-kNN max) each
time. Champion reference = R+M on AE-64 (0.1726 / 0.0963).

| GRU-192 checkpoint | GRU R@10 | R+M+C R@10 [95% CI] | R+M+C MRR | paired-Δ vs champion (R@10) |
|---|---|---|---|---|
| A4 (original) | 0.1230 | 0.1859 [0.1663, 0.2068] | 0.1025 | +0.0133 [+0.0007, +0.0259] >0 |
| config #5 | 0.1230 | 0.1859 [0.1663, 0.2068] | 0.1025 | +0.0133 [+0.0007, +0.0259] >0 |
| rerun 1 (`…6c67b`) | 0.1230 | 0.1859 [0.1663, 0.2068] | 0.1025 | +0.0133 [+0.0007, +0.0259] >0 |
| rerun 2 (`…aefd5`) | 0.1230 | 0.1859 [0.1663, 0.2068] | 0.1025 | +0.0133 [+0.0007, +0.0259] >0 |
| rerun 3 (`…b5244`) | 0.1230 | 0.1859 [0.1663, 0.2068] | 0.1025 | +0.0133 [+0.0007, +0.0259] >0 |
| **pooled** | **0.1230 (spread 0)** | **0.1859 (spread 0)** | **0.1025** | **+0.0133 [+0.0007, +0.0259] every rerun** |

**Crown decision (MET):** the R@10 edge over the prior champion holds across ALL
reproductions — paired-Δ entirely >0 (+0.0133, lower bound +0.0007) every time,
with a MRR lead (0.1025 vs 0.0963, no regression). Per the authorized crown
criterion → **R+M+C on PCA-192 is recorded as the new champion ROW (best on
record).**

Two honest caveats attached to the crown:
- **Reproduction spread is exactly zero** because `seq-nexttrack` training is
  deterministic on this CPU setup — all 5 GRU-192 checkpoints are byte-identical
  (GRU R@10 0.1230 to 4 dp every time). So the "reproduction confirm" verified
  DETERMINISM, not seed-robustness; the only uncertainty quantified is the
  single-split test-session bootstrap.
- The paired-Δ lower CI bound is **+0.0007** — it clears 0 by a whisker on this
  one split. The crown is real by the pre-agreed criterion but the margin over
  the prior champion is thin; a second leak-free test split (or seed-varied legs
  once `seq-blend` is generalized to 3 legs) would harden it.

**Base decision (from the same reruns):** GRU-192's paired-Δ vs the PCA-128 GRU
anchor is **+0.0084 [−0.0014, +0.0182]** — identical every rerun, lower bound
grazes 0. Not consistently >0 → **the Phase-2 base STAYS PCA-128**
(`spotify_tracks_song_pca128` / `seq-20260715-030509`). PCA-192 remains the
representation the champion *row* is measured on, but the single-GRU base for
Phase-2 model work stays PCA-128 (its edge is inside noise).

## Verdict

- **NEW CHAMPION ROW: R+M+C z-blend on PCA-192** (R@10 0.1859 [0.166, 0.207], MRR
  0.1025) — the graded-preferred, content-augmented blend on the rich space,
  confirmed across 5 reproductions. **It is a best-on-record ROW, NOT a promoted
  model:** ranking predictors are train-only (no `predict`) and R+M+C is an
  off-server driver (GRU×Markov×content-kNN thirds) — nothing is
  best-models-promotable until `seq-blend` is generalized to 3 legs + a `predict`
  subcommand lands.
- **Prior champion R+M on AE-64 (0.1726/0.0963)** remains the productization
  anchor (the 2-leg blend that maps onto the app's transition slot) until the
  3-leg R+M+C is registered.
- **Phase-2 base: PCA-128 stands** (GRU-192 edge grazes 0).
- **Combiner:** fixed z-blend ≫ learned XGB stacker. **Best single:** LSTM on PCA-128.

## Best on record after this work (new/changed rows bold)

| model | Recall@10 [95% CI] | MRR | artist@10 / genre@10 | note |
|---|---|---|---|---|
| **★ R+M+C z-blend on PCA-192 (CHAMPION ROW)** | **0.186 [0.166, 0.207]** | **0.103** | 0.331 / 0.394 | GRU×Markov×content-kNN thirds; confirmed 5× (spread 0), paired-Δ vs prior champion +0.013 [+0.0007,+0.026] >0. Train-only + off-server → best-on-record ROW, NOT promoted. `analyze_rmc.py` on `seq-20260715-131139` |
| — R+M+C z-blend on PCA-128 | 0.182 [0.162, 0.203] | 0.099 | 0.332 / 0.396 | content additive on the rich space (+0.015 over R+M-128); paired-Δ vs prior champion straddles 0 |
| — R+M z-blend on AE-64 (prior champion / productization anchor) | 0.173 [0.154, 0.192] | 0.096 | 0.322 / 0.397 | the 2-leg blend that maps to the app; still the deployable anchor |
| — R+M z-blend on PCA-192 | 0.171 [0.151, 0.192] | 0.095 | 0.329 / 0.398 | ≈ prior champion (tie) |
| **GRU LSTM on PCA-128 (best single)** | **0.127 [0.110, 0.145]** | 0.050 | 0.335 / 0.458 | LSTM > GRU on PCA-128 (paired-Δ +0.013 CI>0) |
| GRU on PCA-192 | 0.123 [0.106, 0.140] | 0.053 | 0.338 / 0.450 | reproduced 5× exactly |
| B2 first-order Markov (bar) | 0.107 [0.092, 0.124] | 0.069 | 0.298 / 0.389 | app incumbent |
| XGB stacker (M+G / M+G+A) | 0.080–0.087 | 0.042–0.049 | 0.31 / 0.38 | learned combiner UNDERPERFORMS the z-blend and the GRU |

## Follow-ups

- **Generalize `seq-blend` to 3 legs** so R+M+C is a first-class REGISTERED
  predictor (currently an off-server driver) — prerequisite to making the new
  champion row promotable / app-exportable. THE top follow-up.
- **Harden the crown** with a second leak-free test split (the paired-Δ lower
  bound is +0.0007 on the single split; the reproduction spread is 0 because
  `seq-nexttrack` is deterministic, so seed-robustness was not actually tested).
- Compression + model-family thread otherwise CLOSED: PCA-128 base, LSTM≈best
  single, learned stacker dead, z-blend is the combiner.
- Standing: `predict` subcommand → make the blend best-models-promotable +
  app-exportable.

Driver/scripts: `scratchpad/analyze_rmc.py` (R+M+C both spaces),
`scratchpad/analyze_p2.py` (7-run bootstrap + paired-Δ),
`scratchpad/analyze_confirm.py` (5× GRU-192 reproduction + R+M+C recompute).
GRU legs built from the run-dir `model.pt` checkpoints.
