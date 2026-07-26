# Dual-tower fusion — architecture or just capacity? (view-pairing matrix × fusion depth, param-matched single-GRU control) — 2026-07-25

**Date:** 2026-07-25
**Type:** ARCHITECTURE SCAN — **REFUTED**. Nothing promoted, nothing registered,
champion/leaderboard UNCHANGED. The null is the finding.
**Predictor under test:** `seq-dualgru` (`predictors/seq_dualgru.py`) — brand new,
implemented + registered this session, **never before run on a real dataset**.
Two INDEPENDENT recurrent towers (separate weights) read the prefix in parallel
over configurable causal views (`latent` / `delta` / `cummean`); per-step hidden
states are concatenated and fused by one MLP head predicting the next-track
latent. It reuses `seq_nexttrack.make_batches` + `step_loss` verbatim, so
objective / batching / early-stopping / eval are byte-for-byte the family's.
**Dataset (fixed, all 10 runs):** `seq-20260715-131139` — canonical leak-free
PCA-192 split; 7,154 sessions / 19,402 items; 5,723 train / 1,431 test @ cut
2024-08-24T14:28:02; cold-item rate 0.55.
**Host:** all 10 runs on the SAME remote worker `snappler:4101029` (`queue:true`,
serialized one-at-a-time) — the design's Risk-1 mitigation, verified per run.
**Seed:** 1337. **Noise band:** recall@10 practical CI half-width ±0.015; observed
same-config reproduction spread ±0.0035.

## Goal & baseline

**Question.** A jointly-trained dual-tower GRU was hypothesized to beat a single
recurrent tower *only if the two towers see genuinely different causal views* of
the prefix. Two towers on the same `latent` view should reduce to a
capacity / implicit-ensemble effect that a width-matched single GRU can buy more
cheaply. The scan therefore pairs the **complete upper triangle of the
{latent, delta, cummean}² view matrix** (6 cells at `fusion_layers=1`) with a
**fusion-depth** axis (2 cells at `fusion_layers=0`) and — the axis the original
topology scan lacked — a **parameter-matched single-GRU capacity control**.

**Baseline / reference (C0).** `seq-nexttrack` `gru-infonce-h256` on
`seq-20260715-131139`, on-record recall@10 **0.12299 = 176 of 1,431** next tracks
(`run-20260722-165345-e39c9`, and 5 prior identical reproductions;
`2026-07-15c-nexttrack-phase2-model-sweep.md`,
`2026-07-22-session-holisticness-and-antieager-levers.md`). Re-run in-batch on the
remote host as C0 and reproduced **EXACTLY**: recall@10 0.12299, MRR 0.05341,
artist@10 0.3375, genre@10 0.4500, music@10 0.4558, artist_adj 0.34745,
mood_coh 0.47884, ild 0.45356, **holisticness@10 0.14172**, n_test 1,431.

**Held for all 8 dual arms** (only `view_a`, `view_b`, `fusion_layers` vary):
`hidden_a=256, hidden_b=256, arch_a=gru, arch_b=gru, layers_a=1, layers_b=1,
fusion_hidden=256, loss=infonce, tau=0.07, epochs=40, lr=0.001, dropout=0.1,
batch_size=128, patience=6, seed=1337, k=10, eager_beta=0.0, eager_margin=0.0,
mmr_lambda=1.0, mmr_pool=200`.
**Held for both single-GRU controls** (only `hidden` varies):
`arch=gru, loss=infonce, num_layers=1, bidirectional=false, residual=false,
epochs=40, lr=0.001, dropout=0.1, batch_size=128, patience=6, k=10,
eager_beta=0.0, eager_margin=0.0, mmr_lambda=1.0, mmr_pool=200`
(`tau` is hardcoded 0.07 in `seq_nexttrack.py` → parity with the dual arms is
exact; `seq-nexttrack` exposes no seed).

**Parameter counts — re-measured in this campaign** by instantiating the real
classes at `latent_dim=192` (not estimated), confirming the design's figures
exactly:

| config | params |
|---|--:|
| single GRU h256 (C0) | 394,944 |
| **single GRU h425 (C1, capacity control)** | **871,017** |
| **dual 2×256 + fusion 256×1 (the six f1 arms)** | **871,872** |
| dual 2×256 + `fusion_layers=0` (the two f0 arms) | 789,696 |
| single GRU h400 (reference point) | 789,792 |
| single GRU h512 (B3 arm, not fired) | 1,182,912 |

C1 matches the `fusion_layers=1` dual arms to **0.098 %** — a genuine
param-matched control — and is 10 % *larger* than the `fusion_layers=0` arms,
i.e. deliberately generous to the single tower there.

**The honest bar (stated up front, per the design).** Two boards are kept
separate: (1) **recall@10** — the bar for a standalone single model is the
SINGLE-model board **0.123**, NOT the 3-leg blend champion 0.212 (comparing a
single trained model to the blend would be a category error; dual-tower-as-the-R′-leg
is a `seq-blend` code change and explicitly out of scope). (2) **holisticness@10**
— the current server crown (`domain.toml` `primary`, and
`GET /api/best-models` `"primary_metric":"holisticness@10"`), where the single-GRU
family leads all recall-competent models at H ≈ 0.142.

## Preflight gates

| gate | result |
|---|---|
| **G-a** worker alive | **PASS** — C0 transitioned `queued → running` and every one of the 10 runs shows `claimed_by = snappler:4101029`. One run at a time throughout (verified by polling): no thread oversubscription, and every arm on ONE host. |
| **G-b** remote code freshness | **PASS** — `metrics.holisticness_at_k` is present on all 10 runs, so the post-2026-07-23 `seq_common.py` composite is live on the worker. (Cleared pre-batch by the orchestrator, which found the live `~/lensing-worker/predictors/` checkout STALE and rsync'd the whole `seq_*.py` family — see "Provenance / the stale-checkout finding".) |
| **G-c** Qdrant reachable from remote | **PASS** — `music@10`, `mood_coh@10`, `ild@10`, `artist_adj@10` and the `holisticness@10` composite all populate on C0 and on all 9 others; the content-metric collection loaded. |
| **G-d** paired stats reachable | **PASS** — `GET /api/runs/<C0>/predictions` → HTTP 200, 1,431 rows carrying `row_id`, `actual`, `top_k_ids` **and** per-row `mood_coh` / `ild` / `artist_adj` (so the paired ΔH session bootstrap is computable, not just the recall Δ). |
| **Gate 0** batch validity | **PASS** — C0 recall@10 **0.12299** ∈ [0.1165, 0.1265]; `holisticness_at_k` present and **0.14172** ∈ [0.132, 0.152]. |

**Statistics validated before use.** Recomputed from C0's `predictions.json`:
per-row recall reproduces the stored recall@10 exactly (0.12299091544), and
`clamp(mean mood_coh,0) × mean ild × (1 − mean artist_adj)` = **0.141723** vs the
stored `holisticness_at_k` 0.14172 — so the paired-ΔH product-of-means bootstrap
is measuring the same quantity the server stores.

## Outcome (one line)

**Dual-tower joint end-to-end fusion is REFUTED on this corpus (Rule 3): not one
of the 8 dual arms beats the h256 baseline by paired-Δ CI > 0 — the best dual arm
`latent/latent` at `fusion_layers=0` only TIES it (0.11880 vs 0.12299, Δ −0.0042
straddling 0) — and the param-matched single GRU h425 also ties the baseline
(Δ −0.0021, straddling 0), so capacity is NOT the missed lever either and single-GRU
width is now closed on PCA-192 as it already was on AE-64; this is the FOURTH
independent learned-combiner null on this corpus (XGB stacker, RRF rank-fusion,
oracle blend weights, now representation-level joint fusion) and materially
consolidates "the fixed z-blend is the combiner of record"; two genuine
sub-findings survive — the MLP fusion head is actively HARMFUL (both
`fusion_layers=0` arms beat their `fusion_layers=1` twins by paired-Δ CI > 0,
+0.016 and +0.020, so default `fusion_layers=0`) and `delta`+`cummean` IS a
genuinely complementary view pair (D-DC beats BOTH its single-view parents CI > 0)
just at a hopeless absolute level (0.0867); five arms lift holisticness@10 by
paired-ΔH CI > 0 (D-CC reaches H 0.19715, +0.055) but EVERY one of them fails the
pre-registered ±0.015 recall floor, so Rule 6 does NOT fire and the crown is
untouched — the floor caught exactly the `mood-session` degenerate mode it was
written for. Nothing registered, nothing promoted.**

## Results — Phase A, 10 runs, sorted by recall@10

All runs `succeeded` (**0 failures, 0 interruptions**), all `n_test` = 1,431, all
on `seq-20260715-131139`, all on worker `snappler:4101029`. **Bold = best cell per
metric** (artist_adj: lowest is best; ild / mood_coh / H: highest is best).
recall@10 raw = successes of 1,431. Paired Δ = per-session paired bootstrap vs
C0, 2,000 resamples, rng 1337.

| # | run id | config | params | recall@10 (raw) | paired Δ vs C0 [95% CI] | mrr | artist@10 | genre@10 | music@10 | artist_adj@10 ↓ | mood_coh@10 ↑ | ild@10 ↑ | holisticness@10 ↑ |
|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| C0 | `run-20260725-143143-97c1e-seq-nexttrack` | **ctrl-h256 (BASELINE / GATE)** — single GRU h256 | 394,944 | **0.12299 (176)** | — (reference) | 0.05341 | 0.3375 | 0.4500 | 0.4558 | 0.3474 | 0.4788 | 0.4536 | 0.14172 |
| C1 | `run-20260725-143502-be11b-seq-nexttrack` | ctrl-h425-matched — single GRU h425 (CAPACITY CONTROL) | 871,017 | 0.12089 (173) | −0.00210 [−0.01118, +0.00629] **tie** | 0.05215 | **0.3410** | 0.4577 | **0.4597** | 0.3640 | **0.4832** | 0.4506 | 0.13850 |
| D-LL-f0 | `run-20260725-143502-18b4a-seq-dualgru` | **dual-lat-lat-f0** — latent/latent, `fusion_layers=0` (**D\***, best dual arm) | 789,696 | 0.11880 (170) | −0.00419 [−0.01258, +0.00419] **tie** | **0.05631** | 0.3368 | 0.4563 | 0.4553 | 0.3630 | 0.4798 | 0.4506 | 0.13774 |
| D-LC-f0 | `run-20260725-143502-3e5d4-seq-dualgru` | dual-lat-cummean-f0 — latent/cummean, `fusion_layers=0` | 789,696 | 0.11740 (168) | −0.00559 [−0.01747, +0.00489] **tie** | 0.05087 | 0.3284 | 0.4528 | 0.4458 | 0.3557 | 0.4576 | 0.4630 | 0.13648 |
| D-LL | `run-20260725-143502-12fe6-seq-dualgru` | dual-lat-lat-f1 — latent/latent, `fusion_layers=1` | 871,872 | 0.10273 (147) | −0.02027 [−0.03494, −0.00697] **loses** | 0.04235 | 0.3089 | 0.4577 | 0.4446 | 0.3092 | 0.4348 | 0.4873 | 0.14635 |
| D-LC | `run-20260725-143502-a95c5-seq-dualgru` | dual-lat-cummean-f1 — latent/cummean (the design's "best hope") | 871,872 | 0.09713 (139) | −0.02586 [−0.03913, −0.01258] **loses** | 0.03901 | 0.3040 | 0.4549 | 0.4380 | 0.2997 | 0.4359 | 0.4913 | 0.15000 |
| D-LD | `run-20260725-143502-8d214-seq-dualgru` | dual-lat-delta-f1 — latent/delta | 871,872 | 0.09154 (131) | −0.03145 [−0.04542, −0.01817] **loses** | 0.04029 | 0.2991 | **0.4619** | 0.4385 | 0.2999 | 0.4202 | 0.4988 | 0.14673 |
| D-DC | `run-20260725-143502-e9cfb-seq-dualgru` | dual-delta-cummean-f1 — delta/cummean (no raw-latent tower) | 871,872 | 0.08665 (124) | −0.03634 [−0.05031, −0.02236] **loses** | 0.03379 | 0.2579 | 0.4242 | 0.4125 | 0.2319 | 0.3930 | 0.5413 | 0.16340 |
| D-CC | `run-20260725-143502-88a4c-seq-dualgru` | dual-cummean-cummean-f1 — VIEW ABLATION (centroid only) | 871,872 | 0.06639 (95) | −0.05660 [−0.07198, −0.04123] **loses** | 0.02664 | 0.2460 | 0.4060 | 0.3935 | **0.1899** † | 0.4649 | 0.5235 | **0.19715** † |
| D-DD | `run-20260725-143502-4324f-seq-dualgru` | dual-delta-delta-f1 — VIEW ABLATION (movement only) | 871,872 | 0.05800 (83) | −0.06499 [−0.08246, −0.04892] **loses** | 0.02819 | 0.2523 | 0.4228 | 0.4016 | 0.2363 | 0.3199 | **0.5457** | 0.13335 |

† **Not a real crown.** D-CC's H 0.19715 (the batch maximum, +0.055 over C0 with
ΔH CI entirely > 0) and its minimal artist_adj 0.1899 are bought at recall@10
0.06639 — **95 of 1,431 vs the baseline's 176**, i.e. 0.0566 BELOW C0, nearly 4×
the ±0.015 noise band. This is precisely the `mood-session` failure mode
(H 0.2332 at recall 0.024) that Rule 6's anti-degenerate recall floor exists to
block, and the floor blocked it. See Rule 6 below.

**No arm bolded as winner: there is no winner.** The single-GRU baseline C0 holds
the best recall@10 in the batch, and no dual arm reaches it by the pre-registered
statistic.

**MRR — aggregate no-regression check only.** As the design pre-registered, a
paired MRR Δ is NOT computable from `predictions.json` (it carries `top_k_ids` but
no rank field). This campaign additionally *tested* whether MRR could be
reconstructed from the `top_k_ids` ordering — it cannot (reconstruction gives
0.04605 vs the stored 0.05341), which confirms the design's caution rather than
relaxing it. **No paired MRR CI is claimed anywhere in this report.** On the
aggregate: D-LL-f0 has the batch's HIGHEST MRR (0.05631) — above both C0 (0.05341)
and C1 (0.05215) — the one metric on which a dual arm leads; the two f1
latent-view arms regress MRR sharply (0.042 / 0.039).

## Decision rule — applied mechanically

**D\*** (the dual arm with the highest recall@10) = **D-LL-f0**, 0.11880.

| rule | test | result |
|---|---|---|
| **Gate 0** | C0 recall ∈ [0.1165, 0.1265] and H present ∈ [0.132, 0.152] | **PASS** (0.12299 / 0.14172) — batch valid, verdicts issued |
| **Rule 1 — ARCHITECTURE WIN** | D\* beats **C1** on recall by paired-Δ CI > 0 | **NOT FIRED.** D-LL-f0 vs C1 = **−0.00210 [−0.01118, +0.00769] straddles 0** |
| **Rule 2 — CAPACITY-ONLY** | D\* beats C0 CI > 0, Δ vs C1 straddles/below 0, **and** C1 beats C0 CI > 0 | **NOT FIRED** on both required legs. D\* vs C0 = −0.00419 [−0.01258, +0.00419] straddles 0 (not CI > 0); and C1 vs C0 = −0.00210 [−0.01118, +0.00629] straddles 0 (C1 does NOT beat C0) |
| **Rule 3 — REFUTED** | **no** dual arm beats C0 by paired-Δ CI > 0 | **FIRED.** All 8 checked: 6 arms CI < 0 (lose outright), 2 arms (D-LL-f0, D-LC-f0) straddle 0. **Zero** CI > 0 → dual-tower joint end-to-end fusion is REFUTED on this corpus. Rule 3's rider also holds: **C1 ≈ C0**, which closes single-GRU width on PCA-192 too |
| **Rule 4 — VIEW ATTRIBUTION** | each mixed cell vs **both** single-view parents | **1 of 3 confirmed** — see below |
| **Rule 5 — FUSION DEPTH** | `D-LL` vs `D-LL-f0`, `D-LC` vs `D-LC-f0` | **BOTH f0 arms WIN CI > 0** → the MLP fusion is over-parameterized; **default `fusion_layers=0`** |
| **Rule 6 — CROWN (holisticness@10)** | ΔH CI > 0 vs C0 **AND** recall not below C0 by more than 0.015 | **NOT FIRED for any arm** — 5 arms clear the ΔH leg, all 5 fail the recall floor |

### Rule 4 — view attribution (paired Δ recall@10, each mixed cell vs both parents)

| mixed cell | vs parent 1 | vs parent 2 | verdict |
|---|---|---|---|
| **D-LC** (latent/cummean) | vs **D-LL** −0.00559 [−0.01747, +0.00699] straddles 0 | vs **D-CC** +0.03075 [+0.01747, +0.04472] CI > 0 | **NOT complementary** — beats only the weaker parent; the raw `latent` view carries the cell entirely |
| **D-LD** (latent/delta) | vs **D-LL** −0.01118 [−0.02306, +0.00070] straddles 0 | vs **D-DD** +0.03354 [+0.02027, +0.04682] CI > 0 | **NOT complementary** — same pattern; `delta` adds nothing over a second `latent` tower |
| **D-DC** (delta/cummean) | vs **D-DD** +0.02865 [+0.01468, +0.04333] CI > 0 | vs **D-CC** +0.02027 [+0.00769, +0.03424] CI > 0 | **COMPLEMENTARITY CONFIRMED** — beats BOTH parents CI > 0. The mechanism the hypothesis predicted is REAL; it just operates between two views that are individually hopeless (absolute 0.08665, far below C0) |

### Rule 5 — fusion depth (paired Δ recall@10, f0 minus f1)

| pair | Δ recall@10 [95% CI] | Δ holisticness@10 [95% CI] |
|---|---|---|
| **D-LL-f0 − D-LL** (redundant-view regime) | **+0.01607 [+0.00210, +0.03005] CI > 0** | −0.00861 [−0.01164, −0.00549] CI < 0 |
| **D-LC-f0 − D-LC** (complementary-view regime) | **+0.02027 [+0.00629, +0.03354] CI > 0** | −0.01351 [−0.01712, −0.00999] CI < 0 |

The MLP fusion head does not merely fail to earn its keep — **removing it makes
the model measurably better on recall in BOTH regimes, CI > 0 both times**, at 10 %
fewer parameters (789,696 vs 871,872). Per the pre-registered rule: **default
`fusion_layers=0`** in any follow-up, which makes the model a pure linear read of
the concatenated tower states.

### Rule 6 — crown (paired ΔH vs C0 + the anti-degenerate recall floor)

| arm | holisticness@10 | paired ΔH vs C0 [95% CI] | ΔH leg | recall@10 Δ vs C0 | floor (≤ 0.015 below C0) | Rule 6 |
|---|--:|---|---|--:|---|---|
| **D-CC** | **0.19715** | **+0.05542 [+0.05027, +0.06050]** | CI > 0 ✓ | −0.05660 | **FAIL** (3.8× the band) | **NO** |
| **D-DC** | 0.16340 | +0.02168 [+0.01735, +0.02595] | CI > 0 ✓ | −0.03634 | **FAIL** | **NO** |
| **D-LC** | 0.15000 | +0.00827 [+0.00486, +0.01170] | CI > 0 ✓ | −0.02586 | **FAIL** | **NO** |
| **D-LD** | 0.14673 | +0.00500 [+0.00157, +0.00841] | CI > 0 ✓ | −0.03145 | **FAIL** | **NO** |
| **D-LL** | 0.14635 | +0.00463 [+0.00161, +0.00754] | CI > 0 ✓ | −0.02027 | **FAIL** | **NO** |
| C0 (reference) | 0.14172 | — | — | — | — | — |
| C1 | 0.13850 | −0.00323 [−0.00492, −0.00153] | CI < 0 ✗ | −0.00210 | pass | **NO** |
| D-LL-f0 | 0.13774 | −0.00398 [−0.00561, −0.00232] | CI < 0 ✗ | −0.00419 | pass | **NO** |
| D-LC-f0 | 0.13648 | −0.00524 [−0.00846, −0.00212] | CI < 0 ✗ | −0.00559 | pass | **NO** |
| D-DD | 0.13335 | −0.00837 [−0.01358, −0.00335] | CI < 0 ✗ | −0.06499 | fail | **NO** |

**Rule 6 does not fire for any arm, and the reason is structural, not marginal:
the ΔH leg and the recall floor are ANTI-CORRELATED across the whole batch.**
Every arm that lifts holisticness does so by retreating from exact prediction, and
every arm that holds recall loses holisticness. **holisticness@10 is untouched;
the crown stays with C0's family at H ≈ 0.142 among recall-competent models.**

### Phase B triggers (pre-registered; evaluated, not improvised)

| phase | trigger | fired? |
|---|---|---|
| **B1 — seeds** (seed 7 / 99) | Rule 1 **or** Rule 6 | **NO** — neither fired |
| **B2 — second split** (`seq-20260718-211238`) | only if B1 confirms | **NO** — B1 never fired |
| **B3 — width curve** (`hidden=512`) | Rule 2, **or** C1 beats C0 by paired-Δ CI > 0 | **NO** — Rule 2 did not fire and C1 vs C0 straddles 0 (−0.00210). The width curve is FLAT at the h256→h425 step, so extending it to h512 would be spending a run to re-measure a null |
| **B4 — SAE tower-specialization read** | any mixed-view arm wins **or ties C1 within the band** | **YES** — **D-LC-f0** (mixed view latent/cummean) vs C1 = −0.00349 [−0.01607, +0.00839], well inside ±0.015 → **TIES C1** → B4 fires. Analysis-only; gates no verdict. Handed off (see Follow-ups): `POST /api/interp/model-sae` with `topk=32` on `run-20260725-143502-3e5d4-seq-dualgru`, delegated to the **information-capture-analyst** agent. Feasibility pre-checked: `has_checkpoint = true` on all 10 runs |

**What was saved: NOTHING.** A definition is created only if Rule 1 or Rule 6
fires AND Phase B reproduces it. Neither fired → **no definition registered, no
model promoted, no dataset tag changed, champion and leaderboard UNCHANGED.**

## Findings

1. **Dual-tower joint fusion is refuted — and this is the fourth independent
   learned-combiner null on this corpus.** The record already had three: the XGB
   stacker (0.080–0.087, *below its own GRU base leg*,
   `2026-07-15c-nexttrack-phase2-model-sweep.md`), RRF rank-fusion (−0.029) and
   even ORACLE test-fit blend weights (+0.004, CI straddles 0)
   (`2026-07-18-nexttrack-literature-fit-campaign.md`). Those were all
   **post-hoc, score-level** combiners, and the obvious rebuttal was always "of
   course a bolted-on combiner fails — train the fusion end-to-end so the parts
   co-adapt." This campaign ran exactly that rebuttal at the
   **representation level**, jointly, from scratch, with a param-matched control —
   and it fails too (best dual arm 0.11880, a tie with the 0.12299 baseline; six
   of eight arms lose outright). **The null is now robust across the whole
   combiner design space on this corpus, which materially consolidates "the fixed
   equal-thirds z-blend is the combiner of record."**
2. **Capacity is not the missed lever either, and single-GRU width is now closed
   on PCA-192.** C1 (h425, 871k params — 2.2× C0) scores 0.12089 vs C0's 0.12299,
   paired-Δ −0.00210 straddling 0. This fills a real gap the record flagged:
   width had only ever been scanned on AE-64
   (`2026-07-13-nexttrack-topology-scan.md`, "architecture is not the lever"), and
   the rich PCA-192 space was untested. It behaves the same way. **The
   topology-scan closure now extends to PCA-192.** Because C1 is a genuine
   0.098 % param match to the f1 dual arms, a Rule-3 null cannot be
   mis-attributed to "capacity would have helped" — it demonstrably would not.
3. **The MLP fusion head is actively harmful, not merely useless.** Both
   `fusion_layers=0` arms beat their `fusion_layers=1` twins by paired-Δ CI > 0
   (+0.01607 latent/latent; +0.02027 latent/cummean) at 10 % fewer parameters.
   Mechanistically this is consistent with the whole family arc: the objective is
   cosine/InfoNCE retrieval in the item-latent space, so the head's job is to land
   *in* that geometry — an extra 256-wide non-linear layer between the recurrent
   state and a metric read-out just adds a warp the InfoNCE loss then has to undo.
   Note the trade-off runs the other way on holisticness (both f0 arms have LOWER
   H, ΔH CI < 0): the f1 arms' extra warp diffuses the top-10 (higher ild, lower
   artist_adj) at the cost of exact hits. **Default `fusion_layers=0`.**
4. **Complementarity is real but lands on the wrong pair.** The pre-registered
   Rule-4 matrix — the reason the complete view triangle was worth 6 runs — makes
   this mechanically decidable instead of a judgment call, and it gives a genuinely
   surprising answer. The pair the design nominated as "best hope" (`latent` +
   `cummean`, the slow session centroid that `seq-ann` was built from) is **NOT**
   complementary: D-LC ties D-LL and only beats the centroid-only parent, i.e. the
   raw-latent tower carries the cell and the centroid tower is dead weight. Same
   for `latent` + `delta`. But **`delta` + `cummean` IS complementary** — D-DC
   beats BOTH parents CI > 0 (+0.0287 vs D-DD, +0.0203 vs D-CC). Read: **the
   co-adaptation mechanism the hypothesis proposed genuinely works; it is just
   that the raw `latent` view already subsumes everything the other two views
   carry.** A GRU over raw latents integrates step-to-step movement and running
   mean internally, so pairing it with an explicit computation of either is
   redundant — while two *impoverished* views (movement-only, centroid-only) do
   have non-overlapping information to pool. That is a clean mechanistic
   explanation for the null in (1), not just an absence of effect.
5. **Holisticness and recall are anti-correlated across this whole batch, and the
   anti-degenerate floor earned its place.** Five arms lift holisticness@10 by
   paired-ΔH CI > 0, topping out at D-CC's **0.19715 (+0.055)** — which on the
   naked crown metric would rank second on the entire board, behind only the
   degenerate `mood-session` (0.2332). But D-CC lands **95 of 1,431** exact hits
   against the baseline's **176**. The ordering of the batch by H is almost the
   reverse of its ordering by recall (Spearman-negative by inspection: D-CC, D-DC,
   D-LC top the H column and sit at the bottom of the recall column). **Without
   Rule 6's ±0.015 recall floor this campaign would have "won the crown" with a
   model that is 46 % worse at its actual job** — the exact `mood-session` failure
   mode. Any future holisticness objective on this corpus must be a *constrained*
   one; on the present evidence unconstrained holisticness is maximized by
   predicting less well.
6. **The two `fusion_layers=0` arms are honest ties with both controls, and one
   leads on MRR.** D-LL-f0 (0.11880) and D-LC-f0 (0.11740) straddle 0 against both
   C0 and C1 on recall, and D-LL-f0 posts the batch's best MRR (0.05631 vs C0
   0.05341, C1 0.05215) — the single metric where a dual arm leads. Per the
   noise-band discipline this is **"at least equal, likely neither better nor
   worse"**, NOT a win: the recall Δ is inside the ±0.0035 reproduction spread, and
   no paired MRR CI is computable. It is not a promotion candidate and was not
   registered. It does say the dual-tower *machinery* is sound — the new predictor
   trains, converges and evaluates cleanly at parity with the family — the
   architecture just buys nothing.
7. **`seq-dualgru` works, and its first real-data outing was clean.** 10/10 runs
   succeeded, zero failures, zero interruptions, zero relaunches; all 10 carry the
   full metric suite including the holisticness composite and per-row facets. The
   predictor is leak-free by construction (no bidirectional mode — a backward pass
   would leak the teacher-forcing target; all three views are strictly causal at
   step t, `delta` with x₋₁ = 0), which this report can now state affirmatively
   rather than as a design intention.

## Metric-contradiction reconciliation (explicit campaign deliverable)

The record contained a live contradiction about which metric is primary, and the
approved design required this campaign to state it plainly rather than silently
pick a side. **What the live server actually uses:**

- `GET /api/best-models` returns **`"primary_metric": "holisticness@10"`** — and
  the group's ranking bears it out: rank 1 is `best-seq-mood-20260723-014711-b9172`
  (`seq-mood`) at `metric_value` **0.2332**, ahead of a `seq-blend` at 0.0632. That
  is a holisticness ordering, not a recall ordering (on recall those two sit at
  ≈ 0.024 and ≈ 0.212 respectively — i.e. the ranking is *inverted* relative to
  recall).
- `domain.toml` (uncommitted, dated 2026-07-23) has `[metrics].primary` switched to
  **`holisticness@10`**.
- **`experiments/PROJECT-FACTS.md` still asserted the opposite** in several places:
  its header line says "recall@10 stays primary", §"Target & task" says
  "`[metrics].primary = "recall@10"`", and the 2026-07-20 curation note asserts
  "`[metrics].primary` stays `recall@10`, the best-models group's `primary_metric`
  is `recall@10`". Those statements were **true when written and are now stale.**

**Resolution recorded (no history rewritten):** the facts file's §"Target & task"
and §"Best-models group" now carry a dated note that the SERVER-EFFECTIVE primary
metric is `holisticness@10` as of 2026-07-23, that the earlier recall@10 statements
are superseded, and that the *experimental record's* comparability currency remains
recall@10 (the entire leaderboard is denominated in it). **This campaign judged the
scientific question on recall@10 and the board event on holisticness@10, and
pre-registered both separately** — which is why Rule 3 (refutation) and Rule 6
(crown, with its recall floor) could be applied independently and could not be
traded off against each other after the fact.

**A second, unrelated staleness found while checking this:** the best-models group
is **stale relative to this batch**. `GET /api/best-models` reports
`updated_at`/`selected_at` **2026-07-23T01:54:06** with only **2** auto entries in
a size-12 group, i.e. **the deterministic recompute did not absorb these 10 runs**,
even though all 10 carry `holisticness_at_k` and one of them (D-CC, H 0.19715)
would slot straight in at rank 2 under the current primary metric. So the Risk-8
flood did not happen automatically — but the group is not current either, and
anything that triggers a recompute will flood it. Flagged for best-model-selector,
not actioned here (curation is not the runner's call).

## Best on record after this work (RANKING board — UNCHANGED)

This campaign registered nothing and promoted nothing. The champion row, the
leaderboard and the pins are untouched. recall@10 raw = successes of 1,431.

| model | recall@10 (raw) | holisticness@10 | note |
|---|--:|--:|---|
| **★ R′+M+C′ learned content projection on PCA-192 (CHAMPION — `blend-gru-markov-content-proj`)** | **0.2117 (303)** | 0.0632 | **UNCHANGED.** Not a comparand for this campaign: a 3-leg blend over a supervised projected space vs a standalone single model would be a category error. Dual-tower-as-the-R′-leg remains untested (needs a `seq-blend` code change) |
| — R+M+C content-kNN z-blend on PCA-192 (`blend-gru-markov-content`) | 0.186 | ≈0.068 | prior crown; unchanged |
| — R+M z-blend on AE-64 (productization anchor) | 0.173 | ≈0.068 | unchanged |
| GRU LSTM on PCA-128 (best SINGLE model, `gru-lstm-pca128`) | 0.127 (182) | — | unchanged |
| **GRU infonce h256 on PCA-192 (the SINGLE-model bar this campaign was judged against)** | **0.123 (176)** | **0.1417** | **UNCHANGED and re-reproduced EXACTLY on the remote host** (C0, `run-20260725-143143-97c1e`). Still the best holisticness@10 among recall-competent models |
| — single GRU h425 on PCA-192 (NEW datapoint, this campaign) | 0.1209 (173) | 0.1385 | **TIE with h256** (paired-Δ −0.0021 straddles 0) → single-GRU width is now CLOSED on PCA-192, as it already was on AE-64 |
| — dual-tower GRU 2×256, `fusion_layers=0`, latent/latent (NEW, best dual arm) | 0.1188 (170) | 0.1377 | **TIE with both controls**; batch-best MRR 0.0563. NOT registered, NOT promoted — a tie inside the noise band is not a win |
| — dual-tower GRU 2×256, `fusion_layers=1` (six view cells, NEW) | 0.058–0.1027 | 0.1334–0.1972 | all LOSE to the baseline CI < 0. The MLP fusion head is the main culprit (Rule 5) |
| B2 first-order Markov (bar) | 0.107 (153) | ≈0.065 | unchanged |
| `mood-session` (`seq-mood`) — best-models rank 1 by the current primary metric | 0.0238 (34) | 0.2332 | **DEGENERATE** (near-zero recall). The incumbent that motivated Rule 6's recall floor |

## Pitfalls surfaced / re-confirmed

- **NEW — `seq-dualgru` `fusion_layers=1` is a documented regression mode.** The
  MLP fusion head costs 0.016–0.020 recall@10 (paired-Δ CI > 0 against it in both
  view regimes) for 10 % more parameters. If `seq-dualgru` is ever revisited,
  start at `fusion_layers=0`; do not spend runs re-discovering this.
- **NEW — the raw `latent` view subsumes `delta` and `cummean` for a recurrent
  tower.** Pairing a raw-latent GRU with an explicit movement or running-mean view
  adds nothing (both mixed cells tie their latent/latent parent). `delta`+`cummean`
  ARE mutually complementary (CI > 0 vs both parents) but hopeless in absolute
  terms. Corollary: a *single*-tower `seq-nexttrack` on `delta` or `cummean` is not
  worth a run — the ablation arms already bound it at 0.058–0.066.
- **NEW — unconstrained holisticness@10 is maximized by predicting worse.** Across
  10 arms, H ordering is close to the reverse of recall ordering; D-CC reaches
  H 0.197 at 95/1431 hits. Any holisticness objective needs a hard recall floor
  (Rule 6's ±0.015 band worked exactly as designed and blocked a degenerate crown).
- **NEW — the best-models group can be STALE, not just polluted.** It reported
  `updated_at` 2026-07-23 with 2/12 entries filled after 10 metric-carrying runs
  completed. Do not assume `GET /api/best-models` reflects the latest batch; check
  `updated_at` against the newest run's `finished_at`.
- **NEW — a remote worker checkout can silently diverge from the hub, and the
  symptom is a WRONG NUMBER, not a crash.** See "Provenance" below: the two prior
  remote `seq-nexttrack` h256 runs (`run-20260723-002044-af11a`, `-17cf8`) returned
  0.1201957 on a **stale** `seq_nexttrack.py`; on the refreshed checkout the same
  config returns **0.12299**, bit-identical to the hub. **The "cross-host
  reduction-order offset of −0.0028" recorded in the design was NOT a host effect —
  it was a code difference.** Before trusting any remote number, md5 the predictor
  family against the hub.
- **RE-CONFIRMED — the paired-Δ statistic and the noise band do real work.** Six of
  the ten arms differ from C0 by more than the ±0.015 band and are unambiguous; the
  three interesting ones (C1, D-LL-f0, D-LC-f0) all sit inside ±0.0056, i.e. inside
  even the ±0.0035 reproduction spread, and are correctly called TIES rather than
  a "+0.003 dual-tower win" or a "width regression".
- **RE-CONFIRMED — `queue:true` + one-at-a-time remote execution dodges the 85×
  thread-oversubscription collapse.** 10 torch runs, ~59 min wall clock end-to-end
  (14:31 → 15:30), zero slowdown events, every run `claimed_by` the same worker.
  Per-run times landed at the low end of the design's 8–10 min estimate.
- **RE-CONFIRMED — MRR is not paired-testable from `predictions.json`** (no rank
  field), and it is NOT reconstructible from the `top_k_ids` order either
  (0.04605 vs stored 0.05341). Use it as an aggregate no-regression check only.

## Follow-ups

- **Report-curator (sync mode) — REQUIRED, not automatic.** Fold this campaign into
  `docs/experiments.tex` (+ Spanish `docs/experiments.es.tex`) and rebuild the PDFs.
  The natural home is the combiner-null arc: this is the fourth and strongest null
  (representation-level, jointly trained, param-matched) and it closes the
  "post-hoc combiner" rebuttal. The `2026-07-13` topology-scan section should also
  gain the note that width is now closed on PCA-192, not just AE-64.
- **best-model-selector — REQUIRED (Risk 8), for a different reason than expected.**
  The group did not flood (it did not recompute at all — `updated_at` 2026-07-23,
  2/12 entries). Two things need its judgment: (a) the group is stale relative to
  10 completed metric-carrying runs; (b) if/when it recomputes, **D-CC
  (`run-20260725-143502-88a4c-seq-dualgru`, H 0.19715 at recall@10 0.06639 = 95 of
  1,431) must be excluded as a degenerate-mode entry**, along with the other 9
  campaign runs as validation-only reproductions/ablations, per the standing
  campaign-run de-duplication precedent. Flag any entry with recall@10 < 0.10 as
  degenerate. The incumbent rank-1 `mood-session` (H 0.2332 at recall 0.0238) is
  the same pathology and predates this campaign.
- **B4 SAE tower-specialization read (trigger FIRED, delegated).**
  `POST /api/interp/model-sae` with `topk=32` on **D-LC-f0**
  (`run-20260725-143502-3e5d4-seq-dualgru`, `has_checkpoint = true`), delegated to
  the **information-capture-analyst** agent. `seq_dualgru`'s `model-sae` taps
  `tower_a`, `tower_b` and each `fusion_*` layer separately, so it can directly
  answer the question the recall numbers only imply: **do the two towers capture
  different next-item concepts, or duplicate each other?** Finding 4 predicts
  duplication in the latent/latent and latent/* cells. Registered pitfall to carry
  in: the L1 SAE **will not sparsify** on these dense recurrent states — read
  `n_interpretable_concepts` as saturated and trust ranked `atoms_by_concept` +
  `next_item_decodability`. Analysis-only; gates no verdict here.
- **Do NOT spend runs on:** `seq-dualgru` with `fusion_layers=1` (Rule 5);
  single-GRU width above h425 on PCA-192 (B3 did not fire — the curve is flat);
  a single `seq-nexttrack` tower on `delta` or `cummean` (bounded at 0.058–0.066 by
  the ablation arms); `arch_a`/`arch_b` = LSTM (LSTM 0.1188 < GRU 0.1230 on
  PCA-192); `layers_a`/`layers_b` > 1 (registered depth cliff).
- **Still open, and now the better-motivated direction: dual-tower as the R′ leg
  inside the champion blend.** Out of scope here by design (needs a `seq-blend`
  code change, since its recurrent leg calls `seq_nexttrack.fit` directly). But the
  priority should be read DOWN given this result: the standalone model does not
  work, and the record's own lesson is that single-model gains do not propagate to
  the blend (content being the sole exception). Rule 1 did not fire, so the
  design's "top follow-up if Rule 1 fires" condition was not met.
- **Better-motivated than either:** the record's own standing open direction —
  fit the InfoNCE projection ON a whitened / `std-noaco` / metric space (compose
  the two known-good fixes, `2026-07-18c-nexttrack-representation-exploration.md`).
  Every architecture-side lever on this corpus is now closed (depth, width,
  residual, direction on AE-64; width on PCA-192; dual-tower fusion and fusion
  depth here), while the representation/supervision side is where every real gain
  has come from.
- **A constrained holisticness objective**, if that board is to be pursued: the H
  gains in this batch are all recall-destroying (Finding 5). The MMR λ ≈ 0.9
  eval-time re-rank flagged in `2026-07-22-session-holisticness-and-antieager-levers.md`
  remains the only near-free holisticness lever on record.

## Provenance

- **10 runs**, all `POST /api/runs` with `queue:true` on dataset
  `seq-20260715-131139`, all executed by remote worker **`snappler:4101029`**
  (verified per run via `claimed_by`), serialized one-at-a-time. Wall clock
  14:31:43 → 15:30:29 UTC (~59 min). **0 failed, 0 interrupted, 0 relaunched.**
  No definition was created per config (only a winner would have been saved, and
  there is no winner). Hub load: zero — nothing ran locally.
- **Run id ↔ config mapping** is in the Results table (every row carries its run
  id). C0 = `run-20260725-143143-97c1e-seq-nexttrack`; the other nine share the
  `run-20260725-143502-*` batch stamp.
- **Statistics:** paired per-session bootstrap, 2,000 resamples, rng seed 1337,
  paired by `predictions.json.row_id`, n = 1,431 on every comparison. recall@10 per
  session = `1[actual ∈ top_k_ids]`. holisticness@10 paired ΔH = product-of-means
  recomputed per resample (`clamp(mean mood_coh,0) × mean ild × (1 − mean
  artist_adj)`), **not** a mean of per-row products — validated against C0's stored
  `holisticness_at_k` (0.141723 computed vs 0.14172 stored). Param counts
  re-measured by instantiating `SeqNextLatent` / `DualTowerNextLatent` at
  `latent_dim = 192`.
- **The stale-checkout finding (important for reading prior remote numbers).**
  Before this batch the orchestrator discovered that the live worker directory
  (`~/lensing-worker/predictors/` — the one `run_worker.sh` `cd`s into, NOT the
  instance-path checkout) was **stale**: `seq_common.py` had **zero** occurrences
  of `holisticness_at_k`, `seq_nexttrack.py` differed from the hub by md5, and
  `seq_model_sae.py` / `seq_mood.py` / `seq_continuation_eval.py` were **missing
  entirely** — which would likely have CRASHED every dual-tower run, since
  `seq_dualgru.py` imports `make_batches`/`step_loss` from `seq_nexttrack` and
  always passes `x=` / `eager_beta=` / `eager_margin=`. The whole `seq_*.py` family
  was rsync'd and verified md5-identical to the hub (`seq_common.py`
  a50d02f6742d1305896b13a03fb131f1, `seq_nexttrack.py`
  7a85f1dac3711eea4a3157b60202ac80, `seq_dualgru.py`
  3dc92918477de79f5610ae718245584e), and the worker's `lensing-server` binary was
  refreshed (c56aecf0bedddec4d43261a6d4e5a18c on both).
  **Consequence for the record:** C0 on the refreshed checkout returns **0.12299**,
  bit-identical to the hub's on-record value — **not** the 0.1201957 that the two
  prior remote runs (`run-20260723-002044-af11a`, `-17cf8`) returned. The design's
  "cross-host reduction-order offset of −0.0028" is therefore **withdrawn: it was a
  CODE difference, not a host effect.** Gate 0's band [0.1165, 0.1265] admitted
  both values, so the gate stood either way. **0.12299 is the new same-host
  reference for this code state, and the two earlier remote datapoints are NOT
  code-comparable to this batch** (their recall numbers should be read as
  belonging to a superseded code state; they remain valid as the β0.1/β0.2
  eagerness measurements they were taken for, since those conclusions rest on
  within-batch comparisons).
- **Scratch artifacts:** run/config mapping, cached `predictions.json` per run, the
  full paired-stats matrix and the metric dump are in the campaign scratchpad
  (`mapping.json`, `paired_stats.json`, `allmetrics.json`, `preds/`).
