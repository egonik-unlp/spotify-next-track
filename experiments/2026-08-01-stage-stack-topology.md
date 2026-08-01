# Stage-stack topology — arrangements of per-step MLP × recurrence

**7 runs, all remote on `snappler:1914390` via `queue:true`, 90.0 min total compute,
~91 min wall (00:35:36Z → 02:06:54Z), 0 failed / 0 interrupted, 0 contingency runs used
(budget was ≤10).**

NOTHING PROMOTED AS CHAMPION. NO CHAMPION REGISTRATION. NOT DEPLOYED. Crown metric and
`domain.toml` UNCHANGED. The 7 arms were promoted **for measurement only** — every
`notes` field reads verbatim "promoted to enable walk evaluation; not a crown claim".
One new predictor (`seq-stack`) registered in `registry.toml`; server restarted at the
documented zero-live-runs gate. The orphaned run row was NOT touched.

## Goal

The user's question: does an explicit **stack** of per-step MLP and recurrent stages —
`ann+gru+ann+gru`, `ann+gru+gru+ann`, `ann+gru+gru+gru`, `gru+gru+gru+gru` — beat a
single GRU, on a corpus where five consecutive architecture campaigns have returned
nulls? Plus arm **E** `gru+ann+gru+gru`, chosen by the user to test *why* a leading MLP
hurt at N=1.

**Baseline / in-batch control (C0).** `seq-stack` with `stages=gru`, `hidden=256` =
**394,944 params** — by construction the functional twin of the family's h256 GRU, whose
on-record recall@10 is 0.12299 = **176/1431**. Bit-identity with 176 is NOT available and
its absence is NOT a failure: `seq_nexttrack.fit()` re-seeds on entry, so handing it a
pre-built model offsets the dropout stream even though the init weights match. The
campaign therefore gates on a **functional-equivalence proof** plus an **in-batch
control**, and every verdict below is stated *vs this batch's C0*.

**Dataset** `seq-20260715-131139` (canonical PCA-192 sequence split, 7,154 sessions,
19,402 items, dim 192, train pairs 4,865 / val 858 / **test sessions 1,431**).
`seq-20260718-211238` was NOT used: all 1,431 of its test sessions sit inside split-1's
train set, so it cannot harden anything trained here.

**Held constant across all 7 runs:** dropout 0.1, loss infonce, tau 0.07, epochs 40,
patience 6, lr 0.001, batch_size 128, val_fraction 0.15, seed 1337, k 10, mmr_lambda 1.0
(off), mmr_pool 200, artist_cap 0 (off). Only `stages` / `hidden` / `ann_hidden` vary.

**Capacity equalised to ~1,282,000 params** (mutual spread **0.445%**; A and B identical
by formula; C vs D matched to 0.015%). All 7 counts were re-verified by instantiating
`StackNextLatent` — all 7 matched the design exactly.

## Outcome in one line

**Every stacked arrangement LOSES, on BOTH surfaces, and the axis is now CLOSED: all five
4-stage arms are refuted by paired CI on one-shot recall@10 AND on the generated walk,
the 3.2× capacity control ties, and the one arm that could have explained the mechanism
(E, MLP on a gated hidden state rather than raw signed PCA) loses too — so the damage is
depth/topology itself, not the rectifier's position.**

## Gates — all four green, and two of them earned their keep

| gate | result |
|---|---|
| **P3 functional equivalence** (the one unconditional stop) | **GATE PASS** — `stages=["gru"]` computes the same function as `SeqNextLatent`: max\|Δ\| **0.00e+00** on `forward` AND `predict_next` at T = 1, 2, 5, 17, 40, params **394,944 both sides**. Run BEFORE the registry edit and server restart, so a fail would have cost nothing. |
| **G1 metric completeness** | **GREEN on all 7** — every run carries all five crown facets incl. `artist_conc_at_k`. No worker-image key stripping, no relaunch. |
| **G2 in-batch control** | **PASS** — C0 = **178/1431**, inside the pre-registered band [155, 198], and only **2 hits** off the historical 176. Exactly the small offset the re-seeding argument predicts. No relaunch spent. |
| **G3 walk instrument** | **PASS** — Δ(shipped `gru.onnx` @ a0.4/s0.3 vs C0 @ a0.4/s0.3) straddles 0 on **both** required axes: Δvibe **−0.0033** [−0.0096, +0.0029], Δstride_err **+0.0137** [−0.0063, +0.0370]. The shipped engine and the freshly-trained C0 are interchangeable on the walk's two primary axes, so V2 verdicts vs C0 are not an artefact of the base. (Δground +0.0128 CI>0 — not a G3 criterion.) |

**P4 harness reproduction: GREEN.** The default 5-arm `walk_headtohead.py` invocation
reproduced **all ten** recorded 2026-07-31b means exactly (shipped GRU 0.609 / 0.399 /
−0.089 / 9.925 / 0.345; dual l/cummean 0.610 / 0.144 / −0.053 / 11.425 / 0.351) and the
paired Δs too (champion Δvibe −0.133 CI<0; dual l/l Δvibe +0.030 CI>0; dual l/cummean
Δstride_err −0.255 CI<0, Δgenres +1.50 CI>0, Δdrift +0.035 CI>0, Δvibe +0.001 straddles).
The harness is also **bit-stable under arm addition** — the five default rows came out
identical again with 2, then 7, extra arms present, which is what licenses pairing across
invocations.

## Results — Phase 1, one-shot (n_test 1,431, sorted by recall@10)

Every H below was **recomputed from `predictions.json`** under the live 4-factor
definition and matches the stored value exactly (C1 0.023418 vs stored 0.023418036…;
D 0.018529 vs 0.018529016…).

| arm | run id | stages | params | recall@10 | hits | MRR@10* | H@10 | conc | adj | mood_coh | music@10 | ild | V1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C0 control** | `run-20260801-003536-eaf81-seq-stack` | `gru` h256 | **394,944** | **0.12439** | **178** | **0.04555** | **0.024378** | 0.3589 | 0.3482 | 0.4788 | **0.4552** | 0.4549 | base |
| C1 capacity | `run-20260801-003536-bcedf-seq-stack` | `gru` h537 | 1,280,937 | 0.12159 | 174 | 0.04579 | 0.023418 | 0.3773 | 0.3653 | **0.4856** | 0.4556 | 0.4424 | **TIE** |
| B | `run-20260801-003646-07994-seq-stack` | `ann+gru+gru+ann` h299 | 1,281,407 | 0.08595 | 123 | 0.02554 | 0.017558 | 0.3176 | 0.2639 | 0.3845 | 0.4264 | 0.5112 | REFUTED |
| E | `run-20260801-003646-8cc2d-seq-stack` | `gru+ann+gru+gru` h259 | 1,277,321 | 0.08246 | 118 | 0.02894 | 0.021053 | 0.2704 | 0.2376 | 0.3964 | 0.4310 | 0.5278 | REFUTED |
| C | `run-20260801-003646-dbe68-seq-stack` | `ann+gru+gru+gru` h256 | 1,283,008 | 0.07966 | 114 | 0.02684 | 0.021442 | 0.2626 | 0.2375 | 0.3925 | 0.4324 | 0.5288 | REFUTED |
| D | `run-20260801-003536-d612e-seq-stack` | `gru+gru+gru+gru` h229 | 1,282,821 | 0.07687 | 110 | 0.02566 | 0.018529 | 0.2644 | 0.2326 | 0.3831 | 0.4222 | 0.5279 | REFUTED |
| A | `run-20260801-003646-6e3bf-seq-stack` | `ann+gru+ann+gru` h299 | 1,281,407 | 0.07058 | 101 | 0.02080 | 0.017374 | **0.2626**† | **0.2335** | 0.3821 | 0.4168 | **0.5373** | REFUTED |

\* MRR truncated at 10, recomputed from `predictions.json`. The **stored** `mrr` is over
the FULL ranking and is therefore not pairable from stored predictions; stored values are
C0 0.05264, C1 0.05376, A 0.02740, B 0.03213, C 0.03354, D 0.03241, E 0.03543 — same
ordering. † lowest `conc` is C's 0.2626 and A's 0.2730; the bolded low-eagerness cells
belong to arms that are refuted, which is the point of the recall floor.

**V1 paired bootstrap vs the in-batch C0** (2000 resamples, rng 1337, n = 1,431):

| arm | Δrecall@10 | 95% CI | measured hw | ΔMRR@10 | ΔH | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| C1 | −0.00280 | [−0.01188, +0.00559] | 0.00874 | +0.00025 straddles | −0.00096 | [−0.00216, +0.00016] | **TIE** |
| B | −0.03843 | [−0.05381, −0.02306] | 0.01537 | −0.02000 CI<0 | −0.00682 | [−0.00937, −0.00445] | **REFUTED** |
| E | −0.04193 | [−0.05660, −0.02795] | 0.01433 | −0.01661 CI<0 | −0.00333 | [−0.00585, −0.00089] | **REFUTED** |
| C | −0.04472 | [−0.06010, −0.03005] | 0.01502 | −0.01871 CI<0 | −0.00294 | [−0.00558, −0.00048] | **REFUTED** |
| D | −0.04752 | [−0.06219, −0.03284] | 0.01468 | −0.01988 CI<0 | −0.00585 | [−0.00841, −0.00341] | **REFUTED** |
| A | −0.05381 | [−0.06848, −0.03913] | 0.01468 | −0.02475 CI<0 | −0.00700 | [−0.00967, −0.00445] | **REFUTED** |

No arm meets any WIN clause. Every 4-stage arm loses recall **and** H by CI, and all five
sit **below the 0.108 absolute recall floor** (0.0706–0.0860). Derived per-baseline
relevance floor from C0's own `music@10` = 0.45519: every 4-stage arm is 5.0–8.4% below
it, so none is even relevance-neutral. Losses are 2.5–3.7× the measured half-width — this
is not a noise-band question.

## Results — Phase 2, the walk surface (40 sessions × 20 steps, artist cap 1, paired)

Primary cell **a0.4 / s0.3**, matched for every arm, base = C0. `vibe` is always reported
with `stride_err` beside it, at matched stride, per the standing metric caveat.

| arm | vibe↑ | stride_err↓ | drift~0 | genres↑ | ground↑ | Δvibe (CI) | Δstride_err (CI) | V2 |
|---|---|---|---|---|---|---|---|---|
| GRU shipped a0.4 s0.0 | **0.609** | 0.399 | −0.089 | 9.925 | 0.345 | +0.0376 CI>0 | +0.2833 CI>0 | reference |
| shipped `gru.onnx` a0.4 s0.3 (continuity) | 0.568 | 0.129 | −0.080 | **10.450** | **0.357** | −0.0033 straddles | +0.0137 straddles | **G3 pass** |
| **C0 gru256 a0.4 s0.3** | **0.571** | 0.116 | −0.080 | 10.325 | 0.344 | base | base | base |
| C1 gru537 | 0.561 | 0.122 | −0.082 | 10.625 | 0.359 | −0.0100 straddles | +0.0065 straddles | **indistinguishable** |
| D gggg229 | 0.557 | 0.122 | −0.071 | 10.225 | 0.319 | −0.0140 [−0.0255, −0.0022] | +0.0067 straddles | **REFUTED** ×2 |
| C aggg256 | 0.548 | **0.113** | −0.064 | 10.250 | 0.340 | −0.0231 [−0.0381, −0.0084] | −0.0029 straddles | **REFUTED** |
| B agga299 | 0.545 | 0.126 | **−0.061** | 9.675 | 0.335 | −0.0258 [−0.0421, −0.0108] | +0.0105 straddles | **REFUTED** |
| E gagg259 | 0.540 | **0.109** | −0.084 | 9.225 | 0.335 | −0.0316 [−0.0494, −0.0144] | −0.0072 straddles | **REFUTED** |
| A agag299 | 0.534 | 0.117 | −0.085 | **9.125** | 0.336 | −0.0375 [−0.0559, −0.0214] | +0.0009 straddles | **REFUTED** |

Every 4-stage arm fires the REFUTED clause "Δvibe CI<0 AND Δstride_err not CI<0". **D
fires it twice**, also failing the grounding floor (Δground −0.0250 [−0.0474, −0.0048]
CI<0). A additionally loses genre variety (Δgenres −1.200 [−2.125, −0.300] CI<0). No arm
approaches either WIN clause.

**Stride probe at a0.4** for the only two arms with any directional movement toward a V2
win (C and E were the sole arms with a negative point-estimate Δstride_err):

| arm | s0.2 | s0.3 | s0.4 |
|---|---|---|---|
| C aggg256 — vibe / stride_err | 0.586 / 0.183 | 0.548 / **0.113** | 0.517 / 0.153 |
| E gagg259 — vibe / stride_err | 0.576 / 0.155 | 0.540 / **0.109** | 0.516 / 0.174 |

**Both peak at s0.3 — the cell they were already measured in.** Neither arm is rescued by
re-tuning stride, and at s0.4 both lose vibe CI<0 while stride_err gets *worse*, not
better. This also re-confirms the anchor/stride complementarity from the record (raising
stride past the matched point overshoots).

**Secondary cell a0.8 / s0.5** (the deployed engine's operating point), matched pair for
the single best arm C:

| arm | vibe↑ | stride_err↓ | drift~0 | genres↑ | ground↑ |
|---|---|---|---|---|---|
| C0 gru256 a0.8 s0.5 | **0.624** | 0.150 | −0.057 | **11.225** | **0.355** |
| C aggg256 a0.8 s0.5 | 0.608 | **0.141** | **−0.056** | 11.100 | 0.346 |

Δvibe **−0.0158** [−0.0267, −0.0059] **CI<0**, Δstride_err −0.0094 straddles ⇒ **REFUTED
at the deployed operating point too.** (Worth noting for the showcase: C0 at a0.8/s0.5
reaches vibe 0.624 / stride_err 0.150, a *better* vibe than the deployed dual l/cummean's
0.610 / 0.144 at the same cell — a single-GRU/dual re-comparison, not a claim this
campaign is authorized to make.)

## Epoch diagnostics — and why the dropout-count contingency did NOT fire

| arm | epochs run | best epoch | best val | final val | final train | stop |
|---|---|---|---|---|---|---|
| C0 `gru` h256 | 14 | 8 | 6.5832 | 6.5925 | 5.8406 | early stop |
| C1 `gru` h537 | 14 | 8 | 6.5723 | 6.5811 | 5.8107 | early stop |
| A `ann+gru+ann+gru` | 21 | 15 | 6.5960 | 6.6182 | 5.8194 | early stop |
| B `ann+gru+gru+ann` | **40** | **36** | **6.5378** | 6.5413 | **5.5765** | **ran to cap** |
| C `ann+gru+gru+gru` | 34 | 28 | 6.5842 | 6.5904 | 5.6671 | early stop |
| D `gru+gru+gru+gru` | 34 | 28 | 6.5977 | 6.6074 | 5.6866 | early stop |
| E `gru+ann+gru+gru` | 34 | 28 | 6.5795 | 6.5836 | 5.7048 | early stop |

The pre-authorized contingency required every 4-stage arm to lose V1 **AND stop EARLIER
with higher val loss than C0**. They all lose V1, but **not one stops earlier** — every
4-stage arm trains 21–40 epochs against C0's 14, and **B and E reach a LOWER best val
loss than C0** (6.5378 and 6.5795 vs 6.5832). The trigger fails on both clauses, so **the
2 contingency runs were not spent** and the 4× dropout application is exonerated as the
explanation. The arms are not being regularized or early-stopped out of the race.

**This is the campaign's most interesting number.** B achieves the **lowest validation
InfoNCE loss in the entire batch** (6.5378, better than C0's 6.5832) while retrieving
**55 fewer hits** (123 vs 178). The objective the whole family trains on and top-10
retrieval quality **come apart at depth**: a deeper stack fits "what follows this" better
and *retrieves* worse. Every prior architecture null on this corpus was consistent with
"the extra machinery does nothing"; this one shows the extra machinery does something
measurable to the loss and that it is the *wrong* something.

## V4 contrasts — recorded regardless of arm verdicts

| contrast | question | Δrecall@10 (CI) | ΔH (CI) | reading |
|---|---|---|---|---|
| **C vs D** | leading `ann` at depth 4, params matched to 0.015% | +0.00280 [−0.00699, +0.01258] straddles | **+0.00291 [+0.00109, +0.00479] CI>0** | A leading per-step MLP at depth 4 is **recall-neutral and H-POSITIVE**. It does NOT reproduce the N=1 harm. |
| **A vs B** | MLP placement, params **IDENTICAL** | **+0.01537 [+0.00280, +0.02797] CI>0** (B>A) | +0.00018 straddles | **BRACKETING beats INTERLEAVING.** `ann+gru+gru+ann` beats `ann+gru+ann+gru` at byte-identical capacity. The cleanest topology fact in the batch. |
| **mean(A,B) vs C** | recurrence count (2 vs 3) | +0.00140 [−0.00804, +0.01014] straddles | H 0.017466 → 0.021442 | Number of recurrent stages is **flat** on recall; the third recurrence buys only H (bought by de-eagering below the floor). |
| **C1 vs C0** | capacity, 3.24× params | −0.00280 [−0.01188, +0.00559] straddles | −0.00096 straddles | **TIE on both surfaces.** Capacity is not the lever; **C1 does NOT win, so no verdict is re-referenced to C1.** |
| **E vs A** | rectifier position | **+0.01188 [+0.00140, +0.02236] CI>0** | **+0.00368 CI>0** | E beats the worst arrangement. |
| **E vs B** | rectifier position | −0.00349 straddles | +0.00349 CI>0 | E ties B. |
| **E vs C** | rectifier position | +0.00280 straddles | −0.00039 straddles | E ties C. |

## Findings

1. **The stage-stack axis is CLOSED (V3 satisfied).** Every arm is refuted-or-
   indistinguishable on **both** surfaces with G3 green: the five 4-stage arms are
   refuted by paired CI on one-shot recall AND on the walk, and the capacity control is
   indistinguishable on both. This is the **sixth consecutive architecture null** on this
   corpus, and — unlike the five before it — it is a null that **satisfies the 2026-07-31b
   standing rule**, because it carries a walk-surface read rather than resting on
   one-shot metrics alone. It does not merely fail to close on one surface; it closes on
   two.

2. **The user's mechanism hypothesis for arm E is REFUTED, and that is the most reusable
   fact here.** The record's N=1 finding was that `MLP256→GRU256` loses −0.0140 CI<0
   because "the ReLU destroys signed prefix information the GRU's gated linear input
   transform preserves". E was designed to test the corollary: put the MLP on a **gated
   hidden state** instead of raw signed PCA and the harm should vanish. It does not. E
   reaches 118/1431 — it **ties** B and C, beats only the worst arrangement A, and comes
   nowhere near C0's 178. So half-wave rectification of the *raw* input is **not** the
   mechanism; whatever is wrong survives moving the rectifier downstream of a gate. One
   run bought a clean refutation of a plausible and cheap-to-believe story.

3. **The damage is depth, and it is not capacity, regularization, or optimization
   shortfall.** Three controls close those doors simultaneously: the 3.24× capacity
   control C1 ties C0 on both surfaces (so it is not "bigger is better"); every 4-stage
   arm trains *longer* than C0 with 2 of 5 reaching *lower* val loss (so it is not
   early-stopping or 4× dropout); and the leading-`ann`-at-depth-4 contrast C vs D is
   recall-neutral (so it is not the MLP either). What is left is the depth of the
   recurrent chain itself, and its cost is large — 55 to 77 hits out of 1,431.

4. **Bracketing beats interleaving, at byte-identical capacity.** `ann+gru+gru+ann` beats
   `ann+gru+ann+gru` by +0.01537 recall CI>0 with **the same parameter count by formula**
   — the tightest-controlled topology comparison this campaign could make. Interposing a
   per-step MLP *between* two recurrences is worse than putting MLPs on the outside. This
   is consistent with (3): breaking the recurrent chain in the middle costs more than
   adding depth at the ends.

5. **A leading per-step MLP is NOT harmful at depth 4** (C vs D: recall straddles, ΔH
   +0.00291 CI>0), which **narrows** rather than confirms the N=1 pre-encoder result. The
   record's "leading ann is harmful" claim was measured at N=1 against a single GRU; at
   depth 4 the same addition is free-to-slightly-good. The pre-encoder pitfall should be
   scoped to the shallow case.

6. **Objective and retrieval dissociate at depth** (see the epoch table). B's val InfoNCE
   is the best in the batch and its recall is 55 hits worse than C0's. Any future
   architecture search on this family that selects on validation loss will therefore
   **select the wrong arm**. Select on `recall@10` and the walk, not on the training
   objective.

7. **The eagerness facets improve monotonically as the arms get worse, and the floors
   caught it.** Every 4-stage arm de-eagers substantially (conc 0.359 → 0.263–0.318, adj
   0.348 → 0.233–0.264) and raises `ild` (0.455 → 0.511–0.537); under a naive H-primary
   read some would look interesting. They are all killed by the **0.108 absolute recall
   floor** and by the per-baseline relevance floor (`music@10` 5.0–8.4% below C0's
   0.45519). This is the same mechanism that has now saved four campaigns; the floors
   remain mandatory. **Derive the relevance floor per baseline** — the canonical 0.43
   absolute figure would have passed several of these arms.

8. **Phase 0 — the N=2 `ann+gru` relative was walk-evaluated for the first time, and it
   too is refuted on the walk.** `dual-premlp256-f0` (`run-20260726-161801-a6423`, tower A
   bare / tower B pre-MLP 256, `fusion_layers=0`, recall 0.12369) had been closed on a
   one-shot recall TIE, which under the standing rule says nothing about the walk. At the
   matched cell a0.4/s0.3 vs the shipped GRU: Δvibe **−0.0255 [−0.0432, −0.0096] CI<0**,
   Δstride_err −0.0229 straddles ⇒ **REFUTED**. Its one real gain is **less drift**
   (Δdrift +0.0403 [+0.0086, +0.0719] CI>0, −0.080 → −0.039). At a0.8/s0.5 it reaches
   vibe 0.602 / stride_err 0.146 / ground 0.364, its best cell. This is the **same
   signature the LSTM showed on 2026-07-31b** — better stride/drift, worse vibe — which
   now looks like a general property of adding pre-recurrence machinery rather than
   anything specific to either arm. Its stored H 0.04454 lacks `artist_conc_at_k` (dead
   metric generation) and was NOT compared to the crown board.

9. **The walk harness is now demonstrably a stable instrument.** It reproduced ten
   recorded means from 2026-07-31b exactly, reproduced them again under 2 and then 7
   added arms (so arm addition does not perturb the shared session sample or RNG), and
   G3 showed the shipped ONNX engine and a freshly-trained C0 are interchangeable on
   `vibe` and `stride_err`. Cross-invocation pairing is therefore licensed, which is what
   made the stride probe and secondary cell affordable.

10. **`ground` again discriminated almost nothing** — it separated only D (−0.0250 CI<0)
    out of nine arms, consistent with the recorded arithmetic (between-arm range 0.0060
    against a per-session sd of 0.1417, needing ~2,140 sessions). It is a floor that
    catches degeneracy, not a ranking metric, and it was used only that way.

### Two process findings

11. **The `seq_common.py` md5 gate fired, and it fired on a file the design did not tell
    me to copy.** P2 named `seq_stack.py` and `registry.toml` for copying but named
    `seq_common.py` among the files to *verify* — and verification failed (hub
    `09e7aef666455d49e33f187d73615bda` vs worker `03e14e7c246ba6521ea315805d002157`). The
    divergence was the 2026-07-31 `artist_cap` k-scaling block in `predict_ranking` (+
    `import math`), 19 lines, touching no metric key and inert at `artist_cap=0` / k=10 —
    so this batch would have been valid either way. But that is a *post-hoc* judgement
    that required reading the diff; the rule that caught it is the one worth keeping.
    **Verify the md5 of every file in the family's import closure, not just the ones you
    copied.**

12. **The approved launch body was rejected by the approved registry block.** The design's
    JSON sent `arch`, `bidirectional`, `residual`, `num_layers`, `eager_beta`,
    `eager_margin`; the design's registry block declares none of them, so all three
    chunk-1 POSTs failed with `unknown hyperparameter arch`. Resolved by dropping the six
    keys **after proving it is a no-op** — `seq_stack.DEFAULTS` supplies exactly
    `False/False/1/"gru"/0.0/0.0`, and `seq_stack.load_hp()` returns an **identical**
    resolved dict either way (verified by direct diff: zero differences). The registry
    block was left exactly as approved.

## Best on record after this work — UNCHANGED

No row moves. Nothing here approaches either board.

| board | holder | value | note |
|---|---|---|---|
| **recall@10 (comparability currency)** | `blend-gru-markov-content-proj` + `markov_gate` — `run-20260728-005349-507bb-seq-blend` | **0.23201 = 332/1,431** | untouched; best arm this campaign is C0 at 0.12439 = 178/1,431 |
| **CROWN — holisticness@10** | `run-20260730-235524-93d38-seq-blend` (gate + `artist_cap=3`) | **0.048063** at recall 0.17540 | untouched; best H this campaign is C0's **0.024378**, about half the leader |
| **walk surface (deployed)** | `dualgru.onnx` latent/cummean f0 @ a0.8 / s0.5 | vibe 0.610 / stride_err 0.144 | untouched; not re-litigated here |
| single-GRU h256 reference | this batch's C0 `run-20260801-003536-eaf81-seq-stack` | 0.12439 = 178/1,431, H 0.024378 | new in-batch reproduction of the h256 anchor, 2 hits off the historical 176 |

## Follow-ups

1. **The 3-seed confirm is the recommended next step and was NOT pre-authorized —
   it needs the user's go-ahead.** Six runs (C0 and the best-on-walk arm C at seeds
   {1337, 7, 42}) would convert "refuted on one split, one seed" into a hardened result.
   Given effect sizes of 2.5–3.7 measured half-widths, the outcome is not in serious
   doubt, so this is confirmation rather than discovery — spend it only if the axis is
   worth closing formally in the record.
2. **B is epoch-capped and its recall is therefore a lower bound.** It ran to the 40-epoch
   cap with `patience=6` never triggering (best epoch 36) and was still improving. A
   single `epochs=120` re-run of B would settle whether the arrangement is merely
   under-trained — though finding (6) predicts a better val loss and *not* better recall,
   which is precisely what makes it a clean test of the dissociation.
3. **Scope the pre-encoder pitfall to the shallow case** in light of C vs D: "a leading
   per-step MLP is harmful at N=1" is measured; "harmful at depth 4" is refuted.
4. **The exposure-bias / training-regime axis remains the untested one.** All six
   architecture nulls held the objective, batching and early stopping byte-identical
   while the product rolls 20 steps on its own output. Finding (6) sharpens this: the
   family's one-step teacher-forced InfoNCE is now *demonstrably* misaligned with the
   quantity we rank on. Scheduled sampling on the single GRU at `fusion_layers=0` is
   measurable today via `seq_continuation_eval.py --rollout`.
5. **`best-model-selector` is needed** — the 7 measurement promotions entered the
   server's top-12 auto group (ranks 9, 11, 13, 14 and below at recompute
   2026-08-01T02:07:10Z), including refuted arms at recall 0.07–0.08. The group needs
   re-curation for family diversity and exclusion of refuted arms. The server also
   auto-created `best-seq-stack-20260801-003536-d612e` from arm D (recall 0.07687).
6. **Consider deleting the six refuted measurement promotions** once the walk numbers are
   banked; they exist only to have been walk-evaluated. Not done here — deletion was not
   authorized.
7. **C0 at a0.8/s0.5 out-vibes the deployed dual-tower at the same cell** (0.624 / 0.150
   vs 0.610 / 0.144). Unpaired across families and outside this campaign's mandate, but
   it is a cheap head-to-head the harness can now settle in one invocation.
8. **The orphaned run row `run-20260730-234635-7a297-seq-blend`** (claimed_by
   `snappler:404147`, `/stop` 409s, `DELETE` refuses) is still `running` and still needs a
   direct Postgres `UPDATE`. Not touched — not authorized.
