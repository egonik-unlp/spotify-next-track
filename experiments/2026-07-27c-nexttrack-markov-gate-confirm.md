# 2026-07-27c — `markov_gate`: the Markov leg must ABSTAIN where it has no evidence — native two-split confirm

## Goal

Confirm natively, on two disjoint leak-free splits, the **`markov_gate`** lever
proposed by `2026-07-27b-nexttrack-model-topology-research-memo.md` (a RESEARCH
MEMO — it ran and promoted nothing). The lever is a **per-query evidence gate**,
not a learned combiner: on queries where the last prefix item `u` has ZERO
train-level bigram support (`n_u == 0`), the Markov leg ABSTAINS and the z-blend
divisor renormalizes from `(z_g + w_m·z_m + z_c)/(2+w_m)` over three legs to two.

**Mechanism (why this is not the refuted learned-combiner lane).**
`markov_scorer` (`predictors/seq_baselines.py:172-201`) mixes its bigram row with
an artist/genre back-off via `trust = n_u/(n_u+8)`. At `n_u = 0`, `trust = 0` and
the leg is **PURE back-off** — it carries no track-level evidence at all. But
`_zcand` (`predictors/seq_blend.py:169-172`) is **scale-invariant**: it rescales
whatever vector it is handed to unit variance, so the evidence-free back-off is
re-inflated and speaks at a full one-third weight. On the canonical split that is
**795 of 1,431 test queries = 55.6%**.

This is NOT the refuted B3 oracle reweight. B3 searched **global static scalar**
weights and found +0.004 straddling 0; the bucket decomposition below proves that
no constant can find this effect, because dropping M globally is a TIE (−0.0070)
— the `n_u = 0` gain and the warm-bucket loss cancel exactly.

**Code state (implemented and verified before this campaign, by the
orchestrator, not by this runner):** `predictors/seq_blend.py` gained the
`markov_gate` bool hyperparam (default `False`); `registry.toml` exposes it on
the `seq-blend` block; lensing-server was restarted at 0 live runs and
`GET /api/predictors` confirms the param. All 117 prior runs intact.

**Baseline / anchor — the recall champion.** `blend-gru-markov-content-proj`
(`seq-blend`), the best-on-record row since 2026-07-18b. Fresh in-batch control
`run-20260728-004444-b32ea-seq-blend`: recall@10 **0.21174004192872117
(303/1431)**, MRR 0.12198540476105842, artist@10 0.33263, genre@10 0.40112,
music@10 0.453790, mood_coh 0.398222, ild 0.406374, artist_adj 0.609713,
artist_conc 0.691653, **H 0.008603**.

Held hyperparams (identical in all six arms except the stated overrides):
`alpha 0.5, arch gru, batch_size 128, content false, content_agg max, epochs 40,
hidden 256, k 10, loss infonce, lr 0.001, patience 6, projection true,
projection_epochs 40, projection_lr 0.001, projection_objective infonce,
projection_rank 0, projection_tau 0.07, seed 1337`.

**Datasets.** Canonical `seq-20260715-131139` (PCA-192, 7,154 sessions / 19,402
items, 5,723 train / 1,431 test @ cut 2024-08-24T14:28:02, cold-item rate 0.55)
and the disjoint earlier-holdout second split `seq-20260718-211238` (0.64 cold,
n_test 1,431). No dataset work required.

**Controls re-run FRESH.** C1/C2/C3 were re-trained in this batch rather than
substituted from the archive, so every paired comparison is code-matched to the
same generation.

**PREFLIGHT GATE (hard stop, pre-registered).** C1 had to reproduce recall@10
0.21174, exactly 303/1431, proving `markov_gate=false` left the champion
bit-identical. **It did, to the last digit** (0.21174004192872117 = 303/1431).

## Outcome in one line

**CONFIRMED on both splits and the mechanism is exactly as predicted: gating the
Markov leg to abstain at `n_u = 0` buys paired Δrecall@10 +0.02027 [+0.01048,
+0.03075] on the canonical split and +0.01957 [+0.01048, +0.03005] on the
second, with MRR up on both, ZERO movement in the warm buckets (their top-k
lists are BIT-IDENTICAL), and — unexpectedly — a crown gain too; and stacked
with MMR λ0.7 it produces the first arm on record that is recall-NEUTRAL against
the champion (0.21314 vs 0.21174, 305 vs 303 hits, CI straddles 0) at H 0.030980
= 3.6× the champion's, which would be CROWN-BOARD RANK 2 at nearly double any
incumbent's recall.**

## Results

Six runs, launched serially against the hub (never `queue:true` — see
§Infra finding), base definition `blend-gru-markov-content-proj`.
`ds1 = seq-20260715-131139` (canonical), `ds2 = seq-20260718-211238`.

| id | run id | ds | overrides | recall@10 | hits/1431 | MRR | H@10 | music@10 | ild | artist_adj | artist_conc | mood_coh | artist@10 | genre@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A1** | `run-20260728-005349-507bb-seq-blend` | ds1 | `markov_gate=true` | **0.23201** | **332** | **0.13352** | 0.018699 | 0.49319 | 0.43377 | 0.57477 | 0.59821 | **0.46226** | 0.36688 | 0.46261 |
| A3 | `run-20260728-012030-6d1b7-seq-blend` | ds1 | `markov_gate=true, mmr_lambda=0.7, mmr_pool=200` | 0.21314 | 305 | 0.12852 | **0.030980** | **0.51649** | **0.56770** | **0.51279** | **0.47325** | 0.41885 | **0.38225** | **0.49686** |
| C1 | `run-20260728-004444-b32ea-seq-blend` | ds1 | — (champion; preflight gate) | 0.21174 | 303 | 0.12199 | 0.008603 | 0.45379 | 0.40637 | 0.60971 | 0.69165 | 0.39822 | 0.33263 | 0.40112 |
| C3 | `run-20260728-011245-bfc67-seq-blend` | ds1 | `mmr_lambda=0.7, mmr_pool=200` | 0.19357 | 277 | 0.11560 | 0.015289 | 0.47592 | 0.52010 | 0.57135 | 0.60823 | 0.36433 | 0.34731 | 0.41999 |
| A2 | `run-20260728-010658-39e20-seq-blend` | ds2 | `markov_gate=true` | 0.19008 | 272 | 0.11801 | 0.008414 | 0.42953 | 0.41401 | 0.58672 | 0.64847 | 0.42618 | 0.33543 | 0.39762 |
| C2 | `run-20260728-010113-e2599-seq-blend` | ds2 | — (champion on 2nd split) | 0.17051 | 244 | 0.10498 | 0.003392 | 0.39616 | 0.40177 | 0.61013 | 0.71945 | 0.35364 | 0.29909 | 0.34172 |

No run failed, was interrupted, or was relaunched. (Rows sorted by recall@10;
ds2's absolute recall is lower BY CONSTRUCTION — the 0.64-cold split — so read it
only through the paired Δ.)

### Paired bootstrap (2,000 resamples, rng 1337, n = 1,431, paired by `row_id`)

ΔH is computed **product-of-means per resample** — all five facets re-aggregated
on the SAME resampled index set for both arms, then multiplied — not as a mean of
per-row products.

| comparison | Δrecall@10 [95% CI] | hw | verdict | ΔH [95% CI] | recall floor 0.108 | n_disc (+/−) |
|---|---|---|---|---|---|---|
| **A1 − C1** (gate, canonical) | **+0.02027 [+0.01048, +0.03075]** | 0.01013 | **CI>0 — WIN** | +0.010096 [+0.008491, +0.011669] | both PASS | 57 (+43/−14) |
| **A2 − C2** (gate, 2nd split) | **+0.01957 [+0.01048, +0.03005]** | 0.00978 | **CI>0 — WIN** | +0.005022 [+0.003917, +0.006164] | both PASS | 56 (+42/−14) |
| A3 − C3 (gate, under MMR λ0.7) | **+0.01957 [+0.00908, +0.03005]** | 0.01048 | **CI>0 — WIN** | +0.015691 [+0.013581, +0.017854] | both PASS | 64 (+46/−18) |
| A3 − C1 (gate+MMR vs champion) | +0.00140 [−0.01188, +0.01468] | 0.01328 | **TIE (straddles)** | **+0.022377 [+0.019941, +0.024923]** | both PASS | 96 (+49/−47) |
| C3 − C1 (MMR λ0.7 alone) | −0.01817 [−0.02935, −0.00767] | 0.01084 | CI<0 — costs recall | +0.006686 [+0.005693, +0.007695] | both PASS | 60 (+17/−43) |

**MRR (aggregate no-regression check only — MRR is not paired-testable from
`predictions.json`).** Every gated arm is UP on its control: A1 0.13352 vs C1
0.12199 (+0.01154); A2 0.11801 vs C2 0.10498 (+0.01303); A3 0.12852 vs C3
0.11560 (+0.01292), and A3 is also above the champion C1 (+0.00653). **No
regression anywhere.**

### Bucket decomposition — reproduced NATIVELY, through the real `build_train_transitions`

Bucketed by the train-level bigram support `n_u` of each query's last prefix item
(the exact quantity `markov_scorer` computes at line 178).

**Canonical split, A1 vs C1:**

| bucket | n | C1 | A1 | Δ | n_disc | top-k lists bit-identical? |
|---|---|---|---|---|---|---|
| `n_u = 0` | 795 | 0.1660 (132) | 0.2025 (161) | **+0.0365** | 57 | no |
| `n_u 1–10` | 292 | 0.3116 (91) | 0.3116 (91) | +0.0000 | 0 | **YES** |
| `n_u > 10` | 344 | 0.2326 (80) | 0.2326 (80) | +0.0000 | 0 | **YES** |

**Second split, A2 vs C2:**

| bucket | n | C2 | A2 | Δ | n_disc | top-k lists bit-identical? |
|---|---|---|---|---|---|---|
| `n_u = 0` | 892 | 0.1244 (111) | 0.1558 (139) | **+0.0314** | 56 | no |
| `n_u 1–10` | 322 | 0.2578 (83) | 0.2578 (83) | +0.0000 | 0 | **YES** |
| `n_u > 10` | 217 | 0.2304 (50) | 0.2304 (50) | +0.0000 | 0 | **YES** |

**Canonical split under MMR λ0.7, A3 vs C3:**

| bucket | n | C3 | A3 | Δ | n_disc | top-k lists bit-identical? |
|---|---|---|---|---|---|---|
| `n_u = 0` | 795 | 0.1472 (117) | 0.1824 (145) | **+0.0352** | 64 | no |
| `n_u 1–10` | 292 | 0.2945 (86) | 0.2945 (86) | +0.0000 | 0 | **YES** |
| `n_u > 10` | 344 | 0.2151 (74) | 0.2151 (74) | +0.0000 | 0 | **YES** |

The canonical A1/C1 decomposition matches the memo's banked offline screen
(0.1660→0.2025 on 795 queries; warm buckets unmoved) **to four decimal places and
to the hit**. Stronger than the memo could show: the warm-bucket **top-k lists
are byte-for-byte identical**, so the gate is a surgically pure `n_u = 0`
intervention with literally zero warm-side cost, on three independent
control/arm pairs.

### MMR-validity gate

Required by the registered pitfall: MMR silently no-ops when the content-metric
index is unreachable, so `ild` must differ from the control. **PASS** — C3 ild
0.52010 and A3 ild 0.56770 against C1's 0.40637; and C3 reproduces the
2026-07-26 λ frontier's λ0.7-on-champion recall **0.19357 exactly**.

### Decision-rule application (pre-registered)

- A1 − C1 paired Δrecall@10 CI entirely > 0 on the canonical split — **met**.
- A2 − C2 paired Δ > 0 on `seq-20260718-211238` (CI also entirely > 0) — **met**.
- No aggregate MRR regression on either split — **met** (both up).

**→ `markov_gate` is CONFIRMED.** Nothing was promoted, registered or tagged;
per the design's boundary this report carries a promotion RECOMMENDATION only.

## Findings

**1. The defect is real, is `_zcand`'s, and is worth 29 hits.** The blend's
z-normalization is scale-invariant by construction — it maps every leg's score
vector to unit variance so the three legs are commensurable. That is exactly what
makes it robust to the legs' wildly different score scales, and exactly what
destroys their **confidence**. A Markov leg that has literally never seen the
current track emits a pure artist/genre back-off; `_zcand` re-inflates it to the
same footprint as a GRU that has actually modelled the prefix, and the blend
averages the two as equals. Turning that one-third of the vote off where it is
uninformed converts 43 misses into hits and 14 hits into misses on the canonical
split — net +29, +9.6% relative on the best model on record.

**2. Why no global weight could have found this — and why B3 is not refuted by
this result.** The bucket table is the whole argument. In the warm buckets the
Markov leg is genuinely informative and any global down-weight costs recall
there; in the `n_u = 0` bucket it is pure noise and any global down-weight helps.
The 2026-07-18 oracle test-fit search over **static scalar** weights returned
+0.004 straddling 0 precisely because those two effects cancel — dropping M
globally is a −0.0070 tie. Both results are true simultaneously. The registered
"four independent learned-combiner nulls" pitfall demanded a mechanism that is
"not capacity, not co-adaptation, and not score-level fusion"; **per-query
evidence gating is exactly that third thing** — no parameters are learned, no
weights are fit, and the intervention is conditioned on a *train-side data
statistic*, not on a score. The pitfall should be narrowed accordingly (see
§Record amendments).

**3. The gate is a crown lever as well as a recall lever, which was NOT
predicted.** ΔH is CI>0 in all three paired comparisons (+0.010096 canonical,
+0.005022 second split, +0.015691 under λ0.7), and the facet decomposition says
why: abstaining kills the *bigram-echo* component of the recommendation. On the
canonical split artist_conc drops 0.69165 → 0.59821 (−0.0934) and artist_adj
0.60971 → 0.57477, i.e. the list stops repeating one artist, while mood_coh
*rises* 0.39822 → 0.46226 and music@10 rises 0.45379 → 0.49319. Every one of the
four crown factors moves the right way at once. This is the first lever on record
that de-eagers the champion **without** paying in mood coherence — the exact
double-payment the λ frontier campaign identified as structural ("the champion's
recall comes from artist-adjacency, so de-eagering it also destroys its
mood_coh"). The reason is that here the de-eagering is not a re-rank fighting the
model's own scores; it is the removal of a vote that had no business being cast.

**4. The headline for the crown board: A3 is the first FREE crown arm at
champion-level recall.** Standing alone, MMR λ0.7 on the champion is a priced
lever — C3 − C1 is −0.01817 recall, CI<0, for +0.006686 H. Add the gate and the
recall cost is paid back exactly: A3 − C1 is **+0.00140 [−0.01188, +0.01468], a
statistical tie**, at **H 0.030980** — 3.6× the champion's 0.008603, ΔH +0.022377
CI>0, comfortably above the mandatory 0.108 recall floor at 0.21314. On the crown
board that H sits at **rank 2**, behind only the λ0.7-pool200 single GRU
(0.034171) — which scores it at recall **0.11461**. A3 delivers 96% of the
incumbent crown leader's H at **1.86× its recall**. The gate does not merely add
to MMR; it changes MMR's *tier* on the champion from TIER-2 PRICED to TIER-1
FREE.

**5. Gate and MMR are ADDITIVE, not substitutive.** The gate-alone Δ is +0.02027
and the gate-under-λ0.7 Δ is +0.01957 — indistinguishable, and the bucket
decomposition shows why: MMR re-ranks the top-200 of a scored list, the gate
changes which list gets scored, and they act on disjoint parts of the pipeline.
The `n_u = 0` bucket gains +0.0365 without MMR and +0.0352 with it; warm buckets
are bit-identical in both. Two independent levers that compose.

**6. The best RECALL number on record is A1, and it is not close.** 0.23201
(332/1431) against the champion's 0.21174 (303), paired CI>0 on two disjoint
splits, MRR up on both, artist@10 0.33263 → 0.36688 and genre@10 0.40112 →
0.46261. For context, the champion's own coronation margin over the prior R+M+C
crown was +0.0259 on the canonical split and +0.0426 on the second; this gate is
+0.0203/+0.0196 on top of *that*, from a nine-line change with no new parameters
and no additional training cost.

**7. Honest caveat — the screen was selection-optimistic, this run is the
out-of-sample test.** The memo screened **10** gating variants on the TEST set,
so the banked +0.02027 was optimistic by construction and could not itself be
evidence. What redeems it here: (a) the effect reproduces on a **second, disjoint
0.64-cold split that was not screened on**, at +0.01957 — within 0.0007 of the
canonical figure; (b) it reproduces under a third condition (λ0.7) that was not
screened either; (c) four mechanistically-related variants in the memo all landed
at +0.019–0.020, i.e. the screen was not selecting a lucky corner but measuring a
plateau; (d) the warm buckets do not move by a single hit, which is a *structural*
prediction of the mechanism, not a fitted outcome. The residual risk is that the
`n_u = 0` **bucket boundary** itself was chosen with test knowledge; a soft-gate
(scaling the Markov leg by `trust` before `_zcand` rather than a hard `n_u == 0`
abstention) would test the boundary-free version of the same hypothesis. That is
the top follow-up, not a caveat on the verdict.

**8. Nothing about this is a fluke of one dataset's cold rate.** The second split
is 0.64-cold vs the canonical 0.55, and the `n_u = 0` bucket grows accordingly
(892 vs 795 queries = 62.3% vs 55.6%) while the per-query gain shrinks slightly
(+0.0314 vs +0.0365). Net Δ is nearly identical. The lever's value scales with
how much of the query distribution has no bigram evidence, which is the majority
on any split of this corpus.

## Infra finding (blocking, resolved by routing — no state was changed)

The remote training worker on host `snappler` (`lensing-server worker
--hub-url http://192.168.0.7:8096 --poll-secs 5`, pid 573145) **is live and its
`~/lensing-worker/predictors/seq_blend.py` contains ZERO occurrences of
`markov_gate`** — a stale checkout predating the code change. Had any gated arm
been dispatched there it would have executed the UNGATED path and silently
returned a null, which is precisely the registered "remote-worker checkout can
silently DIVERGE" pitfall.

It could not happen here: remote dispatch is **opt-in** via `"queue": true` on
`POST /api/runs` (`crates/lensing-server/src/api.rs:886`, `runs.rs::enqueue_run`),
and every run in this batch was posted without it, so all six executed on the hub
against the verified code (confirmed by `claimed_by: null` on every run). Runs
were posted **one at a time** rather than as a batch because the hub runs at the
default `--max-runs 2` and the design called for serialization; the documented
~85× thread-oversubscription collapse was avoided. Each blend run took ~5–9 min;
the batch took ~1h05m wall.

**No remediation was performed** — freshening or cycling the remote worker is an
orchestrator action. Until it is done, **any `queue:true` `seq-blend` run will
silently ignore `markov_gate`.**

## Best on record after this work

RANKING board (canonical split `seq-20260715-131139`, n_test 1,431):

| model | recall@10 | hits | MRR | artist@10 / genre@10 | H@10 | status |
|---|---|---|---|---|---|---|
| **★ `blend-gru-markov-content-proj` + `markov_gate` (A1) — NEW BEST RECALL ON RECORD** | **0.23201** | **332** | **0.13352** | 0.36688 / 0.46261 | 0.018699 | `run-20260728-005349-507bb-seq-blend`; paired-Δ vs champion **+0.02027 [+0.01048, +0.03075] CI>0**, hardened on the 2nd split **+0.01957 [+0.01048, +0.03005] CI>0**, MRR up on both. **NOT registered, NOT promoted — recommendation only** |
| — same + MMR λ0.7/pool200 (A3) | 0.21314 | 305 | 0.12852 | 0.38225 / 0.49686 | **0.030980** | `run-20260728-012030-6d1b7-seq-blend`; recall TIE with champion (+0.00140 straddles), **ΔH +0.022377 CI>0** → the first TIER-1 FREE crown arm at champion recall. Crown-board rank 2 |
| — `blend-gru-markov-content-proj` (prior champion row, λ=1.0, ungated) | 0.21174 | 303 | 0.12199 | 0.33263 / 0.40112 | 0.008603 | unchanged; registered + promoted + servable. 10th bit-identical reproduction (`run-20260728-004444-b32ea-seq-blend`) |
| — same + MMR λ0.7/pool200, ungated (C3) | 0.19357 | 277 | 0.11560 | 0.34731 / 0.41999 | 0.015289 | reproduces the 2026-07-26 λ0.7 champion arm exactly; TIER-2 PRICED without the gate |
| — `blend-gru-markov-content` (prior crown, promoted+servable) | 0.186 | — | 0.103 | 0.331 / 0.394 | — | unchanged |
| — R+M z-blend AE-64 (productization anchor) | 0.173 | — | 0.096 | 0.322 / 0.397 | — | unchanged |
| B2 first-order Markov (bar) | 0.107 | — | 0.069 | 0.298 / 0.389 | — | unchanged |

CROWN board (holisticness@10, recall floor 0.108 mandatory) — new entries only:

| arm | H@10 | recall@10 | hits | run id | would-be rank |
|---|---|---|---|---|---|
| **champion + gate + MMR λ0.7/pool200 (A3)** | **0.030980** | **0.21314** | 305 | `run-20260728-012030-6d1b7-seq-blend` | **2** (behind λ0.7-pool200 GRU 0.034171 @ recall 0.11461) |
| champion + gate (A1) | 0.018699 | 0.23201 | 332 | `run-20260728-005349-507bb-seq-blend` | 7 |
| champion + MMR λ0.7/pool200, ungated (C3) | 0.015289 | 0.19357 | 277 | `run-20260728-011245-bfc67-seq-blend` | 8 |

Second split `seq-20260718-211238` (validation only): champion 0.17051 (244) /
MRR 0.10498; + gate 0.19008 (272) / MRR 0.11801.

**Best-models group:** the server's deterministic top-12-by-H recompute ran on
each completion and has already auto-promoted four of this batch's runs — A3 at
**rank 2**, A1 at rank 7, C3 at rank 9, C1 at rank 12. That is server behaviour,
not a campaign action; the group now holds five `seq-blend` entries from two
generations and should be re-curated by the best-model-selector agent.

## Record amendments (reconciled into `PROJECT-FACTS.md`)

1. **The learned-combiner pitfall is NARROWED.** It should forbid *learned
   score-level fusion* (fitted weights, stackers, rank fusion, end-to-end
   representation fusion) — NOT *per-query evidence gating*. B3's oracle null and
   this +0.020 win are both real and are reconciled by the bucket decomposition:
   a global static scalar cannot express a gate whose sign flips with `n_u`.
2. **`_zcand`'s scale-invariance erasing per-leg CONFIDENCE is a general defect**,
   of which the Markov leg's `n_u = 0` back-off is one instance. Any leg with a
   query-dependent notion of "I have no evidence here" is mis-weighted by the
   z-blend the same way. The content and GRU legs have not been audited for this.

## Follow-ups

1. **Register + promote A1 (and evaluate A3 for the crown) — needs user
   approval.** Suggested names `blend-gru-markov-content-proj-gated` and
   `blend-gru-markov-content-proj-gated-mmr-l07`. Both are recommended; A1 as the
   new champion ROW, A3 as the crown candidate that finally makes MMR free on the
   blend. Explicitly out of scope for this campaign by design boundary.
2. **Soft gate — the boundary-free version of the hypothesis.** Scale the Markov
   leg by `trust = n_u/(n_u+8)` BEFORE `_zcand` instead of hard-abstaining at
   `n_u == 0`. This removes the test-chosen bucket boundary (the one residual
   selection concern), is strictly more general, and if it matches or beats the
   hard gate it retires the caveat entirely. 2 runs (canonical + 2nd split) plus
   the two controls already banked here.
3. **Audit the other two legs for the same confidence defect.** Does the content
   leg have a query-dependent evidence measure (e.g. cold-item neighbourhood
   density) that `_zcand` is likewise erasing? Same mechanism, possibly the same
   size of prize.
4. **3-seed confirm of A1.** `seq-blend` exposes `seed`; the projection is
   seed-varied, so 1337/7/99 is a genuine confirm here (the champion's own
   coronation used exactly this). Cheap: 3 runs, ~30 min.
5. **Freshen the `snappler` worker checkout** (orchestrator action) before any
   `queue:true` run of `seq-blend` — see §Infra finding.
6. **Re-run the λ frontier on the GATED champion.** λ0.7 was the only point
   tested here; the gate changed MMR's tier, so the whole λ curve on the blend
   should be re-screened offline from the A1 checkpoint (per the registered
   screen-offline-then-bank rule) and only the 2–3 bankable points run natively.
