# RESEARCH MEMO — which models or topologies can improve the next-track lab — 2026-07-27

> **THIS IS A RESEARCH MEMO, NOT A CAMPAIGN REPORT.** NOTHING WAS RUN on the
> server, NOTHING PROMOTED, NOTHING REGISTERED; the champion, the leaderboard,
> the crown board and the best-models pins are all UNCHANGED by this document.
> It was produced by a 15-agent research workflow (4 scouts → 5 proposal lenses →
> 5 adversarial screeners → synthesis; 2.08M tokens) plus offline scoring against
> frozen checkpoints. Where it reports a number, that number came from an OFFLINE
> screen over stored artifacts — not from a native run. Native confirmation of the
> headline item is the campaign that follows this memo.

## Independently verified by the orchestrator (not just agent-reported)

Re-derived from scratch after the workflow returned, because the headline claim
would otherwise drive a campaign on an agent's say-so:

| check | result |
|---|---|
| `seq_baselines.py:198` — `scores = trust*track + (1-trust)*backoff`; at `n_u=0`, `trust=0` ⇒ the leg is PURE artist/genre back-off | confirmed by reading |
| `_zcand` (`seq_blend.py:169-172`) z-normalizes to unit variance ⇒ rescales that evidence-free vector back to a FULL one-third weight | confirmed by reading |
| test queries whose last prefix item has zero train-bigram support | **795 / 1431 = 0.5556** — reproduced by an independent script, matching the agents' figure exactly |
| ungated control (the self-validating gate: must equal the champion) | **0.21174** ✓ exact |
| gated (`drop M when n_u==0`) | **0.23201 — Δ +0.02027 [+0.01048, +0.03075], CI>0** |
| bucket decomposition | `n_u=0`: 0.1660→0.2025 (795 q). `n_u 1-10`: 0.3116→**0.3116**. `n_u>10`: 0.2326→**0.2326** — the warm buckets are BIT-IDENTICAL |

**NOT verified by the orchestrator** (agent-reported only, treat as provisional):
the +0.01957 second-split figure, the replay channel's +0.02865, the
`latent_gamma` numbers, and the recomputed 0.0706 discordance / hw≈0.0145 noise
floor. The first of these is exactly what the follow-up campaign must establish.

## Caveat on the headline number

The 10 gating variants were screened ON THE TEST SET, so quoting the best arm's
CI is optimistic. Mitigating it: four mechanistically-related variants (V1/V2/V6/V7)
all land at +0.019–0.020; the bucket decomposition is a mechanical explanation
rather than a fit; and the warm buckets do not move by a single hit. The native
two-split confirm is the real gate.

---

# Which models or topologies can improve the results?

**Memo to the owner — 2026-07-27. Decision-grade. Read the headline first; it is not the answer you asked for.**

---

## 1. The honest headline

**"A better topology" is the wrong question, and the record has now answered it six times.** A1 (cell/loss on AE-64), A2 (depth/width/bidirectional), A3 (width on PCA-192), A4 (dual-tower fusion), A5 (MLP fusion head — an *active regression*, +0.016..+0.027 recovered by deleting it), A6 (per-step pre-encoder — a measured **loss**, Δ −0.01398 CI<0, mechanism identified down to `frac_exact_zero = 0.5327 == frac_negative_coords = 0.5327`), and A7 (rollout doesn't separate the family either). Five lenses were pointed at this question this week. Every architecture-shaped proposal they produced came back WEAKENED, SUB_NOISE, or fatally pre-killed. Two came back with *measured negative results* — the split-Markov leg lost at every point of its own pre-registered grid, and the artist-factored softmax head was killed by its own pre-kill run (top-5 artist gating on the real checkpoint gives **Δ exactly 0.0**, because the trained GRU's top-10 already lives inside its own top-5 artists).

Worse for the topology question: the noise floor is **larger than advertised**. Recomputed paired discordance between genuinely different architectures (run-20260715-150442-6c67b vs run-20260726-161801-0df6f) is 101/1431 = 0.0706, giving hw ≈ **0.0145**, not ±0.01. Several of the nulls above were measured with an instrument that cannot resolve +0.01. They are still nulls — every arm was nominally down or flat — but nobody should spend another run looking for a +0.005 architecture effect on a single 1,431-session split.

**Where the evidence actually points: the model is not the constraint. What it is asked to rank against, and which legs are allowed to speak on a given query, are.** Every proposal that survived adversarial screening is a candidate-set or evidence-conditioning change, and the two strongest were *already measured offline before this memo was written*:

| survivor | measured Δrecall@10 | board |
|---|---|---|
| Evidence-gated Markov leg | **+0.02027** canonical [+0.01048,+0.03075] **and +0.01957** on the disjoint 0.64-cold split | champion (0.21174 → 0.23201) |
| Causal cross-session replay channel | **+0.02865** (176 → 217 hits) | single GRU (0.12299 → 0.15164) |

Both are score-time, both are ~12–70 LOC, both were reproduced by an independent adversary. Neither is a topology.

---

## 2. Ranked shortlist

**Ranking criterion:** expected paired Δ on a *named* board, divided by total cost (LOC + native runs + governance risk), with a hard multiplier for candidates already screened offline against a bit-identity-verified control. Measured beats theorized; cheap beats deep; a proposal that needs a new board column pays a penalty.

---

### 1. `markov_gate` — evidence-gate the Markov leg (restore the trust `_zcand` erases)

**Mechanism.** `markov_scorer` computes `trust = n_u/(n_u+8)` at `seq_baselines.py:182`; `_zcand` (`seq_blend.py:169-172`) is scale-invariant and destroys it. On **795/1431 (55.6%)** canonical test queries the last prefix item has *zero* train bigram support, so the leg is pure `0.6·P(artist)+0.4·P(genre)` back-off (solo recall 0.019) yet still contributes a mean max-z of 10.09 (GRU: 3.77) at a full one-third weight. Gate M by `1{n_u>0}` and renormalize the divisor. Zero learned parameters.

**Not a refuted lane.** B3's oracle bound is over **global, static, scalar** weights — and the bucket decomposition is a mechanical proof it cannot reach this: the entire gain sits in the n_u=0 bucket (0.1660 → 0.2025) while the n_u>0 buckets are **bit-identical** (0.3116 and 0.2326 unchanged). Dropping M globally (V5) is a tie (−0.0070) because the two buckets cancel exactly. Not pitfall #1 either — nothing is learned, nothing co-adapts.

**Board & size.** Champion recall **+0.0203 / +0.0196** (both splits, CI>0, independently reproduced). Cold-target recall 0.1891 → 0.2246. Crown **H 0.0086 → 0.0187 (2.17×)**; gated + MMR λ0.7/pool200 screens at recall 0.2131 / **H 0.0310** — rank-2 all-time crown at 1.77× the current crown holder's recall.

**Cost.** ~12 LOC in `predictors/seq_blend.py` (`build_blend_score_fn`) + one bool in `registry.toml:1866-2015`. Serving is free — `train()` and `predict()` both route through the same builder.

**First experiment (4 runs, serialized).** A1 = champion + `markov_gate=true` on `seq-20260715-131139`; A2 = same on `seq-20260718-211238`; A3/A4 = both with `mmr_lambda=0.7, mmr_pool=200`. Preflight: an ungated re-run must reproduce **303/1431**. **Decision:** promote iff paired Δ CI excludes 0 on **both** splits and MRR does not regress in aggregate; crown claims need ΔH as product-of-means per resample + the 0.108 floor + the `ild ≠ control` MMR-validity gate.

---

### 2. Causal cross-session replay channel

**Mechanism.** 76.17% of the 1,431 LLO targets were played in a *strictly earlier* session and are not in the current prefix. Nothing in the family can see this: Markov is a train-frozen bigram, and `seq_baselines.recency_scorer` scores only prefix items — which `seq_common.py:401-404` sets to `-inf`, so its recorded 0.001 is **dead code, not a measurement**. Add a leg scoring candidates by a calibrated prior over age-since-last-play and lifetime count, with history advanced causally as eval walks test sessions.

**Not a refuted lane.** C4 died on the absence of a cross-*user* neighbourhood — orthogonal to intra-user temporal state. C1/C2 died on co-occurrence data volume; recency estimates no co-occurrence. Verified causal: `train_sessions == arange(0,5723)`, `test_sessions == arange(5723,7154)`, chronological split. Hit-Jaccard with the GRU is **0.060** and with the champion **0.057** (vs 0.201 for the GRU/Markov pair that produced a working blend) — this is the "genuinely different signal" exception B4 explicitly names.

**Board & size.** Single GRU **+0.0287 measured** (217 vs 176 hits). Union ceilings: 0.1873 with the GRU, **0.2690** with the champion. Expect **+0.012..+0.035** on the champion. Crown will **fall** (replays are artist-concentrated) — report it, do not claim it.

**Cost & the catch.** ~20 LOC in `seq_common.py:390-411` (keyword-only session index; existing predictors keep bit-identity) + ~70 LOC leg in `seq_blend.py`. **The catch is governance, not code:** this is an information-set amendment. The rankable ceiling moves 0.4493 → 0.7617; three closure rationales (C1/C2/C4) partly rest on "55% cold", which is measurably a *staleness* artifact of freezing the item table at 2024-08-24 against a 22-month test horizon. **This needs its own board column and your sign-off before Stage B.**

**First experiment.** Stage A, **zero training**: z-blend the frozen h256 checkpoint (run-20260725-143143-97c1e) with the online-recency vector; same-code control at w=0. Weight chosen on a *causal* train slice [4292,5723), grid extended to w ∈ {0.5,1.0,1.5,2.0,3.0} (the earlier probe was still rising at the edge). **Decision:** proceed only if Δ > +0.03 against the same-checkpoint control. Mandatory second arm: **exclude the immediately-preceding session** — if the win evaporates, it is 30-minute-gap sessionizer fragmentation, not taste.

---

### 3. `latent_gamma` — spectral temperature on the item latents

**Mechanism.** Cosine over PCA weights coordinate *k* by σ²_k, but next-track pair-correlation is flat-to-rising across the spectrum (coords 0-10: +0.255; 10-32: **+0.403**; 32-64: +0.312) while variance is concentrated (coords 0-10 = 62%). Rescale each coordinate by σ_k^−γ. One scalar, no parameters, no training.

**Not a refuted lane.** D3's V2/V3 standardize the **raw 4-block input before PCA** (producing a different basis, eff. dim ~76); γ rescales the canonical PCA **output**. Not A6 — no parameters, no rectifier, and it moves query and candidate *together*, which is exactly the case A6's expressivity-neutrality proof does not cover. Prospective negative control passed: on AE-64 (flat variance share, flat pair-correlation) it is a null, 0.0636 → 0.0643.

**Board & size — and the reframe that matters more than the number.** Verified: R+M+C 0.1859 → **0.2124** at γ0.75, paired Δ +0.0266 [+0.0140,+0.0377]. But at 0.2194 (γ1.0) vs the champion's 0.21174 this is a **tie**, not a new row. The real finding: **the champion's entire recall contribution is reproducible by a 192-number diagonal rescale** (M+C′ = M+C at γ0.75 = 0.2082 exactly; whitening the already-projected space adds only +0.004). The single-GRU arm is genuinely unmeasured territory (prior +0.01..+0.03 from 18c readout C).

**Cost.** ~15 LOC + 2 registry blocks. **One correction to the design:** the offline win whitens *only the content leg*; `_projected_artifact` swaps the artifact globally (`seq_blend.py:307`, before `fit` at `:313`), so a naive `latent_gamma` at load also retrains the GRU in the whitened space — a different arm. Expose a per-leg `content_gamma` so the banked number is the one native eval reproduces.

**First experiment.** Gates G1 (`seq-nexttrack` γ=0 → 0.12299091544374564 / 176) and G2 (`seq-blend` γ=0 → 0.1859). Then **A1/A2 = seq-nexttrack γ∈{0.75,1.0}** — the only new territory, and the arm worth the runs. Pre-register γ=1.0; read 0.75 as exploratory (the sweep was scored on test). **Decision:** paired Δ vs its own γ=0 twin, CI>0 on canonical, paired Δ>0 on the second split.

---

### 4. `tau` for the GRU + prefix-only negative mask

**Mechanism.** `tau=0.07` has **never been swept for any GRU in this family** — held byte-for-byte through the dual-tower and pre-encoder scans, and absent from the `seq-nexttrack` registry block, so the API physically cannot set it. Separately, ~32.5 of ~1,644 in-batch negatives per row are same-session, holding ~13-17% of the softmax denominator mass at τ=0.07. The *right* subset to mask is not all of them but the ones whose target **already occurred earlier in that session** — those are hard-masked at eval (`scores[prefix] = -inf`), so training against them is pure train/serve misalignment. Blanket same-session masking over-masks the legitimate future candidates and strips the only near-positive negatives.

**Not a refuted lane.** C6 closed *adding* negatives and *reweighting by support* — its α=0.00 / n_neg=8192 arm is effectively uniform over the warm support at ~53% coverage and nulled at 0.1223 vs 0.1230. That bounds the bank's **support and sampling distribution**; it says nothing about **removing** a contaminated subset or about τ. D4 left τ explicitly open (τ0.10 beat the promoted 0.07 by +0.0112, never seed- or split-checked).

**Board & size.** Single-GRU recall, honestly **−0.003..+0.008** per cell — sub-noise individually. The value is the **pre-registered interaction**: if masking works by removing false-negative mass, its benefit must be larger at τ=0.07 (17.5% mass) than τ=0.10 (11.9%), and should *substitute* for the τ gain rather than add. Either result closes the lab's oldest live near-miss.

**Cost.** One line in `registry.toml:1719-1855` for `tau` (default 0.07 ⇒ every existing definition stays bit-identical); ~15 LOC in `seq_nexttrack.py` (`make_batches` yields owner ids; `step_loss` sets masked off-diagonals to −inf). **Available today with zero code:** `seq-embed` exposes `tau` *and* `embed_mode=frozen`, which is a designed `seq-nexttrack` reproduction gate.

**First experiment.** 2×2, {mask off/on} × {τ 0.07, 0.10}, riding inside another batch. Control cell must reproduce 176/1431.

---

### 5. `seq-embed` disposition — constrained residual, or retire the module

**Mechanism.** `predictors/seq_embed.py` is registered (`registry.toml:2329`) with **zero runs** out of 117 on the server. `embed_mode=residual` keeps the frozen content vector as a buffer and trains a per-item delta — the RecSys-2025 content-initialization shape, whose whole finding is that the delta must be **constrained**, which `residual_scale` (init magnitude only, `seq_embed.py:178-180`) does not do.

**Not a refuted lane.** Never executed. Not the free-table claim the data volume kills — cold items retain a retrievable representation by construction.

**Board & size.** Expect a **documented null**, −0.02..+0.01. Its value is closing a registered-but-unrun module and pinning the top of a capacity ladder: shared diagonal (192 numbers, #3) → shared linear (36,864, W) → per-item residual (3.7M). If 3.7M per-item parameters lose to 192 numbers, the item-level representation lane is finished.

**Cost.** Zero to run the existing cells; ~10 LOC for `residual_l2`.

**First experiment.** E0 frozen (gate, judged on **hit count**, tolerance ±2 hits — `seq_embed` gathers latents in-graph while `seq_nexttrack` gathers in numpy, so ULP drift is expected), E1 residual, E3 residual+L2, E4 free/content. **Mandatory cold/warm decomposition** — a per-item table can only help the warm 45%, and pooled recall dilutes any effect by ~55%.

---

### 6. Measurement: folds 3+4 pooled, fold-blocked bootstrap, + the seed diagnostic

**Mechanism.** `train_sessions.u32` is a contiguous arange in every artifact, and `seq-20260718-211238` **is** fold 3 of a rolling origin (train 0..4291). The lab currently spends both splits as a *conjunction* of two CI>0 gates — power 0.369 at a true Δ=+0.010 — when pooling the same two runs gives 0.885.

**Correction to the proposal as filed.** The quoted d=0.0287 is the median over *all* archived pairs and is dominated by near-identical arms; for real architecture pairs d = 0.0706–0.0867. Realistic gain is hw 0.0145 → ~0.0102 (folds 3+4), not 0.0044. And the pooled bootstrap **must be fold-blocked**: 2026-07-27's S4c decomposition found the *checkpoint* component (+0.072) significant while the architecture component (+0.05) was not — a flat session-level resample treats a 4-draw random effect as 5,724 independent draws and will be anti-conservative.

**Cost.** ~150 LOC (`fold_split` in `seq_common.py`, four call sites, two registry params, `tools/pooled_paired_bootstrap.py`). **Footgun:** a `fold` param that silently changes the split while `dataset_id` stays constant is pitfall #6 in a bottle — write the fold into `metrics.json` and the run label, default to the full split.

**Free by-product, and the reason to do it:** folds 1–4 trace the **learning curve** (see §6).

---

### 7. Leg-abstention v2 — reliability scalars, offline screen only

**Mechanism.** `markov_gate` is one instance of a general defect: `_zcand` makes every leg speak at the same volume regardless of whether it has evidence. Give each leg a leg-intrinsic scalar r∈[0,1] and blend `Σ r_l·z_l / Σ r_l`.

**Two corrections that matter.** The GRU's pre-normalization output norm is **not** a confidence proxy — `step_loss` L2-normalizes `p` before the logits, so the loss is exactly invariant to ‖pred‖ and no weight decay exists; measured corr with correctness **+0.0073**, quartiles non-monotone. Use the **top-1-minus-next-10 score margin** instead (corr **+0.294**). And calibrate on *train* queries — min-max across the test set is transductive and not reproducible at serving.

**Board & size.** +0.000..+0.008 on top of the gate; remaining per-query selection headroom is +0.032 (union of solo top-10s 0.2642 vs gated 0.2320). **Offline only**, with a permuted-reliability negative control; native runs only if the real arm wins and the permuted arm ties.

---

### 8. Exogenous artist-level PPMI — Stage 0 audit only

**Mechanism.** The dead bucket is precise: champion recall by bucket is warm 0.2395 (n=643), cold/artist-warm 0.2688 (n=465), **cold/artist-cold 0.0743 (n=323, 24 hits)**. An external multi-user co-occurrence prior restricted to this vocab is the one leg type that removes C4's structural killer rather than renaming it.

**Why it is ranked last.** Coverage, measured: of those 323 targets, **68.4% released after 2016, 63.5% after 2020**, median artist_popularity 38 vs corpus 54. Every openly-downloadable corpus predates that mass (#nowplaying-RS is 2014-16; MPD is no longer distributed). Track-level Stage 0 will most likely return NO-GO. **Pivot to artist-level PPMI over the 4,551 artists** — the bucket has only 169 distinct artists and artist coverage in a 2014-17 corpus is far higher. Report artist- and track-level coverage separately, gate on the post-2020 subset explicitly.

---

## 3. Top 3 — do these first

**Batch 1 (this week, ~6 native runs, serialized `--max-runs 1`, no server restart).**
1. `markov_gate` A1–A4 (both splits, ±MMR λ0.7). Preflight gate: ungated re-run → 303/1431.
2. **Stage A of the replay channel — zero training**, run *in parallel with nothing*, it is a scoring job: frozen-checkpoint z-blend + online recency, plus the exclude-previous-session control.
3. `latent_gamma` gates G1/G2 + arms A1/A2 (single GRU γ∈{0.75,1.0}) — the only genuinely unmeasured cell in that lane.

**These three share one batch cleanly** (1 and 3 are both `seq-blend`/`seq-nexttrack` native runs on `seq-20260715-131139`; 2 costs no training slot). Ride the **τ 2×2** and the **seq-embed E0 gate** as filler cells in the same batch — both are one-line/zero-code and both need the same bit-identity controls you are already paying for.

**Do not add the deltas.** `markov_gate` (+0.0203) and the replay channel (+0.0287) both target the same n_u=0 / staleness-cold mass. The gate lifts n_u=0 from 0.1660 → 0.2025; the replay leg reaches history-warm items that Markov's frozen bigram cannot score. Overlap is likely substantial. **Batch 2 must be the interaction arm**, gate × replay, judged against gate-alone — not against the ungated champion.

**Batch 3 (governance-gated):** the replay learned/hand-built gate, on a **new board column**, with a same-protocol control (GRU with the history channel zeroed), never against the frozen-protocol 0.12299. Do not start it before you have decided the information-set question.

---

## 4. Graveyard

| idea | why it died |
|---|---|
| Full-catalogue / support-restricted softmax CE | Pre-empted by C6's α=0.00 / n_neg=8192 arm (~53% support coverage, uniform): **0.1223 vs 0.1230**. Literature's #1 candidate does not transfer because the item side is frozen. |
| Factored artist softmax head (`gamma·log P(artist)`) | Its **own pre-kill fired**: top-1/2/3/5 artist gating on run-20260725-143143-97c1e gives −0.0196 / −0.0084 / −0.0056 / **exactly 0.0**. The trained GRU's top-10 already lies inside its top-5 artists (mean 5.22 distinct). Only the aux-loss cell (γ=0) survives, as one rider. |
| Split Markov into M_track + M_backoff | **Measured**, every point of its own grid: w=0 −0.0217, 0.1 −0.0112, 0.25 −0.0168, 0.5 −0.0294, 1.0 −0.0384, all CI<0; graded proxy also falls. Separate z-norm structurally destroys `trust` inside the n_u>0 regime. |
| Pool-matched stage-2 reranker | Train-pool positive density **0.5975 vs test 0.3368** — the legs memorize their own sessions, so "pool-matched" is unobtainable and the proposal's own go-gate (≥0.25) passes *because of* the confound. Pitfall #2, unengaged. |
| Mixture-of-vMF / K-vector head | An 8,067-fold LOO **oracle** decomposition already loses (K=2 −0.0017, K=3 −0.0030). Crown escape hatch runs into E3 (every H-lifter retreats from recall) with only 0.015 of floor slack. |
| Distil the blend into the projection | Two of three teacher channels are circular (content-kNN is cosine in the space W transforms; Markov's bigrams are already the InfoNCE target support). The one non-circular channel is the artist/genre back-off — which is the **graded-relevance projection objective**, a cheaper and already-flagged lane. |
| Diffusion / DreamRec / flow-matching | TORS 2026 reproduction: 25% reproducible, tuned baselines win. DreamRec is a cosine regression loss with extra steps, and `loss=cosine` is this lab's weaker objective (0.038-0.041 vs 0.089). |
| Generative retrieval / semantic IDs (TIGER) | Strictly dominated: you already do exact cosine over 15 MB. Cold-start is a named GR weakness and you are 55-64% cold. |
| logQ / popularity debiasing | Popularity baseline is **0.0007**. Nothing to debias. |
| Hard-negative mining | Harmful at low τ; here the hardest negatives are same-album/same-artist tracks — mining false positives. |
| Mode-conditioned input (shuffle / skip-run) | Aims at `artist_adj` while **`artist_conc` is the binding facet** on every arm of the λ-frontier report; the effect is mean-preserving reallocation ⇒ ΔH ≈ 0. Costs a contract extension + Rust + a gated restart. `reason_end` is outcome-adjacent — drop it. |
| Carryover-h0 | Its own channel's explicit, stronger form (content-kNN over causal history) measures **0.0510** and is dominated by the trivial recency scalar. |
| Sub-threshold plays as context | The stated channel doesn't exist (targets unchanged ⇒ no supervision gain); the model can't distinguish a skip from a listen; and expanding `sessions.u32` breaks the `scores[prefix]=-inf` pairing the design rests on. |
| GRU pre-norm as a confidence signal | corr with correctness **+0.0073**; the loss is exactly norm-invariant. |
| Compose projection with whitened/std-noaco space | Already closed by 18c readout A (which re-fits the projection per space): V3 +0.0049 straddling. Reverse composition also measured: +0.004. Correct the *rationale* in PROJECT-FACTS (they are outcome-equivalent, mechanically **opposite** — W makes the space *more* anisotropic, eff. dim 15.1→16.8) but do not re-run it. |

---

## 5. Cross-cutting findings

**(a) Every survivor is a conditioning change, not a capacity change.** Five independent lenses, one convergence: `markov_gate`, the replay channel, leg abstention, the negative mask, and τ are all about *which evidence enters the comparison*, not about the function class. This is the same conclusion the record reached ("every real gain has come from the representation/supervision side"), arrived at from five different directions.

**(b) The learned-combiner anomaly is now mechanically explained.** B3's oracle (+0.0042, straddling) is real *and* `markov_gate` is real, because the oracle searches **global static scalar** weights and the bucket decomposition shows the two buckets **cancel exactly**: dropping M globally gains 0.1660→0.2025 in n_u=0 and loses 0.2326→0.1628 in n_u>10. No constant can find this; a leg-internal evidence flag can. **This dissolves the apparent paradox that a fixed z-blend beats every learned combiner** — the missing degree of freedom was never weight *magnitude*, it was per-query *abstention*, and z-normalization is scale-invariant so it provably cannot see it. Amend pitfall #1 accordingly: it should forbid *learned score-level fusion*, not *per-query evidence gating*.

**(c) The point-estimate head survived a coordinated attack.** Three lenses plus the literature tried to replace it — mixture-of-vMF (oracle-refuted), factored softmax (pre-kill fired), diffusion/generative (actively falsified upstream), full-catalogue CE (pre-empted by C6). The one theoretical result that speaks directly (Kumar & Tsvetkov, vMF) says continuous output has *never* beaten a full softmax, at best matched it. **Keep the cosine-retrieval formulation** — it is the only channel that reaches cold items (C′ cold 0.152 ≈ warm 0.151 vs Markov cold 0.010).

**(d) "55% cold" is a staleness artifact, and this is the single biggest reframe in the material.** Against full causal history only **19.7%** of steps (28.0% of test steps) are genuinely first-ever; 80.3% are replays. The rankable ceiling moves **0.4493 → 0.7617**. The champion is at 47% of the frozen ceiling but 28% of the online one. Three closure rationales lean on the 0.55 figure and should be re-read with this caveat written into PROJECT-FACTS.

**(e) Measurement power is worse than the band implies.** hw ≈ 0.0145 for architecture-scale comparisons, but **0.0004–0.018 for shared-weight eval-time levers** (n_disc as low as 9). Structural consequence: *eval-time and score-time levers are 3–33× cheaper to resolve than training-time ones.* All three of the top-3 recommendations are score-time. That is not a coincidence — it is the rational response to the instrument you have.

**Where the lenses disagreed, and my read:**

- **Literature vs ledger on full-catalogue CE.** The lit lens ranked it #1 with strong external evidence (+27..160% NDCG). The ledger pre-empts it. **My read: the ledger wins.** Klenitskiy's gain channel also reshapes the *item* embeddings; yours are frozen, so the mechanism is halved, and C6's α=0.00 arm already probed ~53% support coverage for nothing. The live remnant is τ, not the bank.
- **Is γ-whitening a new champion?** The representation lens says 0.2194 > 0.21174. **My read: a tie** (+0.008, sub-noise) — but the *mechanism* result is more valuable than a row would have been: a full-rank 192×192 supervised map fit on 73,632 labelled pairs contributes exactly what 192 free numbers contribute. That reframes "the single biggest win on record" as a conditioning fix that happened to be found by supervision, and predicts that further linear metric learning (in-graph W, richer projection targets) is near-exhausted. Rank #3 accordingly, and run the *single-model* arm, which is the only untested cell.
- **How to gate replay-vs-explore.** One lens wants an end-to-end sigmoid off `h_t`; the adversary showed it gets **zero gradient** (the replay prior never enters `step_loss`) *and* that `h_t` doesn't contain the feature that carries the 0.369→0.930 swing (prefix-replay-share is a function of history, not of prefix latents). **My read: use hand-built causal features on a frozen GRU.** That removes the plumbing problem *and* recovers the maximal-pairing precision regime.

---

## 6. What would change the picture

**The cheapest measurement that answers "capacity-saturated or data-limited?" is the rolling-origin learning curve: 8 runs, ~55 minutes serialized, zero new datasets.**

Run `seq-blend projection=true` vs `projection=false` (the +0.026 known-true effect) at folds 1–4 of the rolling origin, where fold *k* trains on sessions [0, 1431k). Fold 4 must reproduce 0.21174 / 0.18588 as the correctness gate. You get three things for the price of one batch:

1. **The learning curve.** Slope of recall between 50% and 100% of train sessions. Flat ⇒ the lab is capacity- or signal-saturated at 5,723 sessions and *no* corpus-growth or supervision-volume work is worth doing (this single number pre-kills a whole class of future proposals). Still climbing ⇒ `pipeline/refresh_corpus.sh` becomes the highest-leverage action available and several method-nulls closed on data volume (C1, C2) are re-openable.
2. **A validated pooled instrument** (folds 3+4, **fold-blocked** bootstrap), hw ~0.0145 → ~0.0102 — which is what everything in §2 items 4–7 needs to be readable at all.
3. **A heterogeneity check**: if the projection's benefit is train-size dependent (it should be — it is *the* cold-reaching channel), that is itself a finding about where the remaining headroom lives.

**Second-cheapest, 5 runs, and long overdue:** the per-seed recall sd *and* per-seed paired discordance for `seq-nexttrack`. The record's "9× bit-identical reproduction" verified **determinism, not seed-robustness** (`SEED = 1337` is hardcoded at `seq_nexttrack.py:99`; `train()` at `:574` never passes one). The `seq-blend` 3-seed runs suggest family sd ≈ 0.002, which is reassuring — but per-*query* seed disagreement has never been measured, and if it is a material fraction of the 0.0706–0.0867 architecture discordance, then a share of every past single-run architecture Δ was seed noise and the ledger needs a retroactive caveat.

**Third, free, and it decides the biggest open governance question:** finish Stage A of the replay channel. If a single untrained scalar with a causally-updated history lifts the champion's *union* ceiling to 0.2690, then the question "which topology?" is not merely the wrong question — it is being asked about a system that is currently forbidden, by an evaluation convention rather than by causality, from seeing 76% of its own answers.