# Tower bank — N parallel bare-GRU towers under a linear head

**10 runs, all remote on `snappler:879965` via `queue:true`, 84.2 min total compute,
86.5 min wall (18:20:53Z → 19:47:24Z), 0 failed / 0 interrupted, 0 of 3 pre-authorized
contingency runs used (budget was ≤13).**

NOTHING PROMOTED AS CHAMPION. NO CHAMPION REGISTRATION. NOT DEPLOYED. Crown metric and
`domain.toml` UNCHANGED. **No definition was saved** — the decision rule's save clause was
not met by any arm. The 8 measurement arms were promoted **for measurement only** — every
`notes` field reads verbatim "promoted to enable walk evaluation; not a crown claim". One
new predictor (`seq-bank`) registered in `registry.toml`; server restarted at the
documented zero-live-runs gate. The orphaned run row was NOT touched.

## Goal

Does a bank of N≥3 **parallel** bare-GRU towers under a linear head buy anything a single
GRU can't — on either surface? The hypothesis was a **dose** one: the corpus's only
measured architecture win (dual `latent/cummean` at `fusion_layers=0`, reopened on the
walk) is *parallel*, and its stated mechanism is "holding a mood needs a FIXED reference
never overwritten by recurrence". If that mechanism is dose-responsive, more fixed-reference
towers alongside enough raw-`latent` towers to hold the recall floor should buy *more* of
what N=2 bought. The independent sub-hypothesis: the raw-`latent` view **subsumes** its
siblings, so N identical `latent` towers buy nothing.

**Dataset** `seq-20260715-131139` (canonical PCA-192 sequence split, 7,154 sessions,
19,402 items, dim 192, train pairs 4,865 / val 858 / **test sessions 1,431**, chronological
cut 2024-08-24T14:28:02, cold-item rate 0.5507). `seq-20260718-211238` was NOT used: all
1,431 of its test sessions sit inside split-1's train set, so it cannot harden anything
trained here. No dataset was built — `walk_headtohead.py::promoted_score` asserts
`item_latents.f32` byte-identity, so a new item space would make every arm unpairable.

**One-shot reference (S1).** `seq-nexttrack` h256 GRU / InfoNCE / τ0.07 / dropout 0.1,
on-record recall@10 `0.12299091544374564` = **176/1431**, H@10 `0.02453832638678665`,
music@10 `0.4557759241147047`, artist_conc 0.3559, params **394,944**. Canonical run
`run-20260726-183333-9a0b2-seq-nexttrack`. Re-run **in batch** as C0.

**Walk reference (S2).** In-batch C0 at the matched cell **a0.4 / s0.3**, 40 held-out
sessions × 20 steps, artist cap 1, paired. Deployed dual `l/cummean` f0 @ a0.8/s0.5:
vibe 0.610 / stride_err 0.144. Real-listening median step cosine 0.261.

**Held constant on every run:** `loss=infonce`, `tau=0.07`, `epochs=40`, `patience=6`,
`lr=0.001`, `batch_size=128`, `val_fraction=0.15`, `seed=1337`, `k=10`, `mmr_lambda=1.0`
(off), `mmr_pool=200`, `artist_cap=0` (off). Only `views` / `hidden` / `dropout` vary.
Every tower is a single-layer bare GRU; the head is a linear map over the concatenated
towers. No fusion MLP, no per-step MLP, no LSTM, no stacked layers, no `eager_beta` —
these are **structurally unreachable**, not merely unused: `seq-bank` declares exactly 15
params and none of `fusion_layers`, `fusion_hidden`, `pre_hidden*`, `pre_layers*`,
`layers*`, `num_layers`, `arch*`, `bidirectional`, `residual`, `eager_beta`,
`eager_margin` is among them.

**Capacity.** All bank arms + C1 matched to **3.00×** the h256 baseline. All counts
re-verified empirically — `seq_bank` emits `n_params` in `metrics.json`, and the
observed values match the design exactly: R2/D2 789,696 (2.000×), B2 1,186,560 (3.004×),
B3 1,184,448 (2.999×), B4 1,187,700 (3.007×), V1/V2/V3 1,184,448 (2.999×), C1 1,182,912
(2.996×, by instantiation), C0 394,944 (1.000×, by instantiation). **Mutual spread across
the seven 3.00× arms including C1: 0.405%** — tighter than the stage-stack campaign's
0.445%, and bracketed on both sides by prior ties (h425 at 2.21×, h537 at 3.24×).

## Outcome in one line

**The dose hypothesis is refuted and the axis is CLOSED: every bank arm from N=2 to N=4 is
statistically indistinguishable from the h256 baseline on one-shot recall, five of eight
LOSE to a param-matched single GRU of the same width budget, adding fixed-reference
`cummean` towers makes the walk strictly worse (V1/V2 refuted on vibe), and the two arms
that earned a walk TRADE at the primary cell both flip to `vibe` CI<0 at the deployed
cell — while the campaign's one real finding is a by-product: at byte-identical params,
**dropout 0.1 beats dropout 0.0 by +0.0049 recall CI>0**, so every `fusion_layers=0` arm
on record, including the deployed engine, trained under-regularized.**

## Gates — all ten green

| gate | result |
|---|---|
| **P1 — md5 the whole import closure** | **PASS**, re-verified before **each** of the four chunk POSTs (identical digest `2c6070753661a311410c76c4e0c36b4d` over the five files every time). It earned its keep twice: `seq_bank.py` was **absent** on the worker and `registry.toml` was the **pre-edit** copy (`bf82b86a…` vs hub `a556ab37…`). Both fixed and re-verified before any POST. Worker binary `3b7c3172a85de4c8f502b3425d8b3bbb` == hub `target/release/lensing-server`, so no re-scp was needed. |
| **P2 — functional equivalence** (the one unconditional stop) | **GATE PASS**, run BEFORE the registry edit and restart, and **re-run independently on the hub** after an infrastructure interruption. Leg (a) `views="latent"` h256 dropout 0.1 vs `SeqNextLatent`: `forward` AND `predict_next` max\|Δ\| **0.00e+00** at T = 1, 2, 5, 17, 40, params **394,944** both sides. Leg (b) `views="latent+latent"` h256 dropout 0.0 vs `DualTowerNextLatent` f0: max\|Δ\| **0.00e+00** at all five T, params **789,696** both sides. |
| **P2c — training-stream identity** (extra, not in the design) | **PASS.** `seq_bank.fit` and `seq_dualgru.fit` run 3 epochs at dropout 0.0 on the same split matched per-epoch train AND val loss to the last digit (e1 6.6571665789665255 / 6.852755815087279; e2 6.144612194104297 / 6.699498798068011; e3 6.020286355658111 / 6.675729464048785) and all **10 state-dict tensors at max\|Δ\| 0.00e+00** under `tower_a.→towers.0.`, `tower_b.→towers.1.`, `fusion.0.→head.`. This is what upgraded G2b from a band to a hard bit-identity gate. |
| **P3 — launch body vs registry block** | **PASS, and it caught a real conflict before it cost a POST.** `seq-nexttrack` declares 19 params but **not** `tau`, `val_fraction` or `seed` — three of the design's "held constant" keys. Rather than invent values, C0's body is `9a0b2`'s stored hyperparams **verbatim** (+`artist_cap:0`) and C1's is the same with `hidden:512`. `tau` is hardcoded `0.07` in `seq_nexttrack.py:86` and consumed as `hp["tau"]`, so parity with the bank arms' explicit `tau:0.07` is **exact, not assumed**. All 10 POSTs were accepted first time; zero `unknown hyperparameter` rejections (three killed POSTs on 2026-08-01). |
| **P4 — server restart gate** | **PASS** — the only `running` row was the known orphan `run-20260730-234635-7a297-seq-blend` (`claimed_by snappler:404147`, started 2026-07-30T23:51:30Z). Re-confirmed at every chunk boundary. The orphan was NOT touched. |
| **P5 — harness reproduction** | **GREEN.** The default 5-arm `walk_headtohead.py` invocation reproduced **all ten** recorded 2026-07-31b means exactly (shipped GRU 0.609 / 0.399 / −0.089 / 9.925 / 0.345; dual l/cummean 0.610 / 0.144 / −0.053 / 11.425 / 0.351) and re-derived the reopening deltas (dual `l/l` Δvibe **+0.0297** CI>0; dual `l/cummean` Δstride_err **−0.2545** CI<0, Δgenres **+1.5000** CI>0, Δdrift **+0.0351** CI>0, Δvibe **+0.0013** straddles). The instrument has not moved. |
| **G1 — metric completeness** | **GREEN on all 10** — every run carries 13 metric keys including `artist_conc_at_k` **and** `music_at_k`. No worker-image key stripping, no daemon cycle, no relaunch. |
| **G2a — C0 bit-identity** | **PASS, 10th reproduction** — C0 = `0.12299091544374564` = **176/1431** exactly, with `music@10` `0.4557759241147047`, H@10 `0.02453832638678665` and `mrr` `0.053411071731908816` all matching `9a0b2`'s stored values to the last digit. |
| **G2b — R2 bit-identity** | **PASS, hard gate, no band needed** — R2 = `0.1187980433263452` = **170/1431** exactly, `music@10` `0.45534726776072254` and `mrr` `0.056307146438214335` also bit-identical to `18b4a`. The band fallback [165, 175] was not used. |
| **G3 — walk instrument** | **PASS** — Δ(shipped `gru.onnx` @ a0.4/s0.3 vs in-batch C0 @ a0.4/s0.3) straddles 0 on **both** required axes: Δvibe **+0.0028** [−0.0007, +0.0067], Δstride_err **+0.0005** [−0.0043, +0.0055]. So S2 verdicts vs C0 are not an artefact of the base. |

## Results — S1, one-shot top-10

Paired bootstrap, 2,000 resamples, rng 1337, n_test 1,431, computed from the stored
per-row `predictions.json` artifacts. Sorted by hits. `H@10` in raw next-track units.

| arm | run id | views / width | params | ×h256 | hits | recall@10 | Δrecall vs **C0** | Δ vs **C1** | H@10 | music@10 | a_conc | S1 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C1** | `run-20260802-183404-91e31-seq-nexttrack` | single GRU h512 | 1,182,912 | 2.996 | **186** | 0.129979036 | +0.006988 [−0.00210, +0.01607] straddles | — | 0.0237953 | **0.45810** | 0.38247 | **S1-TIE** (7th width tie) |
| **B4** | `run-20260802-190415-1d662-seq-bank` | `latent+latent+latent+latent` h211 | 1,187,700 | 3.007 | **186** | 0.129979036 | +0.006988 [−0.00210, +0.01677] straddles | +0.000000 straddles | 0.0244609 | **0.45935** | 0.37163 | **S1-TIE** |
| D2 | `run-20260802-183404-62604-seq-bank` | `latent+latent` h256 **dropout 0.1** | 789,696 | 2.000 | 177 | 0.123689727 | +0.000699 [−0.00769, +0.00839] straddles | −0.006289 straddles | 0.0242558 | 0.45670 | 0.36978 | **S1-TIE** |
| **C0** | `run-20260802-182050-91b54-seq-nexttrack` | single GRU h256 (anchor) | 394,944 | 1.000 | 176 | 0.122990915 | — | −0.006988 straddles | **0.0245383** | 0.45578 | 0.35591 | anchor (G2a) |
| B3 | `run-20260802-190415-90866-seq-bank` | `latent+latent+latent` h256 | 1,184,448 | 2.999 | 172 | 0.120195667 | −0.002795 [−0.01048, +0.00559] straddles | −0.009783 **CI<0** | 0.0235773 | 0.45542 | 0.37635 | **S1-TIE** |
| R2 | `run-20260802-182050-2e6f2-seq-bank` | `latent+latent` h256 **dropout 0.0** | 789,696 | 2.000 | 170 | 0.118798043 | −0.004193 [−0.01258, +0.00419] straddles | −0.011181 **CI<0** | 0.0237849 | 0.45535 | 0.37059 | **S1-TIE** (G2b) |
| B2 | `run-20260802-183404-67185-seq-bank` | `latent+latent` h334 | 1,186,560 | 3.004 | 170 | 0.118798043 | −0.004193 [−0.01328, +0.00349] straddles | −0.011181 **CI<0** | 0.0233569 | 0.45537 | 0.37989 | **S1-TIE** |
| V3 | `run-20260802-192627-17502-seq-bank` | `latent+cummean+cummean` h256 | 1,184,448 | 2.999 | 170 | 0.118798043 | −0.004193 [−0.01607, +0.00629] straddles | −0.011181 straddles | 0.0232003 | 0.44974 | 0.35806 | **S1-TIE** |
| V2 | `run-20260802-192627-836b1-seq-bank` | `latent+cummean+delta` h256 | 1,184,448 | 2.999 | 166 | 0.116002795 | −0.006988 [−0.01957, +0.00489] straddles | −0.013976 **CI<0** | 0.0224536 | 0.44967 | 0.35472 | **S1-TIE** |
| V1 | `run-20260802-190415-b5928-seq-bank` | `latent+latent+cummean` h256 | 1,184,448 | 2.999 | 162 | 0.113207547 | −0.009783 [−0.02166, +0.00140] straddles | −0.016771 **CI<0** | 0.0229874 | 0.45002 | 0.35719 | **S1-TIE** |

**Both mandatory floors pass for every arm.** Absolute recall floor ≥ 0.108 (155/1431):
the worst arm is V1 at 0.11321 = 162, clear. Per-baseline relevance floor (paired
Δ`music@10` vs C0 must not be CI<0): every arm straddles, the most negative being V2 at
−0.006109. No arm was refuted by a floor — a contrast with the stage-stack campaign, where
all five arms fell through the recall floor.

**No arm is S1-WIN** (the rule requires Δ vs C0 CI>0 *and* Δ vs C1 CI>0; no arm achieves
even the first leg). **No arm is S1-REFUTED.** All nine non-anchor arms are **S1-TIE**.

## Results — S2, generated walk

40 held-out sessions × 20 steps, artist cap 1, paired, base = in-batch C0, G3 green.
Primary cell **a0.4/s0.3**. `vibe` is never shown without `stride_err` beside it;
`ground` is used only as a degeneracy floor.

| arm | vibe | stride_err | Δvibe | Δstride_err | Δdrift | Δgenres | Δground | S2 verdict (primary) |
|---|---|---|---|---|---|---|---|---|
| C0 (base) | 0.566 | 0.130 | — | — | — | — | — | base |
| **B2** | 0.567 | 0.139 | +0.0017 straddles | +0.0099 straddles | **+0.0339 [+0.0066, +0.0629] CI>0** | −0.2250 straddles | +0.0044 straddles | **S2-TRADE** |
| **V3** | 0.556 | 0.124 | −0.0094 straddles | −0.0045 straddles | **+0.0286 [+0.0009, +0.0592] CI>0** | −0.4500 straddles | −0.0024 straddles | **S2-TRADE** |
| C1 | 0.563 | 0.130 | −0.0020 straddles | +0.0015 straddles | +0.0194 straddles | −0.3750 straddles | +0.0008 straddles | S2-NULL |
| D2 | 0.559 | 0.129 | −0.0065 straddles | +0.0005 straddles | +0.0104 straddles | −0.7500 straddles | +0.0003 straddles | S2-NULL |
| B4 | 0.566 | 0.139 | +0.0006 straddles | +0.0103 straddles | +0.0164 straddles | −0.7750 straddles | +0.0004 straddles | S2-NULL |
| B3 | 0.557 | 0.135 | −0.0078 straddles | +0.0060 straddles | +0.0146 straddles | **−1.0250 CI<0** | +0.0154 straddles | S2-NULL (genre loss) |
| V2 | 0.551 | 0.127 | **−0.0144 [−0.0290, −0.0010] CI<0** | −0.0019 straddles | +0.0259 straddles | −0.0250 straddles | −0.0148 straddles | **S2-REFUTED** |
| V1 | 0.548 | 0.119 | **−0.0174 [−0.0352, −0.0010] CI<0** | −0.0099 straddles | +0.0227 straddles | **−1.7250 CI<0** | +0.0017 straddles | **S2-REFUTED** |

### Secondary cell a0.8/s0.5 — both TRADEs collapse

The save clause requires the same sign at the deployed cell. It does not hold:

| arm | Δvibe | Δstride_err | Δdrift | Δgenres | Δground | verdict at a0.8/s0.5 |
|---|---|---|---|---|---|---|
| B2 | **−0.0064 [−0.0117, −0.0011] CI<0** | −0.0107 straddles | −0.0052 straddles | −0.1750 straddles | +0.0006 straddles | **S2-REFUTED** |
| V3 | **−0.0112 [−0.0167, −0.0064] CI<0** | −0.0074 straddles | +0.0043 straddles | −0.4000 straddles | −0.0039 straddles | **S2-REFUTED** |

The drift gain that earned both arms their TRADE **does not survive the cell change**
(B2 −0.0052, V3 +0.0043, both straddling), while `vibe` — the non-negotiable line — goes
CI<0 for both. Under the pre-agreed rule, **no definition is saved.**

### Stride probe — all three arms peak at the cell they were already in

Pre-registered for every arm with a negative Δ`stride_err` point estimate (V1, V2, V3),
at a0.4 over s ∈ {0.2, 0.3, 0.4}. Δ`stride_err` vs C0:

| arm | s0.2 | **s0.3** | s0.4 |
|---|---|---|---|
| V1 | +0.0670 CI>0 | **−0.0099 straddles** | +0.0406 straddles |
| V2 | +0.0495 CI>0 | **−0.0019 straddles** | +0.0478 CI>0 |
| V3 | +0.0320 CI>0 | **−0.0045 straddles** | +0.0737 CI>0 |

All three peak at **s0.3**, the cell they were already measured in — the third campaign in
a row where re-tuning stride rescues nothing. And the probe illustrates the registered
coupling caveat cleanly: at s0.2 all three arms' Δ`vibe` becomes *positive-straddling*
(V1 +0.0160, V2 +0.0119, V3 +0.0136) **only by taking visibly shorter steps**
(Δstride_err CI>0 for all three) and losing genre variety (V1 −1.9000, V2 −1.7750,
V3 −1.0000, all CI<0). That is exactly the trade the S2-WIN clause exists to reject.

## Findings

1. **The dose hypothesis is refuted, and it fails in the *opposite* direction to the
   prediction.** Adding fixed-reference `cummean` towers did not buy more of what N=2
   bought — it monotonically destroyed the walk. Ranked by Δvibe at the primary cell:
   B4/B2 (all-`latent`, ≈0) > B3 > V3 (one cummean, −0.0094) > V2 (cummean+delta,
   −0.0144 CI<0) > V1 (−0.0174 CI<0). On S1 the same ordering holds: the three
   `cummean`-bearing arms occupy the **bottom three** places (V1 162, V2 166, V3 170) and
   are the only arms whose Δ`music@10` point estimates are worse than −0.005. The
   mechanism read behind the design — "a fixed reference never overwritten by recurrence"
   — does not survive being dosed. Whatever the N=2 `l/cummean` engine is doing, it is not
   "more cummean is more mood-holding".

2. **The "raw `latent` subsumes its siblings" prediction is confirmed, and extends cleanly
   from N=2 to N=4.** B2 (N=2, h334) 170, B3 (N=3) 172, B4 (N=4) 186 — all straddle C0,
   and the N-axis at matched capacity shows no trend distinguishable from width noise.
   Four independent `latent` towers with different random inits converge on the same
   information a single GRU already has.

3. **The decisive result is that topology and width are interchangeable at this budget.**
   B4 (four h211 towers, 1,187,700 params) and C1 (one h512 GRU, 1,182,912 params) return
   **exactly the same hit count, 186**, with Δ between them of precisely 0.000000
   [−0.00839, +0.00839]. The param-matched control does everything the best bank arm does.
   This is why the S1-WIN rule required the C1 leg: without it, B4's +10 hits over C0 would
   have read as a topology win, when it is fully explained by capacity.

4. **The width curve is flatter than "flat" — C1's +10 hits over C0 is NOT significant.**
   Δ = +0.006988 [−0.00210, +0.01607], straddling. This is the **7th** param-matched width
   tie and it also means the campaign cannot claim h512 as an improvement, despite it being
   the highest single-GRU hit count on record (186 vs h454's 179). Single-GRU width points
   on PCA-192 now read h256→176, h297→**162 (CI<0)**, h425→173, h454→179, h512→186,
   h537→174 — non-monotone, spanning 0.0168 with no trend. Contingency 3 (an h608 bracket)
   did **not** fire because C1 did not lose.

5. **The dropout confound is REAL, and it is the campaign's one positive finding.**
   D2 − R2, params **byte-identical** (789,696 both, `model.pt` 3,162,277 bytes both),
   only `dropout` differing: Δrecall@10 **+0.004892 [+0.000699, +0.009783] CI>0**
   (177 vs 170 hits). Dropout 0.1 **beats** dropout 0.0. Read carefully, this does not
   overturn "f0 beats f1" — it does the opposite of exonerating the *arms*: because the
   f0 path has no active dropout at all (dropout enters `DualTowerNextLatent` only via a
   pre-MLP, an inter-layer gap, or the fusion MLP, all absent at f0/layers=1/pre_hidden=0),
   **every `fusion_layers=0` arm on record — including the deployed engine — trained
   under-regularized and was leaving ≈+0.005 recall on the table.** The three recorded
   f0-vs-f1 gaps (+0.016 / +0.020 / +0.0266) were therefore measured against an f1 that
   *had* regularization the f0 arm lacked, so they are not like-for-like; the true
   topology advantage of f0 is if anything **larger** than recorded. Cheapest actionable
   follow-up on the board: turn dropout on in the deployed engine.

6. **The validation-loss dissociation gets its 4th confirmation, now as a rank
   correlation.** Early-stop epoch and best val InfoNCE across the batch: C0 e14/6.5868,
   C1 e14/6.5653, R2 e14/6.5671, D2 e14/6.5695, B2 e14/6.5660, B3 e14/6.5694,
   B4 e14/6.5657, V1 e21/6.5664, V2 e24/6.5728, V3 e24/6.5813.
   **Spearman(val loss, hits) = −0.2954, p = 0.407** — no usable predictive power. The
   sharpest pairs: B4 (6.5657) and B2 (6.5660) differ by 0.0003 in val loss and **16 hits**;
   V1 has the 4th-best val loss in the batch and the **worst** hit count (162); C0 has the
   **worst** val loss and beats six of the eight bank arms. The decision rule never read
   val loss, and this is why.
   Note R2 (dropout 0.0) reached a *better* val loss than D2 (6.5671 vs 6.5695) while
   retrieving **7 fewer hits** — the dissociation and the dropout finding are the same
   coin.

7. **Both walk TRADEs were cell-local, which is a methodological result in its own
   right.** B2 and V3 each earned a genuine Δdrift CI>0 at a0.4/s0.3 with vibe held — and
   both flipped to vibe CI<0 at a0.8/s0.5 with the drift gain evaporating. Had the design
   not pre-registered the secondary-cell requirement, this campaign would have reported two
   positive walk results and saved two definitions. **The secondary-cell clause is the
   clause that did the work here**; it should stay mandatory in every future walk design.

8. **A process finding: 0 of 3 contingency runs fired, and the epoch trace is why.**
   Contingency 1 (the dropout-confound extension) required R2 to early-stop *later* or
   reach lower train loss than D2 — both stopped at **epoch 14**, so it did not fire.
   Contingency 2 (epoch cap) required an arm to run to the 40-epoch cap with patience never
   triggering — every arm early-stopped (e14 ×7, e21, e24 ×2), so no arm's recall is a
   lower bound. Contingency 3 did not fire (finding 4). I record one honest correction: I
   initially read D2's 15.7-min wall time against R2's 4.7 min as a regularization signal,
   but the epoch traces are identical at e14, so that gap is **shared-CPU wall-time noise,
   not a training signal** — worth remembering that walltime on this worker is not a proxy
   for epochs.

9. **`seq-bank` is a clean instrument and its guard fired as designed.** `parse_views`
   rejects any spec without a `latent` tower (`cummean+cummean`, `delta+cummean`,
   `cummean` all raise `SystemExit`), which makes the registered 0.058–0.087 delta/cummean
   pathologies **structurally unreachable** rather than merely avoided. The predictor also
   emits `views`, `n_towers` and `n_params` into `metrics.json`, which is what let capacity
   matching be verified from the artifacts instead of trusted from the design.

## What was persisted

- **No definition saved.** No arm met `(S1-WIN or S1-TIE) and (S2-WIN or S2-TRADE) and
  same sign at the secondary cell`.
- **8 measurement-only promotions**, every `notes` verbatim
  "promoted to enable walk evaluation; not a crown claim": `bank-c1-gru512`,
  `bank-d2-ll256`, `bank-b2-ll334`, `bank-b3-lll256`, `bank-b4-llll211`,
  `bank-v1-llc256`, `bank-v2-lcd256`, `bank-v3-lcc256`.
- **The best-models group was flooded, as disclosed up front.** The server's deterministic
  recompute auto-created three `best-*` models from this campaign
  (`best-seq-nexttrack-20260802-182050-91b54` at rank 8,
  `best-seq-bank-20260802-190415-1d662` at rank 10,
  `best-seq-bank-20260802-183404-62604` at rank 12) and the group now holds 15 entries.
  **Hand off to best-model-selector**; recommend deleting the 8 measurement promotions now
  that the walk numbers are banked.
- No dataset built, no dataset tagged (no dataset axis was in play).
- `registry.toml` gained the additive `seq-bank` block; `tools/walk_headtohead.py` gained
  `"bank": seq_bank` + the import. The user committed these mid-campaign as **`ed561e4`
  "feat(seq-bank): N parallel GRU towers as one hyperparameter"**; the md5s are unchanged
  from the ones every run and every P1 check used (`seq_bank.py`
  `7c58d53a2d3c3750502050afaabde238`, `registry.toml` `a556ab371e218c45985963de7b0eb7ba`),
  so the batch's provenance is intact. `experiments/PROJECT-FACTS.md` and this report are
  left uncommitted for the user.

## Best on record after this work

| surface | best | value | run / model |
|---|---|---|---|
| one-shot recall@10, `seq-20260715-131139` | `seq-blend` markov-gated champion | **0.23201** | unchanged by this campaign |
| best single-GRU width point (not significant vs h256) | h512 | 186/1431 = 0.129979 | `run-20260802-183404-91e31-seq-nexttrack` |
| best architecture on the walk | dual `latent/cummean` f0 @ a0.8/s0.5 (DEPLOYED, and now known to be **under-regularized**) | vibe 0.610 / stride_err 0.144 | `run-20260725-143502-3e5d4-seq-dualgru` |
| parallel-tower banks N≥3 | **CLOSED — nothing beats a single GRU on either surface** | — | this report |

**Scope of the closure, precisely.** This closes **"N≥3 parallel bare-GRU tower banks at
`fusion_layers=0`"** and only that, to the V3 standing-rule standard (both surfaces, G3
green). It does **not** re-close the dual-tower family, which stays REOPENED and remains
the deployed engine. It is the **second** axis ever closed to that standard, after the
stage-stack axis.

## Follow-ups

1. **Turn dropout on in the deployed engine (1 run, highest value on this board).**
   Finding 5 says the deployed dual `l/cummean` f0 trained with no active dropout and that
   regularizing costs nothing and gains ≈+0.005 CI>0 at byte-identical params. Retrain the
   deployed arm at `dropout=0.1` and re-read both surfaces. This is a *free* improvement to
   the showcase engine, not an architecture bet.
2. **Amend PROJECT-FACTS on the f0 regularization asymmetry** — done in this campaign's
   reconciliation, but the three f0-vs-f1 gap figures on record should be re-stated as
   *not like-regularized* wherever they are quoted as clean topology measurements.
3. **The 3-seed robustness check is NOT pre-authorized and I do not recommend spending it.**
   The trigger technically fired (B2 and V3 reached S2-TRADE at the primary cell), but both
   are refuted at the secondary cell, so 3 seeds would harden a result the rule has already
   rejected. Needs the user either way.
4. **The training regime remains the record's #1 untested axis** — scheduled sampling /
   exposure bias on the single GRU at f0. Finding 6 here is the fourth confirmation that
   the objective is misaligned with what we rank on, and this campaign spent 84 min of
   compute confirming once more that topology is not the lever. Both the designer's own
   expected-value read and the stage-stack Follow-up 4 rank this first.
5. **Record hole to backfill (carried forward, not addressed here).** The "2026-07-31b"
   walk-surface reopening — this corpus's only architecture win, and the evidential
   foundation this whole campaign was built on — has **no primary `experiments/*.md`
   report**. It exists only as PROJECT-FACTS entries, git commit `9430bb3`, and
   `docs/next-track.tex`. Under this project's own convention that the campaign reports are
   primary, the most consequential recent finding is unindexed. It should be written up from
   the commit and the facts entries.
6. **`seq_bank` has a `model-sae` tap emitting `tower_0..tower_{N-1}`** and was never read.
   If anyone wants to know *why* four `latent` towers add nothing, a B4 SAE read is now one
   command away with no code change — the concrete version of "the cummean tower added
   linear directions but ZERO next-item concepts".
7. **Infrastructure: the hub laptop suspended mid-campaign** (server clock jumped 00:36Z →
   18:17Z, process reaped, `/tmp` cleared). The campaign was launched in four small chunks
   specifically to bound that risk, and no run was lost. Sleep inhibition during batches
   would remove the hazard; the orphan row from the 2026-07-30 suspend is still `running`
   in Postgres and still needs a user-authorized `UPDATE` to clear.
