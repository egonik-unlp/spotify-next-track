# Does a learned per-step pre-encoder buy anything? (pre-MLP presence × placement × dose, with the decisive single-tower ablation) — 2026-07-26

**Date:** 2026-07-26
**Type:** ARCHITECTURE SCAN — **REFUTED**, and in the single-tower case actively
**HARMFUL**. Nothing promoted, nothing registered, champion / leaderboard /
best-models pins UNCHANGED. The null is the finding.
**Axis under test:** a **nonlinear per-step re-embedding in front of the
recurrence** — `pre_hidden` / `pre_layers` on `seq-nexttrack` (NEW this session)
and `pre_hidden_a` / `pre_hidden_b` on `seq-dualgru`. Built as
`[Linear(d_in,width), ReLU(), Dropout(dropout)] × pre_layers`, applied per
timestep (strictly causal), with the GRU's `input_size = pre_hidden` when > 0.
**Dataset (fixed, all 10 runs):** `seq-20260715-131139` — canonical leak-free
PCA-192 split; 7,154 sessions / 19,402 items; 5,723 train / 1,431 test @ cut
2024-08-24T14:28:02; cold-item rate 0.55; `latent_dim = 192`.
**Host:** all 10 runs on the SAME remote worker `snappler:4101029` (`queue:true`,
serialized one at a time), verified per run via `claimed_by`. **0 failed / 0
interrupted**, 54 min wall clock (16:14:32 → 17:08:49 UTC).
**Seed:** 1337 (dual arms; `seq-nexttrack` exposes no seed and is deterministic).
**Noise band:** recall@10 CI half-width ±0.015; same-config reproduction spread
±0.0035.

## Goal & baseline

**Question.** The 2026-07-25 dual-tower campaign refuted *view-split* asymmetry,
and the SAE read explained why: `delta` / `cummean` are deterministic lossy
functions of what tower A already had (CKA 0.458, **+0** new next-item concepts).
This campaign asks the strictly stronger version of the question — does a
**learned** per-step re-embedding, which is *not* a function of information the
tower already holds, enlarge the hypothesis space usefully?

The hypothesis has a sharp analytic edge. A GRU's own input transform is already
linear (`W_i·x` per gate), and a *linear* pre-encoder is exactly as expressive as
a bare GRU (`W_i·(Vx) = (W_iV)x`, with no rank constraint since
`pre_hidden ≥ latent_dim`). **The ReLU is therefore the only expressivity
change** — so no linear-pre control run was needed, and a null cannot be
explained away as "the pre-encoder was redundant by construction".

The design's decisive move is the **single-tower ablation (S1)**: a
`MLP256→GRU256` `seq-nexttrack` against a **param-matched plain GRU**. Without
it, any dual-tower win is uninterpretable — pre-encoder or second tower?

**Baseline / reference (C0).** `seq-nexttrack` `gru-infonce-h256` on
`seq-20260715-131139`, on-record recall@10 **0.12299091544 = 176 of 1,431** next
tracks (`run-20260725-143143-97c1e`, plus 5 prior identical reproductions;
`2026-07-25-nexttrack-dual-tower-fusion-scan.md`,
`2026-07-22-session-holisticness-and-antieager-levers.md`,
`2026-07-15c-nexttrack-phase2-model-sweep.md`). Re-run in-batch as C0 and
reproduced **EXACTLY**: recall@10 0.12299091544374564 (bit-identical to the
stored reference, including the final digit), **176 of 1,431 hits**, MRR 0.05341,
artist@10 0.3375, genre@10 0.4500, music@10 0.4558, artist_adj 0.34745,
mood_coh 0.47884, ild 0.45356, **4-factor holisticness@10 0.047086**, n_test
1,431.

**Held for every `seq-nexttrack` arm** (only `hidden`, `pre_hidden` vary):
`arch=gru, loss=infonce, num_layers=1, bidirectional=false, residual=false,
pre_layers=1, epochs=40, lr=0.001, dropout=0.1, batch_size=128, patience=6, k=10,
eager_beta=0.0, eager_margin=0.0, mmr_lambda=1.0, mmr_pool=200` (`tau` hardcoded
0.07 in `seq_nexttrack.py` → exact parity with the dual arms).
**Held for every `seq-dualgru` arm** (only `pre_hidden_a`, `pre_hidden_b`,
`fusion_layers` vary):
`hidden_a=256, hidden_b=256, arch_a=gru, arch_b=gru, view_a=latent,
view_b=latent, layers_a=1, layers_b=1, pre_layers_a=1, pre_layers_b=1,
fusion_hidden=256, loss=infonce, tau=0.07, epochs=40, lr=0.001, dropout=0.1,
batch_size=128, patience=6, seed=1337, k=10, eager_beta=0.0, eager_margin=0.0,
mmr_lambda=1.0, mmr_pool=200`.

**Parameter counts — re-measured in this campaign** by instantiating the real
`SeqNextLatent` / `DualTowerNextLatent` classes at `latent_dim=192` (not
estimated). **All 11 reproduce the design's figures exactly.**

| config | params | note |
|---|--:|---|
| single GRU h256 (C0) | 394,944 | on-record baseline |
| **single MLP256→GRU256 (S1)** | **493,504** | the decisive ablation |
| single GRU h297 (C1) | 494,697 | **−0.241 %** param match to S1 |
| single GRU h425 (on record, `be11b`) | 871,017 | capacity control for T1-f0 / T2 |
| single GRU h454 (C2) | 969,936 | **+0.051 %** param match to T1 |
| dual bare 2×256 f1 (C3) | 871,872 | = last campaign's D-LL |
| dual A0/B128 f1 (T2) | 847,424 | dose-down |
| dual A0/B256 f0 (T1-f0) | 888,256 | |
| **dual A0/B256 f1 (T1, the user's model)** | **970,432** | |
| dual A256/B256 f1 (T4) | 1,068,992 | symmetry test |
| dual A0/B384 f1 (T3) | 1,093,440 | dose-up |

**The honest bar.** Two boards kept separate. (1) **recall@10** — the bar for a
standalone single model is **0.123**; the 3-leg blend champion
`blend-gru-markov-content-proj` (0.2117) is NOT a comparand (category error) and
is held out of every rule. (2) **holisticness@10, 4-factor** (post-2026-07-25):
`clamp(mood_coh,0) × ild × (1−artist_adj) × clamp((music@10 − 0.185)/0.815, 0, 1)`,
product of **run-aggregate means**.

## Preflight gates

| gate | result |
|---|---|
| **P-1** `pre_hidden`/`pre_layers` on `seq-nexttrack` | **PASS** — `GET /api/predictors` shows both; implemented by the orchestrator with the design's conditional-construction contract (no `nn.Module`, no RNG draw at width 0) |
| **P-2** server picked up the params | **PASS** — `seq-nexttrack: [pre_hidden, pre_layers]`, `seq-dualgru: [pre_hidden_a, pre_hidden_b, pre_layers_a, pre_layers_b]`. No runner restart (never a runner action) |
| **P-3** remote worker daemon | **PASS** — `snappler:4101029` live; all 10 runs `claimed_by = snappler:4101029`, one at a time |
| **P-4** live worker checkout freshness | **PASS** — md5 of **all 15** `predictors/seq_*.py` in the LIVE `~/lensing-worker/predictors/` **identical** to the hub, including the freshly-edited `seq_nexttrack.py` (`5e2ec283…`) and `seq_dualgru.py` (`a389b5f0…`) |
| **P-5** Qdrant reachable from worker | **PASS** — `music@10` / `mood_coh@10` / `ild@10` / `artist_adj@10` populate on all 10 runs. Load-bearing: the 4-factor composite is undefined without `music@10` |
| **P-6** per-row facets reachable | **PASS** — `GET /api/runs/<C0>/predictions` → HTTP 200, **1,431 rows** carrying `row_id`, `actual`, `top_k_ids`, `music_sim`, `mood_coh`, `ild`, `artist_adj` |
| **Gate 0** C0 bit-identity | **PASS on every leg** — hits@10 **176 / 1,431** (== reference), recall `0.12299091544374564` (bit-identical incl. last digit), recomputed 4-factor H **0.0470862** ∈ [0.0456, 0.0486] |
| **Gate 1** C3 bit-identity | **PASS bit-identically** — hits@10 **147 / 1,431**, recall `0.10272536687631029` == on-record `run-20260725-143502-12fe6` exactly. The `seq_dualgru` pre-MLP refactor did **not** perturb the bare path → **all dual verdicts are valid** |

**The `pre_hidden` edit did NOT move the family's on-record baseline.** Gate 0
came back bit-identical to the last digit, so the design's "flag that the edit
changed the default code path" clause does **not** apply. The conditional
construction held.

**Statistics validated before use.** Recomputed from C0's `predictions.json`:
per-row means music_sim 0.455776 / mood_coh 0.478838 / ild 0.453564 /
artist_adj 0.347449 → product-of-means 4-factor H = **0.0470862** vs the stored
`holisticness_at_k` **0.0470861** (agree to 7 significant figures). The
paired-ΔH bootstrap therefore measures exactly the quantity the server stores.
A C0-vs-C0 self-comparison returns Δ = 0.0 identically on both metrics.

**Metric-cut note.** The stored `holisticness_at_k` for this batch is the
**4-factor** composite (C0 stores 0.047086 where its 2026-07-25 twin stored the
3-factor 0.141723 — same model, same predictions). Pre-cut runs remain exactly
comparable because all four facets are stored per row; the recomputed board below
re-derives 4-factor H for every facet-carrying run on record.

## Outcome (one line)

**The nonlinear per-step pre-encoder is REFUTED on this corpus (Rule 7): not one
of the six pre-MLP arms beats the h256 baseline by paired-Δ CI > 0 — and the
decisive single-tower ablation S1 (`MLP256→GRU256`, the one arm whose result is
unambiguously attributable to the pre-encoder) does not merely tie but **LOSES
outright**, 0.10901 vs 0.12299, Δ −0.0140 [−0.0266, −0.0014] CI < 0, i.e. 156 hits
against 176 — so since a *linear* pre-encoder is provably expressivity-neutral
here, the ReLU re-embedding is destroying prefix information the GRU's own linear
input transform preserved; the best dual arm T1-f0 only TIES C0 (0.12369, Δ
+0.0007 straddling) and ties the on-record h425 and last campaign's bare
`fusion_layers=0` dual, so nothing on the dual side moves either; the dose curve
is **FLAT** (all three adjacent pairs C3→T2→T1→T3 straddle 0, endpoints straddle),
which would independently have blocked registration even had a win rule fired;
placement is irrelevant at this dose (T1 vs T4 straddles on recall), so keeping
tower A bare is NOT load-bearing; the one genuine positive finding is Rule 6 — the
documented `fusion_layers=1` regression **persists with a pre-encoder and gets
BIGGER**, T1-f0 beating T1 by **+0.0266 [+0.0140, +0.0391]** CI > 0 vs the
previously measured +0.016/+0.020, confirming `fusion_layers=0` as the family
default for a third time; Rule 8 does not fire for any arm (no arm reaches ΔH
CI > 0 vs C0 — the 4-factor grounding held the line this time), while the
degenerate `cummean/cummean` arm still tops the recomputed all-time H board at
0.050436 on recall 0.066, so the ±0.015 recall floor stays mandatory; combined
with the view-split null this closes the **input-side asymmetry** direction as the
view null closed the input-view direction — a FIFTH consecutive architecture null.
Nothing registered, nothing promoted. Only Phase **B4** (analysis-only SAE read)
fires.**

## Results — Phase A, 10 runs, sorted by recall@10

All runs `succeeded` (**0 failures, 0 interruptions**, all `exit_code` 0), all
`n_test` = 1,431, all on `seq-20260715-131139`, all on worker `snappler:4101029`.
**Bold = best cell per metric** (artist_adj: lowest is best; ild / mood_coh / H:
highest is best). recall@10 raw = successes of 1,431. Paired Δ = per-session
paired bootstrap vs C0, 2,000 resamples, rng 1337, n = 1,431.

| # | run id | config | params | recall@10 (raw) | paired Δ vs C0 [95% CI] | mrr | artist@10 | genre@10 | music@10 | artist_adj@10 ↓ | mood_coh@10 ↑ | ild@10 ↑ | H@10 (4-factor) ↑ |
|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| C2 | `run-20260726-161801-7646d-seq-nexttrack` | ctrl-h454-matched — single GRU h454 (capacity ctrl for T1/T3/T4) | 969,936 | **0.12509 (179)** | +0.00210 [−0.00699, +0.01118] **tie** | 0.05389 | **0.3375** | 0.4542 | 0.4572 | 0.3668 | **0.4845** | 0.4466 | 0.045765 |
| T1-f0 | `run-20260726-161801-a6423-seq-dualgru` | dual-A0-B256-f0 — `pre_hidden_b=256`, `fusion_layers=0` (**P\*** = **P\*_dual**, best pre arm) | 888,256 | 0.12369 (177) | +0.00070 [−0.01188, +0.01328] **tie** | **0.05470** | 0.3333 | 0.4647 | **0.4575** | 0.3627 | 0.4528 | 0.4616 | 0.044541 |
| **C0** | `run-20260726-161429-bdcbc-seq-nexttrack` | **ctrl-h256 (BASELINE / GATE 0)** — single GRU h256, `pre_hidden=0` | 394,944 | 0.12299 (176) | — (reference) | 0.05341 | **0.3375** | 0.4500 | 0.4558 | 0.3474 | 0.4788 | 0.4536 | 0.047086 |
| C1 | `run-20260726-161801-7bab3-seq-nexttrack` | ctrl-h297-matched — single GRU h297 (capacity ctrl for S1) | 494,697 | 0.11321 (162) | −0.00978 [−0.01747, −0.00280] **loses** | 0.05199 | 0.3284 | 0.4500 | 0.4513 | 0.3602 | 0.4841 | 0.4474 | 0.045285 |
| S1 | `run-20260726-161801-ad53f-seq-nexttrack` | **single-preMLP256 (DECISIVE ABLATION)** — `hidden=256, pre_hidden=256` | 493,504 | 0.10901 (156) | **−0.01398 [−0.02657, −0.00140] loses** | 0.04463 | 0.3235 | **0.4689** | 0.4499 | 0.3240 | 0.4455 | 0.4815 | **0.047132** |
| C3 | `run-20260726-161801-0df6f-seq-dualgru` | **dual-bare-f1 (GATE 1)** — `pre_hidden_a=b=0`, `fusion_layers=1` | 871,872 | 0.10273 (147) | −0.02027 [−0.03494, −0.00697] **loses** | 0.04235 | 0.3089 | 0.4577 | 0.4446 | 0.3092 | 0.4348 | 0.4873 | 0.046612 |
| T2 | `run-20260726-161801-dc7d2-seq-dualgru` | dual-A0-B128-f1 — dose-down | 847,424 | 0.09993 (143) | −0.02306 [−0.03704, −0.00908] **loses** | 0.04490 | 0.2935 | 0.4410 | 0.4415 | 0.3039 | 0.4291 | 0.4939 | 0.046431 |
| T3 | `run-20260726-161801-ca0b8-seq-dualgru` | dual-A0-B384-f1 — dose-up | 1,093,440 | 0.09993 (143) | −0.02306 [−0.03704, −0.00908] **loses** | 0.04333 | 0.3061 | 0.4570 | 0.4441 | 0.3068 | 0.4198 | 0.4961 | 0.045895 |
| T4 | `run-20260726-161801-bd5b5-seq-dualgru` | dual-A256-B256-f1 — symmetry test | 1,068,992 | 0.09783 (140) | −0.02516 [−0.03913, −0.01118] **loses** | 0.03829 | 0.2991 | 0.4389 | 0.4343 | **0.2845** † | 0.4031 | 0.4986 | 0.043985 |
| T1 | `run-20260726-161801-9450b-seq-dualgru` | **dual-A0-B256-f1 (THE USER'S MODEL)** — `pre_hidden_b=256`, `fusion_layers=1` | 970,432 | 0.09713 (139) | −0.02586 [−0.03985, −0.01258] **loses** | 0.04176 | 0.2970 | 0.4500 | 0.4369 | 0.2962 | 0.4195 | **0.5016** † | 0.045769 |

† T4's minimal `artist_adj` and T1's maximal `ild` are the familiar
de-eagering-by-predicting-worse trade: both sit ~0.025 BELOW C0 on recall, and
neither reaches ΔH CI > 0 (see Rule 8).

**No arm is bolded as winner: there is no winner.** C2's nominal 0.12509 (179
hits) is the batch's top recall but is a **tie** with C0 (+0.0021, inside the
±0.0035 reproduction spread) and is a *control*, not a pre-encoder arm — it
authorizes nothing. The baseline C0 holds.

**T2 and T3 share recall@10 0.09993 (143 hits) — verified a genuine coincidence,
not a duplicated run:** **0 of 1,431** top-10 lists are identical, mean top-10
overlap is **3.82 / 10**, and only **94** of each arm's 143 hits are shared.

**MRR — aggregate no-regression check only.** Per the registered pitfall, a
paired MRR CI is NOT computable from `predictions.json` (no rank field, and the
`top_k_ids` ordering does not reconstruct it). **No paired MRR CI is claimed
anywhere in this report.** On the aggregate, T1-f0 holds the batch's highest MRR
(0.05470 vs C0's 0.05341) — exactly echoing last campaign, where the bare
`fusion_layers=0` dual also led MRR while tying on recall. S1 regresses MRR to
0.04463, its recall loss showing up at the top of the list as well.

## Decision rule — applied mechanically

**P\*_dual** (highest recall@10 among {T1, T1-f0, T2, T3, T4}) = **T1-f0**, 0.12369.
**P\*** (highest among {S1, T1, T1-f0, T2, T3, T4}) = **T1-f0**, 0.12369.

| rule | test | result |
|---|---|---|
| **Gate 0** | C0 hits@10 == 176 / 1,431; 4-factor H ∈ [0.0456, 0.0486] | **PASS** — 176 / 1,431, recall bit-identical to the last digit, H 0.0470862 |
| **Gate 1** | C3 hits@10 == 147 / 1,431 | **PASS** — 147 / 1,431, recall `0.10272536687631029` exactly on-record → dual verdicts VALID |
| **Rule 1 — SINGLE-TOWER PRE-ENCODER WIN** | S1 beats C0 CI > 0 **AND** beats C1 CI > 0 | **NOT FIRED — failed on both legs, the first one badly.** S1 vs C0 = **−0.01398 [−0.02657, −0.00140] CI < 0 (S1 LOSES)**; S1 vs C1 = −0.0042 [−0.0168, +0.0084] straddles 0 |
| **Rule 2 — DUALITY REDUNDANT** | P\*_dual vs S1 does NOT reach CI > 0 | **NOT FIRED** — T1-f0 vs S1 = **+0.01468 [+0.00278, +0.02655] CI > 0**. The second tower is *not* redundant over the single pre-encoded tower — but see the interpretation: it is **rescuing** the pre-encoder's damage, not adding over the baseline |
| **Rule 3 — DUAL + PRE WIN** | P\*_dual beats C0 CI > 0 **AND** its capacity control CI > 0 **AND** C3 CI > 0 | **NOT FIRED — first leg fails.** T1-f0 vs C0 = +0.00070 [−0.01188, +0.01328] straddles 0. (It does beat C3 +0.02096 [+0.00769, +0.03424] CI > 0, but ties its h425 capacity control +0.0028 [−0.0105, +0.0161] and ties last campaign's bare dual f0 +0.0049 [−0.0077, +0.0175]) |
| **Rule 4 — ASYMMETRY** | T1 (asymmetric A0/B256) vs T4 (symmetric A256/B256) | **STRADDLES → placement is irrelevant at this dose.** T1 vs T4 = −0.00070 [−0.01188, +0.01118]. Keeping tower A bare is **NOT** load-bearing on recall. (On H, T1 > T4 by +0.00178 [+0.00011, +0.00358] CI > 0 — a marginal H-only effect at equally-poor recall) |
| **Rule 5 — DOSE RESPONSE** | C3(0) → T2(128) → T1(256) → T3(384) at f1 | **FLAT.** All three adjacent pairs straddle 0 (T2−C3 −0.0028 [−0.0147, +0.0091]; T1−T2 −0.0028 [−0.0154, +0.0091]; T3−T1 +0.0028 [−0.0098, +0.0154]) **and** the endpoints do not separate (T3−C3 −0.0028 [−0.0154, +0.0105]). Per the pre-registered clause, FLAT **blocks registration** even for a win rule — and none fired. **B3 does NOT fire** |
| **Rule 6 — FUSION DEPTH RIDER** | T1-f0 vs T1 | **FIRED, "persists" direction.** T1-f0 − T1 = **+0.02655 [+0.01398, +0.03913] CI > 0**. The documented f1 regression survives the pre-encoder and is **LARGER** than before (+0.0266 vs +0.016/+0.020) → `fusion_layers=0` confirmed as the family default for a third time. The pitfalls entry is strengthened, not amended |
| **Rule 7 — REFUTED** | NO arm in {S1, T1, T1-f0, T2, T3, T4} beats C0 CI > 0 | **FIRED.** All six checked: **five lose CI < 0** (S1, T1, T2, T3, T4), **one ties** (T1-f0 straddles). **Zero** CI > 0 → the nonlinear per-step pre-encoder is REFUTED on this corpus. **B4 fires anyway** via the tie clause (S1 ties C1 within ±0.015; T1-f0 ties C0/h425/D-LL-f0) |
| **Rule 8 — CROWN (4-factor H)** | ΔH vs C0 CI > 0 **AND** recall not > 0.015 below C0 | **NOT FIRED for any arm — the ΔH leg fails for all ten.** No arm reaches ΔH CI > 0 against C0; the closest is S1 at +0.00005 [−0.00153, +0.00161] (straddles). Four arms are CI < 0 (C1, C2, T1-f0, T4). **No crown candidate, so the recall floor was never even reached this time** |

### Rule 5 — the dose curve, as measured

| `pre_hidden_b` @ f1 | arm | recall@10 (raw) | Δ vs C3 (bare) |
|--:|---|--:|---|
| 0 | C3 | 0.10273 (147) | — |
| 128 | T2 | 0.09993 (143) | −0.0028 [−0.0147, +0.0091] straddles |
| 256 | T1 | 0.09713 (139) | −0.0056 [−0.0189, +0.0077] straddles |
| 384 | T3 | 0.09993 (143) | −0.0028 [−0.0154, +0.0105] straddles |

FLAT by the pre-registered test — but note the *nominal* ordering is **downward
then flat**, never upward: every dosed arm sits nominally BELOW the bare dual.
Taken with S1's significant loss, the sign of this axis is consistently negative
across both towers and all four widths. There is no interior peak to chase.

### Rule 8 — the H board leg (paired ΔH vs C0, 4-factor)

| arm | H@10 | paired ΔH vs C0 [95% CI] | ΔH leg | recall floor | Rule 8 |
|---|--:|---|---|---|---|
| S1 | **0.047132** | +0.00005 [−0.00153, +0.00161] | straddles ✗ | (n/a) | **NO** |
| C3 | 0.046612 | −0.00047 [−0.00228, +0.00125] | straddles ✗ | — | **NO** |
| T2 | 0.046431 | −0.00065 [−0.00264, +0.00128] | straddles ✗ | — | **NO** |
| T3 | 0.045895 | −0.00119 [−0.00311, +0.00059] | straddles ✗ | — | **NO** |
| T1 | 0.045769 | −0.00132 [−0.00330, +0.00058] | straddles ✗ | — | **NO** |
| C2 | 0.045765 | −0.00132 [−0.00232, −0.00034] | CI < 0 ✗ | — | **NO** |
| C1 | 0.045285 | −0.00180 [−0.00277, −0.00083] | CI < 0 ✗ | — | **NO** |
| T1-f0 | 0.044541 | −0.00255 [−0.00421, −0.00087] | CI < 0 ✗ | — | **NO** |
| T4 | 0.043985 | −0.00310 [−0.00528, −0.00109] | CI < 0 ✗ | — | **NO** |

**This is the first campaign in which the H board produced no candidate at all** —
a direct consequence of the 4-factor `music@10` grounding factor. Under the old
3-factor composite, five of these arms would have shown large H "wins" (S1 0.14499,
T2 0.14751, T1 0.14809, C3 0.14636, T3 0.14436 vs C0's 0.14172), all bought by
de-eagering at a recall loss. The grounding factor cancels them because
`music@10` falls in lockstep with recall (0.4558 → 0.4343 across the batch).

## Recomputed 4-factor holisticness board (all 32 facet-carrying runs on record)

Re-derived by the runner from stored facets, as the design instructs; every value
the designer pre-computed reproduces exactly. Top 12 shown.

| rank | run | predictor | recall@10 | old 3-factor H | **new 4-factor H** |
|--:|---|---|--:|--:|--:|
| 1 | `run-20260725-143502-88a4c-seq-dualgru` (D-CC, **DEGENERATE**) | seq-dualgru | 0.06639 ⚠ | 0.19715 | **0.050436** |
| 2 | `run-20260723-002044-17cf8-seq-nexttrack` (stale-checkout, not code-comparable) | seq-nexttrack | 0.12020 | 0.14374 | 0.047585 |
| 3 | `run-20260722-165345-353b5-seq-nexttrack` | seq-nexttrack | 0.12369 | 0.14232 | 0.047404 |
| 4 | `run-20260723-002044-af11a-seq-nexttrack` (stale-checkout, not code-comparable) | seq-nexttrack | 0.12020 | 0.14273 | 0.047346 |
| 5 | **`run-20260726-161801-ad53f-seq-nexttrack` (S1, this campaign)** | seq-nexttrack | 0.10901 | 0.14499 | **0.047132** |
| 6 | **`run-20260726-161429-bdcbc-seq-nexttrack` (C0, this campaign)** | seq-nexttrack | 0.12299 | 0.14172 | 0.047086 |
| 6= | `run-20260725-143143-97c1e-seq-nexttrack` (C0 reference) | seq-nexttrack | 0.12299 | 0.14172 | 0.047086 |
| 6= | `run-20260722-165345-e39c9-seq-nexttrack` | seq-nexttrack | 0.12299 | 0.14172 | 0.047086 |
| 9 | `run-20260725-143502-be11b-seq-nexttrack` (h425) | seq-nexttrack | 0.12089 | 0.13850 | 0.046677 |
| 10 | **`run-20260726-161801-0df6f-seq-dualgru` (C3, this campaign)** | seq-dualgru | 0.10273 | 0.14635 | 0.046612 |
| 10= | `run-20260725-143502-12fe6-seq-dualgru` (D-LL f1) | seq-dualgru | 0.10273 | 0.14635 | 0.046612 |
| 12 | `run-20260725-143502-a95c5-seq-dualgru` (D-LC f1) | seq-dualgru | 0.09713 ⚠ | 0.15000 | 0.046567 |

⚠ = below the 0.10 recall sanity line. **The absolute 4-factor board leader
remains the degenerate `cummean/cummean` dual arm** (`88a4c`, H 0.050436 at
recall@10 0.06639 = **95 of 1,431** vs the baseline's 176). The grounding factor
demoted `seq-mood` (H 0.23319 → 0.043881, from rank 1 to rank 24) but did **NOT**
fix `88a4c`. **The ±0.015 recall floor therefore remains mandatory on any H
claim.** For reference, the recall champion `seq-blend` sits at 4-factor H
0.020830 — the H board and the recall board still point in opposite directions.

## Findings

**1. The pre-encoder is not neutral — in the single tower it is actively harmful,
and the analytic setup makes the mechanism identifiable.** S1 loses to C0 by
−0.0140 CI < 0 (156 vs 176 hits). Because a *linear* pre-encoder at
`pre_hidden ≥ latent_dim` is exactly as expressive as a bare GRU (`W_i·(Vx) =
(W_iV)x`, no rank constraint), the ReLU is the only expressivity difference — so
this is not an optimization-difficulty story about extra depth, it is the
rectifier itself. The most likely mechanism: PCA-192 latents are **zero-centred
and near-symmetric by construction**, so a half-wave rectifier zeroes roughly half
of every input vector's coordinates before the recurrence ever sees them. At
`pre_hidden = 256` — only 1.33× the input dimension — there is not enough
over-completeness to re-encode the discarded negative half-space, unlike in the
wide embeddings where ReLU MLPs normally live. The GRU's own gated linear input
transform, by contrast, preserves the full signed latent. **A concrete prediction
this makes, and the natural next probe: the harm should shrink as `pre_hidden`
grows well past 2×192, and should vanish for a linear or GELU/tanh pre-encoder.**
The FLAT dose curve up to 384 (2× the input dim) is weak evidence against a quick
recovery, but 384 is still narrow.

**2. Rule 2 inverted, and the inversion is the attribution.** T1-f0 beats S1
CI > 0 (+0.0147), so the second tower is *not* redundant over a single
pre-encoded tower. But T1-f0 only **ties** the plain h256 baseline. Read together:
the bare tower A in T1-f0 is **rescuing** what the pre-encoded tower B destroyed —
it restores a direct, un-rectified path from the raw latents to the readout,
recovering the baseline but never exceeding it. That is the opposite of the
co-adaptation story the hypothesis wanted: the second tower is a repair, not a
complement. This is consistent with Rule 4 straddling — if tower A's job is to
carry the raw signal, then symmetrizing (T4, both towers rectified) should hurt,
and nominally it does (140 vs 139 hits is a wash, but both sit ~0.025 below C0
whereas T1-f0 with its bare tower reaches 177).

**3. Rule 6 is the campaign's one solid positive: the `fusion_layers=1`
regression is now measured three times and is BIGGER with a pre-encoder.**
+0.0266 [+0.0140, +0.0391] here vs +0.016 and +0.020 in the two 2026-07-25 view
regimes. The mechanism holds and sharpens: the objective is cosine/InfoNCE
retrieval *in* the item-latent space, so every extra nonlinear layer between the
recurrent state and the metric readout adds a warp the loss must undo — and the
more nonlinearity you have already added on the input side, the more the output
side warp costs. `fusion_layers=0` is the family default, full stop. Note also
that T1-f0 (the only pre arm that ties the baseline) is exactly the arm with
**both** the bare tower A **and** `fusion_layers=0` — i.e. the arm with the
shortest un-warped path from raw latents to the readout.

**4. The param-matched width control was a weaker instrument than the design
assumed, and this is a caveat future campaigns should carry.** C1 (h297) came in
at 0.11321, **losing to C0 CI < 0** (−0.0098). The batch's width points are
h256 → 176 hits, h297 → 162, h425 → 173 (on record), h454 → 179: **non-monotone,
spanning 0.0119 in recall with no trend.** So (a) "single-GRU width is flat on
PCA-192" survives as a *trend* claim (h454 ties h256) but the per-width jitter is
~±0.01 on a single split, roughly two-thirds of the noise band; (b) Rule 1's
"beats C1" leg was consequently nearly free, and it is fortunate the rule required
the C0 leg too — S1's verdict rests on the C0 comparison, which is
reproduction-verified to the last digit. **A width-matched control that is itself
a single unreplicated run is not a reliable capacity anchor at this effect size.**

**5. Fifth consecutive architecture null, and the input side is now closed.** The
view-split null (2026-07-25) closed *deterministic* view asymmetry; this closes
*learned* input-side re-embedding. Together with the four learned-combiner nulls,
the pattern is now hard to dismiss: on this single-user, 55 %-cold corpus,
**adding representation machinery around a single GRU over a good latent space
does not help** — the wins on record came from improving the *latent space itself*
(InfoNCE content projection, PCA-192) and from the fixed z-blend. The design
anticipated this outcome and it still teaches: the axis is now quantitatively
bounded (P\* = 0.12369, a tie), the dose curve is flat to 2× input dim, placement
is irrelevant, and the fusion-depth interaction is resolved.

**6. The 4-factor holisticness metric earned its keep this campaign.** It is the
first batch to yield **zero** H candidates, and the counterfactual is stark: five
arms would have shown 3-factor H "wins" over C0, every one of them purchased by
predicting less-eagerly and worse. The `music@10` grounding factor cancels them
because it falls in lockstep with recall. It remains **incomplete**, though —
`88a4c` still tops the absolute board at recall 0.066, so the floor is still
load-bearing.

## Best on record after this work — UNCHANGED

| board | best | value | provenance |
|---|---|--:|---|
| **recall@10 — overall** | `seq-blend` `blend-gru-markov-content-proj` | **0.21174** (303 of 1,431) | `run-20260723-014711-1c1a3`; `2026-07-18b-nexttrack-projection-registration-and-scan.md` |
| **recall@10 — single model** | `seq-nexttrack` `gru-infonce-h256` | **0.12299** (176 of 1,431) | `run-20260725-143143-97c1e`, reproduced **bit-identically** in-batch as `run-20260726-161429-bdcbc` (7th reproduction) |
| **holisticness@10 (4-factor) — recall-competent** | `seq-nexttrack` `gru-infonce-h256` | **0.047086** | same run; batch max among arms clearing the recall line |
| **holisticness@10 (4-factor) — absolute** | `seq-dualgru` D-CC (**DEGENERATE**, recall 0.06639) | 0.050436 | `run-20260725-143502-88a4c`; fails the ±0.015 recall floor by 3.8× the band |
| **`fusion_layers` family default** | `0` | +0.0266 over f1 | **this campaign** (third confirmation, largest effect yet) |

Nothing registered (Rule 7 → no definition created; and Rule 5 = FLAT would have
blocked registration independently). Nothing promoted. Champion, leaderboard and
best-models pins all unchanged.

**best-models group.** The remote-worker recompute defect (found 2026-07-25:
`spawn_recompute` has four hub-side call sites, but the worker writes straight to
Postgres) meant the group was still at `updated_at` 2026-07-25T15:48 with **5 of
12** entries after this batch. A manual `POST /api/best-models/recompute`
absorbed it (→ `2026-07-26T17:11:52`, 12 entries) — and **flooded the group with
this campaign's refuted arms**: 7 of 12 entries are now from this batch,
including **three arms below the 0.10 recall sanity line** (T2 0.09993,
T3 0.09993, T1 0.09713). This is exactly the degenerate-flood risk
the design flagged. **Curation is not the runner's call: handed to
`best-model-selector`.**

## Follow-ups

1. **`best-model-selector` (do first).** The group is auto-flooded with 7 of this
   campaign's 10 runs (C0, C2, C3, S1, T1, T2, T3), three of them below 0.10
   recall. Exclude the refuted arms,
   re-establish family diversity, and consider pinning C0 as the single-model
   anchor. Also worth a decision: `88a4c` (degenerate H board leader) should be
   permanently excluded, not re-litigated every campaign.
2. **Phase B4 — the SAE read (the only Phase-B trigger that fired).** Rule 7's
   tie clause fires it: S1 ties C1 within ±0.015 and T1-f0 ties C0/h425/D-LL-f0.
   Delegate to **`information-capture-analyst`**: `POST /api/interp/model-sae`,
   **`topk=32`**, on **T1-f0** (`run-20260726-161801-a6423`, taps `pre_b`,
   `tower_a`, `tower_b`) and on **S1** (`run-20260726-161801-ad53f`, whose `pre`
   tap `model-sae` now exposes ahead of `recurrent`). All five candidate runs have
   `has_checkpoint=true`. **The exact question:** does `pre_b` / `pre` carry
   next-item concepts the bare recurrent state lacks — and, given Finding 1, is
   the rectified representation measurably *lossier* (lower
   `next_item_decodability` than the bare tower)? The view-split variant's answer
   was **+0** new concepts at the worst genre AUC measured (0.764 vs 0.801).
   Carry the registered pitfall: the L1 SAE **will not sparsify** on dense
   recurrent states → read `n_interpretable_concepts` as saturated, trust ranked
   `atoms_by_concept` + `next_item_decodability`.
3. **The one cheap probe Finding 1 makes worth running (needs a new design, NOT
   authorized here): a nonlinearity/width probe on the single tower.** S1 at
   `pre_hidden` ∈ {576, 768} (3–4× input dim) and a `tanh`/`GELU` variant would
   test the half-wave-rectification mechanism directly and either close the axis
   completely or find the width at which it turns neutral. Note this needs an
   `activation` parameter that does not exist — an orchestrator change.
4. **B5 was NOT fired (Rule 1/3 did not fire), but the dropout confound remains
   un-disentangled** and now cuts the other way: the pre-MLP ends in `Dropout`
   (`seq_dualgru.py:236`), so pre arms get GRU-**input** dropout the bare arms
   lack (`seq_nexttrack` drops the RNN *output* only). S1's loss could therefore
   be *partly* over-regularization rather than the rectifier. **Per the design's
   instruction I did not improvise a `pre_dropout` parameter — and I recommend
   adding one** (`pre_dropout`, default = `dropout`, so existing behaviour is
   unchanged) rather than the matched `dropout=0.0` pair, because the pair
   confounds input- and output-side dropout in a single comparison. That is an
   orchestrator change.
5. **Do not re-spend runs on** `pre_hidden` ≤ 384 with `pre_layers=1` on PCA-192
   (bounded at 0.12369, a tie with baseline, and negative in the single tower);
   `fusion_layers=1` (now 3× confirmed, −0.0266); `layers_*` > 1 (depth cliff);
   symmetric vs asymmetric pre placement at width 256 (Rule 4 straddles).
6. **Width jitter deserves one cheap replication.** Finding 4 shows adjacent
   single-GRU widths differ by up to 0.0119 recall with no trend, which weakens
   every param-matched control this project uses. Re-running h297 and h454 once
   each (`seq-nexttrack` is deterministic, so this needs a second *split*, not a
   second seed — `seq-20260718-211238`) would establish whether the h297 dip is
   real structure or split-specific noise. 2 runs.
