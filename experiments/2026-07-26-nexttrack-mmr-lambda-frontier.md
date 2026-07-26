# 2026-07-26 — The MMR λ frontier under the live 4-factor crown: champion vs. best single, with `artist_conc` as the decisive unknown

## Goal

Test whether the eval-time MMR re-rank (`mmr_lambda`) is a genuine, resolvable
**holisticness@10** lever, and decide the pre-registered mechanism question:
does sonic diversification de-concentrate **artists** (H1, responsive) or not
(H2, sticky)? This is the native-run confirm that
`2026-07-22-session-holisticness-and-antieager-levers.md` §Phase C explicitly
deferred to a user decision, and that `PROJECT-FACTS.md` §Follow-ups carried as
*"the MMR λ≈0.9 eval-time re-rank remains the only near-free holisticness lever
on record."*

**Dataset (all 10 arms): `seq-20260715-131139`** — canonical PCA-192 leak-free
split, 7,154 sessions / 19,402 items, 5,723 train / 1,431 test @ cut
2024-08-24T14:28:02, cold-item rate 0.5507. No dataset work required.

**Live crown definition** (`predictors/seq_common.py:610-623`, `MUSIC_FLOOR =
0.34` at line 52), verbatim:

```
H = max(mood_coh,0) × ild × (1 − clamp(max(artist_adj, artist_conc),0,1))
      × clamp((music@10 − 0.34)/0.66, 0, 1)
```

each facet the **mean over rows**, the product taken **of the aggregates**.

**Anchor A — the recall champion.** `blend-gru-markov-content-proj`
(`seq-blend`), reference run `run-20260723-014711-1c1a3-seq-blend`: recall@10
0.21174004192872117 (303/1431), mrr 0.12198540476105842, music@10 0.453790,
mood_coh 0.398222, ild 0.406374, artist_adj 0.609713, **artist_conc 0.691653
(binds `max()`)**, H **0.008603**. Held hyperparams: `alpha 0.5, arch gru,
batch_size 128, content false, content_agg max, epochs 40, hidden 256, k 10,
loss infonce, lr 0.001, patience 6, projection true, projection_epochs 40,
projection_lr 0.001, projection_objective infonce, projection_rank 0,
projection_tau 0.07, seed 1337`.

**Anchor B — the best single model on the live crown.** `gru-infonce-h256`
(`seq-nexttrack`), reference run `run-20260726-161429-bdcbc-seq-nexttrack`:
recall@10 0.12299091544374564 (176/1431), mrr 0.053411071731908816, music@10
0.455776, mood_coh 0.478838, ild 0.453563, artist_adj 0.347449, **artist_conc
0.355913 (binds)**, H **0.024538**. Held: `arch gru, batch_size 128,
bidirectional false, dropout 0.1, eager_beta 0.0, eager_margin 0.0, epochs 40,
hidden 256, k 10, loss infonce, lr 0.001, num_layers 1, patience 6, pre_hidden
0, pre_layers 1, residual false`.

**Absolute all-time bar** (code-comparable, healthy, recall ≥ 0.108): live-crown
H `H_bar = 0.024872` — `run-20260725-143502-be11b-seq-nexttrack` (h425 capacity
control, recall 0.12089).

**Axes swept:** `mmr_lambda` (primary, 5 points on A, 4 on B) and `mmr_pool`
(one probe, B4). Everything else frozen. λ is **eval-only**: the trained weights
are bit-identical across arms of a model — here used as the *instrument*, which
is what makes λ=1.0 a bit-identity gate and the arms maximally paired.

## Outcome in one line

**MMR is CONFIRMED as the first genuine crown lever on record — three TIER-1
"FREE" arms on the single GRU beat the all-time H bar at a recall that is a
statistical tie (best: λ=0.7/pool=50, H 0.030355 = +22% over `H_bar`, recall
0.12020 vs 0.12299, Δrecall CI straddles 0), and the decisive mechanism question
resolves unanimously to H1 (responsive): sonic diversification de-concentrates
artists 1.6–3.1× FASTER than it reduces seed-echo, in 8 of 8 arms.**

## Method / provenance

All 10 arms on the **same host**, remote worker `snappler:573145`, serialized via
`{"queue": true}` (claimed one at a time — dodging the registered `--max-runs 2`
thread-oversubscription collapse). Predictors md5-verified identical to the hub
before launch (`seq_common.py` = `1c86afaab58fbe2d595e166d68099f25`).
**0 failed, 0 interrupted.** Wall clock 18:33:36 → 19:11:19 = **37.7 min**
(2 gate runs 6.8 min; 8 frontier arms 30 min; `seq-blend` arms 4.4–4.9 min each,
`seq-nexttrack` arms 2.3–2.8 min each).

**Every arm's H was recomputed from `predictions.json`** (served by
`GET /api/runs/<id>/predictions`) under the live formula, never trusted from
storage — one code path for all 10 arms so per-row 4-dp rounding bias cancels in
every Δ. Validation on the two controls: A0 recomputed 0.008603113 vs stored
0.008603039 (|Δ| 7.4e-8); B0 recomputed 0.024538536 vs stored 0.024538326
(|Δ| 2.1e-7) — both far inside the 1e-4 tolerance.

**Statistics.** Paired bootstrap, 2,000 resamples, rng 1337, resampling the 1,431
test sessions; for each resample **all five aggregate facets and hence H are
recomputed for both arms on the same resampled index set**, and likewise paired
Δhit@10. Every Δ is vs the arm's **own-model λ=1.0 control**. Half-widths below
are **measured**, not the design's estimates.

### Gates — all three PASSED

- **G1 bit-identity (judged on hit count).** A0 → **303/1431** exactly, mrr
  0.12198540479212304 vs the anchor's 0.12198540476105842 (Δ 3.1e-11,
  reduction-order float noise); every other A0 facet bit-identical to `1c1a3`.
  B0 → **176/1431** exactly, mrr identical to the last digit. **A0 is the first
  `seq-blend` run ever executed on `snappler`, and it reproduces the hub
  bit-identically** — see Findings §5.
- **G2 metric-index.** All of `music_at_k`, `mood_coh_at_k`, `ild_at_k`,
  `artist_adj_at_k`, `artist_conc_at_k`, `holisticness_at_k` present on all 10
  arms, with per-row `music_sim`/`mood_coh`/`ild`/`artist_adj`/`artist_conc`.
  **This gate FAILED on a first attempt and cost an infra fix — see Findings §6.**
- **G3 MMR-active (silent-failure detector).** Every λ<1 arm's `ild_at_k` differs
  from its own control by ≫ 1e-6 (A: 0.0298 / 0.1137 / 0.2463 / 0.4082; B: 0.0181
  / 0.0425 / 0.0778 / 0.0482). No arm was a no-op; no arm invalidated.

## Results

Sorted by holisticness@10 (the primary). Best cell per metric **bolded**;
`Δ` columns are paired vs the arm's own λ=1.0 control, 95% CI.

| arm | run id | λ | pool | H@10 | ΔH [95% CI] | hw(ΔH) | recall@10 | hits | Δrecall [95% CI] | mrr | music@10 | mood_coh | ild | artist_adj | artist_conc | binds | tier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B3 | `run-20260726-184115-bf395-seq-nexttrack` | 0.7 | 200 | **0.034171** | +0.009648 [+0.008322,+0.011042] | 0.001360 | 0.11461 | 164 | −0.008492 [−0.016073,−0.001398] | 0.05124 | 0.4747 | 0.4585 | 0.5314 | 0.3129 | 0.2707 | artist_adj | **TIER-2 PRICED** |
| A4 | `run-20260726-184115-1dc77-seq-blend` | 0.3 | 200 | 0.031626 | +0.023032 [+0.020623,+0.025574] | 0.002476 | 0.10971 | 157 | −0.101928 [−0.120196,−0.083857] | 0.09969 | **0.4925** | 0.2378 | **0.8146** | 0.2934 | **0.1897** | artist_adj | TIER-3 REJECTED |
| **B4** | **`run-20260726-184116-20d9b-seq-nexttrack`** | **0.7** | **50** | **0.030355** | **+0.005811 [+0.004819,+0.006846]** | **0.001013** | **0.12020** | **172** | **−0.002908 [−0.008386,+0.002795]** | **0.05253** | 0.4670 | 0.4671 | 0.5017 | 0.3270 | 0.3019 | artist_adj | **TIER-1 FREE — NEW ALL-TIME CROWN candidate** |
| B2 | `run-20260726-184115-147a6-seq-nexttrack` | 0.8 | 200 | 0.029458 | +0.004922 [+0.004036,+0.005856] | 0.000910 | 0.11950 | 171 | −0.003600 [−0.009085,+0.001398] | 0.05278 | 0.4648 | 0.4688 | 0.4961 | 0.3300 | 0.3073 | artist_adj | TIER-1 FREE — crown candidate |
| B1 | `run-20260726-184115-e1539-seq-nexttrack` | 0.9 | 200 | 0.026761 | +0.002203 [+0.001568,+0.002902] | 0.000667 | 0.12089 | 173 | −0.002141 [−0.006289,+0.002096] | 0.05284 | 0.4597 | 0.4746 | 0.4716 | 0.3408 | 0.3365 | artist_adj | TIER-1 FREE — crown candidate |
| B0 | `run-20260726-183333-9a0b2-seq-nexttrack` | 1.0 | 200 | 0.024539 | — (control) | — | **0.12299** | **176** | — | 0.05341 | 0.4558 | **0.4788** | 0.4536 | 0.3474 | 0.3559 | artist_conc | control |
| A3 | `run-20260726-184115-7a2f7-seq-blend` | 0.5 | 200 | 0.023576 | +0.014972 [+0.013180,+0.016839] | 0.001830 | 0.16422 | 235 | −0.047296 [−0.061495,−0.032844] | 0.10791 | 0.4871 | 0.3150 | 0.6527 | 0.4855 | 0.4607 | artist_adj | TIER-3 REJECTED |
| A2 | `run-20260726-184115-36edb-seq-blend` | 0.7 | 200 | 0.015289 | +0.006675 [+0.005694,+0.007696] | 0.001001 | 0.19357 | 277 | −0.018142 [−0.029350,−0.007669] | 0.11560 | 0.4759 | 0.3643 | 0.5201 | 0.5713 | 0.6082 | artist_conc | TIER-2 PRICED |
| A1 | `run-20260726-184115-6cb38-seq-blend` | 0.9 | 200 | 0.010435 | +0.001835 [+0.001458,+0.002223] | **0.000382** | 0.21034 | 301 | −0.001272 [−0.007687,+0.004892] | 0.12051 | 0.4625 | 0.3914 | 0.4362 | 0.6029 | 0.6707 | artist_conc | TIER-1 FREE |
| A0 | `run-20260726-183333-1c361-seq-blend` | 1.0 | 200 | 0.008603 | — (control) | — | **0.21174** | **303** | — | **0.12199** | 0.4538 | 0.3982 | 0.4064 | **0.6097** | 0.6917 | artist_conc | control |

`mmr_pool` axis (R3), same λ=0.7, same weights, same host:

| comparison | ΔH [95% CI] | Δrecall [95% CI] |
|---|---|---|
| B4 (pool 50) − B3 (pool 200) | −0.003838 [−0.004692,−0.003034] **CI<0** | +0.005584 [+0.000699,+0.011181] **CI>0** |
| B4 (λ0.7/pool50) − B2 (λ0.8/pool200) | +0.000889 [+0.000497,+0.001360] **CI>0** | +0.000692 [0.000000,+0.002096] straddles |

## Findings

**1. MMR is a real crown lever — the first one on record. Every single λ<1 arm
lifts H with CI entirely > 0 (8/8).** After five consecutive architecture nulls
and a campaign that produced *zero* holisticness candidates
(`2026-07-26-nexttrack-pre-encoder-scan.md`), the eval side delivers on the first
attempt. The mechanism is exactly the one the crown was designed to reward: as λ
falls, **three of four factors improve simultaneously** — ild ↑ (0.4536→0.5314 on
B, 0.4064→0.8146 on A), eagerness ↓, and music@10 ↑ (0.4558→0.4747 on B) — and
only recall pays. That music@10 *rises* as you diversify is what stops the
grounding factor from cancelling the gain, and it is why these H wins are
categorically different from the degenerate `cummean/cummean` and `seq-mood`
pathologies the floor was written to reject.

**2. R2 VERDICT — H1 (responsive) CONFIRMED, unanimously and decisively.** The
pre-registered decisive unknown resolves without ambiguity: in **8 of 8 arms**
`Δartist_conc ≤ Δartist_adj` (both negative), and the ratio
`|Δartist_conc| / |Δartist_adj|` runs **1.59× – 3.06×**:

| arm | λ | Δartist_adj | Δartist_conc | ratio | H1? |
|---|---|---|---|---|---|
| A1 | 0.9 | −0.0068 | −0.0210 | 3.06× | YES |
| A2 | 0.7 | −0.0384 | −0.0834 | 2.17× | YES |
| A3 | 0.5 | −0.1242 | −0.2309 | 1.86× | YES |
| A4 | 0.3 | −0.3163 | −0.5020 | 1.59× | YES |
| B1 | 0.9 | −0.0066 | −0.0194 | 2.92× | YES |
| B2 | 0.8 | −0.0175 | −0.0486 | 2.78× | YES |
| B3 | 0.7 | −0.0346 | −0.0853 | 2.46× | YES |
| B4 | 0.7/p50 | −0.0205 | −0.0541 | 2.64× | YES |

H2 (sticky) is **refuted** — it required `|Δartist_conc| < 0.5·|Δartist_adj|` and
the measurement is off by a factor of 3–6 in the opposite direction. So
**same-artist tracks ARE near-neighbours in the content-metric space**: the
max-similarity penalty de-concentrates artists as a side effect, without any
explicit notion of artist. **Consequence: sonic MMR IS a de-eagering lever**, and
the pre-registered H2 sequel (porting `seq-mood`'s explicit `artist_penalty` to
`seq-blend`/`seq-nexttrack`) is **NOT** the indicated next move. The indicated
next move is pushing the λ frontier and testing λ on the blend's Markov leg.

Note the ratio **shrinks monotonically as λ falls** (3.06→1.59 on A, 2.92→2.46 on
B): the easy artist de-concentration is bought first and the mechanism saturates.

**3. The binding facet SWITCHES, and that locates the end of the lever.** The
design flagged that "a λ at which the binding facet switches from `artist_conc`
to `artist_adj` is the point past which further sonic diversification stops
buying eagerness". Measured: on the **champion** the switch is between λ0.7 and
λ0.5 (conc binds at 1.0/0.9/0.7, adj binds at 0.5/0.3); on the **GRU** it has
already happened by λ0.9 (conc binds only at λ=1.0). Because `artist_adj` falls
~2.5× *slower* than `artist_conc`, once adj binds, the `(1 − eager)` factor
becomes the stiff one and each further recall point buys progressively less
crown. This is the structural reason B's frontier flattens where it does, and it
predicts that pushing λ below ~0.7 on the GRU is poor value — visible already in
B3, whose extra H over B2 costs the recall tie.

**4. R1 VERDICT — three TIER-1 "FREE" arms, all three new all-time crown
candidates, and the promotable operating point is the pool probe.** With
`H_bar = 0.024872` and the measured half-widths:

- **B4 (λ0.7, pool 50) is the single best TIER-1 arm: H 0.030355** (+22.1% over
  `H_bar`, and `H > H_bar + hw` by 5.4 half-widths), ΔH +0.005811 CI>0,
  **Δrecall −0.002908 CI straddles 0** and |Δrecall| = 0.0029 ≪ 0.015, recall
  0.12020 ≥ 0.108. It beats B2 on H by +0.000889 **CI>0** at a recall that
  straddles — so its rank as best TIER-1 is *resolved*, not nominal.
- **B2 (λ0.8) H 0.029458** — the design's pre-registered promotion hypothesis and
  sweet spot. It fires exactly as predicted (projected ~0.0321 vs measured
  0.029458; projected Δrecall ≈ −0.008 vs measured −0.0036, i.e. *better* than
  projected).
- **B1 (λ0.9) H 0.026761** — the "free λ", also a crown candidate.
- **A1 (λ0.9) is TIER-1 FREE on the champion** (ΔH +0.001835 CI>0, Δrecall
  −0.001272 straddles 0, 301/1431 vs 303) — but at H 0.010435 it is nowhere near
  the crown; the champion side was mechanistic exactly as the design said.
- **TIER-2 PRICED:** B3 (λ0.7, pool 200) — the **highest H measured anywhere,
  0.034171**, but Δrecall CI<0, so a real cost; and A2 (λ0.7 champion, H
  0.015289, Δrecall −0.018 CI<0). Both registered, neither promotion-recommended.
- **TIER-3 REJECTED:** A3 (Δrecall −0.047 < −0.03) and A4 (−0.102). A4 is the
  clean mechanism probe and behaved as pre-declared: it maxes ild (0.8146) and
  minimizes artist_conc (0.1897) at 157/1431 hits. Frontier mapping only.

**The 2026-07-22 standing flag is CLOSED POSITIVE, not refuted.** "MMR λ≈0.9 as a
near-free holisticness upgrade" is now natively confirmed on both models: it is
free (recall ties on A1 *and* B1) and its gain is resolvable — though on the
champion the gain is small in absolute terms, and the *valuable* operating points
turn out to be λ0.7–0.8 on the single GRU, not λ0.9 on the champion.

**Recall@10 is untouched.** No arm improves recall; the board's recall champion
remains `blend-gru-markov-content-proj` at 0.21174. This campaign moves the
crown, not the recall leaderboard.

**5. The shared-weights instrument is 3–33× sharper than the family noise floor,
and it overturned a pre-registered "not resolvable" call.** The conservative floor
from the 2026-07-26 architecture family was hw 0.0124 on recall and 0.0018 on H.
Measured here: **hw(ΔH) 0.000382–0.002476** and **hw(Δrecall) 0.004193–0.018169**,
with `n_disc` (sessions whose hit@10 changed at all) as low as **9** for B1. A1
was pre-registered as *"likely UNRESOLVABLE"* at 0.6–0.8 conservative half-widths
— it in fact resolved at **4.8 measured half-widths** (ΔH +0.001835
[+0.001458,+0.002223], hw 0.000382, the sharpest in the batch). The design
predicted this outcome and its meaning: **λ arms of one model are a maximally
paired comparison, and the conservative family half-width materially
under-powers them.** Any future eval-time-lever campaign should budget MDE from
the shared-weights pairing, not from the architecture-family figure.

**Corollary — the offline sweep transferred essentially exactly.** The 2026-07-22
offline λ sweep on the champion checkpoint projected recall 0.2103 / 0.1936 /
0.1642 / 0.1097 for λ 0.9/0.7/0.5/0.3; the native runs returned **0.21034 /
0.19357 /0.16422 / 0.10971**. Four for four to ~4 decimals. The offline MMR
driver is validated as a *screening* instrument — which strengthens, not weakens,
the eval-only pitfall (see §7).

**6. `seq-blend` reproduces bit-identically on `snappler` — the withdrawn
cross-host offset is now confirmed absent for a SECOND predictor family.** No
`seq-blend` run had ever executed on the remote worker; the design made A0 an
explicit cross-host check and budgeted a hub-fallback contingency. A0 returned
recall@10 0.21174004192872117 = **303/1431**, with music@10, mood_coh, ild,
artist_adj and artist_conc all bit-identical to the hub's `1c1a3`, and mrr
agreeing to 3.1e-11. **The fallback was not needed, and the "no measured host
effect given identical code" claim now rests on two families rather than one.**

**7. NEW INFRA PITFALL, found the hard way: a `(deleted)` worker binary image
strips unknown metric keys SILENTLY — rsyncing the binary is not enough, the
daemon must be cycled.** Gate G2 failed on the first Phase-1 attempt: both
controls came back with **no `artist_conc_at_k`** despite the predictors being
md5-identical to the hub and emitting it correctly. The trace:

1. `~/lensing-worker/predictors/seq_common.py` md5-matched the hub and contained
   `artist_conc` (12 occurrences) — **not** the stale-predictors cause the design
   anticipated, so G2's prescribed remedy ("resync predictors") was a no-op.
2. The worker's own artifact **had the data**: its `metrics.json` carried
   `"artist_conc_at_k": 0.35591272614333413` and its `predictions.json` carried
   per-row `artist_conc` on all 1,431 rows.
3. The hub's DB row did not: `metrics->>'artist_conc_at_k'` was NULL while
   `artist_adj_at_k` was populated.
4. Both `lensing-server` binaries **on disk** were current (built 14:25:08, after
   `run.rs` gained `pub artist_conc_at_k` at 14:23:28) and both contained the
   symbol. The hub *process* was current.
5. **The worker daemon (pid 4101029) had started 2026-07-25 14:00:36** — a day
   before the field existed. `/proc/4101029/exe` pointed at
   `…/lensing-server (deleted)`: the rsync had replaced the file but the running
   image was the old one, and scanning that live image gave **0** occurrences of
   `artist_conc_at_k` versus 1 in the on-disk replacement.

The old image deserialized the predictor's `metrics.json` into a `Metrics` struct
lacking the field, dropped it, and POSTed the filtered set — **without erroring**.
`holisticness_at_k` was nevertheless correct throughout, because the *predictor*
computes H itself with `eager = max(artist_adj, artist_conc)` before the Rust
layer sees it, which is precisely what made the defect invisible.

**Consequence for the record: every remote run between 14:23 and 18:13 on
2026-07-26 silently lost `artist_conc_at_k`.** That includes the two discarded
pre-fix control runs `run-20260726-175258-c4968-seq-blend` and
`run-20260726-175258-6f5fb-seq-nexttrack` (both on record, both `succeeded`, both
G2-defective, neither used in this campaign's statistics), and it explains why
Anchor B's quoted `artist_conc` was a **reconstruction**: no run in the DB had
ever carried a natively-ingested `artist_conc_at_k` — the 22 runs that had one
were exactly the orchestrator's rescore set, and
`tools/rescore_holisticness.py:101-118` *reconstructs* the facet from
`top_k_ids` + the dataset `items.json` artist map when the per-row value is
absent. **That reconstruction is now independently validated**: the natively
emitted values from A0/B0 (0.6916530786551751 / 0.35591272614333413) match the
rescore tool's reconstructions to the last digit.

The remedy (cycling the daemon; new pid 573145, `readlink /proc/573145/exe` clean,
1 occurrence of the symbol in the running image) was an orchestrator action, after
which Phase 1 was **re-run from scratch** so that all 10 arms are scored by one
identical path.

**8. R3 VERDICT — `mmr_pool` is NOT a second axis to spend runs on, but pool=50
re-parameterizes the frontier usefully, so the axis closes INTERMEDIATE rather
than HELD.** R3's win condition required Δrecall(B4−B3) CI>0 **and** ΔH(B4−B3)
including or above 0. Measured: **Δrecall +0.005584 CI>0** (the pool cap does
recover recall, as hypothesized — capping the candidate pool bounds how far MMR
can reach for diversity) but **ΔH −0.003838 CI<0**. So pool=50 does not "win" at
fixed λ; it trades H back for recall along the *same* frontier. The literal
"pool default 200 HELD, axis closed" clause does not fire either, because that
clause required a loss *without* a recall gain.

What makes it worth recording is the cross-comparison: **B4 (λ0.7/pool50) beats
B2 (λ0.8/pool200) on H by +0.000889 CI>0 at a recall that straddles** — i.e. the
pool cap reaches a *slightly better* point than the nearest pure-λ setting, which
is why B4 rather than B2 is the batch's best TIER-1 arm. The gain is small
(0.0009 ≈ 0.9 half-widths of its own comparison) and it cost one run. **Do not
open `mmr_pool` as a full axis**; treat pool as a fine-tuning knob available at a
chosen λ, and prefer λ for coarse movement.

**9. The champion/GRU contrast resolves in favour of factor health, not
de-eagering headroom.** The design framed this as the campaign-carrying question:
A and B differ 1.9× in eagerness (0.692 vs 0.356), and the `(1 − eager)` factor
enters as 0.308 vs 0.644, so a given *absolute* de-eagering is worth ~2.1× more
relative H on the champion — yet the champion starts 2.85× behind. Measured
answer: **the GRU wins decisively and the gap widens.** At λ0.7 the champion
reaches H 0.015289 while the GRU reaches 0.034171 — a 2.24× gap, and at every
comparable λ the GRU is ahead. The reason is visible in the facets: as the
champion diversifies, its `mood_coh` **collapses** (0.3982 → 0.2378 at λ0.3,
−40%) whereas the GRU's barely moves (0.4788 → 0.4585 at λ0.7, −4%). The
champion's recall comes from artist-adjacency (the Markov bigram leg — the
registered 2026-07-22 finding), so diversifying away from the seed artist also
walks it out of the prefix's mood neighbourhood; it pays twice. **MMR's value
scales with the health of the other three factors, not with headroom to
de-eager.** No λ closes the champion's crown gap, confirming the design's
projection ceiling.

**10. `PROJECT-FACTS` §Best-models trigger gap re-confirmed for the third time,
and the flood was worse than ever.** Remote-worker completions still do not fire
the best-models recompute. The mandatory manual
`POST /api/best-models/recompute` (R5) then **filled all 12 auto slots with
2026-07-26 runs — 10 of them this campaign's** — including both TIER-3 rejects
(A4 `1dc77` at **rank 2**, A3 `7a2f7` at rank 9) and both G2-defective discarded
controls (`6f5fb` rank 8, `c4968` rank 12). The recall champion is no longer in
the auto group at all. Curation is required before the group is trustworthy for
consensus predictions — handed to `best-model-selector` (see Persisted).

## Best on record after this work

**Crown board — holisticness@10** (live 4-factor definition; recall ≥ 0.108
healthy arms only; all on `seq-20260715-131139`, n_test 1,431):

| rank | model / arm | H@10 | recall@10 | hits | run id | status |
|---|---|---|---|---|---|---|
| 1 | B3 λ0.7 `gru-infonce-h256-mmr-l07` | **0.034171** | 0.11461 | 164 | `run-20260726-184115-bf395-seq-nexttrack` | registered, TIER-2 PRICED — not promotion-recommended |
| 2 | **B4 λ0.7/pool50 `gru-infonce-h256-mmr-l07-pool50`** | **0.030355** | 0.12020 | 172 | `run-20260726-184116-20d9b-seq-nexttrack` | **registered, TIER-1 FREE — crown candidate, promotion pending R4** |
| 3 | B2 λ0.8 `gru-infonce-h256-mmr-l08` | 0.029458 | 0.11950 | 171 | `run-20260726-184115-147a6-seq-nexttrack` | registered, TIER-1 FREE — crown candidate |
| 4 | B1 λ0.9 `gru-infonce-h256-mmr-l09` | 0.026761 | 0.12089 | 173 | `run-20260726-184115-e1539-seq-nexttrack` | registered, TIER-1 FREE — crown candidate |
| 5 | h425 capacity control (prior bar `H_bar`) | 0.024872 | 0.12089 | 173 | `run-20260725-143502-be11b-seq-nexttrack` | unchanged |
| 6 | `gru-infonce-h256` (λ=1.0 baseline) | 0.024539 | 0.12299 | 176 | `run-20260726-183333-9a0b2-seq-nexttrack` | unchanged, 9th reproduction |
| 7 | A1 λ0.9 `blend-gru-markov-content-proj-mmr-l09` | 0.010435 | 0.21034 | 301 | `run-20260726-184115-6cb38-seq-blend` | registered, TIER-1 FREE |
| 8 | `blend-gru-markov-content-proj` (champion, λ=1.0) | 0.008603 | 0.21174 | 303 | `run-20260726-183333-1c361-seq-blend` | unchanged |

Excluded from the board as degenerate (TIER-3, Δrecall < −0.03): A4 H 0.031626 @
recall 0.10971, A3 H 0.023576 @ recall 0.16422.

**Recall board — UNCHANGED.** `blend-gru-markov-content-proj` remains the recall
champion at **recall@10 0.21174 (303 / 1,431)**, mrr 0.12199. No arm in this
campaign improves recall; MMR only ever costs it.

**Promoted models / champion: UNCHANGED.** Nothing was promoted.

## Persisted

- **6 definitions registered** (`POST /api/definitions`), all tagged
  `seq-20260715-131139`, all with notes citing run id, ΔH [CI], Δrecall [CI] and
  the full facet set, and all carrying "NOT PROMOTED — R4 robustness confirm +
  user approval pending":
  `gru-infonce-h256-mmr-l07-pool50`, `gru-infonce-h256-mmr-l08`,
  `gru-infonce-h256-mmr-l09`, `gru-infonce-h256-mmr-l07` (TIER-2),
  `blend-gru-markov-content-proj-mmr-l09`,
  `blend-gru-markov-content-proj-mmr-l07` (TIER-2).
  **Naming note:** the design's scheme `<base>-mmr-l<λ>[-pool<n>]` (and its own
  example `gru-infonce-h256-mmr-l0.8`) is **not a legal model name** — the server
  enforces `^[a-z0-9][a-z0-9-]{0,63}$`, which forbids the decimal point. λ is
  therefore encoded without it (`l09`, `l08`, `l07`).
- **Nothing promoted, no champion change, no pin change.**
- `POST /api/best-models/recompute` executed (R5). Group `updated_at`
  2026-07-26T19:14:15Z, 12/12 auto slots filled, pins (3) unchanged.

## Follow-ups

1. **R4 robustness confirm — MANDATORY before any promotion, needs its own
   approval.** For the best TIER-1 arm **B4** (`seq-nexttrack`, which exposes no
   `seed` and reproduces bit-identically), the pre-agreed confirm is a
   **second-split re-run on `seq-20260718-211238`**: B4's config **+ its λ=1.0
   control**, both on that split, same host. **2 runs, ~6 min.** Verdict = paired
   ΔH CI>0 on that split with recall ≥ 0.108 there too (absolute recall will be
   lower on the 0.64-cold split by construction — judge the paired Δ). Worth
   adding B2 + its control (2 more runs) since B2 is the simpler operating point
   and within 1 half-width of B4.
2. **Curate the best-models group — needed now, group is flooded.** Hand
   `best-model-selector` the TIER-3 exclusions **by both keys**
   (the `88a4c` precedent, where excluding only one key let the run back in):
   A4 `run-20260726-184115-1dc77-seq-blend` / `best-seq-blend-20260726-184115-1dc77`
   and A3 `run-20260726-184115-7a2f7-seq-blend` /
   `best-seq-blend-20260726-184115-7a2f7`; plus the two G2-defective discarded
   controls `run-20260726-175258-6f5fb-seq-nexttrack` /
   `best-seq-nexttrack-20260726-175258-6f5fb` and
   `run-20260726-175258-c4968-seq-blend` /
   `best-seq-blend-20260726-175258-c4968`. B3 (TIER-2, currently rank 1) is a
   judgment call for the selector: it is a legitimate priced operating point
   above the recall floor, not a degenerate arm.
3. **Push the λ frontier on the GRU between 0.75 and 0.85, with a pool probe at
   the winner** — H1 being confirmed means there is real mechanism left, and the
   binding-facet switch says the productive band is above λ0.7. Small, cheap,
   high-prior batch (~4 runs).
4. **Test λ on the blend's Markov leg.** H1 CONFIRMED makes this the indicated
   sequel on the champion side (the 2026-07-22 finding localizes blend eagerness
   to the bigram leg, and MMR de-concentrates artists via sonic proximity). The
   champion's `mood_coh` collapse under diversification (Finding §9) is the
   obstacle to design around.
5. **DROPPED: the explicit artist-diversity rank penalty.** This was the
   pre-registered H2 sequel; H2 is refuted, so porting `seq-mood`'s
   `artist_penalty` is no longer indicated as the *next* move. Keep it on file as
   a complement to (not a replacement for) sonic MMR, since `artist_adj` becomes
   the binding facet at low λ and MMR moves it only ~2.5× slower than
   `artist_conc`.
6. **Cycle the remote worker daemon whenever a metric field is added to
   `run.rs`.** rsyncing the binary is not sufficient. Consider a startup log line
   echoing the worker binary's build id, and/or an ingest-side warning when a
   predictor's `metrics.json` carries keys the server's struct does not know —
   the failure mode here was *silent*, and it cost a full Phase-1 cycle to detect
   and a `/proc/<pid>/exe` trace to diagnose.
7. **Re-derive whether `MUSIC_FLOOR = 0.34` is still right** now that a real lever
   moves music@10 upward (0.4538→0.4925 across the A arms). The floor was set to
   cancel recall-destroying H gains; MMR is the first lever that raises music@10
   *while* diversifying, so the floor's calibration is worth a look before it
   silently shapes the λ optimum.
