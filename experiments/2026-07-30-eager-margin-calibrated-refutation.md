# 2026-07-30 — the eagerness regularizer at a calibrated margin: REFUTED, with a mechanism

**Predictor:** `seq-dualgru` · **Dataset:** `seq-20260715-131139` (canonical PCA-192) · **8 runs, all remote**
(`claimed_by=snappler:3207287`), 60 min compute / ~68 min wall, 0 failed / 0 interrupted / 0 relaunches.
**NOTHING PROMOTED / NOTHING REGISTERED / NOTHING TAGGED.** No champion moved.

## Outcome in one line

**REFUTED (whole axis), with a mechanism.** No arm reached WIN. Every margin that de-eagers (`m=0`)
destroys relevance well before it helps, and every margin calibrated from the consecutive-transition
tail (`m ≥ 0.5843`) is **provably inert — bit-identical loss traces at β up to the schema cap of 100** —
because `cos(pred, x_current)` never reaches the quantiles the calibration was drawn from. `eager_beta`
is not a weak knob; it is **mis-specified and mis-scaled**.

## Goal

`eager_beta` was on record as "a weak knob — not recommended"
(`2026-07-22-session-holisticness-and-antieager-levers.md:215`), but that probe used `eager_margin=0`
at a dose worth ~1–3% of the loss. This campaign asked whether the lever works when placed in the
measured right tail of the transition-cosine distribution at 15–60× that dose. It is the only
**training-time** de-eagering lever in the repo — unlike `mmr_lambda`, which is eval-only and never
reaches the serving path — so it was the one candidate that could actually ship.

Pre-registered primary: paired **Δ`artist_conc@10`** vs an in-batch control (H@10 demoted to secondary
because ∂H/∂music@10 = +0.2062 vs ∂H/∂eager = −0.0378, a 5.46× ratio, would have scored a successful
de-eagering as ≈zero). Pre-registered relevance floor: absolute **music@10 ≥ 0.4300** plus paired
Δmusic@10 CI lower bound > −0.030.

## Gates

| check | result |
|---|---|
| worker `seq_dualgru.py` / `seq_nexttrack.py` md5 | `a389b5f0…` / `5e2ec283…` — both == hub |
| worker binary md5 | `3b7c3172a85de4c8f502b3425d8b3bbb` == hub |
| worker → hub PG / Qdrant / API | 5437 OPEN, 6337 → 200, 8096 → 200 (tailscale `100.107.86.38`) |
| **G1 bit-identity (C0)** | **PASS — exactly 170 hits**, recall `0.1187980433263452`, reconstructed conc `0.370587778554236` (== 18b4a to 15 dp) |
| **G2 relevance-floor availability** | **PASS — 1431/1431 non-null `music_sim` on all 8 arms** |
| **G3 penalty-active** | **PASS: A1, A2, B1. FAIL (inert): B2, B3, B4, B5** |
| best-epoch diagnostic | all arms best-epoch > 3 — none disqualified |

Instrument validated: reconstructed `artist_conc` reproduced C0's stored aggregate exactly (diff 0.00e+00);
recomputed H matched stored H to 1.7e-07.

## Results

| arm | run id | m | β | hits | recall@10 | artist_conc@10 | artist_adj@10 | ild@10 | mood_coh@10 | music@10 | H@10 | best ep/ran | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C0** | `run-20260730-175402-cd006-seq-dualgru` | 0 | 0 | **170** | **0.118798** | 0.370588 | 0.362963 | 0.450610 | **0.479846** | **0.455347** | **0.023785** | 8/14 | CONTROL (G1 pass) |
| A1 | `run-20260730-181405-53088-seq-dualgru` | 0 | 3 | 77 | 0.053809 | 0.240717 | 0.206010 | 0.557169 | 0.346867 | 0.399997 | 0.013339 | 18/24 | **DEGENERATE** |
| A2 | `run-20260730-181405-0bc1a-seq-dualgru` | 0 | 12 | 20 | 0.013976 | **0.133535** | **0.107687** | **0.676164** | 0.221756 | 0.358132 | 0.003569 | 37/40 | **DEGENERATE** |
| B1 | `run-20260730-181405-84da4-seq-dualgru` | 0.40 | 9 | 164 | 0.114605 | 0.367606 | 0.364151 | 0.453001 | 0.463440 | 0.447386 | 0.021601 | 15/21 | no effect |
| B2 | `run-20260730-175402-7c304-seq-dualgru` | 0.5843 | 19 | 170 | 0.118798 | 0.370634 | 0.363103 | 0.450441 | 0.479955 | 0.455210 | 0.023751 | 8/14 | **INERT** |
| B3 | `run-20260730-184556-b08c1-seq-dualgru` | 0.70 | 35 | 170 | 0.118798 | 0.370588 | 0.362963 | 0.450610 | 0.479846 | 0.455347 | 0.023785 | 8/14 | **INERT (bit-identical)** |
| B4 | `run-20260730-175402-0d310-seq-dualgru` | 0.8304 | 92 | 170 | 0.118798 | 0.370588 | 0.362963 | 0.450610 | 0.479846 | 0.455347 | 0.023785 | 8/14 | **INERT (bit-identical)** |
| B5 | `run-20260730-184556-47d7b-seq-dualgru` | 0.5843 | 57 | 170 | 0.118798 | 0.370743 | 0.363103 | 0.450441 | 0.479955 | 0.455379 | 0.023771 | 8/14 | **INERT** |

### Paired statistics vs in-batch C0 (2000 resamples, rng 1337, n=1431)

| arm | Δartist_conc [95% CI] | hw | Δmusic@10 [95% CI] | Δartist_adj CI | ΔH | n_disc | mean\|Δconc\| |
|---|---|---|---|---|---|---|---|
| A1 | **−0.12987** [−0.14354, −0.11605] | 0.0138 | −0.05535 [−0.06606, −0.04465] | [−0.16995, −0.14472] | −0.010445 | 125 | 0.191304 |
| A2 | **−0.23705** [−0.25493, −0.22026] | 0.0173 | −0.09721 [−0.11107, −0.08333] | [−0.27135, −0.23871] | −0.020215 | 166 | 0.282895 |
| B1 | −0.00298 [−0.01085, **+0.00519**] | 0.0080 | −0.00796 [−0.01411, −0.00194] | [−0.00531, +0.00769] | −0.002183 | 62 | 0.093734 |
| B2 | +0.00005 [−0.00028, +0.00039] | 0.0003 | −0.00014 [−0.00037, +0.00000] | [+0.00000, +0.00035] | −0.000034 | **0** | 0.000419 |
| B3 | +0.00000 [+0.00000, +0.00000] | 0.0000 | +0.00000 [+0.00000, +0.00000] | [0, 0] | +0.000000 | **0** | **0.000000** |
| B4 | +0.00000 [+0.00000, +0.00000] | 0.0000 | +0.00000 [+0.00000, +0.00000] | [0, 0] | +0.000000 | **0** | **0.000000** |
| B5 | +0.00016 [−0.00031, +0.00068] | 0.0005 | +0.00003 [−0.00036, +0.00045] | [−0.00021, +0.00049] | −0.000014 | **0** | 0.000901 |

Measured hw(Δconc) on the *active* arms (0.0080–0.0173) matches the design's predicted 0.0102 — the
instrument was correctly powered. The axis still failed.

## Findings

**1. The calibration measured the wrong distribution — this is the campaign's central result.** The
margins were drawn from `cos(x_t, x_{t+1})` (item→item: median 0.262 / p75 0.584 / p90 0.830), but the
penalty acts on `cos(pred_t, x_t)` (**prediction**→current). Those distributions are not comparable.
Reconstructing the hinge from epoch-1 loss inflation at shared init (identical seed ⇒ identical init and
batch order):

| m | β | epoch-1 loss inflation | ⇒ mean `relu(cos−m)` at init | trace ≡ C0? |
|---|---|---|---|---|
| 0 | 3 | +0.406435 | **0.135478** | no |
| 0 | 12 | +0.700075 | 0.058340 (depressed — suppressed within epoch 1) | no |
| 0.40 | 9 | +0.008259 | **0.000918** | no |
| 0.5843 | 19 | +0.000000 | **0.000000** | no (diverges from ep 3, ~1e-6) |
| 0.70 | 35 | +0.000000 | **0.000000** | **YES, all 14 epochs** |
| 0.8304 | 92 | +0.000000 | **0.000000** | **YES, all 14 epochs** |

B3 and B4 reproduced C0's loss to **16 significant digits at every one of 14 epochs**. The ceiling of
`cos(pred, x_current)` over training sits **between 0.5843 and 0.70**, so p75 and p90 of the
real-transition distribution lie *outside the support of the penalized quantity*. The design's hedge —
that the realized penalty would *exceed* nominal P — is empirically inverted by orders of magnitude.

Reason: an InfoNCE-trained prediction is a **hedged near-average over many plausible continuations**, so
its absolute cosine to any single item is bounded low. Eagerness is a **ranking / relative** property
(the current track's neighbours win the top-10), not an absolute-cosine property. **A hinge on absolute
cosine cannot express the album-eager failure mode.**

**2. Where the knob does fire, eagerness and relevance are the same mechanism.** At `m=0` the lever works
spectacularly on the facet — A2 cuts conc 0.371 → 0.134 (Δ −0.237) and drives ild to 0.676 — while
music@10 collapses to 0.358 and recall to 20/1431 hits. There is no separating window: pushing the
prediction off the current track's latent also pushes it off the region where the true next track lives
(27% of real transitions are negative-cosine, mean 0.289), so mood_coh collapses too (0.480 → 0.222).
A1 at β=3 already breaches the floor (music@10 0.400 < 0.4300). The pre-registered failure mode fired
exactly as written.

**3. `fusion_layers=1` remains strictly better than anything this axis offers.** The free on-record
comparator `run-20260726-161801-0df6f` gives Δconc −0.051 at a −0.016 recall cost. No arm here matched
that inside the relevance floor: floor-respecting arms delivered Δconc between −0.003 and +0.0002 (all
CIs straddling or above 0), and the arms beating −0.051 paid 4–8× the recall and breached the floor.

**4. The 2026-07-22 prior is sharpened, not retired.** That probe (β 0.1/0.2, m=0) reported "weak" —
correct in band, wrong in interpretation. β ≤ 0.2 at m=0 sits just *under* the destruction threshold,
and by β=3 the same setting is a relevance destroyer. The entire usable dynamic range at m=0 lies
between β=0.2 and β=3 — a factor of 15, unexplored, bounded above by degeneration and below by "no
effect". The 15–60× dose ladder was **not underpowered; it was aimed past the target.**

**5. H@10's demotion to secondary was load-bearing.** A2 has by far the best facets (conc 0.134, adj
0.108, ild 0.676) and the *worst* H (0.003569) — the music@10 grounding factor annihilates it, exactly
as the 5.46× derivative ratio predicted. Had H been primary, this campaign would have read as a flat
null instead of a mapped frontier.

## Why Phases B and C and the seed-confirm did not run

All three are gated on a Phase A **WIN** in the approved design. There is no WIN. Phase C's arms are
defined *by reference* ("N1 = Phase A winner's margin/β"), so with no winner they are **undefined**.
Applied literally to the arms passing the music floor + best-epoch filter, §10's selection rule would
have yielded N1 = B1 (Δconc −0.003) and N2 = B3/B4 (Δconc exactly 0.000) — i.e. it would have spent 3
runs transferring a **provably inert** setting to `seq-nexttrack`. The runner stopped rather than
improvise; the correct next move is a re-aimed design, not the approved Phase C.

## Best-on-record after this work

| rank | model | run | H@10 | recall@10 | artist_conc@10 | note |
|---|---|---|---|---|---|---|
| 1 | `seq-nexttrack` h256 | `run-20260726-183333-9a0b2` | 0.024538 | 0.122991 | 0.355913 | crown leader, **unchanged** |
| 2 | `seq-dualgru` f0 | `run-20260725-143502-18b4a` | 0.023785 | 0.118798 | 0.370588 | best dual arm, **unchanged**; reproduced bit-identically as C0 today |
| 3 | `seq-dualgru` f1 | `run-20260726-161801-0df6f` | 0.022841 | 0.102725 | 0.319559 | still the cheapest real de-eagering on record (Δconc −0.051) |

## Follow-ups (ranked)

1. **Re-specify the penalty, don't re-tune it.** The failure is expressive, not dosage. Penalize a
   *ranking* quantity — e.g. a hinge on `cos(pred, x_t) − cos(pred, x_{t+1})`, or an explicit
   same-artist mass term over the top-k — not absolute cosine. Needs predictor code.
2. **The one unexplored band on the existing knob: `m=0`, β ∈ [0.2, 3]** (3–4 runs). Would either find
   a narrow safe window or close the axis completely. This is also the only scientifically informative
   Phase C — the 2026-07-22 prior's own margin on its home family `seq-nexttrack`.
3. **Make MMR ship.** Now the strongest remaining de-eagering path *by elimination*:
   `seq_common.predict_ranking` takes no mmr parameter, so `mmr_lambda` enters only via
   `eval_from_scores`. With the training-time lever refuted, the only lever that works is eval-only and
   does not reach the serving path — which also means the best-models group ranks by gains the serving
   path cannot reproduce.
4. **Fix the `spawn_recompute` trigger gap for remote runs** (4th confirmation — `GET /api/best-models`
   still reported `updated_at 2026-07-28T01:39:56` after 8 completed runs). No arm was auto-promoted,
   correctly, since every H ≤ C0's. `POST /api/best-models/recompute` was deliberately NOT run (it can
   auto-promote).
5. **`~/lensing-worker/registry.toml` is stale vs hub** (`b4ef03b6…` vs `33b482cd…`) and lacks
   `seq-embed`'s eager params — harmless here, a live risk for any `seq-embed` remote campaign.

Working artifacts (loss traces, per-row predictions, paired bootstrap output, analysis script) are under
the session scratchpad.
