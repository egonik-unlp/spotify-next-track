# Does anything separate this family on AUTOREGRESSIVE ROLLOUTS? A generation metric (`hold@N`), and five sweeps that say no — 2026-07-27

**Date:** 2026-07-27
**Type:** MEASUREMENT + **REFUTED**. Nothing promoted, nothing registered, champion
/ leaderboard / best-models pins UNCHANGED. **The null is the finding**, and the
sequence of confounds that produced three retracted positives is itself the
deliverable.
**Axis under test:** the regime the product runs in but no metric measures —
**autoregressive rollout**. Every metric on record (`recall@10`,
`holisticness@10` and its four facets, `music@10`, and even the multi-step
`seq_continuation_eval.py`) scores ONE ranked list produced from a REAL prefix.
None feeds a model its own output. A 20-step lab journey does exactly that, so a
trajectory failure — a rollout whose energy decays until a party seed has become a
comedown — is structurally invisible to all of them.
**Origin:** a USER PREFERENCE, not a metric. On playlist `cenamos` (135 tracks,
133 in vocab) the user strongly preferred `seq-dualgru`'s journey over
`seq-nexttrack`'s. `holisticness@10` on those two journeys differed by 0.003.
**Host:** all rollouts local (`POST /api/models/{name}/extend`, 2 run slots) —
no remote worker, no training.
**Seed:** 1337 throughout. Rollout policy held fixed at the lab defaults:
`steps 20, artist_penalty 2, anchor_lambda 1, temperature 0` (deterministic, so
one seed = one sample, no rollout variance), `policy_pool 200`.

## Outcome (one line)

**Rollout fade does NOT separate this family.** At matched training split the two
architectures TIE twice (Δ hold@20 +0.033 [−0.030,+0.093] and +0.051
[−0.012,+0.118]); on the user's OWN 11 playlists the pair they preferred TIES
(+0.004 [−0.088,+0.104]); the only surviving effect is CHECKPOINT-level
(`a1d9d` vs `C0`, same architecture, −0.072 [−0.141,−0.002]) and it does not
transfer to the playlist regime — while per-seed spread on a single model pair
runs ±0.3, i.e. **which seed you hand it matters far more than which of these
models generates**, so any 8–12-seed comparison here is underpowered by
construction.

## What was built

1. **`hold@N`** (`tools/rollout_hold.py`) — a generation metric:
   `fade_a = |mean(2nd half) − mean(1st half)| / band_a` per legible axis
   (energy / valence / acousticness / tempo), `stasis = max(0, jump_p10 − mean
   step jump) / jump_p10`, `hold@N = 1 − clamp(mean_a(fade_a) + stasis, 0, 1)`.
   `band_a` is calibrated on **2,135 real TRAIN sessions** (energy fade p10 0.0141
   / p50 0.0769 / p90 0.1963; valence .0165/.0922/.2341; acousticness
   .0146/.0924/.2665; tempo 1.28/8.05/22.43), so axes with incompatible units
   combine and "a big fade" means big relative to how this listener actually
   listens. Real sessions are NOT flat — median energy fade 0.077 — so a metric
   rewarding zero fade would rank an unnaturally rigid generator above its own
   listener.
2. **`stasis`, the anti-gaming term.** A pure fade metric is degenerate: 20
   near-identical tracks fade by 0 and score 1.0 — precisely the `cummean/cummean`
   / `seq-mood` pathology that `holisticness` needs an EXTERNAL ±0.015 recall
   floor to survive. `stasis` puts the floor INSIDE the metric. Verified on a
   synthetic flatlined rollout: moves-naturally 1.000, at-your-floor 1.000,
   half-floor 0.500, barely-moves **0.048**.
3. **`predictors/seq_continuation_eval.py --rollout`** — offline rollout eval over
   sampled TEST sessions. It IMPORTS the generation loop from `seq_extend`
   (factored out as `seq_extend.rollout()`), never reimplements it, so an offline
   score describes the journeys the lab actually serves. Refactor verified
   BYTE-IDENTICAL (sha of stops+diagnostics+mood_report unchanged, temperature 0).
   On-demand and sampled, like `continuation_eval`, because a rollout costs
   `steps` full-vocab passes per session against the primary path's ONE (~20×).
4. **Playlist-lab readout** (same session): mood report card + comparative
   journey-shape chart + per-step `passed_over`, replacing the eigen/PR readout.
   See the client diff; `holisticness` facets demoted to a `<details>`.

## Results — five sweeps, 120 paired rollouts, 240 journeys

Paired bootstrap over SEEDS (2,000 resamples, rng 1337, 95% percentile CI) plus a
**leave-one-seed-out jackknife**, which did most of the real work here.

| # | sweep | pair | n | Δ hold@20 [95% CI] | LOO | verdict |
|---|---|---|--:|---|---|---|
| S1 | pilot, mood-EXTREME seeds, 10-track prefix (3/8 unanchored) | dc7d2 vs a1d9d | 8 | −0.034 [−0.138,+0.079] | 0/8 | TIE |
| S2 | pilot, mood-EXTREME seeds, 30-track prefix | dc7d2 vs a1d9d | 8 | **+0.314 [+0.107,+0.529]** | 8/8 | (retracted, see S3) |
| S3 | **replication**, RANDOM seeds, 30-track prefix | dc7d2 vs a1d9d | 39 | **+0.122 [+0.044,+0.201]** | 39/39 | (confounded, see S4) |
| S4 | **same-split control** | 18b4a vs 97c1e | 39 | +0.033 [−0.030,+0.093] | 0/39 | **TIE** |
| S4b | same-split, other dualgru | dc7d2 vs 97c1e | 39 | +0.051 [−0.012,+0.118] | 0/39 | **TIE** |
| S4c | **within-architecture** | a1d9d vs 97c1e | 39 | −0.072 [−0.141,−0.002] | — | significant |
| S5 | **the user's own playlists** | dc7d2 vs a1d9d | 11 | +0.004 [−0.088,+0.104] | 0/11 | **TIE** |
| — | `cenamos` alone (the judged seed) | dc7d2 vs a1d9d | 1 | +0.379 (0.623 vs 0.245) | — | outlier |

Mean hold@20, all four models on the IDENTICAL 39 random seeds: dualgru `dc7d2`
0.6488, dualgru `18b4a` 0.6309, nexttrack `C0 97c1e` 0.5983, nexttrack `a1d9d`
0.5266.

**S3 decomposes into a non-significant architecture component (+0.05) and a
significant checkpoint component (+0.072).** Two `seq-nexttrack` models differ
from each other MORE than dual-tower differs from single-GRU at matched split.
`a1d9d` — which the playlist lab preselects into slot B by default — is simply an
unusually poor holder, so the user's original comparison was the best holder of
four against the worst.

### Per-axis, S3 (n=39): only ENERGY, and a FAILED negative control

| axis | dualgru | nexttrack | Δ [95% CI] | |
|---|--:|--:|---|---|
| energy fade | 0.070 | 0.110 | −0.039 [−0.069,−0.009] | dualgru |
| valence fade | 0.082 | 0.105 | −0.022 [−0.048,+0.003] | TIE |
| acousticness fade | 0.094 | 0.125 | −0.031 [−0.073,+0.014] | TIE |
| tempo fade | 5.66 | 6.71 | −1.04 [−3.13,+1.02] | TIE |
| **energy LEVEL** \|mean−seed\| | 0.081 | **0.063** | +0.018 [+0.005,+0.032] | **nexttrack — CONTROL FAILED** |
| valence / acousticness / tempo level, drift | — | — | straddle 0 | TIE (as designed) |

The pre-registered negative controls were the LEVEL deviations and drift. One
moved: nexttrack sits CLOSER to the seed's energy level while fading more. Those
are different properties and the claim had to be re-scoped from "holds the seed
better" to "is more stationary" — which also shows that `hold@N`'s deliberate
exclusion of the level term is a choice that CHANGES the ranking, not a neutral
simplification. At n=8 the acousticness and valence axes had looked significant;
they were noise, and the jackknife said so at the time (4/8 and 4/8).

## Decision rule — applied mechanically

Pre-registered for S3 before results were seen: **CONFIRMED** iff CI entirely > 0
AND the verdict survives ≥90% of LOO refits; **TIE** iff the CI straddles;
**REFUTED** iff CI < 0. S3 met it (39/39). **S4 then invalidated the attribution**
— the rule was sound, the COMPARISON was confounded, and no decision rule
protects against comparing two models that differ in more than one way.

## Findings

1. **Rollout fade does not separate this family.** Combined with the FIVE prior
   architecture nulls this is a sixth, obtained with a purpose-built instrument in
   the deployment regime — the strongest form of the null available, since the
   standing rebuttal to those nulls was "recall@10 can't see session quality".
   It can't; and when something else looks, there is still nothing there.
2. **Per-seed variance dominates model identity.** Individual gaps on ONE model
   pair span +0.39 (`chill&chill`) to −0.27 (`UKG`). Model-level mean difference
   on the user's playlists: +0.004. Consequence for the product: prefer serving an
   ENSEMBLE and letting the listener choose (`POST /api/best-models/predict`) over
   hunting a single better generator; and budget ≥40 seeds for any future rollout
   comparison, because 8–12 cannot resolve anything at this variance.
3. **`hold@N` is a working instrument, just not a model ranker.** It fired on a
   real checkpoint difference (S4c) and on `cenamos` (0.62 vs 0.25) that
   `holisticness@10` scores as a 0.003 gap, and it correctly returned TIE three
   times. Recommendation: keep it as a per-journey LAB readout; do NOT wire it as
   a leaderboard column. Ground truth for it is listener preference and that
   currently stands at **n=1 blind vote**. Promoting an unvalidated composite to
   primary is how the crown ended up needing an external recall floor; replacing
   one unvalidated composite with another repeats the mistake.
4. **`PROJECT-FACTS`' "best open direction" is STALE — compose-the-representations
   is CLOSED.** That line says to fit the InfoNCE projection on a whitened /
   `std-noaco` / metric space. `2026-07-18c` readout A **already re-fits** the
   projection per space (its §Design says so explicitly): V3 `std-noaco` scored
   +0.0049, straddling 0, and the report concludes **"whitening and the learned
   projection are the SAME fix."** Do not re-run it.
5. **The evaluation regime, not the topology, is the untested axis.** All five
   architecture nulls held the objective/batching/early-stopping "byte-for-byte
   the family's" — one-step teacher forcing. The product rolls 20 steps on its own
   output. That train/serve mismatch (exposure bias) is the one thing those nulls
   could not have detected. Scheduled sampling on the single GRU at
   `fusion_layers=0` is the indicated next architecture-adjacent test, and it is
   now measurable via `--rollout`.
6. **Acoustic coverage is a real sampling constraint.** 74% of the corpus carries
   `af_*`; unfiltered test-session sampling drops ~25% of rollouts as unscorable
   (the sweeps above pre-filtered seeds for ≥60% prefix coverage, which is a mild
   selection effect worth stating).

## Pitfalls surfaced / re-confirmed

- **NEW — a cross-split comparison masquerading as an architecture result.** The
  lab's two default preselects are trained on DIFFERENT splits (`dc7d2` on
  `seq-20260715-131139`, `a1d9d` on `seq-20260715-030509`; identical vocab and
  item latents, different train/test assignment). Comparing them attributes
  checkpoint+split differences to architecture. **Always pair models from the same
  batch/split when the claim is architectural** — the campaign's own C0/D-LL-f0
  pair existed the whole time and additionally TIES on recall, so neither is the
  better predictor.
- **NEW — mood-extreme seed selection inflates effects ~2.5×.** S2's +0.314
  became S3's +0.122 on a random sample. Pick seeds at random, or state the
  selection.
- **NEW — an anti-gaming term can be too weak to do its job.** `stasis` first
  normalised the shortfall by the band WIDTH; because this listener's jump band is
  wide (0.21–0.98), a flatlined rollout scored **0.740 — ABOVE a real generator's
  0.477**. Normalise by the FLOOR. A guard beatable by the degeneracy it targets
  launders that degeneracy as a good score.
- **NEW — a metric that returns `None` must drop the PAIR and report the smaller
  n.** First version crashed on a zero-coverage rollout; silently dropping one arm
  would have broken the pairing.
- **NEW — `grep -v` on a diagnostic channel hides the diagnosis.** A profiling
  pass returned "0 usable playlists" because it ran under an interpreter without
  `qdrant_client`, so `load_mood_table` returned all-NaN — and the warning saying
  exactly that was filtered out by the invocation's own `grep -v '"kind"'`.
- **RE-CONFIRMED — the jackknife earns its keep at n<40.** It killed two of three
  per-axis positives at n=8 and flagged S4/S5 as unsupported at 0/39 and 0/11.
- Infra: `pkill -f "<pattern>"` matches the invoking shell's own command line and
  kills it before the intended target — use PIDs. Backgrounding a job inside a
  harness-tracked shell reports completion for the WRAPPER, not the job.

## Best on record after this work (RANKING board — UNCHANGED)

Nothing promoted, nothing registered, no metric wired. Champion stays
`blend-gru-markov-content-proj` (recall@10 0.21174). Crown board unchanged. The
live-crown standings relevant to the dual-tower question, for the record:
best legitimate dualgru arm **0.02378** vs single-GRU C0 **0.02454** — a 0.0008
gap against `domain.toml`'s stated ~0.0018 crown CI half-width, i.e. **a TIE, not
a loss**, and `domain.toml` itself says the family's whole H spread (0.003 = 1.8
half-widths) makes 7 of 8 adjacent arms unresolvable. `seq-dualgru` was refuted on
`recall@10`, which `domain.toml` now records as "no longer load-bearing"; it ties
on the live primary; and it shows nothing on the rollout axis. There is no
measured reason to prefer it, on any metric including one built for the purpose.

## Follow-ups

1. **Blind preference votes** — the binding constraint on everything above. The
   lab now records them (`localStorage lab.votes.v1`, shuffled sides, wins over
   appearances). Until n is enough to discriminate, `hold@N`'s equal axis weights
   are a placeholder and its validity is unestablished.
2. **Scheduled sampling** on the single GRU (`fusion_layers=0`), scored with
   `--rollout`. Finding 5.
3. **Change the lab's slot-B preselect** off `a1d9d` — it is the worst holder of
   the four models measured (0.5266 vs C0's 0.5983) and it is what every default
   lab comparison starts from.
4. Attention encoder for the LONG-prefix regime only, and only after (2) — a GRU's
   fixed hidden state is a plausible bottleneck at 133-item playlist prefixes and
   is not covered by the five nulls, but recall@10 on short session prefixes
   cannot see it either way.
5. `PROJECT-FACTS` line 4 above needs correcting where it names
   compose-the-representations as open.
