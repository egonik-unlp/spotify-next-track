# Settling the f0 dropout question — a seed grid, and the walk read on the deployed pair

**4 new runs (2 reused), all remote on `snappler:879965` via `queue:true`, 27.5 min new
compute, 0 failed / 0 interrupted, 2 chunks of 2. Plus 3 walk invocations on the hub
(no training).**

Direct continuation of `2026-08-02b-dropout-on-the-deployed-view-pair.md`, whose two
follow-ups the user authorized together. That report's verdict ("NOT REPLICATED, but
UNDERPOWERED — status OPEN") is **superseded by Phase A below**: the question is no longer
open in the direction that mattered.

NOTHING DEPLOYED. NO CHAMPION REGISTRATION. NO DEFINITION SAVED. Crown metric and
`domain.toml` UNCHANGED. `models.toml` not hand-edited. **No server restart.** Orphan row
`run-20260730-234635-7a297-seq-blend` untouched. Two models promoted **measurement-only**
(`notes` verbatim "promoted to enable walk evaluation; not a crown claim").

## Goal

Two questions, pre-registered before looking:

**Phase A — is the f0 dropout effect real on the deployed `latent/cummean` view pair?**
`2026-08-02b` measured +0.002795 [−0.002795, +0.007704] at seed 1337 — straddling, with a
half-width (0.005250) *larger than the whole D2−R2 effect it was sent to reproduce*
(+0.004892). A seed grid replaces "one split, 16 discordant rows" with a seed-averaged
instrument.

**Phase B — does E1 hold the walk at the deployed operating point?** A recall gain is
worthless if the showcase engine's mood-holding collapses; two tower-bank arms earned a
TRADE at the matched cell and flipped to Δvibe CI<0 at exactly a0.8/s0.5.

**Dataset** `seq-20260715-131139` only (canonical PCA-192 sequence split, test sessions
**1,431**). `seq-20260718-211238` NOT used (total leak against split-1-trained arms). **No
dataset was built** — `walk_headtohead.py::promoted_score` asserts `item_latents.f32`
byte-identity, so a new item space would make every arm unpairable.

## Grid — 6 cells, of which 4 are new

`dropout ∈ {0.0, 0.1} × seed ∈ {1337, 7, 42}` on `seq-bank`, `views=latent+cummean`,
`hidden=256`.

**E0 and E1 from `2026-08-02b` ARE the seed-1337 cells and were REUSED, not re-run.** This
was verified mechanically rather than assumed: their stored hyperparams were diffed against
the grid cell key by key — same predictor, same dataset, `seed: 1337`, **zero differing
keys and zero extra keys** on both. Re-running them would have produced bit-identical
models (the training path is deterministic given seed and init, as G-E0 established by
reproducing the deployed engine's ranked list on all 1,431 rows). So the grid cost 4 runs,
not 6.

**Held constant on all six:** the same 15 declared `seq-bank` params as in `2026-08-02b` —
`views=latent+cummean`, `hidden=256`, `loss=infonce`, `tau=0.07`, `epochs=40`,
`patience=6`, `lr=0.001`, `batch_size=128`, `val_fraction=0.15`, `k=10`, `mmr_lambda=1.0`,
`mmr_pool=200`, `artist_cap=0`. Only `dropout` and `seed` vary.

## Outcome in one line

**The dropout effect on the deployed view pair does NOT survive seeds — the signs disagree
(+4 / −4 / −13 hits at seeds 1337 / 7 / 42) and the seed-averaged estimate is
−0.003028 [−0.006988, +0.000932], i.e. if anything NEGATIVE; crucially the seed-averaged
instrument's half-width (0.003960) is SMALLER than the D2−R2 effect it hunted (0.004892),
so this is a resolution-sufficient non-finding, not another underpowered one — the positive
effect is EXCLUDED. And the by-product is the bigger result: this is the family's FIRST
seed-variance measurement, the seed range at `dropout=0.1` (12 hits / 0.008386) EXCEEDS the
sharpest single-hyperparameter half-width on record (0.005250), and the D2−R2 finding that
launched this whole line was itself SINGLE-SEED with an effect size (7 hits) equal to the
seed range measured here — so it must be re-flagged as unconfirmed. On the walk, E1 holds
vibe at the deployed cell (Δvibe +0.0024, not CI<0) and would qualify as a redeployment
candidate — but Phase A removed the reason to redeploy, and E0 reproduced the deployed
engine to EXACTLY 0.0000 on all five walk axes at both cells, the strongest harness
continuity check this project has run.**

## Gates

| gate | result |
|---|---|
| **P1 — worker md5 parity on the import closure** | **PASS, re-verified before EACH of the two chunk POSTs** (verified, not assumed, as instructed): identical digest over the five files both times — `seq_bank.py` `7c58d53a…`, `seq_common.py` `09e7aef6…`, `seq_nexttrack.py` `1f995226…`, `seq_dualgru.py` `d263c425…`, `registry.toml` `15fd3837…` (combined `ff7ed04de25b1eee32ab56f4d219a561`). All 4 new runs `claimed_by snappler:879965` — asserted programmatically, no hub fallback. |
| **G1 — metric completeness** | **GREEN on all 6 cells** — every arm carries `recall_at_k`, `music_at_k`, `holisticness_at_k`, `artist_conc_at_k`, `mood_coh_at_k`, `ild_at_k`, `mrr`. No missing `music@10` anywhere, so no arm is a silently-suspended run. |
| **Grid integrity** | **PASS** — all six arms share the identical 1,431-row split (row_id sets asserted equal), and each arm's stored `seed`/`dropout` was asserted to equal its intended cell. |
| **G3 — walk instrument** | **PASS.** Δ(promoted GRU h256 `9a0b2` @ a0.4/s0.3 vs shipped `gru.onnx` @ a0.4/s0.3) straddles on both required axes: Δvibe **−0.0028** [−0.0067, +0.0007], Δstride_err **−0.0005** [−0.0055, +0.0043]. Same magnitudes as the tower-bank campaign's G3 with the sign flipped (base/arm order reversed), i.e. reproduced. |
| **Harness reproduction** | **GREEN.** `real median step 0.261` and the default arms' means reproduce the record exactly — GRU shipped a0.4/s0.0 **0.609 / 0.399 / −0.089 / 9.925 / 0.345**, GRU tuned a0.4/s0.3 **0.568 / 0.129 / −0.080 / 10.450 / 0.357**, deployed dual `l/cummean` a0.8/s0.5 **0.610 / 0.144 / −0.053 / 11.425 / 0.351**. |

## Phase A results — the seed grid

Paired bootstrap, 2,000 resamples, rng 1337, n=1,431, from per-row `predictions.json`.

| seed | `dropout` | run id | hits | recall@10 | H@10 | music@10 | a_conc |
|---|---|---|---|---|---|---|---|
| 1337 | 0.0 | `run-20260802-214618-625eb-seq-bank` (reused) | 168 | 0.117400419 | 0.02152039 | 0.445823 | **0.36638** |
| 1337 | 0.1 | `run-20260802-214618-a38be-seq-bank` (reused) | 172 | 0.120195667 | 0.02195565 | 0.446890 | 0.36535 |
| 7 | 0.0 | `run-20260802-221545-b63d0-seq-bank` | 166 | 0.116002795 | 0.02378805 | 0.450159 | 0.34785 |
| 7 | 0.1 | `run-20260802-221545-4ee07-seq-bank` | 162 | 0.113207547 | 0.02344442 | 0.447337 | 0.34580 |
| 42 | 0.0 | `run-20260802-222820-99ff7-seq-bank` | **173** | **0.120894479** | **0.02405385** | **0.454388** | 0.34896 |
| 42 | 0.1 | `run-20260802-222820-1a607-seq-bank` | 160 | 0.111809923 | 0.02357464 | 0.450809 | 0.34136 |

**Within-seed paired Δrecall@10 (dropout 0.1 − dropout 0.0):**

| seed | Δhits | Δrecall@10 | 95% CI | half-width | Δmusic@10 | n_disc | flag |
|---|---|---|---|---|---|---|---|
| 1337 | **+4** | **+0.002795** | [−0.002795, +0.007704] | 0.005250 | +0.001068 [−0.002067, +0.004160] | 16 | straddles |
| 7 | **−4** | **−0.002795** | [−0.008386, +0.002795] | 0.005590 | −0.002823 [−0.006521, +0.000984] | 18 | straddles |
| 42 | **−13** | **−0.009085** | [−0.018868, +0.000699] | 0.009783 | −0.003580 [−0.008739, +0.001342] | 51 | straddles |

**Across-seed:** simple mean of the three per-seed deltas **−0.003028**; sd of those deltas
**0.005943**; range **0.011880**. Seed-averaged per-session instrument (per-session hit rate
averaged over the 3 seeds, then paired): **Δrecall@10 −0.003028 [−0.006988, +0.000932],
half-width 0.003960, straddles**; Δmusic@10 −0.001778 [−0.004157, +0.000476] straddles.

**Floors.** Absolute recall ≥ 0.108: worst arm 0.111810 → PASS. Δ`music@10` not CI<0: every
per-seed and the seed-averaged estimate straddles → PASS. No arm is refuted by a floor.

### Verdict, against the pre-registered rule

- **SETTLED-POSITIVE is EXCLUDED.** The rule required sign consistency across all three
  seeds *and* an across-seed estimate excluding 0. Signs are **+ − −**; the across-seed
  estimate straddles and leans **negative**. Not close.
- **The primary branch is STILL UNRESOLVED, and I am reporting it as such rather than
  upgrading it.** Both of that branch's clauses fire: signs disagree, **and** the across-seed
  spread swamps the effect (sd of the per-seed deltas 0.005943 is ~2× the |across-seed mean|
  0.003028). SETTLED-NULL's second clause is met (per-seed intervals consistently straddle)
  but its first is not in spirit — the across-seed estimate is not centred *on* 0, it is
  centred slightly negative with 0 near the interval's upper edge. So the sign of the effect
  is genuinely unresolved.
- **But the question that motivated the campaign IS settled, and this must not be blurred.**
  The **resolvable floor** here is a half-width of **0.003960**, which is *smaller* than the
  D2−R2 effect being hunted (**+0.004892**). Had that effect existed on this view pair, this
  instrument would have resolved it as CI>0. It came back the other sign. So: **there is no
  free recall to recover by regularizing the deployed view pair.** That is a resolution-
  sufficient non-finding, categorically different from `2026-08-02b`'s underpowered one.

## Phase A by-product — the family's first seed-variance measurement

`seq-nexttrack` **does not declare `seed`**, so every noise figure on record for this family
is a between-config or reproduction spread — seed variance has never been measurable.
`seq-bank` declares it, so these are the first numbers:

| config | hits across seeds {1337, 7, 42} | mean recall | **sd** | **range** |
|---|---|---|---|---|
| `latent+cummean` h256 `dropout=0.0` | 168 / 166 / 173 | 0.118099 | **0.002520** | **0.004892** (7 hits) |
| `latent+cummean` h256 `dropout=0.1` | 172 / 162 / 160 | 0.115071 | **0.004493** | **0.008386** (12 hits) |
| pooled over all six arms | 160–173 | — | 0.003656 | 0.009085 (13 hits) |

Every one of the six pairwise same-config seed-to-seed deltas **straddles 0** (largest
−0.008386 at dropout 0.1, seed 42 − seed 1337), so no seed is an outlier in the
significance sense — the spread is ordinary seed noise, which is exactly what makes it
dangerous.

🚨 **Flagging this prominently, as instructed: at `dropout=0.1` the seed RANGE (0.008386,
12 hits) EXCEEDS the sharpest single-hyperparameter half-width on record (0.005250), and
the seed sd (0.004493) is of the same order.** Consequences, in descending severity:

1. **The D2−R2 finding that launched this entire line was single-seed, and its effect size
   (+0.004892, 7 hits) exactly equals the seed range measured here at `dropout=0.0` and is
   smaller than the range at `dropout=0.1`.** It must be re-flagged as **unconfirmed**: a
   CI>0 at one seed is not evidence of an effect when a config's own seed range spans the
   effect. (Strictly, the seed variance measured here is on `latent+cummean`; carrying it to
   `latent+latent` is an inference from an otherwise-identical config, not a measurement.)
2. **Every single-seed verdict in the record inherits this caveat** — including the ties.
   Ties are safe (a tie plus noise is still a tie); it is the *narrow wins* that are exposed.
   The tower-bank campaign's five "LOSE to C1 CI<0" results (−0.0098 to −0.0168) are mostly
   larger than this seed range and so survive; the +0.0049 dropout win does not.
3. **New standing recommendation:** any future win claim on a `seed`-declaring predictor
   with |Δ| < ~0.009 recall should carry a 3-seed check before it is quoted as a win.

## Phase B results — the walk

40 held-out sessions × 20 steps, artist cap 1 (harness-enforced), paired bootstrap 2,000
resamples rng 1337, base = the **deployed** engine
(`best-seq-dualgru-20260725-143502-3e5d4`) at the cell under test. `vibe` is reported only
with `stride_err` beside it; `ground` is read as a degeneracy floor only.

### The free instrument check — E0 vs the deployed engine

E0 is bit-identical to the deployed engine on all 1,431 one-shot rows, so its walk must be
flat. It is, at **both** cells, on **all five** axes:

| cell | Δvibe | Δstride_err | Δdrift | Δgenres | Δground |
|---|---|---|---|---|---|
| a0.8/s0.5 | **+0.0000 [+0.0000, +0.0000]** | **+0.0000** | **+0.0000** | **+0.0000** | **+0.0000** |
| a0.4/s0.3 | **+0.0000 [+0.0000, +0.0000]** | **+0.0000** | **+0.0000** | **+0.0000** | **+0.0000** |

Not "small" — identically zero, with zero-width intervals, per session. This is a stronger
continuity check than G3: it validates the walk harness, the `promoted_score` path, the
Postgres→`data/models` promotion materialization, **and** the cross-family `seq_bank` ↔
`seq_dualgru` substitution simultaneously, on the generated-walk surface rather than the
one-shot one. No walk verdict from the last two campaigns needs re-examination on this
evidence.

### Means

| arm | vibe ↑ | stride_err ↓ | drift ~0 | genres ↑ | ground ↑ |
|---|---|---|---|---|---|
| deployed `l/cummean` f0 @ a0.8/s0.5 | 0.610 | 0.144 | −0.053 | 11.425 | 0.351 |
| E0 d0.0 @ a0.8/s0.5 | 0.610 | 0.144 | −0.053 | 11.425 | 0.351 |
| **E1 d0.1 @ a0.8/s0.5** | **0.613** | **0.132** | −0.058 | 11.150 | 0.347 |
| deployed @ a0.4/s0.3 | 0.551 | 0.126 | −0.061 | 9.900 | 0.351 |
| E0 d0.0 @ a0.4/s0.3 | 0.551 | 0.126 | −0.061 | 9.900 | 0.351 |
| E1 d0.1 @ a0.4/s0.3 | 0.551 | 0.130 | **−0.084** | 10.000 | 0.352 |

### Paired Δ, E1 vs the deployed engine

| cell | Δvibe | Δstride_err | Δdrift | Δgenres | Δground |
|---|---|---|---|---|---|
| **a0.8/s0.5 (deployed, PRIMARY)** | **+0.0024 [−0.0044, +0.0082] straddles** | −0.0129 [−0.0319, +0.0040] straddles | −0.0046 [−0.0287, +0.0148] straddles | −0.2750 [−0.8250, +0.2500] straddles | −0.0042 [−0.0164, +0.0093] straddles |
| a0.4/s0.3 (matched) | +0.0007 [−0.0067, +0.0080] straddles | +0.0036 [−0.0167, +0.0225] straddles | **−0.0226 [−0.0401, −0.0056] CI<0** | +0.1000 [−0.5500, +0.7250] straddles | +0.0013 [−0.0118, +0.0164] straddles |

**Phase B verdict: E1 PASSES the stated candidacy test — Δvibe is not CI<0 at a0.8/s0.5
(it is nominally positive), so E1 does not repeat the tower-bank collapse.** Read `vibe`
with `stride_err`: at the primary cell E1's stride_err is *lower* by 0.0129 (better,
straddling) while vibe is *higher*, so the vibe reading is not the "took tiny steps" artifact
the record warns about — it is at worst neutral and possibly a small genuine improvement.
`ground` never moves (degeneracy floor intact at both cells).

**The one real walk cost is at the matched cell: Δdrift −0.0226 CI<0.** E1's end-quarter
vibe falls 0.084 below its start-quarter vs the deployed engine's 0.061 — a real
regression in mood-holding *over the course of the walk* at a0.4/s0.3, which does **not**
appear at the deployed cell (−0.0046, straddles). Under the standing one-cell rule this is
the mirror image of the tower-bank failures: an effect at one cell that does not hold at
the other, so it is not a closure — but it is the only axis where E1 is measurably worse
than what ships, and it is on the axis this project cares about.

## Findings

**1. The f0 dropout defect is real as code and empty as an opportunity.** The inertness is
not in doubt — `2026-08-02b` demonstrated it by having a `seq-bank` twin at explicit
`dropout=0.0` reproduce the deployed engine's ranked list on all 1,431 rows, and this
campaign reproduced it again on the walk at exactly 0.0000. What is now settled is that
*fixing* it buys nothing measurable on the deployed view pair: the seed-averaged estimate
is negative-leaning with a resolvable floor finer than the effect being sought.

**2. The campaign's most valuable output is a noise number, not a verdict.** A 12-hit seed
range inside one fixed config, on a family where seed variance had never been measurable,
recalibrates how every narrow single-seed win in the record should be read — including the
one that motivated this campaign. This is the third time in three campaigns that the
methodological by-product outweighed the stated hypothesis (val-loss ranking, wall-clock
non-signal, now seed variance).

**3. `dropout=0.1` increases seed variance on this config** (sd 0.004493 vs 0.002520;
range 12 hits vs 7). Small-n, so treat as suggestive rather than established — but it is
the opposite of the usual "regularization stabilizes" intuition, and it is consistent with
the seed-42 cell being the one that swings hardest (−13 hits, 51 discordant rows).

**4. The one-shot and walk surfaces disagree about E1, mildly and legibly.** One-shot
across seeds: no gain, possibly a small loss. Walk at the deployed cell: vibe and
stride_err both nominally better, nothing CI<0. Walk at the matched cell: drift genuinely
worse. This is the record's standing pattern — the surfaces are not substitutes — and it is
why the campaign ends with no recommendation to ship rather than with a trade-off argument.

**5. The best-models selector's concurrent pass did exactly the right thing and cost
nothing.** It de-flooded the group to 12 entries and **excluded** both of this campaign's
measurement promotions (`dropout-e0-latent-cummean-d00`,
`dropout-e1-latent-cummean-d01`) along with the 8 tower-bank `bank-*` ones. The model
**directories survive** (exclusion is a group-membership decision, not a deletion), so every
walk number above was computed against intact artifacts and none had to be re-promoted. The
deployed `best-seq-dualgru-20260725-143502-3e5d4` remains pinned.

## Best on record after this work

| surface | best | value | run / model |
|---|---|---|---|
| one-shot recall@10, `seq-20260715-131139` | `seq-blend` markov-gated champion | **0.23201** | unchanged |
| best single-GRU width point (not significant vs h256) | h512 | 186/1431 = 0.129979 | `run-20260802-183404-91e31-seq-nexttrack` (unchanged) |
| **best architecture on the walk** | **dual `latent/cummean` f0 @ a0.8/s0.5 — STILL THE DEPLOYED ENGINE, and no longer describable as "a known-defective under-regularized version of itself"** | vibe 0.610 / stride_err 0.144 | `run-20260725-143502-3e5d4-seq-dualgru` |
| best `latent/cummean` f0 one-shot point | `dropout=0.0`, **seed 42** | 173/1431 = 0.120894 | `run-20260802-222820-99ff7-seq-bank` (a seed-lottery high, not a win) |
| dropout at `fusion_layers=0` | **NO recall effect on `latent+cummean` (positive effect excluded); the `latent+latent` CI>0 result is now UNCONFIRMED (single-seed, effect ≈ seed range)** | — | this report |
| **seed variance, `seq-bank` `latent+cummean` h256** | **sd 0.002520 (d0.0) / 0.004493 (d0.1); range 7 / 12 hits** | **first measurement on this family** | this report |

**No champion changed. Nothing was deployed.**

## Follow-ups

1. **Close the dropout line. Do not spend more runs on it.** Both surfaces have now been
   read, the positive effect is excluded at a resolution finer than the effect, and the
   remaining sign ambiguity is worth less than the runs it would cost. If anyone wants the
   sign, it needs ~10 seeds, not 3 — and the payoff is a number near zero.
2. **Re-seed the `latent+latent` D2−R2 contrast (2 runs) IF and only IF anyone intends to
   quote that +0.0049 CI>0 result again.** It is the one live claim in the record that this
   campaign's noise measurement directly undermines. Cheaper than the alternative of
   leaving a probably-spurious win in the facts file — but it is optional, because the
   facts file now flags it.
3. **Adopt the 3-seed rule for narrow wins on `seed`-declaring predictors** (|Δrecall| <
   ~0.009). This is the reusable output; it is recorded in `PROJECT-FACTS.md` as a standing
   rule rather than a suggestion.
4. **Measure seed variance once on `seq-blend`, where the champion lives** (3 runs). The
   champion's margins (+0.013 to +0.025) are comfortably above the seed range measured here,
   so this is not urgent — but the champion was crowned with seeds 1337/7/99 and its
   *gated* variant's +0.0014-to-+0.025 spread would benefit from a same-family seed figure
   rather than one imported from `seq-bank`.
5. **The `Δdrift −0.0226 CI<0` at the matched cell is the only unexplained result here.**
   If the dual-tower family is revisited on the walk, that is where to look: dropout on the
   concatenated hidden state appears to cost mood-holding *late in the walk* specifically,
   which no one-shot metric can see.

## Provenance

`registry.toml` unchanged since `2026-08-02b` (`15fd3837de3b5e983735b2dc4cc52346`);
`predictors/seq_bank.py` unchanged since the tower-bank campaign
(`7c58d53a2d3c3750502050afaabde238`). No predictor code was written. Two measurement-only
promotions created (`dropout-e0-latent-cummean-d00`, `dropout-e1-latent-cummean-d01`), both
subsequently excluded from the best-models group by the concurrent selector pass, which is
the intended handling. All 4 new runs' artifacts (4 each) are in Postgres via the worker
upload path. Walk outputs: `walk-primary-a08s05.json`, `walk-matched-a04s03.json`,
`walk-g3.json` (scratchpad; means and paired Δ reproduced in full above). This report and
the `experiments/PROJECT-FACTS.md` reconciliation are left uncommitted for the user.
