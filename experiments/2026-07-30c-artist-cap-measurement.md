# 2026-07-30c — the HARD per-artist cap: the first de-eagering lever that works

**Type:** off-run measurement + serving-path implementation (no training runs).
**Dataset:** `seq-20260715-131139` (canonical PCA-192, n_test 1431).
**NOTHING PROMOTED / NOTHING REGISTERED / NOTHING TAGGED.** No champion moved.

## Outcome in one line

A **hard per-artist cap on the top-k** cuts `artist_conc@10` from **0.692 → 0.113** and raises
`holisticness@10` from **0.0086 → 0.0481** (5.6×, above the board's current best of 0.0342) while
*improving* `music@10` — at a cost of **52 of 1431 exact hits**. It is the only lever tested in this
project that de-eagers the one-shot top-10 without breaching the relevance floor, and it works
precisely because it **cannot be outscored**.

## Why this was tried

Two soft levers were refuted in the preceding 24h:

- **`eager_beta`** (train-time, `2026-07-30-eager-margin-calibrated-refutation.md`) — REFUTED across
  5 margins × a 60× dose range. Mis-specified: hinges on absolute `cos(pred, x_current)`, which a
  hedged InfoNCE prediction never drives above ~0.6.
- **`mmr_lambda`** (eval-time, now also serving-time after the 2026-07-30b fix) — works where the pool
  is sonically varied, but measured **inert on the pathological case**. On `rock chabon ahre` the model
  ranks 59 of its top 60 candidates as one artist and the next artist first appears at **rank 75**;
  MMR's entire diversity budget there is worth `0.3 × 0.26 ≈ 0.078` of normalized score, nowhere near
  enough to pay for a 70-rank relevance drop.

Both failures share one mechanism: **a soft penalty in score space loses to a steep relevance
gradient.** A cap does not compete on score at all — the same property `seq_extend`'s
`artist_cooldown` already relies on ("a cooldown cannot be outscored, which is what a listenable
playlist actually needs").

## Implementation

`seq_common.artist_cap_order(order, artist_ids, k, cap)`, applied **after** MMR in both
`eval_from_scores` and `rank_topk`/`predict_ranking` — so eval and serving agree by construction,
avoiding the divergence class fixed earlier the same day. `artist_cap` threaded into the train-time
eval **and** the serving predict of all five declaring families (`seq_nexttrack`, `seq_blend`,
`seq_dualgru`, `seq_embed`, `seq_mood`). Default `None` ⇒ every existing run and model is
**byte-identical**. Needs NO sonic vectors, so unlike MMR it ports to a thin client unchanged.

**BEST-EFFORT, not absolute** (verified by unit test): the scan walks the WHOLE order looking for
admissible artists, but where a prefix's candidate space is exhausted the remaining slots are
backfilled with deferred items rather than returning fewer than `k`. Unit-verified: identity when off,
identity when `cap >= k`, full permutation, prefix-stable, unknown-artist (`id < 0`) never capped.

## Split measurement (n=1431, paired on BIT-IDENTICAL cached scores, 2000 resamples, rng 1337)

Scores are computed once per model and memoized, so every cap arm sees the same score vectors — the
pairing is exact, not merely matched.

### Champion (deployed, ungated)

| cap | recall@10 | hits | MRR | artist_conc | artist_adj | ild | mood_coh | music@10 | H@10 |
|---|---|---|---|---|---|---|---|---|---|
| off | **0.211740** | 303 | 0.121985 | 0.691653 | 0.609713 | 0.406374 | 0.398222 | 0.453790 | 0.008603 |
| 5 | 0.178896 | 256 | 0.119693 | 0.252613 | 0.337247 | 0.563417 | 0.358768 | 0.474291 | 0.027258 |
| 3 | 0.160727 | 230 | 0.116915 | 0.122975 | 0.215304 | 0.613641 | 0.339311 | 0.476787 | 0.033862 |
| 2 | 0.138365 | 198 | 0.112313 | 0.061946 | 0.150384 | 0.639717 | 0.328459 | 0.474317 | 0.036331 |

### Champion + `markov_gate` (the deployable candidate)

| cap | recall@10 | hits | MRR | artist_conc | artist_adj | ild | mood_coh | music@10 | H@10 |
|---|---|---|---|---|---|---|---|---|---|
| off | **0.232006** | 332 | 0.133521 | 0.598214 | 0.574773 | 0.433765 | 0.462264 | 0.493189 | 0.018699 |
| 5 | 0.198463 | 284 | 0.131137 | 0.231338 | 0.335849 | 0.580367 | 0.413910 | 0.505703 | 0.040056 |
| **3** | **0.175402** | **251** | **0.127788** | **0.113440** | 0.217191 | 0.631939 | 0.386846 | **0.505762** | **0.048063** |
| 2 | 0.150943 | 216 | 0.122627 | 0.057365 | 0.152690 | 0.660122 | 0.368732 | 0.503767 | 0.051175 |

### Paired deltas vs cap=off

| model | cap | Δartist_conc | Δhit-rate | Δmusic_sim |
|---|---|---|---|---|
| ungated | 5 | −0.43905 [−0.45549, −0.42314] CI<0 | −0.03284 [−0.04472, −0.02096] CI<0 | **+0.02050 [+0.01469, +0.02692] CI>0** |
| ungated | 3 | −0.56867 [−0.58738, −0.55011] CI<0 | −0.05101 [−0.06569, −0.03702] CI<0 | **+0.02300 CI>0** |
| ungated | 2 | −0.62970 [−0.64907, −0.61067] CI<0 | −0.07338 [−0.09015, −0.05660] CI<0 | **+0.02053 CI>0** |
| gated | 5 | −0.36688 [−0.38445, −0.35104] CI<0 | −0.03354 [−0.04472, −0.02236] CI<0 | **+0.01251 CI>0** |
| gated | 3 | −0.48477 [−0.50477, −0.46653] CI<0 | −0.05660 [−0.07058, −0.04333] CI<0 | **+0.01257 CI>0** |
| gated | 2 | −0.54085 [−0.56157, −0.52198] CI<0 | −0.08106 [−0.09645, −0.06569] CI<0 | **+0.01058 CI>0** |

## Findings

**1. `music@10` goes UP, CI>0, at every cap on both models.** This is the campaign's most important
result and it was NOT predicted — the pre-registered worry was that a cap would de-eager by degrading
into generic filler. The opposite happens: the items promoted from rank 75+ are *more* musically apt to
the true continuation than the same-artist tracks they displace. The cap trades **literal** relevance
for **graded** relevance. Every arm clears the board's `music@10 ≥ 0.43` floor comfortably
(0.474–0.506), so none is degenerate.

**2. Only exact recall@10 regresses.** At `gate + cap 5`, versus the DEPLOYED status quo: recall
0.2117 → 0.1985 (−19 hits) but MRR 0.1220 → **0.1311 (up)**, artist_conc 0.692 → **0.231**,
ild 0.406 → **0.580**, mood_coh 0.398 → **0.414**, music@10 0.454 → **0.506**, H 0.0086 → **0.0401**.

**3. The cap does NOT pay for itself out of the gate's gain.** `markov_gate` is +29 hits; the mildest
cap costs −48 on the gated model. So `gate + cap` is net **−52 hits vs status quo at cap 3**. Anyone
who cares about exact recall should read this as a trade-off, not a free win. The project's stated
objective is holisticness, where it is a 5.6× gain.

**4. Every cap arm clears the board's recall floor** (0.108): 0.151–0.198. These H values are
therefore NOT instances of the "unconstrained holisticness is maximized by predicting worse" pitfall.
`gate + cap 3` at H **0.048063** would top the crown board by ~40% at **1.5× rank 1's recall**.

**5. The cap binds only where there is a problem** — see the spot-check: on a already-diverse list it
is a literal no-op at every cap value.

## Spot-check on three real user playlists (distinct artists in top-10 / `artist_conc`)

Same code path (`artist_cap_order` over the model's own ranking). `rock chabon ahre` and
`PREVIA NO NORMIE` are IDENTICAL between the two models because `markov_gate` is inert there.

| playlist | model | cap off | cap 5 | cap 3 | cap 2 |
|---|---|---|---|---|---|
| french touch (17/20) | champion | 8 / 0.044 | 8 / 0.044 | **8 / 0.044** | 8 / 0.044 |
| french touch | + gate | 4 / 0.467 | 6 / 0.222 | **7 / 0.089** | 8 / 0.044 |
| rock chabon ahre (17/27) | either | **1 / 1.000** | 4 / 0.267 | **5 / 0.156** | 6 / 0.089 |
| PREVIA NO NORMIE (126/169) | either | 3 / 0.622 | 3 / 0.356 | **4 / 0.200** | 6 / 0.089 |

Two qualitative results:

- **`rock chabon ahre` is FIXED** — the case that defeated both `eager_beta` and MMR outright. From
  ten consecutive Gustavo Cerati tracks to, at cap 3: Cerati ×3, Soda Stereo ×3, Babasonicos ×2,
  Patricio Rey ×1, La Portuaria ×1.
- **The cap exactly repairs `markov_gate`'s only regression.** The gate alone takes french touch from
  8 distinct artists to 4 (conc 0.044 → 0.467). At cap 3 it is back to 7 / 0.089, and at cap 2 to
  8 / 0.044 — identical to the deployed champion. So the gate's accuracy gain becomes available
  without its diversity cost.
- On the ungated champion's french touch list the cap changes **nothing at any value** (8 / 0.044
  throughout) — it never fires on an already-varied list.

## Recommendation

**`markov_gate=true` + `artist_cap=3`** as the next deployment candidate: `artist_conc` 0.692 → 0.113,
H 0.0086 → 0.0481, MRR and music@10 both up, `rock chabon ahre` repaired, at −52/1431 exact hits.
`artist_cap=5` is the conservative alternative (−19 hits vs status quo, MRR still above today's).

Because the cap needs no sonic vectors it is the ONLY diversity lever that ports to the app's WASM
champion unchanged — MMR would require shipping the 517-d `balanced` dial (~48 MB vs the ~6 MB
`sonic` dial currently baked, and the app baked the WRONG dial for MMR parity anyway).

## CONFIRMING RUNS — EXACT REPRODUCTION (added 2026-07-30c, same day)

Three registered runs on the canonical split, all `queue:true` on `snappler:434418`, launched from the
gated champion's registry-declared hyperparams with only `artist_cap` varying. **Every reported figure
reproduced to every digit**, so the off-run measurement above transfers exactly and the plumbing is
verified end-to-end (registry → API validation → remote worker → train-time eval → stored metrics):

| arm | run id | cap | hits | artist_conc@10 | H@10 | music@10 | MRR |
|---|---|---|---|---|---|---|---|
| **C0r** (bit-identity gate) | `run-20260730-235524-0ec1d-seq-blend` | 0 | **332** | 0.5982141470611072 | 0.018699174464215984 | 0.4931888239949002 | 0.13352076243938396 |
| **A1r** | `run-20260730-235524-93d38-seq-blend` | **3** | **251** | **0.11344048450966691** | **0.048062885636393195** | 0.5057618871551192 | 0.12778786906380157 |
| **A2** | `run-20260730-234635-a6252-seq-blend` | 5 | **284** | 0.23133783678857056 | 0.040055574262463825 | 0.5057033213634163 | 0.13113735533387036 |

C0r's `holisticness_at_k` is **bit-identical** to the original gated run `run-20260728-005349-507bb`
(`0.018699174464215984`), so `markov_gate` reproduces as well. `A1r` at H **0.048063** would rank 1 on
the crown board by ~40% over the incumbent 0.034171, at 1.5x its recall.

**⚠ THE FIRST LAUNCH ATTEMPT WAS INVALID — and the bit-identity gate is the only reason it was caught.**
The initial C0 (`run-20260730-234635-505ae-seq-blend`, now a misleading row in the record: its
hyperparams say `markov_gate=true` but its metrics are the UNGATED champion's — 303 hits / conc
0.6916530786551751 / H 0.008603) ran against a **stale worker checkout**:
`~/lensing-worker/predictors/seq_blend.py` and `seq_common.py` contained **ZERO** occurrences of
`markov_gate` OR `artist_cap`, md5s differing from the hub. Both hyperparams were silently ignored —
a WRONG NUMBER, not a crash. Fixed by rsyncing all predictor `.py` files + `registry.toml` (md5 parity
then verified). Killing the worker to stop the in-flight bad arm left
`run-20260730-234635-7a297-seq-blend` **orphaned at `running`** — `POST /runs/{id}/stop` returns 409
("is not running": it only tracks LOCAL runs) and `DELETE` refuses ("running on a worker"), so that row
is stuck and can only be cleared by a direct Postgres `UPDATE`. **Lesson: md5 the predictor files for
THE FAMILY BEING LAUNCHED immediately before POSTing, not at design time** — the preceding campaign had
verified `seq_dualgru.py`/`seq_nexttrack.py`, which said nothing about `seq_blend.py`.

Best-models: still stale at `2026-07-28T01:39:56` after all three runs; no cap arm auto-promoted
(**5th confirmation** of the `spawn_recompute` trigger gap on remote batches). `POST
/api/best-models/recompute` deliberately NOT run — it auto-promotes.

## SECOND-SPLIT HARDENING — CONFIRMED (2026-07-31)

4 runs on the disjoint earlier-holdout split **`seq-20260718-211238`** (0.64 cold vs the canonical
0.55, byte-identical item space, n_test 1431), all `queue:true`. Pre-registered before results: two
legs, both required — Δ`artist_conc` ≤ −0.05 with paired CI<0, **and** Δ`music@10` paired CI lower
bound **> 0** (graded relevance must still IMPROVE, not merely avoid collapse).

| arm | run id | hits | artist_conc | music@10 | H@10 | MRR |
|---|---|---|---|---|---|---|
| S0 ungated cap-off | `run-20260731-001833-23767` | **244** | 0.7194502678779409 | 0.39616282243809336 | 0.0033919636485836347 | 0.10498196225675074 |
| **S1 gated cap-off** (in-batch control) | `run-20260731-001833-8a90d` | **272** | 0.6484664958459508 | 0.4295280074321936 | 0.008413694670203916 | 0.11800747296189296 |
| **S2 gated cap-3** | `run-20260731-001833-37102` | 207 | **0.1214690581566892** | **0.4559654869716959** | 0.03052934903328233 | 0.11394412064031198 |
| S3 gated cap-5 | `run-20260731-001833-758dd` | 226 | 0.24574889354763577 | 0.4487262477232358 | 0.02392898420044787 | 0.11600092325777545 |

Paired vs the in-batch S1 control (2000 resamples, rng 1337, n=1431):

| arm | Δartist_conc | Δmusic@10 | Δhit-rate |
|---|---|---|---|
| **cap 3** | **−0.52699** [−0.54546, −0.50825] **CI<0** | **+0.02644** [+0.01919, +0.03368] **CI>0** | −0.04542 [−0.05870, −0.03284] CI<0 |
| cap 5 | −0.40273 [−0.41911, −0.38609] CI<0 | **+0.01920** [+0.01349, +0.02513] **CI>0** | −0.03215 [−0.04263, −0.02236] CI<0 |

**VERDICT: HARDENED — both legs hold.** The unexpected `music@10` GAIN reproduces on a colder,
disjoint window (+0.026 here vs +0.013 canonical), so "the cap trades LITERAL relevance for GRADED
relevance" survives replication and is now the claim to stand behind. Cross-split consistency: every
effect same sign and order of magnitude, absolute conc at cap 3 within 0.008 of canonical (0.121 vs
0.113); the recall cost is SMALLER here and the relevance gain LARGER.

Bonus — **third independent `markov_gate` reproduction**: S0/S1 reproduced their on-record split-2
figures exactly (244 / 272 hits; S0 MRR `0.10498196225675074` to the digit) and the gate's paired
effect came out **+0.01957 [+0.01048, +0.03005] CI>0**, matching the recorded +0.01957 exactly. Note
the gate ALSO lowers conc on this split (−0.071 CI<0) and raises music@10 (+0.033 CI>0) — opposite in
sign to its french-touch behaviour, so its diversity effect is query-dependent, not uniformly bad.

⚠ **The canonical split's absolute `music@10 ≥ 0.43` floor does NOT transfer** — S1 sits at 0.4295,
just under it, on a legitimately colder window. Pre-registering the rule on the paired Δ rather than an
absolute threshold is the only reason this did not read as a false failure. Derive floors per-baseline.

## PORTED TO THE APP (2026-07-31)

Both levers are now in `../app`'s WASM champion (`worker-core/src/champion.rs`) — pure post-processing,
**no `champion.bin` rebuild, no new vectors, no retraining**:

- `markov_n_u()` recovers the gate's `n_u` from the RAW bigram counts already baked in (this is exactly
  why `build_champion.py` emits raw counts rather than normalized probabilities).
- `apply_artist_cap()` mirrors `seq_common.artist_cap_order` including the best-effort backfill, and
  treats `"?"`/empty artist as UNKNOWN → never capped, matching `_relevance_ids`' −1 class. (The
  canonical vocab has 0 such items; the guard is for a corpus refresh. Without it the port would cap
  all unknown-artist items as one artist while Python leaves them alone.)
- `blend_rank()` stays UNGATED/UNCAPPED so the pre-existing parity test keeps pinning the original
  arithmetic; policy goes through `blend_rank_opts()`. `Champion::extend` — the app's ONLY
  recommendation surface — serves `RankOpts::deployed(k)`.

**k-SCALING, settled by measurement (2026-07-31).** The app accepts `k` up to 50
(`worker/index.ts:164`, default 10). A FIXED cap 3 at k=50 drives artist_conc to **0.021** (5× past the
hardened operating point) and reaches base rank **p95 278**; scaling the cap to preserve the SHARE
reaches only **p95 156** and holds conc at **0.118**. Scaled is both gentler on relevance and faithful
to what was validated, and the two are IDENTICAL at k=10. Implemented as **30% of slots, floor 3** —
`Champion::effective_artist_cap` in Rust and in `seq_common.predict_ranking` server-side, so both paths
agree by construction.

**Parity gate: 4/4 tests pass.** `topk_matches_python_reference` passes **unchanged** (proves the
ungated path is byte-identical after the refactor); new `deployed_policy_matches_python_reference`
matches the Python reference exactly on all 5 fixture prefixes, generated from the same model dir
`champion.bin` is built from with `markov_gate=true` + `artist_cap=3`; new `artist_cap_scales_with_k`
pins max-per-artist at 3/6/15 for k=10/20/50. WASM rebuilt (`worker_core_bg.wasm`
`b606f710…` → `62bc99f7…`), full `npm run build` green. **NOT deployed and NOT committed.**

## PORTED TO infinite-playlist AS A SECOND ENGINE (2026-07-31)

`clients/infinite-playlist` turned out to share NOTHING with `../app`: it runs a
single GRU as ONNX in the browser (`gru.onnx`, `seq-nexttrack` + `eager_beta 0.1`)
with its own autoregressive walk. So none of the blend work transferred there
directly — `markov_gate` is meaningless without a Markov leg.

**What its own evaluation showed** (real shipped artifacts, walk reproduced exactly):
its soft artist penalty was **effectively binary** — at 0 you get artist blocks (runs
to 13), and at ANY value ≥0.25 every 30-step walk returned **30 distinct artists,
max run 1, conc 0.000**. A hard cap on top was a literal no-op. So the shipped
default already solved artist repetition, and the cap's value there is a *product*
one: it reaches the middle the slider could not (cap 3 → 16–23 artists, runs of 2–3).

**Shipped:** the soft penalty is replaced by a hard journey-wide cap exposed as a
"shape" control — Explore (1) / Balanced (2) / Continue (3) / Blocks (uncapped).
Explore reproduces the old default **exactly** on all three playlists (30 artists,
run 1, conc 0.000, same genre counts), so existing behaviour is preserved.

**Plus the champion as a SECOND ENGINE.** `champion.rs`/`transit.rs` are vendored
**byte-identical** from `../app` (only the `Meta` import path differs, verified by
diff) so they re-sync with a plain copy; the new autoregressive loop lives in a
separate `walk.rs` using only champion.rs's public API. It runs **Worker-side**
against a 22.52 MiB `champion.bin` read through the `[assets]` binding, so the
browser downloads **nothing** extra — one round-trip per batch, not per track, and
the Worker re-seeds from the whole sequence so batches stay coherent. Loaded lazily,
so the GRU engine never pays for it. `tools/bake_champion.py` is a thin wrapper
around `../app/tools/build_champion.py` rather than a second builder, so the bundle
stays the one `../app`'s parity test pins.

⚠ **THE CHAMPION ENGINE IS DEGENERATE UNCAPPED — 30 tracks by ONE artist** on an
Argentine-rock seed (`worker-core/tests/journey.rs`), against the GRU's 6 artists on
the same seed. The cap is load-bearing for this engine, not cosmetic: cap 1 → 30
artists, cap 2 → 19 (run 2), cap 3 → 19 (run 3). This is the walk-surface analogue
of its split `artist_conc` 0.598 vs the GRU's 0.356. The "Blocks" option is labelled
to say so.

New tests (3/3 pass): cap is respected journey-wide, no repeats, nothing from the
seed, capping never reduces variety, and both gate settings produce well-formed
journeys (the gate is inert on that particular seed — its last item has bigram
support). `npm run build` + `tsc --noEmit` clean.

### MOVED IN-BROWSER — the Worker-side version is unshippable on the FREE plan

Worker smoke test (real `wrangler dev`) PASSED — cold 4.67 s incl. the 22.5 MiB load,
**warm 0.93/0.94/0.94 s per batch of 8** (~117 ms/track), cap held exactly, cap 1 gave
8/8 distinct artists, unknown seeds → clean 422. **But that ~0.94 s is CPU, and the
Cloudflare FREE plan caps CPU at 10 ms per invocation — ~100× over. Batching cannot
rescue it: one step is already ~117 ms, 12× over.** The deployment target is on the
free plan, so the Worker path is dead there (it stays wired at `/api/champion` for a
paid deployment).

**The engine therefore runs IN THE PAGE**, where no CPU cap applies; the cost becomes a
one-time download. Same `--target web` wasm the Worker uses (272 KB), copied to
`public/wasm/` by `npm run copy:wasm`, loading the same `champion.bin`. Verified from
Node against the real artifacts: **wasm init 22 ms, champion.bin parse 89 ms, journey
750–1284 ms per batch of 8** — and the journey is **IDENTICAL track-for-track to the
Worker's output**, confirming the port is faithful. Uncapped in-browser reproduces the
degeneracy (1 artist / 8 slots).

**Prefetch, so the download is not a UX penalty.** The bundle is pulled in the
background as soon as the user has any seed (`renderSeeds` → `warmChampion`), i.e.
during the time they are already spending choosing seeds — and unconditionally, with
streamed progress, when they explicitly pick the champion. Skipped on `Save-Data` or
`2g` unless explicitly chosen, since silently spending ~23.6 MB of someone's data plan
on a mode they may never use is not our call. First failure retires the option for the
session (relabelled "unavailable on this deployment") and continues on the GRU rather
than re-failing every batch.

Two asset-serving facts found by measurement, both handled: Cloudflare's asset server
**does not always send `content-length`** (wrangler dev does not), so the progress
readout falls back to "N MB so far"; and its default is
`Cache-Control: max-age=0, must-revalidate`, which would revalidate 23.6 MB on every
visit — a `public/_headers` file now caches the bundle `immutable` for a year, made
safe by requesting `champion.bin?v=<manifest.champion.built_at>` so a rebake changes
the URL. `manifest.json` is deliberately left uncached so clients learn about rebakes.

**NOT deployed, NOT committed.**

## Caveats
- **Single split, single seed.** No second-split hardening, no seed variation.
- **`k` interacts with the cap.** Measured at `k=10` only; `cap >= k` is identity by construction.
- The open **eval-vs-serving row-level divergence** (`2026-07-30b`) is unaffected by this work and
  still means production output will be close to, not identical with, these numbers.

## Follow-ups

1. Register + run `blend-gru-markov-content-proj` + `markov_gate` + `artist_cap ∈ {3,5}` on the
   canonical split for an in-batch confirmation, plus the disjoint `seq-20260718-211238` split.
2. Scan `artist_cap` on the crown leader `seq-nexttrack` h256 — its base `artist_conc` is 0.356 vs the
   blend's 0.692, so the cap's effect there is likely smaller but cheaper in recall.
3. Port gate + cap into `../app`'s `champion.rs` (both are pure post-processing; the gate needs `n_u`,
   which `build_champion.py` already emits as raw bigram counts).
4. `album_cap` — the same construction on `album_adj`, the original "album-eager" complaint.
