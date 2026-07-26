# Next-track item-VECTOR representation exploration — is the item latent space hindering prediction? — 2026-07-18 (c)

A representation bake-off. The 2026-07-15 compression axis established that latent
DIM (peak ~192) dominates method, but never asked whether the *composition* of the
192-d item vector — the equal-weighted 4-block PCA (text / numeric / acoustic /
categorical) — is itself a handicap. An EVR preflight (see below) said it might be:
full-matrix PCA-192 spends **87% of its variance budget** reconstructing the
numeric+acoustic blocks to R²=1.000 while the signal-carrying text block reaches
only R²=0.717. 8 alternative item-latent spaces (whitening × acoustic-drop,
text-only, text+categorical, and the musical-distance metric geometry) were built
parity-gated to the champion split — vocab/split byte-identical to
`seq-20260715-131139`, **only `item_latents.f32` differs** — and each is evaluated
here against the PCA-192 champion. **Validation-only: this pass returns a VERDICT;
promotion is the coordinator's call (nothing promoted / re-promoted here).**

Source of the datasets: `experiments/PROJECT-FACTS.md` §"ITEM-VECTOR
BLOCK-COMPOSITION EXPLORATION" (built 2026-07-18 evening). Prior campaigns:
`2026-07-18-nexttrack-literature-fit-campaign.md` (the projection champion),
`2026-07-18b-nexttrack-projection-registration-and-scan.md` (registration + scan),
`2026-07-15-nexttrack-compression-representation-bakeoff.md` /
`2026-07-15b-...dim-extension.md` (the dim/PCA-vs-AE axis).

## Goal & baseline

- **Question:** does the item VECTOR representation hinder next-track prediction —
  and if so, does *whitening*, *dropping acoustic*, or the *musical-distance metric
  geometry* give a better retrieval space than the equal-weighted PCA-192?
- **Baseline / champion:** `blend-gru-markov-content-proj` (R′+M+C′ learned content
  projection, InfoNCE / full-rank / τ0.07) and `gru-infonce-h256` (frozen GRU
  single), both on the **PCA-192 champion split** `seq-20260715-131139`
  (`spotify_tracks_song_pca192`; 7,154 sessions / 19,402 items; chronological split
  5,723 train / 1,431 test @ cut 2024-08-24; cold-target rate 0.55). On-record
  champion baseline: proj-blend R@10 **0.21174** / MRR 0.12199 / artist@10 0.3326 /
  genre@10 0.4011; frozen GRU R@10 **0.12299** / MRR 0.0534 / artist@10 0.3375 /
  genre@10 0.4500. **Both baselines were re-run in THIS batch** (so the paired-Δ is
  same-batch) and reproduced the on-record numbers to 4 decimals.
- **EVR preflight (already established, motivates the design):** the RAW input
  variance budget is 44% acoustic + 43% numeric vs only 2.5% text (text is
  L2-normalized → tiny per-dim scale), so full PCA-192 reconstructs numeric+acoustic
  to R²=1.000 (top-10 PCs = 65% of variance, all popularity/era/loudness with ~0
  text loading) while text reaches only R²=0.717; per-column standardize (whitening)
  flips the captured budget to 73% text / 21% categorical (text R² 0.905).
  Hypothesis: a whitened / acoustic-dropped / metric space is a better retrieval
  geometry.
- **Two readouts per variant** (both re-fit/train on each space, so each space is
  tested under both the winning recipe AND raw): **(A) champion projection**
  `blend-gru-markov-content-proj` (the InfoNCE projection re-fits on that space's
  train pairs → the space's ceiling under the winning recipe) and **(C) frozen GRU
  single** `gru-infonce-h256` (no learned reweighting → raw representation quality,
  the bake-off readout).
- **Noise band (single split):** ~±0.015 on recall@10 (bootstrap half-width).
  Verdict statistic = **paired per-session Δ on exact hit@10** (2,000 bootstrap
  resamples, rng 1337) vs the same-readout baseline; a lift needs the paired-Δ CI
  **entirely >0**.
- **Serialization:** all 18 runs launched strictly **one at a time** (documented
  thread-oversubscription pitfall: >1 concurrent torch predictor tanks throughput
  ~85×). Zero >1 in flight, **zero failures, zero relaunches**, n_test uniformly
  1,431. (Batch note: the harness killed the driver process after run 14; it was
  restarted for the final 4 — no run was affected, no double-launch, the killed
  run 14 V7/C had already succeeded on-server and was backfilled from its metrics.)

## Outcome (one line)

**The item representation is NOT hindering the deployed champion but IS hindering
the raw model: NO variant beats the champion on readout A with paired-Δ CI>0 (the
learned InfoNCE projection already re-weights away the loud acoustic/numeric
directions), while SIX of 8 variants beat the frozen GRU on readout C by paired-Δ
+0.025…+0.036 (whitening and/or dropping acoustic lifts the raw representation,
confirming the EVR preflight) — the gain is fully absorbed by the projection, so
the champion PCA-192 space holds and nothing is a promotion candidate.**

## Results — READOUT A: champion projection (`blend-gru-markov-content-proj`)

Baseline = **full** on `seq-20260715-131139` (`run-20260718-223848-d593e-seq-blend`).
Best cell per metric **bold**; paired-Δ CI>0 column = does the paired-Δ 95% CI clear
0. The champion baseline holds (**bold** row).

| variant (space) | dataset_id | recall@10 [95% CI] | MRR | artist@10 | genre@10 | paired-Δ hit@10 [95% CI] | CI>0 | run_id |
|---|---|---|---|---|---|---|---|---|
| **full — PCA-192 champion (baseline, holds)** | seq-20260715-131139 | **0.2117** [0.1908, 0.2327] | 0.1220 | 0.3326 | 0.4011 | (baseline) | — | run-20260718-223848-d593e-seq-blend |
| V1 noaco (drop acoustic) | seq-20260718-222742 | 0.2089 [0.1880, 0.2292] | 0.1188 | 0.3312 | 0.3976 | −0.0028 [−0.0133, +0.0077] | no | run-20260718-224718-fb487-seq-blend |
| V2 std (whiten all) | seq-20260718-222754 | 0.2082 [0.1873, 0.2292] | 0.1208 | 0.3368 | 0.4004 | −0.0035 [−0.0154, +0.0084] | no | run-20260718-225548-f455d-seq-blend |
| V3 std-noaco (whiten + drop acoustic) | seq-20260718-222807 | 0.2166 [0.1950, 0.2383] | 0.1193 | 0.3403 | 0.4032 | +0.0049 [−0.0077, +0.0168] | no | run-20260718-230418-55314-seq-blend |
| V4 textcat (text+categorical) | seq-20260718-222820 | 0.1950 [0.1747, 0.2152] | 0.1114 | 0.3305 | 0.3969 | −0.0168 [−0.0287, −0.0056] | **neg** | run-20260718-231318-1a295-seq-blend |
| V5 balanced-pca192 (metric geom) | seq-20260718-222834 | 0.2159 [0.1943, 0.2369] | **0.1256** | 0.3396 | 0.3997 | +0.0042 [−0.0063, +0.0147] | no | run-20260718-232119-4fd6c-seq-blend |
| V7 std-txtcatup (whiten + txt/cat up, aco soft-drop) | seq-20260718-222848 | 0.2055 [0.1845, 0.2264] | 0.1180 | 0.3382 | 0.4039 | −0.0063 [−0.0189, +0.0056] | no | run-20260718-233019-08f1c-seq-blend |
| V10 text-only | seq-20260718-222901 | 0.1999 [0.1789, 0.2194] | 0.1171 | **0.3466** | **0.4144** | −0.0119 [−0.0245, +0.0007] | no | run-20260718-234110-2c5e1-seq-blend |
| V6 sonic-64 (64-d, below dim peak) | seq-20260718-222905 | 0.1936 [0.1740, 0.2146] | 0.1119 | 0.3312 | 0.3962 | −0.0182 [−0.0314, −0.0049] | **neg** | run-20260718-235010-4520d-seq-blend |

Best readout-A cell: recall@10 V3 0.2166 (sub-noise, CI straddles 0); MRR **V5 0.1256**;
artist@10 **V10 0.3466**; genre@10 **V10 0.4144**.

## Results — READOUT C: frozen GRU single (`gru-infonce-h256`)

Baseline = **full** on `seq-20260715-131139` (`run-20260718-224518-49721-seq-nexttrack`).
Best cell per metric **bold**; the best raw-rep lift row (V5) **bold**.

| variant (space) | dataset_id | recall@10 [95% CI] | MRR | artist@10 | genre@10 | paired-Δ hit@10 [95% CI] | CI>0 | run_id |
|---|---|---|---|---|---|---|---|---|
| full — PCA-192 champion (baseline) | seq-20260715-131139 | 0.1230 [0.1069, 0.1398] | 0.0534 | **0.3375** | 0.4500 | (baseline) | — | run-20260718-224518-49721-seq-nexttrack |
| V1 noaco (drop acoustic) | seq-20260718-222742 | 0.1481 [0.1300, 0.1670] | 0.0674 | 0.3305 | 0.4507 | +0.0252 [+0.0091, +0.0405] | **YES** | run-20260718-225218-9a37f-seq-nexttrack |
| V2 std (whiten all) | seq-20260718-222754 | 0.1530 [0.1342, 0.1726] | 0.0686 | 0.3340 | 0.4556 | +0.0300 [+0.0112, +0.0489] | **YES** | run-20260718-230048-23b6c-seq-nexttrack |
| V3 std-noaco (whiten + drop acoustic) | seq-20260718-222807 | 0.1572 [0.1384, 0.1775] | 0.0695 | 0.3361 | 0.4710 | +0.0342 [+0.0154, +0.0545] | **YES** | run-20260718-230948-d3d4a-seq-nexttrack |
| V4 textcat (text+categorical) | seq-20260718-222820 | 0.1565 [0.1377, 0.1754] | 0.0641 | 0.3270 | 0.4458 | +0.0335 [+0.0154, +0.0510] | **YES** | run-20260718-231818-c1b8e-seq-nexttrack |
| **V5 balanced-pca192 (metric geom — best raw-rep lift)** | seq-20260718-222834 | **0.1586** [0.1405, 0.1775] | **0.0730** | 0.3291 | 0.4493 | **+0.0356 [+0.0189, +0.0517]** | **YES** | run-20260718-232649-61137-seq-nexttrack |
| V7 std-txtcatup (whiten + txt/cat up, aco soft-drop) | seq-20260718-222848 | 0.1488 [0.1307, 0.1684] | 0.0623 | 0.3277 | **0.4717** | +0.0259 [+0.0056, +0.0461] | **YES** | run-20260718-233549-212f7-seq-nexttrack |
| V10 text-only | seq-20260718-222901 | 0.1377 [0.1195, 0.1558] | 0.0647 | 0.3138 | 0.4179 | +0.0147 [−0.0049, +0.0336] | no | run-20260718-234640-141a6-seq-nexttrack |
| V6 sonic-64 (64-d, below dim peak) | seq-20260718-222905 | 0.1216 [0.1048, 0.1391] | 0.0479 | 0.3019 | 0.4403 | −0.0014 [−0.0168, +0.0126] | no | run-20260718-235540-31702-seq-nexttrack |

Best readout-C cell: recall@10 **V5 0.1586**; MRR **V5 0.0730**; artist@10 **full
0.3375**; genre@10 **V7 0.4717**.

## Verdict 1 — EXACT recall@10 (the primary question)

**Readout A (champion ceiling): NO variant beats the PCA-192 champion with paired-Δ
CI entirely >0.** The two nominally-positive spaces — V3 std-noaco (+0.0049) and V5
balanced-pca192 (+0.0042) — have paired-Δ CIs that straddle 0 (sub-noise ties). Two
spaces are significantly WORSE: V4 textcat (−0.0168, CI<0 — dropping numeric+acoustic
costs the projection real signal) and V6 sonic-64 (−0.0182, CI<0 — the expected
64-d handicap, below the ~192 dim peak). **⇒ The item vector representation is NOT
hindering the deployed champion.** The learned InfoNCE projection re-fits a metric
map over train consecutive pairs and already re-weights away the loud
acoustic/numeric directions, so re-basing the input geometry (whitening / drop /
metric) buys nothing at the champion's ceiling. **No promotion candidate; the
champion `blend-gru-markov-content-proj` on PCA-192 holds.**

**Readout C (raw representation): SIX of 8 variants beat the frozen GRU baseline with
paired-Δ CI entirely >0** — V1 noaco +0.0252, V2 std +0.0300, V3 std-noaco +0.0342,
V4 textcat +0.0335, V5 balanced +0.0356, V7 std-txtcatup +0.0259. Best raw-rep lift:
**V5 balanced-pca192 +0.0356 [+0.0189, +0.0517]** and **V3 std-noaco +0.0342 [+0.0154,
+0.0545]**. V10 text-only (+0.0147) grazes 0 (not significant); V6 sonic-64 ties
(−0.0014, 64-d). **⇒ At the RAW representation level the full-matrix PCA-192 geometry
IS a handicap** — suppressing the loud acoustic/numeric directions lifts the frozen
GRU by ~+0.025 to +0.036 (a +20–29% relative jump off the 0.123 base), **confirming
the EVR-preflight hypothesis**.

**The cross-readout read (as the design asked):** the whitened variants (V2/V3/V7)
lift the FROZEN GRU (C) but NOT the champion (A). Per the pre-registered
interpretation, this means **the projection already compensates for the loud
directions** — it is NOT the strong result (they do not lift both readouts). The
representation problem is real and the projection is precisely the fix for it; the
two are the same lever, and the champion has already pulled it.

## Verdict 2 — GRADED (artist@10 / genre@10)

The wanted whitened/metric graded lift shows up, mostly on genre and mostly on the
frozen GRU:

- **genre@10, readout C:** V7 std-txtcatup **0.4717** and V3 std-noaco **0.4710** vs
  baseline 0.4500 (+0.022 / +0.021 point) — the two whitened, text-up-weighted /
  acoustic-suppressed spaces are the genre-band leaders. V2 std 0.4556 also up. (These
  are point deltas; predictions.json carries no genre map, so no paired-Δ — flagged
  per the design.) artist@10 on readout C is flat-to-down (the frozen GRU's artist@10
  is already high at 0.3375 and no space improves it).
- **readout A, the graded-preferred space is V10 text-only:** it posts the highest
  artist@10 (**0.3466** vs baseline 0.3326) AND genre@10 (**0.4144** vs 0.4011) of any
  readout-A cell — while its exact recall drops (−0.0119). A text-only geometry trades
  exact hits for band/vibe matching under the projection, exactly the design's caveat
  ("a text/metric space may lift graded MORE than exact — a WANTED outcome, flag it").
  V3 std-noaco is the only space nominally positive on BOTH exact (A +0.0049) and
  graded (artist@10 0.3403 / genre@10 0.4032, both ≥ baseline).

**Best exact-recall variant:** readout A — none beats the champion CI>0 (nominal best
V3 std-noaco 0.2166, sub-noise); readout C — **V5 balanced-pca192 0.1586, paired-Δ
+0.0356 CI>0**. **Best graded variant:** V7 std-txtcatup / V3 std-noaco (genre@10 ~0.47
on the frozen GRU); V10 text-only for graded on the projection (artist@10 0.3466 /
genre@10 0.4144). **Does any variant beat the champion CI>0 on readout A? NO.**

## Findings

- **Does whitening help?** YES for the raw model, NO for the champion. Per-column
  standardize (V2) lifts the frozen GRU +0.0300 CI>0 (and whiten+drop V3 +0.0342,
  the biggest of the whitening family), confirming the EVR story that the loud,
  behaviorally-inert acoustic/numeric directions crowd out the signal-carrying text.
  But the champion's InfoNCE projection already learns this reweighting from data, so
  whitening the input first is redundant at the champion ceiling (V2 −0.0035, V3
  +0.0049 — both sub-noise). Whitening and the learned projection are the SAME fix.
- **Does dropping acoustic help?** For the raw model yes (V1 noaco +0.0252 CI>0), and
  it stacks with whitening (V3 std-noaco is the top whitening-family cell on C). For
  the champion it's neutral (V1 −0.0028). But dropping numeric+acoustic entirely (V4
  textcat) HURTS the champion (−0.0168 CI<0) even though it helps the raw GRU (+0.0335)
  — the numeric block carries something the projection uses that the frozen GRU can't
  exploit; keep numeric, drop/whiten only acoustic.
- **Does the metric geometry help?** V5 balanced-pca192 (the 517-d musical-distance
  `balanced` metric reduced to PCA-192) is the single best RAW-rep space (C +0.0356
  CI>0, best recall AND MRR on readout C) and even nudges the champion's MRR (A 0.1256
  vs 0.1220) and recall (0.2159, sub-noise) — the most promising alternative geometry,
  but still does not clear the champion's exact-recall CI. V6 sonic-64 (the 64-d
  whitened-AE sonic metric) is dominated by its dimensionality (both readouts ≈ the
  AE-64 control, as forewarned).
- **Exact vs graded.** The representation levers move the GRADED axis (genre band /
  artist band) more cleanly than exact recall: whitening lifts frozen-GRU genre@10 to
  ~0.47, and text-only lifts the projection's artist@10/genre@10 to the readout-A
  maxima — while exact recall at the champion ceiling is immovable. This mirrors the
  standing note that the metric is *sonic* / intrinsic (Spearman ≈0.04 vs
  co-listening): re-basing the geometry helps band-matching but not exact
  next-track adjacency, which the supervised projection owns.
- **Net:** the equal-weighted PCA-192 item vector is a genuine handicap for a raw
  model, and the 2026-07-18 learned projection is exactly the right response to it —
  this campaign independently validates WHY the projection is the champion (it is the
  representation fix, learned) and confirms there is no free lunch left in re-basing
  the input geometry. The most promising open direction is composing the two: fit the
  projection ON a whitened / std-noaco / metric space (see Follow-ups).

## Best on record after this work (RANKING board — UNCHANGED)

Validation-only pass; **the champion ROW is unchanged** and nothing was promoted.

| model | recall@10 [95% CI] | MRR | artist@10 / genre@10 | note |
|---|---|---|---|---|
| **★ R′+M+C′ learned content projection on PCA-192 (CHAMPION — HOLDS)** | **0.212** [0.191, 0.233] | **0.122** | 0.333 / 0.401 | `blend-gru-markov-content-proj`, best-models rank 1, seed-robust ×3, second-split hardened. THIS pass re-ran it in-batch (0.21174) and confirmed no alternative item space beats it on readout A (paired-Δ CI>0) — the item representation is not a residual bottleneck at the champion ceiling. |
| — frozen GRU on PCA-192 (raw-rep baseline) | 0.123 [0.107, 0.140] | 0.053 | 0.338 / 0.450 | `gru-infonce-h256`; the readout-C baseline. Beaten by 6 alt item spaces (+0.025…+0.036 CI>0) — the raw representation IS hindered, but the gain does not survive the projection. |
| — best alt RAW-rep space: V5 balanced-pca192 (frozen GRU) | 0.159 [0.141, 0.178] | 0.073 | 0.329 / 0.449 | musical-distance metric geometry; +0.0356 [+0.0189,+0.0517] over the frozen-GRU baseline. NOT a board row (raw single-model readout; ≪ champion) — recorded as the representation finding. |

## Follow-ups

- **Compose the two fixes: fit the projection on a whitened / std-noaco / metric
  space.** V3 std-noaco is the only space nominally positive on BOTH readouts and
  lifts frozen-GRU genre; V5 balanced is the best raw-rep space. The projection
  currently starts from raw PCA-192 and re-derives the whitening internally — test
  whether a pre-whitened / metric input lets it reach FURTHER (raises the champion
  ceiling) or is redundant (confirms it already extracts this). The highest-value
  next experiment.
- **Full ZCA whitening** (this pass used diagonal per-column standardize; PROJECT-FACTS
  flagged full ZCA as the obvious follow-up if V2/V3 win — they DID win on readout C).
- **Graded-relevance projection objective** (open follow-up): V10 text-only maximizes
  the projection's artist@10/genre@10 — a natural input for a projection fit toward
  artist/genre adjacency rather than exact next-track adjacency.
- **Keep numeric, whiten/drop only acoustic:** V4 textcat (drop numeric+acoustic)
  regressed the champion — do not drop the numeric block; the win is whitening +
  acoustic-drop (V3 std-noaco), not text-only.
- Do NOT pursue V6 sonic-64 at 64-d (dimensionality-dominated); if the sonic metric is
  worth chasing, reduce it to ~192-d first.
