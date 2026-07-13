# PCA-dimension scan on the leakage-free recipe (Campaign #2)

**Date:** 2026-06-07. Runner: experiment-runner agent. Replaces `2026-06-07-INTERIM-pca-dim-scan.md`.

## Goal

Find the PCA dimensionality that minimizes habit-fit MAE for the leakage-free
recipe, for both the champion family (xgboost) and the linear reference
(ridge). Scan dims {16, 32, 64, 96, 128, 160} × 6 split seeds {42, 7, 84, 123,
256, 999}, two models each (xgboost via `noiseband-xgb`, ridge via
`noiseband-ridge`). 60 new runs; the dim=128 column is **reused** from
Campaign #1 (no rerun).

**Baseline (reused, verified this campaign):**
- Champion: xgboost on ds-20260607-172307-p128-s42 family, promoted as `habit-fit`.
  6-split mean MAE **5.0495** (σ_split 0.155), mean R² 0.650.
- Ridge reference at p128: 6-split mean MAE **6.7556** (σ_split 0.191), mean R² 0.446.
- Held params: predictor hyperparams fixed per family; only `pca.dims` and the
  split seed vary.
- **Decision band:** ±0.18 MAE (= √2 · 2 · σ_split/√6, K=6 splits).
- Harness gate (C#1): s42×128 xgb control = 5.1691 = champion 5.169. PASSED.
- Re-verification this campaign: reused 128 means reproduce to 4 sig figs
  (xgb 5.0495 vs stated 5.050, σ 0.1554 vs 0.155; ridge 6.7556 vs 6.756,
  σ 0.1908 vs 0.191). Baseline confirmed intact.

## Outcome (one line)

**Rule 4 (EXTEND-UP) fired:** dim=160 beats the 128 champion by +0.25 MAE on
6/6 R²-confirmed splits, but it sits at the top edge of the scan — champion
stands UNCHANGED, no promotion; follow-up scan {192, 200} queued.

## Results — dim × model mean MAE (6 split seeds)

MAE in raw habit-fit units. **Bold** = best cell per metric per model. Champion
column (128) is reused from Campaign #1.

| model | dim | mean MAE | σ_split | SE (σ/√6) | mean R² | margin vs 128 |
|---|---|---|---|---|---|---|
| xgboost | 16 | 6.2966 | 0.2966 | 0.1211 | 0.5561 | −1.2471 |
| xgboost | 32 | 6.5516 | 0.2626 | 0.1072 | 0.4627 | −1.5021 |
| xgboost | 64 | 5.6348 | 0.2949 | 0.1204 | 0.5675 | −0.5853 |
| xgboost | 96 | 5.0766 | 0.1965 | 0.0802 | 0.6436 | −0.0271 (tied) |
| xgboost | 128 (reused) | 5.0495 | 0.1554 | 0.0634 | 0.6498 | — (baseline) |
| xgboost | 160 | **4.7995** | 0.2039 | 0.0832 | **0.6914** | **+0.2500** |
| ridge | 16 | 8.5835 | 0.3373 | 0.1377 | 0.1834 | −1.828 |
| ridge | 32 | 8.2476 | 0.2430 | 0.0992 | 0.2357 | −1.492 |
| ridge | 64 | 7.0311 | 0.2494 | 0.1018 | 0.4274 | −0.276 |
| ridge | 96 | 7.0382 | 0.1857 | 0.0758 | 0.4034 | −0.283 |
| ridge | 128 (reused) | **6.7556** | 0.1908 | 0.0779 | **0.4461** | — (baseline) |
| ridge | 160 | 7.0453 | 0.2296 | 0.0937 | 0.3579 | −0.290 |

"margin vs 128" is positive when the dim is *better* (lower MAE) than the 128
champion. All 60 new runs + 12 reused runs succeeded; 0 failed, 0 explosions.

### Per-split detail at the decision boundary (xgboost: 160 vs 128)

| seed | 160 MAE | 128 MAE | 160 R² | 128 R² | 160 R² ≥ 128 R²? |
|---|---|---|---|---|---|
| 42 | 4.9256 | 5.1691 | 0.6921 | 0.6346 | yes |
| 7 | 5.0132 | 5.2552 | 0.6811 | 0.6188 | yes |
| 84 | 4.4509 | 4.8119 | 0.7466 | 0.6756 | yes |
| 123 | 4.6829 | 5.0614 | 0.6966 | 0.6759 | yes |
| 256 | 4.9028 | 5.0355 | 0.6468 | 0.6275 | yes |
| 999 | 4.8216 | 4.9638 | 0.6853 | 0.6665 | yes |

dim=160 wins on R² on **6/6** splits — the Rule-1 R²-non-loss condition is met
cleanly. The MAE margin (+0.25) lands in the Rule-5 fragile band [0.18, 0.36].

## Findings

**Rule trace (applied verbatim).** The only dim d≠128 with xgb mean MAE lower
than 5.050 by more than the ±0.18 band is **dim=160** (+0.2500, R² non-loss on
6/6 splits). That single qualifying dim is exactly the top of the scan range,
which is the case Rule 4 names explicitly: "IF dim=160 beats 128 by >0.18, do
NOT promote; flag follow-up to scan {192, 200}." Rule 4 is the specific
boundary carve-out and governs here — **no promotion**. (Rule 1 and Rule 4 both
match the dim=160 data; the design's dim-160 carve-out resolves the overlap in
favor of not locking in a recipe at the unexplored top edge of the range. Rule
5's fragile flag would have applied to any promotion in the 0.18–0.36 band; it
is recorded for the follow-up.)

**The curve is still descending at the top of the range.** xgb mean MAE falls
monotonically from 96 → 128 → 160 (5.077 → 5.050 → 4.800) and R² rises
monotonically (0.644 → 0.650 → 0.691). 96↔128 is flat (Δ 0.027, well inside
±0.18) but 128→160 is a clean step down. There is no sign of saturation at 160;
the optimum is plausibly beyond 160, which is precisely why promoting 160 now
would be premature.

**Non-monotone low end.** xgb is non-monotone at the bottom: dim=32 (6.552) is
*worse* than dim=16 (6.297). At 16–32 dims the fixed tree hyperparams
(tuned for 128-dim inputs) appear mismatched to the much narrower feature
space; the model is under-capacity-matched rather than the data being
information-poor. Trade-off: low dims are cheap to build but cost ~1.5 MAE.

**Rule 2 secondary (recommend, do NOT promote).** The lowest dim whose xgb mean
is within ±0.18 of the 128 champion is **dim=96** (5.077, Δ −0.027), and it is
below 128. Per Rule 2 this is the **cheapest statistically-tied dim**:
switching the standard build recipe from 128→96 would cut PCA/build cost with
no measurable MAE penalty. This is a recommendation only — no API change made.

**Family-dependence note.** The optimal dim differs by family. xgboost's
argmin is **160** (and still falling); ridge's argmin is **128**, with ridge
*degrading* past 128 (160 = 7.045 > 128 = 6.756). Ridge — a fixed-capacity
linear model — saturates earlier and is hurt by the extra noisy PCA directions
that the trees exploit. The non-leakage signal that xgboost extracts from
dims 129–160 is not linearly accessible. Practical consequence: a single global
"best PCA dim" is wrong; tree families want more components than linear ones.

**Failures.** None. All 72 contributing runs (60 new + 12 reused) succeeded;
no explosions, no spurious stops, no relaunches needed. xgboost and ridge
remain explosion-free, consistent with the noise-band campaign's pitfall entry.

## Re-measured noise band (secondary deliverable)

σ_split per dim per model (sample stdev over the 6 split seeds):

| dim | σ_split xgb | σ_split ridge |
|---|---|---|
| 16 | 0.297 | 0.337 |
| 32 | 0.263 | 0.243 |
| 64 | 0.295 | 0.249 |
| 96 | 0.197 | 0.186 |
| 128 | 0.155 | 0.191 |
| 160 | 0.204 | 0.230 |

**σ_split grows materially at low dim.** At 128 the bands are 0.155 (xgb) /
0.191 (ridge), the values on record. At 16–64 dims σ_split roughly *doubles*
for xgboost (up to ~0.30) and is ~0.25–0.34 for ridge — split-luck variance is
larger in low-dimensional feature spaces. By 96 and 160 the bands are close to
the 128 values (0.20 ± 0.04). So the existing ±0.4 single-split significance
rule, calibrated at 128, **does NOT generalize downward**: at dims ≤ 64 a
single-split xgb improvement needs to exceed ~0.6 MAE (2·0.30) to be credible,
not 0.4. At 96 and 160 the ±0.4 rule remains a reasonable guard (2σ ≈ 0.41).
**Recommendation:** apply ±0.4 only for dims ≥ 96; widen to ±0.6 below that, or
re-measure per dim.

## Best on record after this work

No champion change. The promoted `habit-fit` model is unchanged.

| model | dataset | MAE | RMSE | R² | source |
|---|---|---|---|---|---|
| xgboost (promoted as `habit-fit`) | ds-20260607-172307-p128-s42 | 5.17 | 11.19 | 0.63 | bootstrap run |
| ridge | ds-20260607-172307-p128-s42 | 6.85 | 13.98 | 0.43 | bootstrap run |
| baseline-median | ds-20260607-172307-p128-s42 | 10.08 | 19.54 | −0.11 | bootstrap run |

Note for the leaderboard reader: the p160 xgboost recipe (6-split mean 4.800
MAE, R² 0.691) is a **stronger candidate** than the current champion but is
intentionally NOT promoted pending the extend-up scan — promoting at the scan
edge risks locking in a sub-optimal recipe. See 2026-06-07-noise-band-study.md
for the band derivation and this report for the dim curve.

## Follow-ups (ranked)

1. **EXTEND-UP scan {192, 200} (Rule 4 trigger, highest priority).** xgb MAE is
   still falling at 160 with rising R². Scan dims {192, 200} × the same 6 split
   seeds for xgboost. If the curve flattens, the flat region's *cheapest* dim
   becomes the promotion candidate; if it keeps falling, extend further. The
   PCA word2vec space is 200-dim (EVR@128 ≈ 0.9995), so 200 is the hard ceiling.
2. **Fragile-band confirmation of p160 (Rule 5).** Whatever dim wins the
   extend-up scan, the margin over 128 is in the 0.18–0.36 fragile band — run a
   3-seed × 6-split (training-seed × split-seed) confirmation before promoting,
   per the noise-band campaign's "prefer multi-split" guidance.
3. **Recipe simplification 128→96 (Rule 2).** If build cost matters, adopt
   dim=96 as the standard build recipe: statistically tied with 128 (Δ −0.027,
   inside ±0.18) and cheaper. Decouple this from the champion question.
4. **Re-tune xgb hyperparams for low dims.** The 16↔32 non-monotonicity
   suggests the tree params are mistuned for narrow inputs; a small
   depth/n_estimators sweep at dim ≤ 32 could recover the low-cost regime — low
   priority (low dims are not competitive regardless).
5. **Per-family dim policy.** Record that tree families want ≥160 dims while
   linear families saturate at 128; bake this into future dataset-axis designs
   so we stop assuming a single global best dim.

## Provenance / reconciliation

- 97 runs on server, all succeeded. Accounted for: 60 campaign-2 dim-scan runs
  (matched to cells strictly by run_id via /tmp/run_map.json) + 12 reused C#1
  p128 runs + 25 unrelated runs on the p128-s42 datasets (3 bootstrap, 11
  noise-band internal-seed reruns, 11 p128-curation A/B runs on
  ds-20260607-203256-p128-s42). None of the 25 touch the dim-scan datasets, so
  the matrix is uncontaminated. The "~11 extra" were the curation A/B campaign,
  not a 128 control re-run or p32 schema tests.
- Two orphan p32 datasets (ds-20260607-202855-p32-s42,
  ds-20260607-202902-p32-s42) from schema discovery have **zero** runs against
  them; harmless, no DELETE endpoint (405). Tracked in lineage as orphans.
