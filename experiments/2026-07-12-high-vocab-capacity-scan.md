# burn-deep-classifier capacity scan on the HIGH-VOCAB rotation dataset (2026-07-12)

Goal: find out whether **capacity** (width/depth) lifts the burn-deep-classifier
ANN family into the tree tier on the HIGH-VOCAB rotation matrix
`ds-20260612-143014-p64-s42` (497 cols, 23,529 rows, 18,823/4,706 @ split seed
42; artist cap 300 / genre 120). High vocab is where ANNs are actually
competitive — `embeddings` reached 0.7228 (bottom of the tree tier) but was
never depth/width-tuned, and `mlp` (0.7129) was never capacity-tuned. Two
capacity ladders, one axis each (hidden geometry), `lr` held at each topology's
on-record best and `embed_dim` held at 8 (saturated on record):

- **Embeddings ladder (B1-B3, PRIMARY)** — ed8, lr 0.0005: [128,64] control /
  [256,128] wider / [256,128,64] deeper.
- **MLP ladder (B4-B6, SECONDARY / cross-family read)** — lr 0.001: [128,64]
  control / [256,128] wider / [256,128,64] deeper.

All arms: `activation=relu`, `dropout=0.1`, `epochs=250`, `patience=20`,
`batch_size=256`, `val_fraction=0.15`. Fixed baselines: on-record embeddings
best **0.7228** (ed8/lr0.0005/[128,64]); incumbent saved def
`burn-deep-embeddings-highvocab` (ed16/lr0.001/[128,64]) **0.7192**; on-record
mlp **0.7129**; in-dataset tree tier catboost **0.7252** / xgboost **0.7299**.
Noise band 2σ_AUC = **0.0126**. Lineage: `2026-07-12-pyramid-vs-rect-sae.md`;
`archive/2026-06-12-high-vocab-topology.md`; `2026-06-15-vocab-cap-scan-gbdt.md`.

Outcome in one line: **CEILING — capacity is saturated for both topologies at
high vocab.** Neither widening nor deepening beats either sub-family's [128,64]
control by >1 band (deepening embeddings to 3 layers actively HURTS, −0.0152 ≈
1.2 bands); the best config stays the on-record embeddings ed8/lr0.0005/[128,64]
(reproduced here at 0.72274, Δ −0.0001 vs 0.7228). Best mlp sits within a band of
best embeddings → **capacity, not the embeddings mechanism, was the lever**; the
ANN family remains ~0.2–0.6 band below the tree tier. No definition saved,
incumbent saved def unchanged; best arm B1 promoted for the record (+ auto SAE).

## Results (all on ds-20260612-143014-p64-s42, 497 cols; sorted by AUC)

| arm | topology | hidden | AUC | logloss | accuracy | brier | run |
|---|---|---|---|---|---|---|---|
| **B1 (control, WINNER)** | embeddings | [128,64] | **0.72274** | **0.61043** | 0.66426 | **0.21140** | run-20260712-171008-065bd-burn-deep-classifier |
| B6 | mlp | [256,128,64] | 0.71800 | 0.61266 | 0.66065 | 0.21276 | run-20260712-171008-7c302-burn-deep-classifier |
| B2 | embeddings | [256,128] | 0.71798 | 0.61323 | 0.66511 | 0.21246 | run-20260712-171008-cf746-burn-deep-classifier |
| B5 | mlp | [256,128] | 0.71576 | 0.61220 | **0.66638** | 0.21249 | run-20260712-171008-fc3e0-burn-deep-classifier |
| B4 (mlp control) | mlp | [128,64] | 0.71229 | 0.61694 | 0.65852 | 0.21447 | run-20260712-171008-c093f-burn-deep-classifier |
| B3 | embeddings | [256,128,64] | 0.70750 | 0.61671 | 0.66001 | 0.21444 | run-20260712-171008-786b4-burn-deep-classifier |

All 6 runs succeeded; no failures, no relaunches. Launched simultaneously at
17:10 UTC into an empty queue (0 running / 0 queued at launch), ~2 min wall on 2
concurrent slots. `embed_dim` recorded as 8 on every arm; hyperparams verified
against the design post-launch.

### Decision rule (applied in order)

- **Step 0 — reproduction gates (both PASS, batch trustworthy).** Embeddings:
  B1 0.72274 ≥ 0.7102 — reproduces the on-record embeddings best (0.7228) to 4
  digits (Δ −0.00006). MLP: B4 0.71229 ≥ 0.7003 — reproduces the on-record mlp
  (0.7129) to 3 digits (Δ −0.00061). Near-, not bit-determinism confirmed.
- **Step 1 — no overfit signature.** Max logloss 0.61694 (B4) ≪ the 0.8 pitfall
  threshold; all six in the trees' calibrated regime (logloss ~0.610–0.617,
  brier ~0.211–0.214). Early stopping (patience 20) held for the 3-layer nets —
  the registered "deeper burn-deep-clf overfits at the full budget with patience
  0" pitfall did NOT trigger, as designed.
- **Step 2 — no material win.** Bar 0.7318 (incumbent + >1 band). Best arm B1
  0.72274 is 0.0091 below the bar → no SAVE, no mandatory 3-seed confirm.
- **Step 3 — no promising arm.** No embeddings arm reaches the [0.7228, 0.7318)
  band (B1 0.72274 sits fractionally below 0.7228 and is itself the control; B2
  0.71798, B3 0.70750 are lower). No mlp arm both beats B4 by >1 band and reaches
  the embeddings tier (≥0.7192): B6 beats B4 by only +0.00571 (sub-band) and is
  itself 0.71800 < 0.7192.
- **Step 4 — CEILING (default verdict, fires).** Neither sub-family's non-control
  arms beat their control by >1 band: embeddings B2 −0.00476 / B3 −0.01524 (both
  below control, B3 by >1 band); mlp B5 +0.00347 / B6 +0.00571 (both sub-band).
  Capacity is saturated for both topologies; embeddings [128,64] remains the
  family's best config; **no new definition, no dataset tag.** Best arm B1
  promoted for the record → model `emb-128-64-ed8-highvocab` (auto-queued SAE).
- **Step 5 — cross-family read (report only).** Best mlp B6 0.71800 ≥ best
  embeddings B1 0.72274 − 0.0126 = 0.71014 → **capacity, not the embeddings
  mechanism, was the lever**: at their best geometries the two families are tied
  within one band (Δ +0.00474 ≈ 0.38 band; control-to-control B1−B4 = +0.01045 ≈
  0.83 band, also sub-band).

## Findings

- **Capacity is a spent lever for burn-deep-clf at high vocab.** Both ladders are
  flat-to-negative in geometry. Widening [128,64]→[256,128] moves nothing
  out-of-band (emb −0.0048, mlp +0.0035); deepening to [256,128,64] helps mlp a
  touch (+0.0057, sub-band) but HURTS embeddings by −0.0152 (≈1.2 bands, the only
  out-of-band move in the batch). The on-record [128,64] geometry is already at
  or past the family's efficient frontier here.
- **The embeddings "mechanism" is not a distinguishable lever — capacity is.**
  On record `embeddings` beat `mlp` by +0.0099 at [128,64] (sub-band); this
  campaign reproduces that ordering (B1−B4 = +0.0105) but confirms it stays
  inside the band at every geometry, and the best mlp (B6) closes to within
  +0.0047 of the best embeddings (B1). Reading Step 5: the earlier "embeddings
  wins at high vocab" is an in-band tie, not a mechanism win — the two topologies
  encode the high-vocab identity signal about equally well once you let mlp use
  its capacity.
- **Deeper embeddings overfit the geometry, not the budget.** B3 ([256,128,64])
  is the worst arm (0.70750) with normal logloss (0.6167) — this is not the
  patience-0 calibration blowup; early stopping kept it calibrated, but the extra
  depth over a learned entity-embedding table dilutes ranking. A registered-pitfall
  cousin: the embeddings topology does not want a third layer.
- **The ANN family stays ~0.2–0.6 band below the tree tier.** Best ANN B1
  0.72274 vs catboost 0.7252 (−0.0025, in-band, bottom of the tree tier — exactly
  the on-record embeddings position) and vs xgboost 0.7299 (−0.0072 ≈ 0.57 band).
  Capacity does not cross the gap; consistent with the GBDT vocab-cap scan finding
  that identity-resolution QUALITY, not raw capacity (of model or vocab), is the
  remaining lever.
- **Incumbent saved def stands.** B1 0.72274 beats the saved
  `burn-deep-embeddings-highvocab` (ed16/lr0.001, 0.7192) by only +0.0035
  (sub-band) — no supersede. Note B1's config (ed8/lr0.0005) is the *actual*
  best-known embeddings config and is distinct from the saved (ed16/lr0.001) one;
  it now has a promoted, SAE-bearing record.

## Information capture (SAE)

Per-layer sparse-autoencoder read (2× overcomplete atoms, l1 0.0015, 40 epochs,
`compare_embedding: true`) of three promoted arms — the winner
`emb-128-64-ed8-highvocab` (B1, 0.72274), the best mlp `mlp-256-128-64-highvocab`
(B6, 0.71800), and the deep-embeddings `emb-256-128-64-highvocab` (B3, 0.70750) —
all read on the shared dataset `ds-20260612-143014-p64-s42` (497 cols). Analyses
reused from the auto-queued SAEs (no recompute). The SAE shows the AUC-tied
cross-family pair capture the *same* decodable taste-fit signal with a *different
concept allocation*, and that 3-layer embeddings degrades by diluting an
already-weaker learned input — not by starving capacity or dropping embedding
signal.

| model | best-layer util | Σ concepts | segments repr. | dropped | peak linear R² | AUC |
|---|---|---|---|---|---|---|
| emb-128-64-ed8-highvocab (WINNER) | 100% | 108 | 245/245 | 0 | 0.149 | 0.72274 |
| mlp-256-128-64-highvocab | 100% | 200 | 245/245 | 0 | 0.141 | 0.71800 |
| emb-256-128-64-highvocab | 100% | 181 | 243/245 | 0 | 0.135 | 0.70750 |

util is 100% (0 dead atoms) for every layer of every model — codes are dense (l0
≈ 22–69% of atoms active/track), so capacity is never the bottleneck and
utilization does not discriminate. peak linear R² is the best-layer test
R²_target for rotation; Σ concepts is summed interpretable-concept count across
hidden layers. dropped = 0 for all: `compare_embedding` ran, but the dataset SAE
surfaced **0 target-correlated atoms** above the 0.3 concept bar
(`embedding_diff.n_target_atoms=0` for all three) — the standing dataset-SAE null
on this corpus (no clean monosemantic taste-fit atom at this data scale), so the
dropped-signal axis is inconclusive **by construction**, not a model difference.

- **The cross-family tie is same signal, different allocation.** MLP [256,128,64]
  (0.71800) and embeddings [128,64] (0.72274) plateau at the same decodability
  ceiling (peak linear R²_target 0.141 vs 0.149; model-output R² 0.145 vs 0.150)
  and both give every one of the 245 identity segments a dedicated separating
  atom. The difference is layout only: the deeper/wider MLP forms nearly 2× the
  nameable concepts (Σ200) but they **peak in the middle 128-d layer (55→113→32)
  and collapse through its 64-d neck**, while the 2-layer winner reaches the
  identical ceiling with far fewer, monotonically-built concepts (33→75, peaking
  in its final bottleneck). This is the `2026-07-12-pyramid-vs-rect-sae.md`
  consolidation-vs-proliferation pattern reproduced at high vocab **and across
  families** — invisible to AUC. The ~0.008 decodability edge tracks the winner's
  marginally higher AUC but is suggestive, not a mechanism win.
- **Deepening embeddings hurts by dilution, not starvation.** B3 is *not*
  capacity-starved — 0 dead / 0 rare atoms, full utilization, var-explained
  0.967–0.989, drops no embedding concepts. Instead its *own learned
  entity-embedding input* is already weaker (input probe R²_target 0.123 vs the
  winner's 0.133 on the same rows) and the extra depth never recovers it:
  decodability runs 0.132→0.127→0.135 (a mid-trunk *dip* below its own first
  layer) vs the winner's clean 0.147→0.149. The middle layer over-generates
  concepts (91, above the winner's peak of 75) with no decodable gain — redundant,
  not productive — and even train-side fit is *lower* (train R²_log 0.216 vs
  0.234). So the −0.0152 penalty is under-/mis-fit from a worse-conditioned
  representation, **not** the patience-0 overfit blowup. This is the mechanistic
  face of the campaign's capacity-saturation ceiling: a third layer over a learned
  embedding table dilutes ranking.
- **Caveats.** Utilization and segment coverage are near-saturated for all three
  (dense codes at the default l1 0.0015), so the discriminating axes are concept
  *allocation* and *decodability*, not raw capacity. The dropped-vs-embedding axis
  is uninformative here (dataset-SAE null). Read on one dataset/seed — the
  allocation contrast is a repeatable pattern (now seen twice); the ~0.008
  decodability gaps are suggestive, not decisive.

Analyses (reused, auto-queued on promotion; reopenable in the UI **Saved
analyses** tool), shared dataset `ds-20260612-143014-p64-s42`:
`interp-run-20260712-171243-f65f7-msae` (emb-128-64-ed8-highvocab),
`interp-run-20260712-171915-c52e4-msae` (mlp-256-128-64-highvocab),
`interp-run-20260712-171915-fe7db-msae` (emb-256-128-64-highvocab). B6 and B3 were
promoted (`mlp-256-128-64-highvocab`, `emb-256-128-64-highvocab`) to enable this
comparison.

## Best on record after this work

HIGH-VOCAB board (ds-20260612-143014-p64-s42, 497 cols) — champion and saved def
unchanged:

| model | AUC | logloss | note |
|---|---|---|---|
| xgboost-classifier (champion config) | **0.7299** | 0.6022 | in-dataset AUC champion |
| catboost-classifier (defaults) | 0.7252 | 0.6068 | 2nd tree family, in-band |
| burn-deep emb ed8/lr5e-4 [128,64] — **B1 this campaign** | 0.72274 | 0.61043 | best ANN; reproduces on-record 0.7228; promoted `emb-128-64-ed8-highvocab` |
| burn-deep-classifier emb ed16/lr0.001 [128,64] | 0.7192 | 0.6087 | **SAVED `burn-deep-embeddings-highvocab`** (unchanged) |
| burn-deep mlp lr0.001 [128,64] | 0.7129 | 0.6166 | on-record mlp |

No definition saved this campaign; no dataset tag added (tagging rides a saved
winner). Global rotation AUC champion unchanged (DS-META xgboost 0.7429,
`xgboost-classifier-rotation`). One model promoted for the record:
`emb-128-64-ed8-highvocab` (run-20260712-171008-065bd), carrying an auto-queued
per-model SAE; auto-promotion displaced no registered model.

## Follow-ups

- **Resolution QUALITY, not capacity.** Both this scan and the GBDT vocab-cap
  scan point the same way: bigger models and bigger vocab caps are spent levers.
  Next identity work should target followers/popularity interactions or a genre
  hierarchy, not width/depth or higher caps.
- **Per-model SAE read — DONE** (see "Information capture (SAE)" above). The
  cross-family tie is same-signal / different-allocation and the deepening penalty
  is dilution not starvation; the entity-embedding table consolidates identity
  concepts (108) while the wider/deeper mlp proliferates then collapses them
  (200), reproducing the `2026-07-12-pyramid-vs-rect-sae.md` pattern across
  families. Dropped-embedding-signal axis inconclusive (dataset-SAE null).
- **lr / embed_dim re-scan at [128,64]** is the only unexplored corner for
  embeddings (geometry is now saturated, embed_dim saturated at 8 on record, lr
  fixed at the on-record best here) — low expected value, but the last untested
  axis if the family is revisited.
- No 3-seed confirm queued — no arm cleared the save bar, so the s17/s101
  rebuilds specified as a conditional follow-up are NOT triggered.
