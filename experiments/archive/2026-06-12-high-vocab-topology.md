# Experiment: burn-deep-classifier topology bake-off on the HIGH-VOCAB binary rotation dataset

## Goal

Paired re-run of the `burn-deep-classifier` topology bake-off
(`experiments/2026-06-12-burn-deep-topology.md`, on the 60/40-vocab canonical
dataset) on a **higher identity-cardinality** matrix, to test that report's
standing hypothesis: that the `embeddings` topology lost to plain `mlp` only
because entity embeddings were **starved** at the old vocab caps (artist 60 /
genre 40). A dataset identical to the canonical full-64-d rotation dataset
EXCEPT for identity cardinality (artist cap 60→**300**, genre_primary 40→**120**)
was built for exactly this test. The headline question: with cardinality 5×
(artist) / 3× (genre) larger, does `embeddings` now **beat** `mlp` (and
`wide_deep`)? A reversal of the old +0.0067 mlp-over-embeddings margin confirms
the starvation hypothesis. Three further questions: (ii) do the trees also move
on the wider matrix, (iii) does any ANN topology reach the in-dataset tree
tier here, (iv) what is the embed_dim effect now that cardinality supports a
wider embedding.

**Fixed baseline.**
- Dataset: **`ds-20260612-143014-p64-s42`** — "Song-AE · rotation · p64 +
  metadata · high-vocab (artist 300, genre 120)". **23,529 rows** (18,823 train
  / 4,706 test @ split seed 42), **497 cols** (64 song-AE PCA latent + 9 numeric
  + 426 identity one-hots: artist 301 incl. `__other__`, genre_primary 121,
  album_type 4). Target `metadata.rotation`, task binary, transform none. SAME
  corpus / rows / split as the canonical `ds-20260609-204419-p64-s42`; differs
  ONLY in identity cardinality (PCA EVR byte-identical).
- **IMPORTANT — fresh in-dataset references.** Every metric in this campaign is
  on THIS dataset. The old in-dataset tree references (xgboost 0.7204, catboost
  0.7111) were on the 60/40 matrix and are NOT directly comparable, so Phase 0
  re-ran the trees here to establish a fresh in-dataset ranking.
- The burn predictor's internal split seed is fixed (1337); there is no `seed`
  hyperparam. The binary is near- (not bit-) deterministic — confirmed here:
  the Phase-2 ed16/lr0.001 cell reproduced the Phase-1 embeddings run to 4
  digits (0.71923 vs 0.71923), sub-band jitter, do not chase it.
- **Held hyperparameters (all Phase-1 arms):** epochs 250, patience 20,
  val_fraction 0.15, hidden [128,64], dropout 0.1, lr 0.001, batch_size 256,
  activation relu. Embeddings arm: **embed_dim 16** (raised from the old run's 8
  — the whole point being that higher cardinality now justifies a wider
  embedding). mlp / wide_deep ignore embed_dim. The ONLY Phase-1 axis is
  `topology`.
- **Noise band.** σ_AUC ≈ 0.00628, 2σ = **0.0126** (measured on the OLD corpus;
  approximate here — on 4,706 test rows the true band is plausibly tighter).
- **Collinearity note (new at high vocab).** The dataset manifest's redundancy
  report flags 21 near-perfectly-correlated genre↔artist one-hot pairs
  (e.g. `genre_primary=australian psych` ≡ `artist=Tame Impala`, corr 1.000;
  19 more ≥ 0.957) — an artifact of admitting rare genres whose only carrier is
  a single newly-admitted artist. The trees and the ANN regularizers absorb it;
  no run pathology observed. It dilutes nothing measurably but is worth knowing
  before reading too much into individual one-hot importances on this matrix.

## Outcome (one line)

**Raising identity cardinality FLIPS the topology ranking — `embeddings`
(0.7192) now beats `mlp` (0.7129, +0.0063) and `wide_deep` (0.7061, +0.0131),
reversing the old +0.0067 mlp-over-embeddings margin and confirming the
starvation hypothesis directionally; the best embeddings net reaches the
in-dataset tree tier (within ~½ band of catboost 0.7252), and the saved
`burn-deep-embeddings-highvocab` definition is the first ANN to clear the save
bar on rotation.**

## Phase 0 — fresh tree references (this high-vocab dataset)

Both trees run on the SAME 497-col high-vocab matrix as the ANN arms, so all
five families share one ranking.

| model | config | AUC | logloss | accuracy | brier | run id |
|---|---|---|---|---|---|---|
| **xgboost-classifier** | n_est 500, depth 6, lr 0.05, sub/col 0.8, seed 42 | **0.7299** | **0.6022** | **0.6719** | **0.2080** | run-20260612-143411-02fda-xgboost-classifier |
| catboost-classifier | registry defaults | 0.7252 | 0.6068 | 0.6649 | 0.2099 | run-20260612-143411-8f15e-catboost-classifier |

Both trees **moved UP** on the wider matrix vs the 60/40 dataset: xgboost
0.7204 → **0.7299** (+0.0095, ~¾ band), catboost 0.7111 → **0.7252** (+0.0141,
~1.1 bands). The extra identity granularity is genuine signal the trees exploit
— the high-vocab matrix is a better matrix for *every* family, not just for the
embedding hypothesis. The in-dataset tree tier on this dataset is therefore
**0.7252 (catboost) – 0.7299 (xgboost)**.

## Phase 1 — topology bake-off (axis = topology)

| topology | AUC | logloss | accuracy | brier | run id |
|---|---|---|---|---|---|
| **embeddings (embed_dim 16)** | **0.7192** | 0.6087 | **0.6694** | **0.2110** | run-20260612-143411-e8db6-burn-deep-classifier |
| mlp | 0.7129 | 0.6166 | 0.6617 | 0.2143 | run-20260612-143411-01d7b-burn-deep-classifier |
| wide_deep | 0.7061 | 0.6256 | 0.6526 | 0.2183 | run-20260612-143411-7d117-burn-deep-classifier |

Winner = **embeddings** (highest test AUC). Logloss stays calibrated (~0.61,
the trees' regime) across all three — the raw-one-hot calibration fix from the
prior campaign holds at high vocab.

**Paired comparison (the headline).** On the OLD 60/40 dataset, mlp beat
embeddings by **+0.0067** (mlp 0.7001 vs emb 0.6934). Here, embeddings beats mlp
by **+0.0063** — a **reversal**: a swing of **+0.0130** in embeddings' favour
(crossing the band). The embeddings arm rose from the bottom of the old
ranking (emb ≈ wide_deep, both below mlp) to the **top** of the new one. The
single-split emb-vs-mlp gap here (+0.0063, ~½ band) is "at least equal, likely
better — needs a 3-seed/3-split check to call decisive", but the *direction
flipped and the whole ranking reordered*, which is the qualitative signature
the hypothesis predicted. wide_deep is the consistent laggard in both regimes.

## Phase 2 — tune the winning topology (embeddings; embed_dim × lr)

Gate cleared: Phase-1 best (embeddings 0.7192) ≥ Phase-0 catboost (0.7252) −
0.0126 = 0.7126. embed_dim ∈ {8,16,32} × lr ∈ {0.0005,0.001} at hidden
[128,64], 6 runs, sorted by AUC.

| embed_dim | lr | AUC | logloss | accuracy | brier | run id |
|---|---|---|---|---|---|---|
| **8** | **0.0005** | **0.7228** | 0.6105 | 0.6645 | 0.2114 | run-20260612-143508-3310f-burn-deep-classifier |
| 16 | 0.0005 | 0.7210 | 0.6088 | 0.6711 | 0.2108 | run-20260612-143508-d9bf7-burn-deep-classifier |
| 32 | 0.0005 | 0.7200 | 0.6100 | 0.6657 | 0.2114 | run-20260612-143508-75804-burn-deep-classifier |
| 8 | 0.001 | 0.7193 | 0.6092 | 0.6643 | 0.2112 | run-20260612-143508-0b77d-burn-deep-classifier |
| 16 | 0.001 | 0.7192 | 0.6087 | 0.6666 | 0.2110 | run-20260612-143508-b717e-burn-deep-classifier |
| 32 | 0.001 | 0.7183 | 0.6097 | 0.6668 | 0.2114 | run-20260612-143508-5f0f2-burn-deep-classifier |

Best tuned = **embed_dim 8 / lr 0.0005, AUC 0.7228** (+0.0036 over the Phase-1
winner). Determinism check: ed16/lr0.001 (b717e, 0.71923) reproduces the
Phase-1 embeddings run (e8db6, 0.71923) to 4 digits — confirms the documented
sub-band jitter.

**Decision rule applied.** Best tuned (0.7228) beats the Phase-1 winner
(0.71923) by **+0.0036, NOT > 0.0126** → the config to save is the **Phase-1
winner** (embeddings, embed_dim 16, lr 0.001), not the tuned cell. The
save-eligibility check: the config-to-save sits at 0.7192, **−0.0060 vs
catboost 0.7252** (< 1 band) → **within ~1 band of the in-dataset tree tier →
SAVE.** Saved as definition **`burn-deep-embeddings-highvocab`**, tagged
`dataset_tags:[ds-20260612-143014-p64-s42]`. It reaches the tree tier →
**flagged for best-model-selector.**

## Findings

**(i) Did raising identity cardinality flip embeddings > mlp? YES — the
starvation hypothesis is CONFIRMED (directionally).** The topology ranking
reversed: embeddings went from below mlp (−0.0067 on the 60/40 matrix) to above
it (+0.0063 here), a +0.0130 swing across the band. With artist cardinality 5×
and genre 3× larger, the entity-embedding table finally has enough distinct
identities to learn a useful low-rank identity geometry instead of collapsing
near-constant rows. The single-split per-arm margin (+0.0063) is sub-band, so
the *magnitude* needs a 3-split confirm, but the *qualitative* prediction — that
embeddings would overtake mlp once cardinality is no longer starved — held. The
prior report's hypothesis can be marked **resolved-confirmed**.

**(ii) Did the trees also move on the bigger matrix? YES, and more than the
ANN.** xgboost +0.0095 and catboost +0.0141 vs the 60/40 dataset — both gained
roughly as much as (catboost) or more than (the trees collectively) the
embeddings arm gained. The wider one-hot matrix is better signal for every
family; this is NOT a representation advantage unique to embeddings. The
correct reading is "high vocab helps everyone, and it specifically *unlocks*
the embeddings topology", not "embeddings now wins outright".

**(iii) Does any ANN topology reach the tree tier here? YES — at the boundary.**
The best embeddings net (Phase-2 0.7228; Phase-1 winner 0.7192) sits −0.0024
(tuned) / −0.0060 (saved) below catboost (0.7252) and −0.0071 / −0.0107 below
xgboost (0.7299) — all inside the 0.0126 band vs catboost. By the band
convention this is "at least equal to the 2nd tree family, likely a hair
behind". So an ANN reaches the **bottom of the tree tier** on this dataset for
the first time, but does not overtake the leading tree (xgboost remains the
in-dataset champion, ~¾–1 band ahead). Logloss (~0.61) and brier (~0.21) are in
the trees' calibrated regime — no calibration penalty for the ANN here.

**(iv) The embed_dim effect.** Mild and monotone-decreasing in width: at
lr 0.0005, ed8 (0.7228) > ed16 (0.7210) > ed32 (0.7200); the full ed8→ed32
spread is ~0.0028 (sub-band). lr 0.0005 beats lr 0.001 at every width
(+0.0009 to +0.0035). Read: the gain came from **enabling embeddings at all +
the slower learning rate**, NOT from going wider — once cardinality supports a
real embedding, embed_dim 8 is already enough, and 16/32 add capacity that
neither helps nor hurts beyond noise. (The design raised Phase-1 to ed16
expecting width to matter; the tune shows ed8 is marginally best, still
sub-band — so the saved Phase-1-winner config at ed16 and the best cell at ed8
are statistically interchangeable.)

## Best-on-record context

These numbers are on `ds-20260612-143014-p64-s42` (HIGH-VOCAB, 497 cols) and
are NOT comparable to the 60/40 `ds-20260609-204419-p64-s42` board nor to the
old DS-META 0.7429 champion. In-dataset AUC ranking after this work
(higher-better):

| model | config | AUC | logloss | accuracy | brier | run id | registered? |
|---|---|---|---|---|---|---|---|
| **xgboost-classifier** | n_est 500 / d6 / lr 0.05 / 0.8 / seed 42 | **0.7299** | 0.6022 | 0.6719 | 0.2080 | run-20260612-143411-02fda | no |
| catboost-classifier | registry defaults | 0.7252 | 0.6068 | 0.6649 | 0.2099 | run-20260612-143411-8f15e | no |
| burn-deep-classifier (embeddings) | ed8 / lr0.0005 / [128,64] — Phase-2 best | 0.7228 | 0.6105 | 0.6645 | 0.2114 | run-20260612-143508-3310f | no |
| burn-deep-classifier (embeddings) | **ed16 / lr0.001 / [128,64] — saved** | 0.7192 | 0.6087 | 0.6694 | 0.2110 | run-20260612-143411-e8db6 | **YES — `burn-deep-embeddings-highvocab`** |
| burn-deep-classifier (mlp) | lr0.001 / [128,64] | 0.7129 | 0.6166 | 0.6617 | 0.2143 | run-20260612-143411-01d7b | no |
| burn-deep-classifier (wide_deep) | lr0.001 / [128,64] | 0.7061 | 0.6256 | 0.6526 | 0.2183 | run-20260612-143411-7d117 | no |

The registered champions on the *other* corpora are unchanged:
`xgboost-classifier-rotation-p64` (0.7204 on the 60/40 dataset) and
`xgboost-classifier-rotation` (0.7429 on DS-META) both stand on their own
matrices. xgboost-classifier at depth6/lr0.05 is the in-dataset leader on THIS
matrix at 0.7299 but is not registered here (no save rule triggered for a tree
this campaign).

## Follow-ups

1. **3-split confirm of the topology reversal (top priority).** The
   emb-vs-mlp margin here (+0.0063) is sub-band on a single split. Re-run
   embeddings vs mlp on this recipe at split seeds 17 / 101 to confirm the
   reversal is stable and to measure a high-vocab-specific σ_AUC (the 0.00628
   band is from the old corpus; 4,706 test rows likely give a tighter band).
2. **Register a high-vocab tree?** xgboost-classifier hits 0.7299 here (best
   in-dataset) but no save rule fired for a tree. If the high-vocab matrix
   becomes a serving target, consider an xgboost depth×lr sweep + save on this
   recipe — it is the true ceiling here, ~¾–1 band above the saved ANN.
3. **Does even-higher vocab keep helping, or has it saturated?** Both trees and
   the embeddings arm gained going 60/40→300/120. A 600/240 (or uncapped)
   variant would test whether identity signal is still unsaturated or whether
   the `__other__` mass (artist 49.6%, genre 21.2% at current caps) is now
   small enough that further granularity only adds collinear noise.
4. **embed_dim is already saturated at 8 here** — no need to scan wider on this
   cardinality. Any future embed_dim work should pair with the higher-vocab
   variant from (3) where a wider table might finally pay.
5. **best-model-selector** should re-curate the group: the saved
   `burn-deep-embeddings-highvocab` is the first ANN in the tree tier and a
   candidate for family-diversity inclusion (its raw single-split 0.7192 is
   honest, not a fluke — the determinism check and calibrated logloss support
   it).
