# Next-track Phase-1: item-representation bake-off (AE vs PCA × latent dim) — 2026-07-15

Which item **compression** — linear PCA vs nonlinear autoencoder, at latent dim
32 / 64 / 128 — is the best space for the sequence GRU to predict into and
retrieve candidates over? The champion's space is one arbitrary choice: a 64-dim
multimodal AE of the leak-safe content source (`spotify_tracks_song_ae` = AE-64).
We baked 5 alternative spaces from the SAME leak-safe 384-dim source
(`spotify_tracks_content`) and scored each under a **frozen GRU readout**
(`gru-infonce-h256`; `arch=gru, loss=infonce, hidden=256, num_layers=1,
dropout=0.1, epochs=40, patience=6, lr=0.001, batch=128, k=10`). Only the baked
item latent space varies.

**Baseline / CONTROL** — AE-64, artifact `seq-20260713-014226` (7,154 sessions /
19,402 items; chronological split 5,723 train / 1,431 test @ cut 2024-08-24;
cold-item rate 0.55). On-record GRU-infonce h256: R@10 0.096 [0.081, 0.112],
MRR 0.041, artist@10 0.294, genre@10 0.450. Champion R+M z-blend on AE-64 =
R@10 0.173 [0.154, 0.192], MRR 0.096 (the blend-confirmation reference). Sources:
`2026-07-13-nexttrack-sequence-sweep.md`, `2026-07-14-nexttrack-graded-relevance-and-new-legs.md`.

Every representation was fit unsupervised over all 23,529 items using ONLY
intrinsic content/acoustic/catalog fields (text-384 + z-scored numerics + af_*
acoustics + genre/album_type multi-hot); no behavioral/target field. Each new
seq artifact passed the exact **parity gate** (n_sessions=7154, n_items=19402,
train=5723, test=1431, cut=2024-08-24T14:28:02+00:00), so the comparison is
apples-to-apples. Bootstrap 95% CI = 2,000 resamples over the 1,431 test
sessions; paired-Δ = per-session R@10 difference vs the in-batch AE-64 control.

## Outcome (one line)

**Latent DIM — not method — is the lever: PCA-128 (`spotify_tracks_song_pca128`, `seq-20260715-030509`) is the single-GRU winner (R@10 0.1146, paired-Δ +0.019 CI entirely >0, MRR gain), carried to Phase 2; but the win does NOT survive into the blend (0.167 ≈ champion 0.173) and the training-free content-kNN probe does not corroborate it — so nothing is promoted, the AE-64 champion blend stands, and the "AE protects small acoustic blocks" hypothesis is refuted.**

## Results — GRU-single (frozen `gru-infonce-h256`), primary readout

Parity gate: control R@10 **0.0957 ∈ [0.081, 0.112] — PASS** (reproduces the
on-record 0.096 within CI; MRR/artist/genre also reproduce exactly). Ranking is
valid.

| config | run_id | R@10 [95% CI] | paired-Δ vs AE-64 | MRR | artist@10 | genre@10 | artist_mrr | content-kNN R@10 |
|---|---|---|---|---|---|---|---|---|
| **★ PCA-128 (WINNER)** | run-20260715-042100-a1d9d | **0.1146 [0.0985, 0.1321]** | **+0.0189 [+0.0042, +0.0335]** *>0* | **0.0489** | **0.3326** | 0.4521 | **0.2644** | 0.0636 |
| PCA-64 (runner-up) | run-20260715-042100-8851d | 0.1041 [0.0881, 0.1209] | +0.0084 [−0.0049, +0.0217] tie | 0.0416 | 0.3117 | 0.4598 | 0.2345 | 0.0580 |
| AE-64 (CONTROL) | run-20260715-042059-dd261 | 0.0957 [0.0811, 0.1111] | — (control) | 0.0411 | 0.2942 | 0.4500 | 0.2181 | 0.0636 |
| AE-128 | run-20260715-042059-e96ae | 0.0929 [0.0783, 0.1076] | −0.0028 [−0.0147, +0.0091] tie | 0.0398 | 0.3026 | **0.4640** | 0.2288 | 0.0601 |
| AE-32 | run-20260715-042059-d3f9e | 0.0545 [0.0426, 0.0664] | −0.0412 [−0.0552, −0.0273] *<0* | 0.0274 | 0.2376 | 0.3955 | 0.1630 | **0.0720** |
| PCA-32 | run-20260715-042059-b1242 | 0.0412 [0.0314, 0.0517] | −0.0545 [−0.0685, −0.0405] *<0* | 0.0224 | 0.2187 | 0.3816 | 0.1293 | 0.0468 |

All 6 runs `succeeded` (exit 0, n_test=1431). Best cell per metric bolded.

### Blend confirmation (winner only) — NOT a board event

`seq-blend` (`blend-gru-markov`; α=0.5, h256, infonce, seed 1337) on the PCA-128
artifact:

| model | run_id | R@10 [95% CI] | MRR | artist@10 | genre@10 | verdict |
|---|---|---|---|---|---|---|
| R+M z-blend on **PCA-128** | run-20260715-044920-36dce | 0.1670 [0.1481, 0.1866] | 0.0923 | 0.3291 | 0.3983 | below champion on BOTH; CIs overlap |
| R+M z-blend on AE-64 (on-record CHAMPION) | (prior) | 0.173 [0.154, 0.192] | 0.096 | 0.322 | 0.397 | stands |

Decision rule step 5: the PCA-128 blend does NOT beat the champion on both R@10
and MRR beyond non-overlapping CIs (it is slightly *lower* on both). **Not a
board event → nothing promoted, champion blend stands, the winning
representation is still carried to Phase 2.**

## Findings

1. **Latent dimensionality is the dominant lever — method is secondary.** The
   two 32-dim spaces (AE-32 0.0545, PCA-32 0.0412) sit far *below* the 64-dim
   control; the 128-dim spaces are best. For PCA the trend is monotone
   (0.041 → 0.104 → 0.115 as dim 32 → 64 → 128). 32 dims simply starve the
   space, regardless of compressor.

2. **At high dim, linear PCA beats the nonlinear AE — the hypothesis is
   refuted.** PCA-128 (0.1146) > AE-128 (0.0929), and PCA-64 (0.1041) ≥ AE-64
   (0.0957). The design bet on the AE's equal-weight-per-block loss preserving
   small acoustic/catalog blocks that PCA's text-dominated global variance would
   starve. But AE-32 preserves numeric+acoustic at R²=1.00 (build log) and is
   near the *bottom* of the GRU ranking — so acoustic preservation is irrelevant
   to next-track retrieval. What helps is capturing the high-variance
   text/catalog structure (PCA-128 cum EVR 0.982), and giving the GRU a
   *variance-ordered, decorrelated* target geometry that is easier to regress
   into than the AE's entangled equal-weight latents. PCA's global variance is a
   feature here, not the liability the hypothesis assumed.

3. **At low dim (32) the AE beats PCA** (AE-32 0.0545 > PCA-32 0.0412): the
   nonlinear bottleneck squeezes more usable structure into 32 dims than the
   linear 92.3%-EVR projection. So the AE's advantage is a *low-capacity*
   phenomenon; it evaporates and reverses once dim is generous. Both 32-dim
   spaces are poor.

4. **Content-kNN divergence — FLAGGED.** The training-free probe (pure cosine
   retrieval in each space, max-agg) is flat across representations (~0.06) and
   does NOT track the GRU ranking: PCA-128's probe R@10 (0.0636) exactly *ties*
   the AE-64 control (0.0636), and AE-32 tops the probe (0.0720) despite being
   near-worst under the GRU. So the intrinsic retrieval richness of these spaces
   is similar; the GRU differences come from the *learnability / predictability*
   of the target geometry, not from raw retrieval richness. Per the decision
   rule this divergence is flagged, not auto-promoted — PCA-128 is a better space
   *for the trained GRU to predict into*, not a demonstrably richer space in the
   absolute. (Probe cross-check: content-kNN on AE-64 reproduces the on-record
   0.0636 exactly.)

5. **The single-GRU win does NOT propagate to the blend.** Despite the PCA-128
   GRU leg being clearly stronger than the AE-64 GRU leg (0.1146 vs 0.096), the
   R+M blends are statistically indistinguishable (0.167 vs 0.173, overlapping
   CIs, PCA-128 slightly lower). The Markov leg is representation-independent
   (item co-occurrence, not latents); the blend's power is signal *diversity*
   between latent-similarity and co-occurrence. The extra tracks the stronger
   PCA-128 GRU gets right evidently overlap more with what Markov already
   supplies, so a better GRU space buys little at the blend level. **Key Phase-2
   caution: representation gains measured on the single GRU do not automatically
   translate to the production blend.**

Net: the compression axis is essentially spent for the *blend* (no board event),
but PCA-128 is a genuine improvement for the *single GRU* and is the sanctioned
Phase-2 base space. The improvement is dimensionality-driven and should be read
as "give the GRU a higher-dim, variance-ordered target", not "PCA is magic".

## Best on record after this work (RANKING board — next-distinct, leak-free split)

Champion unchanged. New/changed rows in **bold**.

| model | Recall@10 [95% CI] | MRR | artist@10 / genre@10 | note |
|---|---|---|---|---|
| ★ R+M z-blend α=0.5 on **AE-64** (CHAMPION) | 0.173 [0.154, 0.192] | 0.096 | 0.322 / 0.397 | unchanged; the bar to beat; 3-seed confirmed |
| — R+M z-blend on **PCA-128** | 0.167 [0.148, 0.187] | 0.092 | 0.329 / 0.398 | blend on the new winner space; ≈ champion (overlapping CI, slightly lower) — NOT a board event |
| **GRU infonce h256 on PCA-128 (best SINGLE model)** | **0.1146 [0.099, 0.132]** | **0.049** | 0.333 / 0.452 | **new best single model; beats AE-64 GRU, paired-Δ +0.019 CI>0; the Phase-2 base space** |
| GRU infonce h256 on PCA-64 | 0.1041 [0.088, 0.121] | 0.042 | 0.312 / 0.460 | ties control by CI (Δ straddles 0) |
| GRU infonce h256 on AE-64 (prior best single) | 0.0957 [0.081, 0.111] | 0.041 | 0.294 / 0.450 | the control; reproduced exactly |
| GRU infonce h256 on AE-128 | 0.0929 [0.078, 0.108] | 0.040 | 0.303 / **0.464** | ties control; best genre@10 |
| B2 first-order Markov (bar) | 0.107 [0.092, 0.124] | 0.069 | 0.298 / 0.389 | app incumbent |
| GRU infonce h256 on AE-32 | 0.0545 [0.043, 0.066] | 0.027 | 0.238 / 0.396 | 32-dim starves the space |
| GRU infonce h256 on PCA-32 | 0.0412 [0.031, 0.052] | 0.022 | 0.219 / 0.382 | weakest; PCA-32 worst space |

Deliverable to Phase 2: **winning representation = (`spotify_tracks_song_pca128`,
`seq-20260715-030509`)**; runner-up PCA-64 (`spotify_tracks_song_pca64`,
`seq-20260715-030503`). Nothing promoted (Phase 1 promotes nothing absent a
3-seed-confirmed board event; ranking predictors are train-only and not
best-models-promotable regardless).

## Follow-ups

- **Phase 2 runs on PCA-128** (`seq-20260715-030509`). But re-baseline the *blend*
  on PCA-128 early — the single-GRU gain did not carry into the R+M blend, so
  confirm whether any Phase-2 model actually lifts the blend before treating
  PCA-128 as a win at the product level.
- Try **dim > 128** (PCA-192/256): PCA R@10 is still climbing at 128 (0.104→0.115
  from 64→128) — the dimensionality ceiling is not yet found for the single GRU.
- The content-kNN/GRU divergence suggests the GRU is limited by target
  *learnability*, not space richness — worth a short InfoNCE-temperature /
  negatives scan on PCA-128 (the deferred tuning follow-up) now that the space is
  chosen.
- Compression axis for the *blend* is documented spent (no board event); do not
  re-open method×dim for the blend without a new mechanism.

Prerequisite builds: `pipeline/build_song_ae.py` (AE-32/128, `--dst`/`--artifact-prefix`),
`pipeline/build_song_pca.py` (PCA-32/64/128), `pipeline.corpus.sequences`
(`--latent-collection`/`--latent-dim`). Each new seq artifact was produced under
`data/seq/` and registered into `data/datasets/` (this instance's established
sequence-dataset path — no API exists to ingest an externally-produced sequence
artifact; the control is present in both dirs identically).
