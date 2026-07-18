# Next-track Phase-1b: compression dim extension (PCA-192/256, AE-192) — 2026-07-15

Follow-on to the bake-off (`2026-07-15-nexttrack-compression-representation-bakeoff.md`),
which found (a) latent DIM dominates method and (b) PCA R@10 was still climbing
at 128. This extends the sweep upward to locate the real optimum before Phase 2
commits a base space: **PCA-192, PCA-256** (does the PCA curve keep climbing or
peak?) and **AE-192** (does PCA ≥ AE hold at higher dim?). Same protocol: all
reps fit from the leak-safe `spotify_tracks_content` source; frozen GRU readout
`gru-infonce-h256`; each new seq artifact passed the exact parity gate
(7154 / 19402 / 5723 / 1431 @ cut 2024-08-24T14:28:02); bootstrap 95% CI = 2,000
resamples over the 1,431 test sessions.

**Reference = PCA-128** (`spotify_tracks_song_pca128`, `seq-20260715-030509`,
run `run-20260715-042100-a1d9d`; R@10 0.1146 [0.099, 0.132], MRR 0.0489), the
current Phase-1 winner. Paired per-session Δ is measured against it (its and the
control's existing `predictions.json` reused). GRU-single R@10 is the selection
yardstick; no blend/promotion in this leg.

## Outcome (one line)

**The PCA dim curve PEAKS at ~192 (R@10 0.123) and turns over by 256 (0.117) — it was not still climbing; PCA-192 is the point-optimum but misses the paired-Δ-CI-entirely-positive bar by a whisker (Δ +0.0084 [−0.0007, +0.0189]), so by the mechanical rule PCA-128 STANDS as the Phase-2 base, with PCA-192 flagged as the confirm-and-promote candidate. PCA ≥ AE holds strongly at 192 (0.123 vs 0.098).**

## Results — GRU-single (frozen `gru-infonce-h256`), paired-Δ vs PCA-128

| config | run_id | R@10 [95% CI] | paired-Δ vs PCA-128 | MRR | artist@10 | genre@10 | content-kNN R@10 |
|---|---|---|---|---|---|---|---|
| **PCA-192 (point-optimum)** | run-20260715-131240-1bbdb | **0.1230 [0.1069, 0.1405]** | +0.0084 [−0.0007, +0.0189] tie (whisker) | **0.0534** | **0.3375** | 0.4500 | 0.0685 |
| PCA-256 | run-20260715-131240-b4d6e | 0.1167 [0.1006, 0.1335] | +0.0021 [−0.0084, +0.0133] tie | 0.0502 | **0.3375** | **0.4584** | **0.0699** |
| PCA-128 (REFERENCE, prior winner) | run-20260715-042100-a1d9d | 0.1146 [0.0985, 0.1321] | — (reference) | 0.0489 | 0.3326 | 0.4521 | 0.0636 |
| PCA-64 | run-20260715-042100-8851d | 0.1041 [0.0881, 0.1209] | −0.0105 [−0.0245, +0.0014] tie | 0.0416 | 0.3117 | 0.4598 | 0.0580 |
| AE-192 | run-20260715-131240-7fdb2 | 0.0978 [0.0825, 0.1132] | −0.0168 [−0.0294, −0.0042] *<0* | 0.0401 | 0.3054 | 0.4577 | 0.0657 |
| AE-64 (control) | run-20260715-042059-dd261 | 0.0957 [0.0811, 0.1111] | −0.0189 [−0.0335, −0.0042] *<0* | 0.0411 | 0.2942 | 0.4500 | 0.0636 |

All 3 new runs `succeeded` (exit 0, n_test=1431). Best cell per metric bolded.
AE-192 build R²: text 0.520, numeric 0.999, acoustic 0.999, categorical 0.953;
PCA cum EVR — 192: 0.9922, 256: 0.9973.

## Decision (mechanical rule)

Rule: *new Phase-2 base = the highest-dim rep that beats PCA-128 on R@10 with
paired-Δ CI entirely >0 AND no MRR loss; else PCA-128 stands.*

- PCA-192: MRR gain (0.0534 > 0.0489) ✓, but R@10 paired-Δ CI = [−0.0007, +0.0189]
  **includes 0** (by 0.0007) → does NOT clear "entirely >0".
- PCA-256: R@10 paired-Δ CI [−0.0084, +0.0133] straddles 0 → fails.
- All others tie or lose.

**No rep clears the bar → PCA-128 STANDS as the Phase-2 base**
(`spotify_tracks_song_pca128`, `seq-20260715-030509`). PCA-192
(`spotify_tracks_song_pca192`, `seq-20260715-131139`) is the **point-optimum**
and a near-miss (a whisker inside 0 on the lower CI bound); it is the natural
confirm-and-promote candidate — a confirmatory re-run reproducing the +0.008 edge
would move the base to PCA-192. Nothing promoted here (GRU-single yardstick; no
blend leg; ranking predictors are train-only anyway).

## Findings

1. **The PCA R@10-vs-dim curve PEAKS at ~192, then declines.** Full trend:
   32 → 0.041, 64 → 0.104, 128 → 0.115, **192 → 0.123**, 256 → 0.117. The steep
   gains are done by 64; 128→192 adds a little; 256 gives it back. So the earlier
   "still climbing at 128" reading resolves to a **peak/plateau at 128–192** with
   a turn-over by 256 — extra variance past ~192 (PCA-256 EVR 0.997) is
   low-signal / noise dimensions the GRU cannot exploit and that slightly dilute
   the target. **The dim ceiling is found: ~192.** MRR tracks the same shape
   (peak 0.0534 at 192).

2. **PCA ≥ AE holds — and the gap is widest at 192** (PCA-192 0.123 vs AE-192
   0.098, paired-Δ of AE-192 vs PCA-128 is −0.017, CI entirely <0). AE plateaus
   at ~0.09–0.10 from dim 64 upward (64 → 0.096, 128 → 0.093, 192 → 0.098) —
   adding AE capacity does essentially nothing, because the AE's equal-weight
   reconstruction objective keeps spending capacity on the tiny numeric/acoustic
   blocks (R² already 0.999 at 192) rather than the high-variance text/catalog
   structure the GRU needs. The bake-off's refutation of "AE protects the small
   blocks → better retrieval" is reconfirmed at higher dim.

3. **Content-kNN probe still ⟂ the GRU ranking.** The training-free probe stays
   flat (~0.064–0.070 across all high-dim reps) and does not reproduce the GRU's
   peak at 192 — PCA-256 even edges PCA-192 on the probe (0.0699 vs 0.0685) while
   losing on the GRU. Consistent with the bake-off: the GRU dim-optimum is about
   target *learnability/predictability*, not intrinsic retrieval richness of the
   space. Divergence flagged, as before.

4. **Same "does not carry to the blend" caveat applies** (not tested here): the
   PCA-192 single-GRU edge over PCA-128 (+0.008) is smaller than the noise the
   bake-off already showed the R+M blend absorbs, so a blend re-baseline on the
   chosen space (128 or 192) remains a Phase-2 prerequisite before treating any
   dim gain as a product win.

## Best on record after this work (single-model rows only; champion unchanged)

Blend champion (R+M z-blend on AE-64, R@10 0.173) unchanged. Best-single-model
row updated. New/changed rows in **bold**.

| model | Recall@10 [95% CI] | MRR | artist@10 / genre@10 | note |
|---|---|---|---|---|
| **GRU infonce h256 on PCA-192 (point-best single)** | **0.123** [0.107, 0.141] | **0.053** | 0.338 / 0.450 | peak of the dim curve; +0.008 over PCA-128 but Δ-CI grazes 0 (not significant) — confirm-candidate for Phase-2 base |
| GRU infonce h256 on PCA-256 | 0.117 [0.101, 0.134] | 0.050 | 0.338 / 0.458 | past the peak; ties PCA-128 |
| GRU infonce h256 on PCA-128 (Phase-2 base) | 0.115 [0.099, 0.132] | 0.049 | 0.333 / 0.452 | the mechanically-confirmed Phase-2 base space |
| GRU infonce h256 on AE-192 | 0.098 [0.083, 0.113] | 0.040 | 0.305 / 0.458 | AE plateau; loses to PCA-128 (Δ-CI <0) |

Phase-2 base deliverable: **PCA-128** (`spotify_tracks_song_pca128`,
`seq-20260715-030509`); **PCA-192** (`spotify_tracks_song_pca192`,
`seq-20260715-131139`) is the point-optimum / confirm-and-promote candidate.

## Follow-ups

- **Confirm PCA-192 vs PCA-128** before Phase 2 hard-commits: the +0.008 edge is a
  whisker from significance. `seq-nexttrack` has no seed, but a couple of repeat
  runs (or a paired eval on a re-split) would settle whether PCA-192 clears the
  bar; if it reproduces, move the base to PCA-192.
- **Dim ceiling is ~192** — do not sweep higher (256 already regresses).
- Phase 2 caveat stands: re-baseline the R+M blend on the chosen space before
  reading any single-GRU dim gain as a product win.
