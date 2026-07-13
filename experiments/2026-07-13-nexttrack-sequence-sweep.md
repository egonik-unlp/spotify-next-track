# Next-track sequence model vs the first-order Markov bar — 2026-07-13

First experiment of the `spotify-next-track` instance. Question: does a
session-context sequence model (GRU/LSTM over the ordered prefix, predicting the
next track's 64-dim song-AE latent, retrieved by cosine) beat the first-order
Markov transition model the sibling app already ships — **forward**, on a
leak-free session split?

## Setup

- **Artifact:** `data/seq/seq-20260713-014226` (produced by `pipeline/corpus/sequences.py`
  from `sessions.parquet` + `id_map.json` + the `spotify_tracks_song_ae` 64-dim
  latents). **7,154 sessions / 19,402 items**; chronological-by-session-start
  split 5,723 train / 1,431 test (cut 2024-08-24). **Cold-item rate 0.55** — 55%
  of test-session target tracks never appear in any training session (the
  leak-free forward reality; absolute recall is modest by construction).
- **Eval:** leave-last-out per test session — prefix = all but last, truth =
  last item; rank the full vocab by cosine to the predicted latent, exclude
  prefix items (next-distinct / Markov-parity). Metrics: Recall@10, Recall@20,
  MRR, hit@10, each with a bootstrap 95% CI (2,000 resamples) + a **paired**
  per-session Δ vs the B2 bar.
- **Bar (B2):** first-order Markov — `track_bigram` + artist/genre back-off
  (`transit.rs::affinity`, backoff_k=8/artist 0.6/genre 0.4), rebuilt on TRAIN
  sessions only. Floors: B0 popularity, B1 recency.

## Results

| arm | R@10 [95% CI] | R@20 | MRR | paired Δ vs B2 [95% CI] |
|---|---|---|---|---|
| **B2 Markov (bar)** | **0.1069** [0.092, 0.124] | 0.1516 | **0.0694** | — |
| M3 gru h128 **infonce** | 0.0894 [0.076, 0.104] | 0.1335 | 0.0388 | −0.0175 [−0.037, +0.002] → **TIE** |
| M2 gru h256 cosine | 0.0412 [0.031, 0.051] | 0.0643 | 0.0199 | −0.066 [−0.084, −0.047] loses |
| M1 gru h128 cosine | 0.0391 [0.029, 0.049] | 0.0545 | 0.0162 | −0.068 [−0.085, −0.051] loses |
| M4 lstm h128 cosine | 0.0377 [0.029, 0.048] | 0.0531 | 0.0142 | −0.069 [−0.087, −0.050] loses |
| B0 popularity / B1 recency | 0.0007 | 0.0014 | 0.0008 | −0.106 loses |

## Verdict — NO WIN; first-order transitions hold (a TIE at best)

Decision rule required beating B2 on **both** Recall@10 and MRR beyond
non-overlapping CIs. The best arm (M3, InfoNCE) reaches a **statistical tie with
the Markov bar on Recall@10** (paired-Δ CI includes 0) but **loses on MRR**
(0.039 vs 0.069) and on Recall@20 (0.133 vs 0.152). So: not beaten → **no 3-seed
confirm, no deploy.** The bigram ranks the exact next track at least as well.

Reads:
1. **Loss objective is the dominant lever** — InfoNCE ≈ **2.3× cosine** on
   Recall@10 (0.089 vs 0.039); capacity (h256) and cell (LSTM) barely moved the
   cosine arms. Any further sequence work should start from InfoNCE.
2. **A latent + single cosine hop is a harder route to the *exact* next track
   than a co-listening bigram.** Co-listening adjacency directly memorizes "what
   follows what"; the GRU must instead land near the right point in a 64-d
   content space, then retrieve — losing on precise ranking (MRR).
3. **Consistent with the parent instance's taste-drift finding** — 55% cold
   items, identity/adjacency-driven behaviour; content-latent generalization
   only reaches parity on top-10 recall.
4. **Product implication:** the app keeps its Markov `W_TRANS` term. The GRU's
   value is as a **complement** — candidate generation / an "extend-this-seed
   into a session" demo (Phase 4), where top-10 parity + content-space coherence
   matter more than exact-next MRR — not a replacement.

## Follow-ups
- **Blend** GRU-infonce candidate scores with the Markov bigram (the app's A*
  already normalizes a transition term) — the natural next arm; a blend could
  plausibly beat either alone.
- InfoNCE tuning (temperature, negatives, longer training) — the only lever that
  moved the needle; a focused scan is warranted before concluding the family.
- The "extend-a-session" demo (Phase 4) is viable now on M3.

## Provenance
Off-server sweep on the sequence artifact (`predictors/seq_nexttrack.py` +
`seq_baselines.py`, torch 2.12 cpu). Arms M1–M4 at 60 epochs / patience 8; B0/B1/B2
via `seq_baselines.py`. CI script: bootstrap over test sessions (recall@10 marginal
+ paired Δ vs B2). No model promoted; the sequence result is on this instance's
own ranking board, separate from the parent's rotation-AUC leaderboard.
