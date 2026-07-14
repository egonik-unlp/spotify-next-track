# PROJECT FACTS — Spotify Next-Track (session recommendation)

Living, agent-maintained roll-up of empirical knowledge for the **next-track /
session-recommendation** instance. The `experiments/*.md` campaign reports are
PRIMARY; this file indexes them. Newest report wins.

> Lineage: cloned from `spotify-predict-engagement` (the rotation/taste-fit lab)
> on 2026-07-13 and retargeted to a SEQUENCE / next-item ranking goal. The
> rotation-dated `experiments/*.md` files carried over are INHERITED parent
> lineage (kept for reference; the full parent roll-up is preserved as
> `INHERITED-rotation-PROJECT-FACTS.md`). They describe a different target
> (rotation, AUC) and do not apply here. This instance's record starts below.

## Target & task

Predict, within a listening session, **which track the user plays next** — a
next-item RETRIEVAL task. The model consumes the ordered session prefix and
predicts the next track's 64-dim song-AE latent; candidates are retrieved by
cosine over the corpus. `domain.toml`: `task = "ranking"`, `[sequence]`
(session_id / ts / 30-min gap / 30s skip threshold / last-item label /
latent_source = `spotify_tracks_song_ae`), `[metrics].primary = "recall@10"`.
Ranked by Recall@k / MRR / hit-rate on a leak-free session split. Does NOT share
the parent's rotation-AUC leaderboard.

## Framework capability added here (build-concrete-first; upstream later)

Additive sequence/ranking support in this instance's `crates/` (Phase 2):
ranking metrics (`Metrics.recall_at_k/mrr/hit_rate` + `MetricsSpec` maps +
`compute_ranking_metrics`), `SequenceDataset` (parallel artifact family, pointwise
`Dataset` untouched), `Task::Ranking` + `[sequence]` domain block, the
`pipeline/corpus/sequences.py` offline producer, and the `seq-nexttrack` torch GRU
predictor + baselines. To be generalized + upstreamed to the mother lensing
framework (Phase 5).

## Best on record (RANKING board — next-distinct, leak-free session split)

Artifact `data/seq/seq-20260713-014226` (7,154 sessions / 19,402 items;
chronological split 5,723 train / 1,431 test @ cut 2024-08-24; cold-item rate
0.55). Source: `experiments/2026-07-13-nexttrack-sequence-sweep.md`.

| model | Recall@10 [95% CI] | MRR | note |
|---|---|---|---|
| **★ R+M z-blend α=0.5 (CHAMPION)** | **0.173** [0.154, 0.192] | **0.096** | GRU-infonce × train-only Markov; BEATS the bar, paired-Δ +0.066 [+0.049,+0.083], 3-seed confirmed. `seq-blend` / `blend-gru-markov` |
| B2 first-order Markov (bar) | 0.107 [0.092, 0.124] | 0.069 | `transit.rs::affinity`, train-only; the app's incumbent |
| — R+A+M z-blend | 0.168 | 0.094 | adding the ANN leg (redundant w/ GRU) slightly HURTS the champion |
| — XGB stacker M+R / M+R+A | 0.102 / 0.089 | 0.056 / 0.057 | learned meta-learner UNDERPERFORMS the z-blend (barely clears Markov) |
| GRU infonce **h256** (best single model) | 0.096 [0.080, 0.110] | 0.041 | width; ties B2 on R@10, loses MRR |
| GRU infonce depth-2 h128 | 0.092 [0.076, 0.106] | 0.040 | depth flat vs 1-layer |
| GRU infonce h128 (1-layer) | 0.089 [0.076, 0.104] | 0.039 | sweep's infonce arm, reproduced |
| GRU infonce depth-3 h128 | 0.083 [0.069, 0.096] | 0.037 | depth-3 HURTS (loses to bar) |
| GRU infonce bidirectional h128 | 0.048 [0.037, 0.059] | 0.023 | leak-free last-item objective (~13× fewer train targets); loses |
| GRU h128 cosine (1-layer) | 0.039 | 0.016 | cosine ≈ 0.4× infonce |
| B0 popularity / B1 recency | 0.001 | — | floors |

Read: **the GRU×Markov z-blend is the CHAMPION and the first model to beat the
bar** (R@10 0.107→0.173, MRR 0.069→0.096, paired-Δ CI entirely >0, 3-seed
confirmed) — because the two legs rank *different* correct next-tracks (latent
similarity vs co-occurrence counts), fusing them ~doubles hit rate. **Single
models never beat the bar** across the loss + topology sweeps (InfoNCE ~2.3×
cosine is the only mover; topology flat/spent; latent-retrieval caps ~0.09–0.10).
**Further ensembling does NOT beat the blend** (`2026-07-14-nexttrack-blend-and-ensemble.md`):
the ANN leg is redundant with the GRU (R+A+M 0.168 ≈ champion; R+A ≈ GRU alone),
and the XGB stacker UNDERPERFORMS the simple z-blend (0.102/0.089 — pointwise
training doesn't capture the ranking objective). The win is signal DIVERSITY not
model count; the ceiling is now the 55% cold-item rate + the need for GENUINELY
NEW signal (content/acoustic features, longer/cross-session context, a different
candidate source), not more fusion. PRODUCTIZATION GAP: ranking predictors are
train-only (no `predict`) → the champion isn't best-models-promotable and can't
export to the app yet; adding a `predict` subcommand unlocks both (the blend maps
onto the app's A* W_TRANS slot). Sources: `2026-07-14-nexttrack-blend-and-ensemble.md`,
`2026-07-13-nexttrack-topology-scan.md`.

## Noise / significance

No pre-registered band for these retrieval metrics; every claim uses a bootstrap
95% CI over test sessions (2,000 resamples) + a PAIRED per-session Δ vs the B2 bar
(the decision statistic). A WIN needs beating B2 on BOTH Recall@10 AND MRR beyond
non-overlapping CIs, then a 3-seed confirm before any deploy.

## Pitfalls / infra

- **Wrong-collection trap:** sequence latents MUST come from
  `spotify_tracks_song_ae` (64-dim song-AE), never the default `spotify_tracks`
  (200-dim behavioral co-listening embedding — soft-leaks replay).
- **Cold items dominate forward eval:** 55% of test targets unseen in train —
  absolute recall is modest by construction; judge vs the B2 bar, not intuition.
- **Runtime:** own port **8100** + own Postgres DB **`nexttrack`** in the shared
  container; `LENSING_DISABLE_PATHFINDER=1`; reads the SHARED corpus Qdrant
  (6335). Never restart the parent's 8096.

## Follow-ups
- Blend GRU-infonce candidate scores with the Markov bigram (matches the app's A*
  transition-term slot; likely beats either alone).
- InfoNCE tuning scan (temperature / negatives / longer training).
- Phase 4: "extend-a-session" showcase demo in `../app`.
- Phase 5: generalize + upstream the sequence/ranking framework support.
