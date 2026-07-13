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
| **B2 first-order Markov (bar)** | **0.107** [0.092, 0.124] | **0.069** | `transit.rs::affinity`, train-only; the app's incumbent |
| GRU infonce **h256** (topology best) | 0.096 [0.080, 0.110] | 0.041 | width; TIES B2 on R@10 (paired-Δ CI incl. 0), loses MRR |
| GRU infonce depth-2 h128 | 0.092 [0.076, 0.106] | 0.040 | depth flat vs 1-layer |
| GRU infonce h128 (1-layer) | 0.089 [0.076, 0.104] | 0.039 | sweep's infonce arm, reproduced |
| GRU infonce depth-3 h128 | 0.083 [0.069, 0.096] | 0.037 | depth-3 HURTS (loses to bar) |
| GRU infonce bidirectional h128 | 0.048 [0.037, 0.059] | 0.023 | leak-free last-item objective (~13× fewer train targets); loses |
| GRU h128 cosine (1-layer) | 0.039 | 0.016 | cosine ≈ 0.4× infonce |
| B0 popularity / B1 recency | 0.001 | — | floors |

Read: **no WIN — first-order co-listening transitions hold, across BOTH loss and
topology sweeps.** InfoNCE is the dominant lever (~2.3× cosine); **network
topology is flat/spent** — width (h256) is the only marginal mover (R@10 0.096,
still a tie vs the bar and losing on MRR), depth is flat then HURTS at 3 layers,
residual adds nothing, and bidirectional collapses (mostly an objective artifact:
the leak-free last-item objective starves training ~13×). The transferable
latent-retrieval signal caps ~R@10 0.09–0.10 regardless of architecture. The
GRU's value is a COMPLEMENT (candidate generation / "extend-a-session" demo), not
a replacement. The only unexplored lever is the training objective / candidate
framing — esp. **blending GRU-infonce candidate scores WITH the Markov bigram**
(the app's A* transition slot) — the most promising path to actually beat it.
Source: `experiments/2026-07-13-nexttrack-topology-scan.md`.

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
