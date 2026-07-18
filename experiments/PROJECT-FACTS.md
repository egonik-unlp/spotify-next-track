# PROJECT FACTS — Spotify Next-Track (session recommendation)

Living, agent-maintained roll-up of empirical knowledge for the **next-track /
session-recommendation** instance. The `experiments/*.md` campaign reports are
PRIMARY; this file indexes them. Newest report wins.

_Last updated: 2026-07-15 after `2026-07-15c-nexttrack-phase2-model-sweep.md` (incl. the R+M+C-on-PCA-192 confirmation / crown)._

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

**Graded-relevance metrics (first-class since 2026-07-14):** alongside exact
recall@k / mrr / hit@k, the eval credits landing the right band/vibe —
**artist@k** (some top-k candidate shares the truth's artist), **genre@k**, and
**artist_mrr** — over the same prefix-excluded ranking. `[metrics].columns =
[recall@10, artist@10, genre@10, mrr, hit@10]`; wired through
`seq_common.eval_from_scores` → Rust `Metrics`/`MetricsSpec` → `domain.toml` → UI.
These are the HONEST scorecard: exact recall is modest by construction (55% cold
items), but the champion lands the right **artist ~32%** / **genre ~40%** of the
time in the top-10 (artist_mrr ~0.30) — far more useful than the 17% exact number.
Source: `2026-07-14-nexttrack-graded-relevance-and-new-legs.md`.

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
0.55). Source: `experiments/2026-07-13-nexttrack-sequence-sweep.md`. The CHAMPION ROW
(best on record) is now R+M+C on the rich PCA-192 space (confirmed 5×); the prior
champion / productization anchor is R+M on AE-64. Sources:
`2026-07-15-...bakeoff`, `2026-07-15b-...dim-extension`, `2026-07-15c-...phase2-model-sweep`.

| model | Recall@10 [95% CI] | MRR | artist@10 / genre@10 | note |
|---|---|---|---|---|
| **★ R+M+C z-blend on PCA-192 (CHAMPION ROW — best on record)** | **0.186** [0.166, 0.207] | **0.103** | 0.331 / 0.394 | GRU×Markov×content-kNN thirds; CONFIRMED 5× (GRU-leg reproductions, spread 0), paired-Δ vs prior champion **+0.013 [+0.0007,+0.026] every rerun** + MRR lead. Train-only + off-server (`analyze_rmc.py`) → best-on-record ROW, NOT a promoted model. Caveats: reproduction spread 0 (seq-nexttrack deterministic → tested determinism not seed-robustness); paired-Δ lower bound +0.0007 (thin, single split). `seq-20260715-131139`; `2026-07-15c` |
| — R+M+C z-blend on PCA-128 | 0.182 [0.162, 0.203] | 0.099 | 0.332 / 0.396 | content additive on the rich space (+0.015 [+0.004,+0.028] over R+M-128); paired-Δ vs champion straddles 0 |
| — R+M z-blend α=0.5 on AE-64 (prior champion / PRODUCTIZATION ANCHOR) | 0.173 [0.154, 0.192] | 0.096 | 0.322 / 0.397 | GRU-infonce × train-only Markov, 3-seed confirmed; the 2-leg blend that maps onto the app's transition slot — remains the deployable anchor until the 3-leg R+M+C is registered. `seq-blend` / `blend-gru-markov` |
| — R+M+C (+content-kNN) on AE-64 (graded-preferred) | 0.174 [0.154, 0.194] | 0.101 | **0.330** / 0.400 | on AE-64 the content leg only TIED the champion on exact recall (paired-Δ +0.0014) — the rich PCA space makes it pay off (see PCA-192 R+M+C) |
| — R+M z-blend on PCA-192 | 0.171 [0.151, 0.192] | 0.095 | 0.329 / 0.398 | ≈ champion (tie, paired-Δ −0.0014); rich-space R+M. `run-...96f53` |
| — R+M z-blend on PCA-128 | 0.167 [0.148, 0.187] | 0.092 | 0.329 / 0.398 | ≈ champion (tie); the single-GRU gain does NOT propagate to the R+M blend |
| B2 first-order Markov (bar) | 0.107 [0.092, 0.124] | 0.069 | 0.298 / 0.389 | `transit.rs::affinity`, train-only; the app's incumbent |
| — R+A+M z-blend | 0.168 | 0.094 | — | adding the ANN leg (redundant w/ GRU) slightly HURTS the champion |
| — R+M+I (+item2vec) | 0.135 | 0.087 | 0.327 / 0.399 | co-listening item2vec is a noisy low-ceiling leg — DRAGS the blend down |
| — R+M2 / R+M+M2 (2nd-order Markov) | 0.173 | 0.099 / 0.098 | 0.322 / 0.321 | trigram ≈ neutral |
| **GRU LSTM on PCA-128 (best SINGLE model)** | **0.127** [0.110, 0.145] | 0.050 | 0.335 / 0.458 | LSTM > GRU on PCA-128 (paired-Δ +0.013 [+0.0007,+0.024] CI>0); new best single, still ≪ the blend. `run-...546d0` |
| GRU infonce h256 on PCA-192 | 0.123 [0.107, 0.140] | 0.053 | 0.338 / 0.450 | reproduced twice (15b + 15c confirm); paired-Δ vs PCA-128 grazes 0 |
| GRU infonce h256 on PCA-128 (Phase-2 base) | 0.115 [0.099, 0.132] | 0.049 | 0.333 / 0.452 | the confirmed Phase-2 base space; `spotify_tracks_song_pca128` / `seq-20260715-030509` |
| GRU infonce h256 on PCA-256 | 0.117 [0.101, 0.134] | 0.050 | 0.338 / 0.458 | past the dim peak |
| GRU infonce h256 on PCA-64 | 0.104 [0.088, 0.121] | 0.042 | 0.312 / 0.460 | ties AE-64 control |
| GRU infonce h256 on AE-192 | 0.098 [0.083, 0.113] | 0.040 | 0.305 / 0.458 | AE plateau; PCA≥AE gap widest at 192 |
| GRU infonce h256 on AE-64 (control) | 0.096 [0.080, 0.110] | 0.041 | 0.294 / 0.450 | bake-off control; reproduced exactly |
| GRU infonce h256 on AE-128 | 0.093 [0.078, 0.108] | 0.040 | 0.303 / 0.464 | ties AE-64 control |
| ANN (pooled MLP) on PCA-128 | 0.090 [0.076, 0.105] | 0.039 | 0.299 / 0.435 | non-recurrent pooled MLP; loses to the GRU (paired-Δ <0) |
| XGB stacker (M+G / M+G+A) on PCA-128/192 | 0.080–0.087 | 0.042–0.049 | 0.31 / 0.38 | learned combiner UNDERPERFORMS the z-blend AND its own GRU base leg — reconfirmed on the rich space |
| GRU infonce h256 on AE-32 / PCA-32 | 0.055 / 0.041 | 0.027 / 0.022 | — | 32-dim starves the space |
| GRU infonce depth-2/3, bidir, cosine | 0.039–0.092 | — | — | topology sweep; all ≤ h256 (see topology-scan) |
| content-kNN (song-AE cosine) | 0.064 [0.050,0.077] | 0.032 | 0.272 / 0.405 | weak standalone; the additive C leg (see R+M+C) |
| item2vec (train-only SGNS) | 0.052 | 0.021 | 0.166 / 0.312 | co-listening too sparse; weakest leg |
| B0 popularity / B1 recency | 0.001 | — | — | floors |

Read: **R+M+C (GRU×Markov×content-kNN) on PCA-192 is the CHAMPION ROW / best on
record** (R@10 0.186, MRR 0.103, confirmed 5×, paired-Δ vs the prior champion
+0.013 CI>0) — a best-on-record ROW, not a promoted model. **The prior champion
R+M z-blend on AE-64** (0.173/0.096, 3-seed confirmed) remains the productization
anchor (the 2-leg blend that maps to the app). **Single models never beat
the bar**; the representation sweeps raised the single-model ceiling to ~0.12–0.13
(PCA dim ~128–192; LSTM edges GRU) but singles stay ≪ the blend. **Combiners:
the FIXED z-blend is the winner — the learned XGB stacker UNDERPERFORMS it badly**
(0.08–0.09, loses even to its GRU base leg; reconfirmed on the rich PCA space in
`2026-07-15c`), and adding redundant legs (ANN, item2vec) does not help. The one
genuinely additive signal is CONTENT: **R+M+C (GRU×Markov×content-kNN) — which
only TIED the champion on AE-64 — on the rich PCA-192 space reaches R@10 0.186 /
MRR 0.103, the strongest on record, paired-Δ-beating the champion on R@10
(+0.013 CI>0)**. It cleared the paired-Δ statistic (the one that crowned the prior champion) but
NOT the stricter non-overlapping-marginal-CI bar (its CI overlaps the prior
champion); the authorized 5× GRU-leg-rerun confirm settled it (spread 0, paired-Δ
>0 every rerun) → crowned as the new champion ROW.
PRODUCTIZATION GAP: ranking predictors are train-only (no `predict`) and R+M+C is
an off-server driver → nothing is best-models-promotable; a confirmed win = a new
champion ROW. Sources: `2026-07-15c-nexttrack-phase2-model-sweep.md`,
`2026-07-14-nexttrack-blend-and-ensemble.md`, `2026-07-13-nexttrack-topology-scan.md`.

## Item-representation / compression axis + Phase-2 model family (2026-07-15)

Sweeps of the item latent space (AE vs PCA × dim 32/64/128/192/256) and the model
family on the chosen spaces, all fit from the leak-safe `spotify_tracks_content`
source, frozen `gru-infonce-h256` readout (representation legs) / frozen held
config (model-family legs). Sources: `2026-07-15-...bakeoff`,
`2026-07-15b-...dim-extension`, `2026-07-15c-...phase2-model-sweep`.

- **Latent DIM dominates method and PEAKS at ~192.** PCA R@10 vs dim:
  32→0.041, 64→0.104, 128→0.115, 192→0.123 (peak), 256→0.117 (turns over). Dim
  ceiling ≈ 192; do not sweep higher.
- **Linear PCA beats the nonlinear AE at every dim ≥64, gap widest at 192**
  (PCA-192 0.123 vs AE-192 0.098). "AE protects small acoustic/catalog blocks →
  better retrieval" is REFUTED and reconfirmed at 192 (AE preserves num/acoustic
  at R²≈0.999 yet plateaus ~0.09–0.10). The GRU learns from high-variance
  text/catalog structure + a decorrelated, variance-ordered target geometry.
- **Phase-2 base = PCA-128** (`spotify_tracks_song_pca128` / `seq-20260715-030509`).
  PCA-192 GRU (`..._pca192` / `seq-20260715-131139`) reproduced its point-edge 5×
  (0.123, spread 0) but its paired-Δ vs PCA-128 grazes 0 (+0.008 [−0.001,+0.019])
  every time — inside noise, so PCA-128 STAYS the base. (PCA-192 is the space the
  CHAMPION ROW R+M+C is measured on, but the single-GRU base for model work is
  PCA-128.)
- **Model family (Phase 2, on PCA-128/192):** LSTM > GRU as the best SINGLE model
  (PCA-128 LSTM 0.127 vs GRU 0.115, paired-Δ +0.013 CI>0); ANN (pooled MLP) loses
  to the GRU; the **learned XGB stacker badly underperforms the fixed z-blend**
  (0.08–0.09) — pointwise training doesn't capture the ranking objective. The
  fixed z-blend is the combiner of record.
- **Content is additive on the RICH space.** R+M+C beats same-space R+M by +0.015
  (PCA-128) / +0.015 (PCA-192), CI>0 both; R+M+C-192 (0.186/0.103) is strongest
  on record and paired-Δ-beats the champion on R@10 — the graded-preferred,
  content-augmented blend. The 2026-07-14 lesson ("single-model/representation
  gains don't carry to the blend") holds for the R+M pair (rich-space R+M ≈
  champion) but the CONTENT leg is the exception that adds genuinely new ranking
  signal, and it pays off more on the higher-EVR space.
- **Content-kNN probe ⟂ GRU ranking** (training-free probe flat ~0.06–0.07 across
  spaces; does not track the GRU dim-optimum). Judge on trained-model metrics.

## Representation / dataset lineage

- `spotify_tracks_content` (384-dim) — leak-safe compression SOURCE (intrinsic
  content/acoustic/catalog only; no behavioral/target fields).
- `spotify_tracks_song_ae` (AE-64) — original champion item space; control.
- `spotify_tracks` (200-dim) — behavioral co-listening embedding; **soft-leaks
  replay, NEVER use as item space.**
- NEW 2026-07-15: `..._ae32/ae128/ae192`, `..._pca32/pca64/pca128/pca192/pca256`
  — 23,529 pts each. PCA cum EVR 32/64/128/192/256 =
  0.923/0.959/0.982/0.992/0.997. Builders `pipeline/build_song_ae.py`
  (`--dst`/`--artifact-prefix`), `pipeline/build_song_pca.py`.
- Seq artifacts (all pass the exact parity gate 7154/19402/5723/1431 @
  cut 2024-08-24T14:28:02): AE-32 `seq-20260715-030446`, AE-128
  `seq-20260715-030452`, AE-192 `seq-20260715-131150`, PCA-32
  `seq-20260715-030457`, PCA-64 `seq-20260715-030503`, PCA-128
  `seq-20260715-030509` (Phase-2 base), PCA-192 `seq-20260715-131139`
  (candidate), PCA-256 `seq-20260715-131145`. Produced by
  `pipeline.corpus.sequences --latent-collection <c> --latent-dim <k>` into
  `data/seq/`, then registered into `data/datasets/` (the established path — no
  API ingests an externally-produced sequence artifact).

## Noise / significance

No pre-registered band; every claim uses a bootstrap 95% CI over test sessions
(2,000 resamples) + a PAIRED per-session Δ vs the reference (B2 bar / same-space
GRU anchor / champion). A board WIN needs beating the reference on BOTH Recall@10
AND MRR **beyond non-overlapping marginal 95% CIs** (the strict campaign bar),
then a 3-seed confirm; the PAIRED-Δ CI-entirely-> 0 is the more powerful
alternative statistic (it crowned the current champion) and can DISAGREE with the
marginal-CI bar — e.g. R+M+C-192 clears paired-Δ vs champion but its marginal CI
overlaps (→ candidate, not confirmed). `seq-nexttrack` exposes no seed; the AE-64
control reproduced R@10 0.096 across runs essentially exactly (run-to-run noise
well within the ~±0.015 CI half-width), so the reproduction clause substitutes for
seed-variation in the 3-seed confirm. The ~0.008–0.015 gaps between adjacent
high-dim PCA spaces and between the top blends are at/inside this noise.

## Pitfalls / infra

- **Wrong-collection trap:** sequence latents MUST come from a content-derived
  song space (`..._song_ae` / the new `..._pca*`/`..._ae*`), never the default
  `spotify_tracks` (200-dim behavioral embedding — soft-leaks replay).
- **Compression + combiner:** dim dominates method, peaks ~192; PCA≥AE ≥dim 64;
  a single-GRU/representation win does NOT imply a blend win (re-baseline the
  blend) EXCEPT the content leg, which adds genuinely new signal; the learned XGB
  stacker underperforms the fixed z-blend; content-kNN probe ⟂ GRU. See the
  compression/model-family section.
- **Two win-criteria can disagree:** non-overlapping marginal CIs (strict) vs
  paired-Δ CI>0 (powerful). Report both; a paired-Δ win with overlapping marginal
  CIs is a board-event CANDIDATE needing confirm, not an auto-crown. NB: for a
  seedless deterministic predictor (`seq-nexttrack`) the run-to-run reproduction
  confirm has ZERO spread — it verifies determinism, not seed-robustness; harden
  such a crown with a second test split.
- **Sequence-dataset registration:** `sequences.py` writes to `data/seq/`; the
  server reads datasets from `data/datasets/`. A new seq artifact must be copied
  into `data/datasets/<id>/` to be runnable (there is no ingest API).
- **Two execution slots / thread oversubscription:** running >1 torch predictor
  concurrently on a 12-core box oversubscribes threads (11/proc) and tanks
  throughput ~85× (~6 min/epoch vs ~4.5 s solo). Serialize (`--max-runs 1`).
  (Server restart is an ORCHESTRATOR action with user approval, never a runner
  action.)
- **Cold items dominate forward eval:** 55% of test targets unseen in train —
  absolute recall is modest by construction; judge vs the B2 bar.
- **Runtime:** own port **8100** + own Postgres DB **`nexttrack`** in the shared
  container; `LENSING_DISABLE_PATHFINDER=1`; reads the SHARED corpus Qdrant.
  Never restart the parent's 8096.

## Follow-ups
- ~~Blend GRU-infonce with the Markov bigram~~ DONE — the champion.
- ~~Item-representation bake-off + dim extension~~ DONE — PCA-128 base, dim peaks ~192.
- ~~Phase-2 model-family sweep (LSTM/ANN/stacker) + R+M+C on rich space~~ DONE —
  combiner = fixed z-blend; LSTM best single; R+M+C-192 crowned (see below).
- ~~CONFIRM R+M+C on PCA-192~~ DONE — confirmed 5× (spread 0, paired-Δ vs prior
  champion +0.013 >0 every rerun) → **CROWNED as the new champion ROW** (best on
  record; train-only + off-server, NOT promoted). Base stays PCA-128.
- **Harden the R+M+C-192 crown** with a second leak-free test split — the crown
  rests on a single split with paired-Δ lower bound +0.0007 and zero reproduction
  spread (deterministic GRU).
- **Register R+M+C as a 3-leg `seq-blend`** — prerequisite to making the champion
  row promotable / app-exportable (currently off-server driver only).
- **Generalize `seq-blend` to 3 legs** so R+M+C is a first-class registered
  predictor (currently off-server) — prerequisite to a promotable champion row.
- Confirm PCA-192 vs PCA-128 as base (paired-Δ still grazes 0) — folds into the
  R+M+C confirm above (same GRU reruns).
- Phase 4: "extend-a-session" showcase demo in `../app`.
- Phase 5: generalize + upstream the sequence/ranking framework support.
- `predict` subcommand → make ranking models best-models-promotable + app-exportable.
