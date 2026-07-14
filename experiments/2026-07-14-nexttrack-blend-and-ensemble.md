# Next-track: GRU×Markov blend beats the bar; further ensembling does not — 2026-07-14

Follow-up to the sweep (`2026-07-13-nexttrack-sequence-sweep.md`) and topology scan
(`2026-07-13-nexttrack-topology-scan.md`), which found the loss objective was the
only mover and topology was flat — the best single model (GRU-infonce) *tied* the
first-order Markov bar on Recall@10 but lost on MRR. This campaign asked: can we
BEAT the bar by combining models, and how far do ensembles go?

## Setup
- Artifact `seq-20260713-014226` (7,154 sessions / 19,402 items; leak-free
  chronological split; cold-item rate 0.55). Leave-last-out, next-distinct eval;
  Recall@10/@20, MRR, hit@10; bootstrap 95% CIs + paired Δ.
- Legs (each → full-vocab candidate scores): **M** = train-only Markov bigram +
  artist/genre back-off (the bar); **R** = GRU-infonce-h256 (session recurrence →
  next-latent → cosine); **A** = ANN bag-of-prefix MLP (feed-forward twin of R).
- Ensembles: weighted z-norm blend (`α·z(leg)+…` over candidates) and an XGB
  stacker (GBDT meta-learner over per-candidate features `[leg scores +
  log-popularity + recency + artist/genre match]`, negatives sampled, train-only).

## Result 1 — GRU×Markov blend BEATS the bar (the win)

| model | Recall@10 [95% CI] | MRR | paired Δ vs Markov |
|---|---|---|---|
| **R+M z-blend α=0.5 (CHAMPION)** | **0.1726** [0.154, 0.192] | **0.0963** | **+0.0657 [+0.049, +0.083]** |
| Markov bar (M) | 0.1069 [0.092, 0.123] | 0.0694 | — |
| GRU alone (R) | 0.0957 | 0.0411 | loses |

The blend beats the bar on Recall@10 AND MRR, **paired-Δ CI entirely > 0** →
decisive win. Robust plateau across α=0.25–0.75, and **3-seed confirmed** (R@10
0.1705 / 0.1726 / 0.1726 at seeds 17 / 101 / 1337). ~**+61% Recall@10, +39% MRR**
over the bar. Mechanism: the GRU (latent similarity) and Markov (co-occurrence
counts) rank *different* correct next-tracks, so fusing them roughly doubles hit
rate. Registered `seq-blend` predictor + `blend-gru-markov` definition (champion).

## Result 2 — further ensembling does NOT beat the blend

| ensemble | Recall@10 | MRR | vs champion |
|---|---|---|---|
| R+M z-blend (champion) | 0.1726 | 0.0963 | — |
| R+A+M z-blend | 0.1684 | 0.0943 | slightly worse |
| R+A z-blend | 0.0894 | 0.0377 | worse (≈GRU alone) |
| XGB stacker M+R (RNN+XGB) | 0.1020 | 0.0561 | much worse |
| XGB stacker M+R+A (ANN+RNN+XGB) | 0.0887 | 0.0572 | much worse |

- **The ANN leg is redundant with the GRU.** Both are neural latent-predictors →
  same signal; adding A to the champion doesn't help (R+A+M 0.168 ≈ R+M) and R+A
  is basically the GRU (0.089).
- **The XGB stacker underperforms the simple z-blend** (0.102 / 0.089 vs 0.173) —
  counter to "stacking beats blending." Its pointwise binary-with-negatives
  training doesn't optimize the ranking objective as well as directly fusing
  normalized scores; it barely clears Markov alone.
- **The win is about signal DIVERSITY, not model count.** Markov's different *kind*
  of signal (counts) complements the GRU's latent similarity; piling on more of
  the same kind (another NN) or a tree fuser adds nothing.

## Verdict
`R+M z-blend` (0.173) is the champion; no ensemble beats it. Ceiling reached with
these legs. The remaining headroom is bounded by the **55% cold-item rate** and
requires GENUINELY NEW signal (richer per-track features, longer/cross-session
context, a different candidate source), not more model fusion.

## Follow-ups
- New signal, not new fusion: e.g. content/acoustic features per candidate,
  time-of-day/session-type context, or session-graph neighbors.
- Productization: the ranking predictors are train-only (no `predict`), so the
  champion isn't best-models-promotable and can't yet export to the app. Adding a
  `predict` subcommand to `seq-blend` unlocks both — and the blend maps directly
  onto the app's A* transition slot (W_TRANS).

## Provenance
Off-server; blend + ensemble drivers reuse `seq_common.eval_from_scores`, the
trained GRU checkpoint (no retrain), `seq_ann`/`seq_stacker`, `seq_baselines`
Markov. CIs: bootstrap over test sessions + paired Δ. No new champion; `seq-blend`
stands.
