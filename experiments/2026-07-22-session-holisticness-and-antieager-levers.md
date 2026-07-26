# Session-holisticness metrics + anti-eager levers (MMR λ / eager β) — 2026-07-22

**Date:** 2026-07-22
**Type:** MEASUREMENT + exploratory. **Nothing promoted** — the new metric suite
is display-only, the primary metric stays recall@10, and the north-star
(holisticness) decision is explicitly deferred to the user.
**Dataset (fixed):** `seq-20260715-131139` (canonical leak-free PCA-192 split;
7,154 sessions / 19,402 items; 1,431 test sessions; cold-item rate 0.55).
**Seed:** 1337. **Noise:** recall@10 practical half-width ≈ ±0.015 (a Δ inside
that band is a TIE).

## Goal & baseline

- **Question:** quantify the "predictions are too album-eager / bigram-like"
  symptom with the newly-added display-only holisticness metrics, test the user's
  *GRU-beats-LSTM-because-more-eager* hypothesis, and measure whether two new
  Phase-2 levers (MMR eval re-rank `mmr_lambda`; anti-eager training regularizer
  `eager_beta`) reduce eagerness at acceptable recall cost.
- **Reference champion:** `blend-gru-markov-content-proj` (R′+M+C′ seq-blend),
  on-record recall@10 **0.2117** = **303 of 1,431** next tracks (see
  `2026-07-18-nexttrack-literature-fit-campaign.md`,
  `2026-07-18b-nexttrack-projection-registration-and-scan.md`). Reproduced
  EXACTLY in this batch (0.21174).
- **New display-only metric columns** (added to `predictors/seq_common.py`
  `eval_from_scores`, wired through `run.rs`/`domain.rs`/`domain.toml`/UI like
  music@10; primary stays recall@10):
  - **artist_adj@10** — fraction of top-10 sharing the SEED track's artist;
    **LOWER = better** (the "parrots the current artist" eagerness rate).
  - **album_adj@10** — same on (artist,album); **OMITTED on this dataset**
    (its `items.json` lacks the album field) — EXPECTED, not an error; covered
    by a separate album-carrying dataset outside this campaign.
  - **mood_coh@10** — mean cosine of top-10 to the prefix mood centroid in the
    content-metric space; higher = better.
  - **ild@10** — intra-list diversity (mean pairwise sonic distance) of the
    top-10; higher = less duplicative.
  - **suffix_recall@10 / cont_prec@10** — multi-step continuation coverage,
    produced ONLY by the on-demand harness `predictors/seq_continuation_eval.py`
    (Phase E), not by normal runs.
- **Sanity gate — PASSED:** music@10 / mood_coh@10 / ild@10 populate on all
  blend and GRU runs (proves the content-metric Qdrant vectors at :6337 loaded);
  the MMR/mood/ild levers are therefore live, not no-ops.
- **Two new levers (both OFF by default, verified byte-identical off):**
  MMR re-rank at eval (`mmr_lambda`, 1.0=off; `mmr_pool` 200) on seq-nexttrack +
  seq-blend; anti-eager training regularizer (`eager_beta`, 0.0=off;
  `eager_margin` 0.0) on seq-nexttrack.

## Outcome (one line)

**Eagerness TRACKS recall inside the blend family (champion 0.2117 is the MOST
artist-eager at artist_adj 0.610) but the eagerness lives in the MARKOV bigram
leg, NOT the RNN — the single GRU/LSTM are far LESS eager (artist_adj ≈ 0.345)
and statistically TIED with each other (recall 0.123 vs 0.119, artist_adj 0.347
vs 0.344), so the GRU-beats-LSTM-because-eager hypothesis is NOT supported;
of the two levers the eval-time MMR re-rank is the STRONG knob (λ=0.9 buys ild
0.406→0.436 essentially free within the recall noise band, and diversifying even
RAISES music@10) while the anti-eager training regularizer β is a WEAK knob
(β 0→0.2 moves artist_adj only 0.347→0.340 at flat recall) — nothing promoted;
λ≈0.9 flagged as a future promotion candidate for the user.**

## Phase A — holisticness fingerprint across the model family (sorted by recall@10)

Lower artist_adj@10 = less eager; higher ild@10 / mood_coh@10 = more holistic.
All runs `n_test` = 1,431. **Bold = best cell per metric** (artist_adj: lowest;
ild/mood_coh: highest). recall@10 raw = successes of 1,431.

| run id | config | recall@10 (raw) | mrr | artist@10 | genre@10 | music@10 | artist_adj@10 ↓ | mood_coh@10 ↑ | ild@10 ↑ |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| run-20260722-165345-82a8f-seq-blend | **A1 champion R′+M+C′ (blend-gru-markov-content-proj)** | **0.2117 (303)** | **0.122** | 0.333 | 0.401 | 0.454 | 0.610 | 0.398 | 0.406 |
| run-20260722-165345-28dbb-seq-blend | A2 R+M (blend-gru-markov) | 0.1712 (245) | 0.095 | 0.329 | 0.398 | 0.443 | 0.579 | 0.399 | 0.403 |
| run-20260722-165345-e39c9-seq-nexttrack | A3 GRU alone (gru-infonce-h256, β=0) | 0.1230 (176) | 0.053 | **0.338** | 0.450 | **0.456** | 0.347 | **0.479** | 0.454 |
| run-20260722-165345-bb387-seq-nexttrack | B LSTM (arch=lstm, else matched to A3) | 0.1188 (170) | 0.049 | 0.328 | **0.455** | 0.449 | 0.344 | 0.458 | 0.463 |
| run-20260722-165345-047cc-seq-markov | A4 markov (first-order bigram) | 0.1069 (153) | 0.069 | 0.298 | 0.389 | 0.407 | 0.543 | 0.310 | 0.462 |
| run-20260722-165345-34614-seq-popularity | A5 popularity | 0.0007 (1) | 0.001 | 0.014 | 0.085 | 0.185 | **0.003** † | −0.041 | **0.898** † |

† **Degenerate, not a real win.** Popularity ignores the seed entirely, so its
top-10 shares the seed's artist essentially never (artist_adj ≈ 0) and is maximally
spread (ild 0.898) — but it lands ≈ 0 recall and NEGATIVE mood coherence
(−0.041, anti-correlated with the prefix mood). This proves artist_adj/ild are NOT
trivially "gamed low/high by a good model": you max them by being random. The
meaningful low-eager / high-diversity models are the single RNNs (A3/B).

## Phase B — GRU vs LSTM eagerness (matched hyperparams: h256, InfoNCE, 40ep, lr 1e-3)

| run id | arch | recall@10 (raw) | mrr | artist_adj@10 ↓ | mood_coh@10 ↑ | ild@10 ↑ |
|---|---|--:|--:|--:|--:|--:|
| run-20260722-165345-e39c9-seq-nexttrack | GRU | 0.1230 (176) | 0.053 | 0.347 | 0.479 | 0.454 |
| run-20260722-165345-bb387-seq-nexttrack | LSTM | 0.1188 (170) | 0.049 | 0.344 | 0.458 | 0.463 |

Recall Δ (GRU − LSTM) = **+0.0042, inside ±0.015 → TIE.** artist_adj 0.347 vs
0.344 (Δ +0.003, negligible), ild 0.454 vs 0.463, mood_coh 0.479 vs 0.458. The
GRU is NOT measurably more eager than the LSTM on this split — **the
GRU-wins-because-more-eager hypothesis is NOT supported here.**

## Phase C — MMR λ sweep on the champion (OFFLINE re-rank, `mmr_pool`=200)

**Design change vs launch:** the 4 Phase-C training runs (`5ff57` λ0.9, `5c208`
λ0.7, `39848` λ0.5, `0b796` λ0.3) were **CANCELLED (status `failed`) — not failed
experiments.** MMR is a pure eval-time re-rank, so retraining the champion per λ
is wasteful; Phase C was computed OFFLINE from the champion A1's checkpoint
instead (artifact `scratchpad/mmr_sweep_results.json`, `validation_ok: true`:
λ=off reproduced A1's recall@10 0.2117 exactly). LOWER artist_adj / HIGHER ild =
less eager.

| λ | recall@10 (raw) | Δrecall vs off | mrr | artist@10 | music@10 | artist_adj@10 ↓ | mood_coh@10 ↑ | ild@10 ↑ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| off (1.0) | **0.2117 (303)** | — | **0.122** | 0.333 | 0.454 | 0.610 | **0.398** | 0.406 |
| 0.9 | 0.2103 (301) | −0.0014 (TIE) | 0.121 | 0.341 | 0.462 | 0.603 | 0.391 | 0.436 |
| 0.7 | 0.1936 (277) | −0.0181 | 0.116 | 0.347 | 0.476 | 0.571 | 0.364 | 0.520 |
| 0.5 | 0.1642 (235) | −0.0475 | 0.108 | **0.359** | 0.487 | 0.486 | 0.315 | 0.653 |
| 0.3 | 0.1097 (157) | −0.1020 | 0.100 | 0.358 | **0.492** | **0.293** | 0.238 | **0.815** |

Trade-off curve: **λ=0.9 is essentially free** (recall Δ −0.0014, well inside the
±0.015 band) yet already lifts ild 0.406→0.436 and trims eagerness 0.610→0.603.
**λ≈0.7** trades ~2 recall points (Δ −0.0181, just outside the band) for a large
ild gain (0.406→0.520) and a real eagerness drop (0.610→0.571). λ=0.5/0.3 cut
recall hard (down to 157 hits at λ0.3) but crush eagerness (artist_adj 0.610→0.293)
and maximize ild (0.815). **Notable:** music@10 *rises monotonically* as you
diversify (0.454→0.492) — diversifying the list surfaces more sonically-adjacent
near-misses, so "musical closeness" and "list diversity" are not in tension here;
exact-recall is the only thing traded.

## Phase D — anti-eager regularizer β sweep on the GRU (`eager_margin`=0.0)

β0.1 and β0.2 were re-launched as remote-worker runs (the "snappler" box) to
parallelize past the local 2-slot bottleneck (below) — new run_ids; the original
local `c3419`/`13035` are superseded/killed and ignored.

| run id | β | recall@10 (raw) | mrr | artist@10 | artist_adj@10 ↓ | mood_coh@10 ↑ | ild@10 ↑ |
|---|--:|--:|--:|--:|--:|--:|--:|
| run-20260722-165345-e39c9-seq-nexttrack | 0.0 | 0.1230 (176) | 0.053 | **0.338** | 0.347 | **0.479** | 0.454 |
| run-20260722-165345-353b5-seq-nexttrack | 0.05 | **0.1237 (177)** | 0.053 | 0.337 | 0.346 | 0.478 | 0.455 |
| run-20260723-002044-af11a-seq-nexttrack | 0.1 | 0.1202 (172) | 0.053 | 0.335 | 0.344 | 0.477 | 0.456 |
| run-20260723-002044-17cf8-seq-nexttrack | 0.2 | 0.1202 (172) | 0.051 | 0.336 | **0.340** | 0.475 | **0.458** |

**Weak lever.** Across β 0→0.2, artist_adj moves only 0.347→0.340 (Δ −0.0074) and
ild only 0.454→0.458, with recall flat (all four inside ±0.015 of each other).
Penalizing near-copy prediction of the current track's latent barely shifts
eagerness — the single GRU is already low-eager (0.347), so there is little copy
behavior left to penalize. Not a useful holisticness knob at these settings.

## Phase E — multi-step continuation (on-demand harness, `-m 3 -k 10`, n_continuation = 1,172)

`predictors/seq_continuation_eval.py` on `data/datasets/seq-20260715-131139`,
markov baseline vs the trained GRU checkpoint
`data/runs/run-20260722-165345-e39c9-seq-nexttrack` (A3).

| model | suffix_recall@10 | cont_prec@10 |
|---|--:|--:|
| markov (first-order bigram) | 0.0869 | 0.0236 |
| GRU (gru-infonce-h256, A3) | **0.0963** | **0.0273** |

The GRU beats the bigram on **multi-step** continuation on BOTH metrics
(+0.0094 suffix_recall, +0.0037 cont_prec) — i.e. the RNN's advantage widens
when you look past the immediate next track: the bigram's single-step
co-occurrence structure doesn't compound into good 3-step continuations, whereas
the GRU's learned sequence state does. (This is the mirror of the single-step
board where the bigram-carrying blend wins raw recall@10.)

## Findings

1. **Eagerness tracks recall INSIDE the blend family — but the eagerness is the
   Markov leg, not the RNN.** Champion R′+M+C′ (recall 0.2117) is the most
   artist-eager model measured (artist_adj 0.610); R+M is next (0.579). Yet the
   single GRU/LSTM sit at artist_adj ≈ 0.345 and the standalone first-order
   **markov bar is 0.543**. So the champion's high eagerness is inherited from the
   Markov bigram leg (which parrots the current artist via raw co-occurrence),
   *not* from the neural leg. **Mechanistic takeaway: the blends buy their
   recall@10 lead by being artist-adjacent (bigram-like), and that is precisely
   what the holisticness metrics flag.**
2. **GRU ≈ LSTM, and neither is notably eager (hypothesis NOT supported).**
   Recall 0.123 vs 0.119 (TIE within ±0.015), artist_adj 0.347 vs 0.344. The
   user's "GRU wins because it parrots the current artist harder than the LSTM"
   does not hold on this PCA-192 split — they are indistinguishable on both recall
   and eagerness.
3. **RNNs are the mood-coherent, low-eager models.** The single GRU/LSTM have the
   HIGHEST mood_coh (0.479 / 0.458) and LOWEST real-model artist_adj (0.347 /
   0.344) — they retrieve mood-consistent, non-parroting continuations, they just
   land the exact next track less often. Markov has moderate eagerness (0.543) but
   the LOWEST real-model mood coherence (0.310): bigram co-occurrence is neither
   diverse nor mood-aware, it just repeats the artist.
4. **artist_adj@10 / ild@10 are not trivially gamed** — the popularity baseline
   maxes ild (0.898) and minimizes artist_adj (0.003) yet scores ≈ 0 recall and
   NEGATIVE mood_coh (−0.041). The metrics reward genuine holistic retrieval, not
   randomness.
5. **MMR (eval re-rank) is the STRONG lever; β (training regularizer) is WEAK.**
   MMR λ=0.9 is essentially free (recall TIE, ild +0.030, eagerness −0.007) and
   λ≈0.7 buys a big ild gain (+0.114) and real de-eagering (−0.039) for ~2 recall
   points; and diversifying RAISES music@10. β 0→0.2 moves artist_adj only −0.007
   at flat recall. If the user later decides to trade exact-recall for
   holisticness, the knob to reach for is MMR λ, applied at eval, not β at train.
6. **GRU compounds better over multi-step continuation** than the bigram
   (suffix_recall 0.096 vs 0.087, cont_prec 0.027 vs 0.024).

## Best on record after this work (RANKING board — UNCHANGED)

Primary metric stays recall@10; this campaign promoted nothing and does not
change the champion or leaderboard.

| model | recall@10 (raw) | note |
|---|--:|---|
| **★ R′+M+C′ learned content projection on PCA-192 (CHAMPION — `blend-gru-markov-content-proj`)** | **0.2117 (303)** | UNCHANGED. This campaign re-confirmed 0.21174 and newly fingerprinted it as the MOST artist-eager model (artist_adj 0.610), sourced from its Markov leg. |
| — R+M+C content-kNN z-blend on PCA-192 (`blend-gru-markov-content`) | 0.186 | prior crown; unchanged |
| — R+M z-blend on AE-64 (productization anchor) | 0.173 | unchanged |
| GRU/LSTM single on PCA-192 (this batch) | 0.123 / 0.119 | least-eager, most mood-coherent; TIE with each other |
| B2 first-order Markov (bar) | 0.107 (153) | high eagerness (artist_adj 0.543), low mood_coh (0.310) — the eagerness source |

## Candidate settings flagged for a future USER promotion decision (promote nothing now)

- **MMR λ ≈ 0.9 on the champion at eval** — a near-free holisticness upgrade
  (recall TIE within noise, ild 0.406→0.436, eagerness 0.610→0.603, music@10
  0.454→0.462). If a holisticness objective is ever adopted, this is the
  lowest-cost first step. λ≈0.7 is the "meaningful de-eagering for ~2 recall pts"
  operating point. Requires a user decision + a 3-seed / native-run confirm before
  any promotion.
- **eager_beta: not recommended** as a lever at these settings (too weak to matter).

## Pitfalls surfaced / re-confirmed

- **MMR λ is an EVAL-ONLY re-rank — never retrain per λ.** Retraining the champion
  per λ (as originally launched) is pure waste; sweep it offline from ONE trained
  checkpoint (`mmr_sweep_results.json` reproduced λ=off exactly). The 4 cancelled
  Phase-C runs are documented as cancelled-for-this-reason, not failed experiments.
- **Blend eagerness is sourced from the MARKOV leg, not the RNN** (single RNN
  artist_adj ≈ 0.345 vs markov 0.543 vs champion 0.610). Any future de-eagering
  should target the bigram leg / the fusion, not the neural model.
- **Thread oversubscription re-confirmed (registered pitfall).** Launching the
  batch with the server running >1 torch predictor concurrently tanked throughput
  to ~6.7 min/epoch (vs ~4.5 s solo) — the documented ~85× slowdown; the local
  2-slot batch would have taken ~10–20 h, which is why β0.1/β0.2 were moved to the
  remote worker. Serialize (`--max-runs 1`) or offload to a remote worker for
  batches like this.

## Follow-ups (not done)

- **Report-curator (sync mode):** fold this campaign into `docs/experiments.tex`
  (+ Spanish `docs/experiments.es.tex`) — the holisticness metric suite + the
  MMR-λ trade-off curve figure. Not auto-updated.
- If the user adopts a holisticness objective: run the MMR λ≈0.7–0.9 operating
  point through the API natively (not offline) + 3-seed confirm before promoting.
- Album-carrying dataset: re-run the fingerprint there to populate album_adj@10
  (absent here by construction).
- Target the Markov leg for de-eagering (e.g. down-weight same-artist bigram mass)
  since that is where the eagerness lives.
