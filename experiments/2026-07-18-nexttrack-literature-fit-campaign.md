# Next-track literature-fit campaign: a learned content projection is a new best-on-record; session-kNN / rank-fusion / combiner / GRU-negative lanes all close — 2026-07-18

Motivated by a bibliography survey of the *established* (non-bleeding-edge)
session-recommendation literature (Jannach–Ludewig session-kNN line; sequential
rules / factorized Markov; GRU4Rec training tricks; NARM/SASRec attention;
rank-fusion & learning-to-rank; content/cold-start-aware session rec), this
campaign asks which classic methods actually improve the FIT to *this* problem —
a **single-user**, session-based, next-**distinct**-track **retrieval** task with
**55% cold test targets** on a single leak-free split. Three tiers were
pre-registered:

- **Tier A** — improve the *signal/estimation* of legs we already have: (A1) a
  supervised **learned content projection** (metric learning over the content
  latent), (A2) principled **Markov smoothing** (absolute-discounting /
  Kneser-Ney continuation).
- **Tier B** — (B1) a **ranking-aware combiner** (coordinate-ascent linear
  reweight + RRF, subsuming LambdaMART) and (B2) a gated **V-SKNN** session-kNN
  probe.
- **Tier C** — **GRU4Rec importance negatives** (Hidasi & Karatzoglou 2018) on
  the InfoNCE GRU leg.

## Setup

All arms measured **off-server** (the sanctioned driver path that crowned the
current champion) on the PCA-192 champion artifact
`data/seq/seq-20260715-131139` (7,154 sessions / 19,402 items; 1,431 test
sessions; cold-target rate 0.55), reusing `seq_common.eval_from_scores` +
`seq_baselines.markov_scorer` + `seq_models.content_knn_scorer` +
`seq_nexttrack.fit` UNMODIFIED. The harness **reproduces the champion exactly**
— R+M+C equal-thirds z-blend = R@10 **0.1859**, MRR **0.1025**, artist@10
0.3305, genre@10 0.3941 (matches `run-20260718-123738-38725-seq-blend`), and the
leg standalones match the board (M 0.1069, R/GRU 0.1230, C 0.0685). Bootstrap 95%
CI = 2,000 resamples over the 1,431 test sessions; **paired per-session Δ vs the
champion** is the decision statistic. Legs: **R** = GRU-infonce h256 (the
champion leg), **M** = first-order Markov (the bar), **C** = content-kNN(max).
Scratch drivers under `scratchpad/litfit/` (`_harness.py`, `tier_{a1,a2,b,c}.py`,
`verify_a1.py`).

## Outcome (one line)

**NEW BEST-ON-RECORD ROW (candidate): a supervised LEARNED CONTENT PROJECTION.**
Fitting a linear map of the PCA-192 content space toward next-track adjacency
(InfoNCE over train consecutive pairs) and retrieving both the content leg (C′)
and the GRU (R′) in that projected space gives **R′+M+C′ z-blend R@10 0.2117
[0.191, 0.233], MRR 0.122 — paired-Δ vs the champion +0.0259 [+0.012, +0.041],
entirely >0**, and it is **seed-robust** (3 seeds: +0.026 / +0.025 / +0.029, all
CI>0 — a genuine seed-varied confirm, stronger than the champion's
determinism-only one) and **reaches cold items** (C′ scores cold targets 0.152 ≈
warm 0.151, vs Markov's cold 0.010 — genuine content generalization, not
transition memorization). **Every other lane closed:** the ranking-aware combiner
(even the ORACLE test-fit weights: +0.004, CI straddles 0), RRF (−0.029), V-SKNN
(drags the blend −0.038), Markov smoothing (a real but tiny standalone bar bump
0.107→0.110 that washes out in the blend, −0.002), and GRU importance negatives
(+0.005 standalone that washes out, blend 0.1845 tie) all fail to beat the fixed z-blend champion — exactly as the
literature-fit survey predicted for a single-user, cold-heavy corpus. It remains
train-only + off-server → a best-on-record ROW, not a promoted model; the one
outstanding gate is the standing second-split hardening (true for every crown
here).

## Tier A1 — learned content projection (THE WIN)

**Design.** The content-kNN leg and the GRU both retrieve over the *unsupervised*
PCA-192 geometry (variance-ordered, target-agnostic). Learn a linear map W of
that space by InfoNCE over TRAIN consecutive pairs (u→v, self-loops dropped —
Markov convention), so cosine in W-space reflects "likely to follow." Leak-free
(train pairs + intrinsic content latents only; never the split). Variants:
symmetric single W (drop-in content leg C′) vs directional two-tower Wq/Wk; rank
∈ {192, 64}. Then optionally retrain the champion GRU to retrieve in the
projected space (R′) — the shared-latent gain the survey predicted.

**Results (PCA-192, vs champion 0.1859).**

| model | R@10 [95% CI] | MRR | artist@10 | genre@10 | paired-Δ vs champion |
|---|---|---|---|---|---|
| R+M+C champion (thirds) | 0.1859 [0.166, 0.207] | 0.1025 | 0.331 | 0.394 | — (reference) |
| **★ R′+M+C′ (both projected)** | **0.2117 [0.191, 0.233]** | **0.1220** | 0.333 | 0.401 | **+0.0259 [+0.012, +0.041] >0** |
| R+M+C+C′ (add proj-content, 4-leg) | 0.2061 [0.185, 0.227] | 0.1148 | 0.345 | 0.410 | +0.0203 [+0.013, +0.029] >0 |
| R+M+C′ (swap C→C′, 3-leg) | 0.1971 [0.177, 0.218] | 0.1107 | 0.334 | 0.400 | +0.0112 [−0.000, +0.023] candidate |
| R′ proj-GRU (single model, 3 seeds) | 0.1726 / 0.1719 / 0.1761 | — | warm 0.159 / cold 0.184 | — | vs base GRU 0.1230: **+0.05** |
| C′ proj-content (single leg, sym r192) | 0.1516 | 0.0708 | 0.305 | 0.401 | vs unsup C 0.0685: **+0.083** |
| C′ dir two-tower (r192) | 0.1349 | 0.0552 | 0.291 | 0.393 | symmetric wins |

**Verification (the load-bearing part).** Two adversarial checks, both passed:

- **Warm/cold decomposition** (anti-memorization). Split test targets into warm
  (truth appears in some train session) vs cold (unseen). The learned content leg
  C′ scores **cold 0.152 ≈ warm 0.151** — it reaches never-seen items as well as
  seen ones. A memorizing leg would collapse on cold (Markov: warm 0.226 / **cold
  0.010**). So C′ is genuine *content generalization*, the one channel that
  reaches the 55% cold majority. The blend R′+M+C′ lifts BOTH warm (0.240) and
  cold (0.189).
- **Seed robustness** (3 seeds 1337/7/99). The projection has random init +
  shuffled batches, so this is a real seed test (unlike the deterministic GRU).
  C′ standalone = 0.1516 / 0.1495 / 0.1516; R+M+C+C′ = 0.2061 all three; R′+M+C′ =
  0.2117 / 0.2110 / 0.2145 with paired-Δ +0.026 / +0.025 / +0.029, **every CI
  entirely >0**.

Read: the supervised content projection is the first genuinely-additive NEW
signal since the content leg itself — and it works for the reason the literature
review flagged: content is the only channel with cold reach, and replacing the
unsupervised PCA geometry with a target-fit one lifts *both* legs that share it.
The symmetric map beats the directional two-tower; rank 192 = rank 64 (the gain
is the supervision, not the capacity). Single-model note: the projected GRU R′
alone (~0.176) nearly matches the *old champion blend* and crushes the prior best
single (LSTM 0.127) — a large single-model-ceiling lift as a side effect.

## Tier A2 — principled Markov smoothing (null in the blend)

The current Markov leg is already Dirichlet smoothing (`trust = n_u/(n_u+8)`).
Absolute-discounting + a **Kneser-Ney continuation-count floor** is a genuinely
better *estimator*: standalone R@10 **0.1069 → 0.1104** (the continuation floor is
what helps; μ-tuning is flat). But it does NOT survive the blend: **R+Ms+C =
0.1838, Δ −0.0021 [−0.006, +0.001]** (straddles 0) — the content leg already
covers the marginal cold mass the smoothing recovers. Reconfirms "single-leg
gains don't carry." (Both standalone numbers sit inside the ±0.015 noise band, so
even the bar bump is only suggestive; a cleaner deployable Markov, not a champion
move.)

## Tier B — ranking-aware combiner + V-SKNN (both lanes close)

**B1 combiner.** The 3-leg blend was hard-coded to equal-thirds
(`seq_blend.py:180`), never optimized. Coordinate-ascent reweight over R/M/C:
even the **ORACLE** weights (fit directly on the *test* labels — cheating) reach
only **0.1901, Δ +0.0042 [−0.004, +0.013]** (straddles 0) — i.e. *no honestly
achievable weight beats equal-thirds on recall beyond noise.* **RRF** rank-fusion
*hurts* recall (0.1572, −0.029, CI<0) — it discards the leg-score magnitude the
z-blend keeps. Fitting weights on a train-val slice is **corrupted** (the count
legs memorize their own training data → the fit collapses to pure-Markov
[0,1,0]); the oracle bound is the clean answer, and it says the combiner lane is
tapped. LambdaMART is subsumed (a boosted tree over the same 3 legs can only add
overfit capacity). *One real sub-signal:* the oracle/RRF reweights lift the
GRADED metrics (artist@10 → 0.358/0.366, genre@10 → 0.430/0.456) by upweighting
R+C over M — the graded axis is reweight-movable even though exact recall is not.

**B2 V-SKNN.** Recency-decayed IDF session-neighbor kNN (Ludewig & Jannach 2018).
Standalone **0.0671** (≈ content-kNN, below the 0.107 Markov bar); artist@10 0.199
(worst leg). Blended it DRAGS the champion (R+M+C+V = 0.1474, −0.038, CI<0; R+V+C
= 0.1202) — the item2vec pattern. The single-user setting strips the cross-user
collaborative neighborhood that drives its multi-user benchmark wins, and it is
structurally warm-only against 55% cold targets. Session-kNN / STAN / VSTAN lane
**retired** with a pre-registered null, as designed.

## Tier C — GRU importance negatives (null)

The InfoNCE bank is currently only the in-batch targets — and since frequent
items appear as targets more often, those negatives are already ~popularity^1
weighted. The GRU4Rec "top-k gains" trick appends M extra shared negatives
sampled ∝ support^α to the bank (implemented by monkeypatching
`seq_nexttrack.step_loss`, reusing `fit()`'s seeding/val-split/early-stop
verbatim; sampler seeded for determinism). Standalone sweep (vs base GRU 0.1230):
n_neg=2048/α=0.75 → 0.1265; n_neg=8192/α=0.00 → 0.1223; n_neg=8192/α=1.00 →
**0.1279** (best) — at most +0.005, inside the ±0.015 noise band. Blend
re-baseline of the best variant: **R″+M+C = 0.1845, Δ −0.0014 [−0.008, +0.005]**
(straddles 0) — ties the champion. The single-model tick does not carry to the
blend (the documented rule), and the only genuinely-new degree of freedom over
the existing popularity^1 in-batch negatives (a larger bank / tail de-biasing)
buys nothing here. Null, as pre-registered.

## Verdict

- **A supervised learned content projection is a new best-on-record row**
  (R′+M+C′ 0.2117, +0.0259 paired-Δ CI>0, seed-robust ×3, cold-reaching) — the
  one literature-fit direction that attacks the actual binding constraint
  (cold-start) rather than piling redundant warm co-occurrence legs or reshuffling
  the same three legs. It clears every bar the current champion cleared, more
  robustly.
- **The flashy borrowed methods are dead ends here, exactly as the survey
  predicted:** session-kNN (single-user kills the collaborative edge; warm-only
  vs 55% cold), rank fusion / learned combiners (no headroom over the fixed
  z-blend — even the oracle can't beat it on recall; the *only* additive gains
  come from new SIGNAL, not smarter fusion), and GRU-side loss tricks
  (importance negatives give a within-noise standalone bump that washes out in
  the blend).
- **The one movable secondary axis is graded relevance** — reweighting toward R+C
  lifts artist@10/genre@10 (though it trades exact recall), and the learned
  projection also nudges genre@10 up. Worth a graded-optimized variant.

## Follow-ups

- **Register the learned content projection as a first-class leg / predictor** so
  R′+M+C′ becomes a reproducible champion row (currently an off-server driver,
  like the R+M+C it beats). The projection is a ~30s deterministic-seed-robust
  linear map; C′ is a drop-in for `content_knn_scorer` over projected latents; R′
  is the existing GRU trained on the projected space.
- **Harden the crown on a second leak-free split** — the standing gate for every
  crown here (needs re-sessionization via `pipeline/corpus/sequences.py` /
  dataset-architect). This crown is already seed-robust and cold-reaching, so the
  second split is the last remaining check.
- **Sweep the projection** — objective (BPR vs InfoNCE), τ, epochs, and a
  graded-relevance projection objective (fit toward artist/genre adjacency) for a
  graded-preferred variant.
- **Is R′ (projected single GRU, ~0.176) worth promoting on its own?** It nearly
  matches the *old* champion blend as a single model — the cleanest servable
  next-track model the lab has, if a `predict` path is added.

## Provenance

Off-server drivers `scratchpad/litfit/{_harness,tier_a1,tier_a2,tier_b,tier_c,
verify_a1}.py` on `data/seq/seq-20260715-131139` (PCA-192), predictor venv
(torch 2.12.0+cpu). Harness reproduces `run-20260718-123738-38725-seq-blend`
(champion) exactly. Bibliography survey: Ludewig & Jannach 2018 UMUAI; Ludewig
et al. 2021 UMUAI (VSTAN); Garg et al. 2019 SIGIR (STAN); Kamehkhosh/Jannach/
Ludewig 2017 (SR); Rendle 2009 (BPR); Hidasi & Karatzoglou 2018 CIKM (top-k
gains); Cormack et al. 2009 (RRF); Chen & Goodman 1999 (KN smoothing). No
server restart, no `models.toml` / `data/` hand-edits.
