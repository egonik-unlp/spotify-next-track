# Next-track topology scan (depth / width+residual / bidirectional) — 2026-07-13

Follow-up to `2026-07-13-nexttrack-sequence-sweep.md`. That sweep found the loss
objective (InfoNCE) was the dominant lever and the best GRU tied the first-order
Markov bar on Recall@10 but lost on MRR. This scan asks: **does network TOPOLOGY
(depth, width+residual, bidirectionality) lift the sequence model over the Markov
bar?** Loss is held at InfoNCE (the sweep's winner) so topology is the only axis.

## Setup

- **Artifact:** `data/seq/seq-20260713-014226` (7,154 sessions / 19,402 items;
  chronological split 5,723 train / 1,431 test; cold-item rate 0.55).
- **Held:** loss = InfoNCE, cell = GRU, lr 1e-3, epochs 40, patience 6, k=10.
- **Bar (B2):** first-order Markov (train-only) — R@10 **0.107** / MRR **0.069**.
- **Predictor extension (this campaign):** `seq-nexttrack` gained `num_layers`
  (stacked), `residual` (manual inter-layer residual stack), and `bidirectional`.
  Leak note: a bidirectional RNN under per-step teacher forcing would leak future
  items, so the bi arm uses a **leak-free last-item objective** (encode the full
  prefix → predict only the held-out next item), NOT per-step teacher forcing.
- **Eval:** leave-last-out, next-distinct; Recall@10/@20, MRR, hit@10 with
  bootstrap 95% CIs (2,000 resamples) + a paired per-session Δ vs B2.

## Results

| arm | R@10 [95% CI] | MRR | R@20 | paired Δ vs B2 [95% CI] |
|---|---|---|---|---|
| **B2 Markov (bar)** | **0.1069** [0.092, 0.124] | **0.0694** | 0.1516 | — |
| T3 width h256 | 0.0957 [0.080, 0.110] | 0.0411 | 0.1398 | −0.0112 [−0.031, +0.009] tie |
| T1 depth-2 h128 | 0.0922 [0.076, 0.106] | 0.0399 | 0.1412 | −0.0147 [−0.035, +0.005] tie |
| T4 depth-2 h256 +res | 0.0908 [0.076, 0.106] | 0.0382 | 0.1398 | −0.0161 [−0.036, +0.004] tie |
| T0 1-layer h128 (ctrl) | 0.0894 [0.075, 0.104] | 0.0388 | 0.1335 | −0.0175 [−0.036, +0.002] tie |
| T2 depth-3 h128 | 0.0825 [0.069, 0.096] | 0.0372 | 0.1195 | −0.0245 [−0.043, −0.006] loses |
| T5 bidirectional h128 | 0.0482 [0.037, 0.059] | 0.0225 | 0.0720 | −0.0587 [−0.077, −0.041] loses |
| B0 popularity | 0.0007 | — | — | −0.106 loses |

## Verdict — NO WIN; topology is a flat/spent lever

No arm beats B2 on both Recall@10 AND MRR (the decision rule) → no 3-seed, no
deploy. Reads:

1. **Best arm = width (h256), R@10 0.0957 — a TIE with the bar on Recall@10**
   (paired-Δ CI includes 0) but it LOSES on MRR (0.041 vs 0.069) and on R@20. So
   the bigram still ranks the exact next item better.
2. **Depth doesn't help and 3 layers hurts.** 1→2 layers is flat (0.089→0.092,
   in-band); **depth-3 drops to 0.083** (out-of-band below the bar). Residual over
   depth-2-h256 (0.091) adds nothing vs depth-2-h128 (0.092).
3. **Width is the only marginal mover** (+0.007 over the 1-layer control), still a
   tie vs the bar.
4. **Bidirectional collapses (0.048) — mostly an OBJECTIVE artifact, not
   directionality.** The leak-free last-item objective supplies ~1 training target
   per session vs teacher forcing's ~13 next-step pairs, so it trains on ~13× less
   signal. Do NOT read this as "bidirectionality is bad"; it's the causal-safe
   objective that starves it. A fair bi test would need a different leak-free
   training scheme (e.g. masked-item), out of scope here.

**Overall:** architecture is not the lever for next-track on this corpus. The
transferable latent-retrieval signal caps at ~R@10 0.09–0.10 across depth / width
/ residual / direction, below the co-listening bigram. Consistent with the whole
arc: the loss objective moved the needle (InfoNCE ~2.3× cosine), capacity/topology
do not. First-order co-listening transitions remain the bar.

## Follow-ups
- The remaining unexplored lever is the **training objective / candidate framing**
  (masked-item / sampled-softmax / a blend of GRU-infonce candidate scores WITH
  the Markov bigram — the app's A* already normalizes a transition term). A blend
  is the most promising path to actually beat the bigram.
- Not architecture: this scan closes depth/width/residual/direction for the family.

## Provenance
Off-server scan (`predictors/seq_nexttrack.py` with the new topology params;
`registry.toml` `seq-nexttrack` extended). 6 arms, 40 epochs each, sequential,
OMP_NUM_THREADS=8. CI: bootstrap over test sessions (`scratchpad/topo_ci.py`).
No model promoted; result on the ranking board only.
