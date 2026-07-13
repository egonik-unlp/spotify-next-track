# INTERIM — Pyramid-MLP architecture scan (p64 binary rotation)

Status: 2/6 arms done (M0, M1). M2–M5 still training (CPU-contended, 6 concurrent; load avg ~27). This is a snapshot; the final report replaces this file.

## Run id ↔ config mapping
| arm | run_id | top_width | n_layers | decay | realized hidden | status |
|---|---|---|---|---|---|---|
| M0 re-baseline | run-20260612-123155-8bfab-pyramid-mlp-classifier | 256 | 3 | 0.5 | [256,128,64] | succeeded |
| M1 wide-shallow | run-20260612-123155-0ace5-pyramid-mlp-classifier | 512 | 2 | 0.5 | [512,256] | succeeded |
| M2 deep pyramid | run-20260612-123155-79128-pyramid-mlp-classifier | 256 | 4 | 0.5 | [256,128,64,32] | running |
| M3 steep taper | run-20260612-123155-0df76-pyramid-mlp-classifier | 512 | 3 | 0.33 | [512,169,56] | running |
| M4 gentle taper | run-20260612-123155-d3834-pyramid-mlp-classifier | 384 | 3 | 0.8 | [384,307,246] | running |
| M5 wide-deep | run-20260612-123155-01f07-pyramid-mlp-classifier | 512 | 4 | 0.6 | [512,307,184,110] | running |

Dataset: ds-20260609-204419-p64-s42 (full-64-d binary rotation, 18823/4706, 177 cols). Held: alpha 1e-4, lr_init 1e-3, max_iter 800, seed 42.

## Headline reframing (early)
The design assumed the bake-off's 0.6781 was an under-training artifact (300-iter cap). **It was not.** progress.jsonl shows M0 [256,128,64] CONVERGED at **73 iters** (well under the 800 cap), reproducing AUC 0.6781 / logloss 2.4085 / brier 0.3251 EXACTLY. So 0.6781 is the architecture's genuine convergence plateau on these features, not a budget artifact. Raising max_iter 300→800 changed nothing because the optimizer never hit the cap.

## Results so far (sorted by AUC)
| arm | hidden | AUC | logloss | acc | brier | iters | converged |
|---|---|---|---|---|---|---|---|
| M1 | [512,256] | 0.6792 | 2.0584 | 0.633 | 0.3216 | 75 | yes (<cap) |
| M0 | [256,128,64] | 0.6781 | 2.4085 | 0.633 | 0.3251 | 73 | yes (<cap) |

Tree incumbents (same dataset, bake-off): xgboost 0.7204, catboost 0.7111, ngboost 0.6917.

## Decision-rule status
Winner TBD (need M2–M5). Save threshold: best converged MLP AUC >= ~0.683 (within ~3 bands of the tree tier). So far best is 0.6792 — BELOW threshold. If M2–M5 do not clear 0.683, persist no definition and report MLP family confirmed below trees.

## Resume instructions
Re-read this file + `curl /api/runs/<id>` for M2–M5. progress.jsonl per run dir gives realized hidden + iters + converged. Then assemble final table, apply rule, write `2026-06-12-pyramid-mlp-arch-scan.md`, reconcile PROJECT-FACTS.md, delete this interim file.
