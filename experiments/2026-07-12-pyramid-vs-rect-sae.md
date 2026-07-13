# burn-deep-classifier: pyramid vs uniform-MLP representation read via per-model SAE (2026-07-12)

Goal: try a **pyramid** deep-classifier and at least one **MLP alternative** on
the canonical full-64-d rotation dataset, and use the **per-model SAE** feature
to infer what each net carves into its own hidden representation — a read the AUC
leaderboard can't give. Fixed baseline: dataset `ds-20260609-204419-p64-s42`
(177 cols, 23,529 rows, 18,823/4,706 @ seed 42), `topology=mlp`, `lr=5e-4`,
`activation=relu`, seed 1337 (trainer-fixed), differing ONLY in the hidden
geometry. Reference tier: the on-record burn-deep-classifier mlp
(0.7001–0.7038, `2026-06-12-burn-deep-topology.md`); xgboost champion on this
dataset 0.7204.

Outcome in one line: **at equal AUC (~0.699, tied) the pyramid consolidates its
interpretable concepts into the 64-d bottleneck (72→42 across its last two
layers) while the uniform net keeps a broader concept set to the end (66→54) —
a representational difference invisible to the leaderboard. No definition saved
(sub-champion, tied tier).**

## Framework change enabling this (committed separately)

The per-model SAE (`POST /api/interp/model-sae`) previously existed only for
`burn-mlp` (a regressor) and was undeclared in `registry.toml`, so no
classification MLP could be probed. This campaign ports the SAE head into
`predictor-burn-deep-clf` (new `model_sae.rs` + `forward_capture` /
`capture_stage_activations` in `model.rs` + a `model-sae` subcommand), reusing
the shared `lensing-sae` engine and `lensing-interp` probes unchanged, and
declares `model_sae_args` for `burn-deep-classifier`. `supports_model_sae` now
holds for the family; **promotion auto-queues** a per-model SAE for it. Binary
classifiers only (the output reference stage reads `P(class==1)`); any trunk
topology is analyzable.

## Results (all on ds-20260609-204419-p64-s42)

### Models (early-stopped, patience 10 / val_fraction 0.15)
| config | AUC | logloss | accuracy | run | model |
|---|---|---|---|---|---|
| **pyramid mlp [256,128,64]** | 0.69857 | 0.62536 | 0.64046 | run-20260712-164130-672c5 | `pyramid-mlp-256-128-64` |
| **rect mlp [128,128,128]** | **0.69937** | **0.62441** | **0.64896** | run-20260712-164130-6b093 | `rect-mlp-128-128-128` |

Δ AUC 0.0008 ≪ 0.0126 band → **statistically tied**; both in the on-record
burn-deep mlp tier (0.7001–0.7038, in-band) and ~1.7 bands below the xgboost
champion (0.7204). Calibration healthy (logloss ~0.625, brier regime).

### No-early-stop control (data point: the family overfits at the full budget)
| config | AUC | logloss | run |
|---|---|---|---|
| pyramid [256,128,64], 100 ep, patience 0 | 0.68325 | **1.21761** | run-20260712-163958-3db60 |
| rect [128,128,128], 100 ep, patience 0 | 0.68308 | **0.95902** | run-20260712-163958-cf7df |

Run to the full 100 epochs with no early stopping, both 3-layer nets **overfit**:
AUC drops ~0.015 and logloss blows up to 0.96–1.22 (ranking survives,
calibration destroyed) — the same signature as the sklearn pyramid-mlp, but
here it is a *training-budget* artifact, not a ceiling: early stopping restores
the calibrated ~0.699 / 0.625 regime. **Deeper burn-deep-clf nets need
`patience`.**

### Per-model SAE (2× overcomplete atoms, l1 0.0015, 40 epochs; auto-queued)
Analyses: `interp-run-20260712-164205-d3e43-msae` (pyramid),
`interp-run-20260712-164205-72f33-msae` (rect).

| metric | Pyr L1 (256) | Pyr L2 (128) | Pyr L3 (64) | Rect L1 (128) | Rect L2 (128) | Rect L3 (128) |
|---|---|---|---|---|---|---|
| utilization | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dead / rare | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| var explained | 0.966 | 0.986 | 0.991 | 0.985 | 0.984 | 0.981 |
| **interp concepts** | 11 | **72** | 42 | 17 | 66 | **54** |
| segments repr | 105/105 | 104/105 | 103/105 | 105/105 | 104/105 | 103/105 |

Linear decodability of rotation by depth (test R²): input 0.111 → both stay flat
at ~0.11–0.12 (pyramid peaks slightly at L1 0.124; rect flatter). Model-output
reference R² ~0.122 both.

## Findings

- **Same signal captured, different allocation.** Both nets barely lift linear
  decodability above the raw input (~0.111 → ~0.12) and post identical AUC —
  they capture the *same amount* of decodable taste-fit structure. The SAE
  difference is purely in layout.
- **The pyramid's narrowing forces concept consolidation.** Interpretable
  concepts peak in its middle 128-d layer (72) then compress to 42 in the 64-d
  bottleneck — the geometry squeezes ~40% of concepts out at the neck. The
  uniform net spreads them more evenly (17→66→54) and *retains* more concepts
  (54) in its final layer because it has the width to.
- **Neither is starved.** Both fully utilize every atom (0 dead / 0 rare across
  all layers at 2× width), give essentially every identity segment (~103–105/105)
  a dedicated separating atom, and reconstruct >96% of activation variance.
  Codes are dense (l0 ≈ 47–94% of atoms active per track) — distributed, not
  sparse concepts, consistent with the framework's light default l1 0.0015.
- **Interpretation:** at equal AUC the pyramid and the uniform MLP encode the
  same rotation signal, but the pyramid consolidates its interpretable structure
  into a compressed bottleneck while the uniform net keeps a broader concept set
  to the end — a distinction the AUC leaderboard cannot see.

## Best on record after this work

Unchanged. xgboost-classifier remains the champion (DS-META 0.7429; this-corpus
0.7204). The two new burn-deep-clf mlp models (0.699) sit in the existing ANN
tier (tied with the on-record 0.7001–0.7038), below the ~0.71 save bar → **no
definition saved**, no champion change. Both promoted (kept for their saved SAE
analyses); auto-promotion did not displace any registered model.

## Follow-ups

- Higher-l1 SAE (sparser atoms) for a cleaner concept read — the default 0.0015
  yields dense codes (l0 ~50%); a sparsity scan would sharpen the concept counts.
- `compare_embedding` run (dropped-signal diff vs a dataset SAE) to quantify the
  taste-fit embedding signal each geometry leaves on the table.
- Deeper/narrower pyramids ([512,128,32]) to test whether harder bottlenecks
  keep consolidating concepts or start dropping segment representation.
- Wire `probe_args` (layer-probe) for burn-deep-classifier too, so the family
  shows up in `GET /api/interp/models` (currently gated on probe_args; model-sae
  works regardless via the auto-queue / direct endpoint).
