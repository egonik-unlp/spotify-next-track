---
name: information-capture
description: Compare how much next track-relevant information competing sequence next-track models (the GRU and its ANN feed-forward twin / blend legs) capture inside their own hidden activations, using a sparse autoencoder (SAE) trained per hidden layer via the pg-server interpretability API. Use when the user wants to compare next-track models on information capture — capacity/utilization, interpretable next-item concepts, per-concept-class representation, concept-vs-decodability by depth, and the next-item concepts a model drops — beyond leaderboard holisticness@10. Runs `POST /api/interp/model-sae` and distills a side-by-side comparison. For a full campaign write-up, delegate to the information-capture-analyst agent.
user-invocable: true
argument-hint: "[model ...] [compare-embedding]"
allowed-tools:
  - Read
  - Bash(curl *)
  - Bash(sleep *)
  - Bash(python3 *)
---
<!-- GENERATED from agents-src/skills/information-capture/SKILL.md by agents-src/render.py — edit the template (and domain.toml), not this file; then run `zig build render-agents`. -->

Compare **information capture** across next-track models (the recurrent GRU and
its feed-forward ANN twin — the blend's base learners): how much next-item
structure each trained net actually re-represents inside its own hidden layers,
read with a sparse autoencoder (SAE). Because a next-track model has no scalar
target, the SAE's atoms are read against the **next item** — the artist / genre
/ album / sonic-continuity the holisticness suite cares about. This answers a
question the leaderboard can't: two models can post the same holisticness@10
while one *captures* far more next-item structure (and drops fewer next-item
concepts) than the other.

This is an **agent-layer wrapper** over the existing interpretability API
(`POST /api/interp/model-sae`); it computes nothing itself — it orchestrates the
jobs and distills their JSON into a comparison. The engine lives in
`predictors/seq_model_sae.py` (shared by `seq_ann.py` / `seq_nexttrack.py`).

## What "information capture" means here

Rows are `(session, step)` pairs: the activation at each step paired with the
teacher-forcing target — the **next item**. The per-model SAE reads the *model's*
side of the next-track task: for each hidden layer it reports

- **capacity** — `utilization` (fraction of atoms that ever fire), `dead_atoms`,
  `rare_atoms`, `l0_mean` (atoms active per step), and `var_explained` (how well
  the sparse code reconstructs the layer). Low utilization / high dead count =
  the layer is using a small slice of its width.
- **n_interpretable_concepts** — atoms whose activation separates a next-item
  concept class (artist / genre / album) above the engine's separation bar; a
  proxy for how many nameable next-track features the layer has formed.
- **segments** — per next-item `genre` class value, whether a
  dedicated atom *separates* it (`represented: true/false`). Surfaces
  rare-class structure the bulk fit buries.
- **next_item_decodability** — how linearly decodable the next item's
  `genre` is from the raw activations (`acc`/`auc` vs a majority
  `baseline_acc`), joined per layer with the interpretable-concept count, so you
  can see where decodable structure and nameable concepts form. The model's own
  retrieval recall@10 anchors it (`model_recall_at_10`).
- **dropped_vs_next_item** — frequent next-item concept classes that NO atom at
  a layer represents: next-track structure the model leaves on the table.

## Ground rules

- The server must be running (default `http://localhost:8096`; `zig build serve`).
  Analysis is **read-only** — it never mutates models, runs or datasets. Never
  restart pg-server (live training runs die on restart); never write under
  `data/`.
- Operates on **promoted models** (`data/models/<name>/`), not runs — promote a
  run first (`model-definitions` skill / `POST /api/models`) if you want to
  analyze it. The server auto-promotes top runs as `best-<predictor>-<slug>`, so
  a campaign's leaders are usually already promotable targets.
- Only predictors that declare `model_sae_args` in `registry.toml` are
  analyzable — the sequence next-track families (`seq-nexttrack` GRU,
  `seq-ann`), which expose a per-step hidden-activation space. `GET
  /api/interp/models` marks each with `supports_model_sae`; a model without it
  is not an error, just unsupported. Bidirectional GRUs have no per-step state
  and are refused.
- **Comparisons are only meaningful across models analyzed on the same dataset**
  (same `latent_dim` sequence artifact). Pass an explicit `dataset` so every
  model is read on identical sessions; otherwise each defaults to its own
  training dataset and the numbers are not comparable.
- Percentage-style reads rendered as %; keep run/dataset ids verbatim.

## Workflow

Analyses are **persisted**: every `model-sae` run is saved to the
`interp_analyses` table (mirrored from `data/interp/<id>/`) and served back by
id, so the first move is always to **reuse, not recompute**. Promoting an MLP
model also **auto-queues** a per-model SAE (embedding-diff read, no atom labels),
so a campaign's leaders usually already have a saved analysis by the time you
look.

### 0. Reuse persisted analyses (don't recompute)

```sh
curl -s 'localhost:8096/api/interp/analyses?tool=model-sae' | python3 -m json.tool
```
Each row carries `id`, `model`, `dataset_id`, `predictor`, `config`, `status`,
`source` (`auto`/`manual`) and `created_at` (no heavy `result`). For every target
model, if a `done` row exists on the shared dataset with an adequate `config`
(e.g. it already has `compare_embedding` when you need the dropped-signal diff),
fetch its stored result instead of launching a job:

```sh
curl -s localhost:8096/api/interp/analyses/<id> | python3 -m json.tool   # `.result` is the full analysis JSON
```
Only launch a new job (step 2) for models with no suitable saved analysis, or
when you need a richer config than the saved one (add `compare_embedding` or
`label_atoms`). Cite the analysis `id` in the report so it can be reopened in the
UI's **Saved analyses** tool.

### 1. Discover analyzable models

```sh
curl -s localhost:8096/api/interp/models | python3 -m json.tool
```
Each entry carries `name`, `predictor`, `dataset_id`, `n_cols`, `hidden`,
`activation`, and `supports_model_sae`. Keep the ones with
`supports_model_sae: true` that share a dataset width; agree the target set and
the shared `dataset` with the user (or, from the analyst agent, take them from
the campaign).

### 2. Launch a per-model SAE (async — only for models with no saved analysis)

```sh
curl -s localhost:8096/api/interp/model-sae -H content-type:application/json -d '{
  "model": "<name>",
  "dataset": "<shared ds-id>",
  "layers": "",
  "n_atoms": 0,
  "compare_embedding": false
}'
```
Body fields: `model` (required), `dataset` (default = model's training set;
**set it explicitly for comparisons**), `layers` (comma-sep 1-based hidden
layers, empty ⇒ all), `n_atoms` (`0` ⇒ 2× layer width), `l1`, `epochs`,
`label_atoms` (LLM auto-interp of top atoms — no-op unless the server has
`OPENAI_API_KEY`), `compare_embedding` (adds the dropped-signal diff). Returns
`{"job_id": "interp-…"}`. SAE jobs are serialized behind the shared build-slot
semaphore, so launch them and poll rather than expecting parallelism. The result
is persisted automatically, so a later look reuses it via step 0.

### 3. Poll each job

```sh
curl -s localhost:8096/api/jobs/<job_id> | python3 -m json.tool
```
`sleep 15` between rounds (these jobs take minutes; longer with
`compare_embedding`). `state` goes `running` (with a live `stage` mirrored from
the engine) → `done` (analysis JSON under `.result`) or `failed` (`.error`).

### 4. Distill the result

The `.result` JSON has top-level `model_dir`, `dataset_id`, `predictor`,
`hidden`, `latent_dim`, `config`, `model_recall_at_10`, `segment_field`,
`layers[]` (each with `capacity`, `n_interpretable_concepts`,
`next_item_decodability`, `atoms_by_concept`, `segments`,
`dropped_vs_next_item`), `concept_vs_decodability`, and `embedding_diff`. Pull
the per-layer capacity + concept counts and, per model, its best-layer
utilization, total interpretable concepts, concept classes represented, and
dropped-concept count.

### 5. Present the comparison

Build a side-by-side table, one row per model, and interpret it against
holisticness@10 — not instead of it:

```
| model | predictor | best-layer util | Σ concepts | classes repr. | dropped | peak next-item AUC | recall@10 | holisticness@10 |
```

Read it: does the leaderboard leader also capture the most next-item structure,
or is it winning while dropping next-item concepts a rival keeps? Where in depth
do concepts form vs. where the next item becomes decodable? Which rare concept
classes does each model represent? Flag any model whose capacity looks
degenerate (near-zero utilization, no interpretable concepts) — a likely
mis-load or a collapsed layer, not a finding.

## Delegate to the information-capture-analyst agent

The workflow above is for a quick, user-driven comparison of a handful of
models. For a campaign write-up — many models, a ready-to-paste report section
and a `experiments/PROJECT-FACTS.md` note — spawn the **information-capture-analyst** agent
with the model set and shared dataset id. The experiment-runner spawns it
automatically when a campaign trains next-track models with hidden activations
(GRU / ANN).
