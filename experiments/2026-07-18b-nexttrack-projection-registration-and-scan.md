# Next-track learned-content-projection: registration + promotion of a servable champion, and an on-server objective/rank/tau ablation scan — 2026-07-18 (b)

Execution follow-through on the 2026-07-18 literature-fit campaign
(`2026-07-18-nexttrack-literature-fit-campaign.md`), which found a NEW
best-on-record ROW — a supervised **learned content projection** R′+M+C′ on
PCA-192 (R@10 ~0.2117, seed-robust ×3, cold-reaching) — but left it an
*off-server driver*, not registered or promoted (the same productization gap the
R+M+C crown had before its own registration). Between that campaign and this one,
the projection was implemented as a first-class capability of the `seq-blend`
predictor (new hyperparams `projection`, `projection_rank`,
`projection_objective`, `projection_tau`, `projection_epochs`, `projection_lr`;
`projection=true` implies `content=true`). Mechanism: fit a linear metric map of
the item-latent space from TRAIN consecutive pairs (InfoNCE or BPR), then the GRU
(R′) and the content-kNN leg (C′) retrieve in the projected space; Markov (M) is
unchanged; the map is saved as `projection.npz` in the run dir and baked into the
promoted model for serving. This report does two things: (1) reproduces the
off-server win on-server, **registers + promotes** it as a servable champion, and
(2) runs the pre-approved **objective / rank / τ ablation scan** to confirm the
default config is the robust one to promote. A later **second-split hardening**
pass (added 2026-07-18 evening, see the section of that name) then CONFIRMS the
crown on a disjoint, colder leak-free split.

## Goal & baseline

- **Dataset:** `seq-20260715-131139` — PCA-192 champion space
  (`spotify_tracks_song_pca192`; 7,154 sessions / 19,402 items; chronological
  split 5,723 train / 1,431 test @ cut 2024-08-24; cold-target rate 0.55).
- **Predictor:** `seq-blend`. **Definition:** `blend-gru-markov-content-proj`
  (`predictor=seq-blend`, `hyperparams={projection:true}`; schema defaults fill
  the rest → objective infonce, rank 0 = full (192), τ 0.07, projection_epochs
  40).
- **References:** the previously-promoted+servable champion
  `blend-gru-markov-content` (R+M+C, R@10 **0.1859** / MRR 0.1025,
  `run-20260718-123738-38725-seq-blend`), and the off-server projection driver
  (~0.2117) from the lit-fit campaign.
- **Noise band (single split):** ~±0.015 on recall@10 (bootstrap half-width);
  adjacent configs differing by <0.015 are ties.
- **Serialization:** runs launched strictly **one at a time** (documented
  thread-oversubscription pitfall: >1 concurrent torch predictor tanks
  throughput ~85×). No run ever had >1 in flight; all 7 (verify + 5-cell scan)
  ran clean, zero failures, zero relaunches. Walltime ≈ 6 min/run.

## Outcome (one line)

**The learned content projection reproduced on-server EXACTLY (R@10 0.21174, MRR
0.12199 — matches the off-server 0.2117 / 0.122) and is now a REGISTERED +
PROMOTED + SERVABLE champion, `blend-gru-markov-content-proj`, topping the
best-models group (0.212 vs the R+M+C model's 0.186).** The
objective/rank/τ scan CONFIRMS the promoted default (**InfoNCE / full-rank / τ
0.07**) as the robust config: **InfoNCE ≫ BPR** (BPR collapses to ~0.15–0.16,
below even the old R+M+C champion), **full-rank ≥ rank-64** (rank-64 −0.006, a
tie), and **τ is mildly sensitive** (0.05 → 0.193, 0.07 → 0.212, 0.10 → 0.223) —
but the nominal τ-0.10 top-cell beats the default by only +0.0112 on recall,
**inside** the noise band, so the decision rule does not re-promote it. Default
HELD. **And (2026-07-18 evening) the crown HARDENS on the second split:** on the
disjoint 0.64-cold earlier-holdout `seq-20260718-211238` the projection beats the
R+M+C crown by paired-Δ **+0.0426 [+0.0280, +0.0573] (CI entirely >0)** with no
MRR regression — CONFIRMED (see "## Second-split hardening").

## Step 1–3 — reproduce, register, promote

The verify run (`run-20260718-191539-1b89c-seq-blend`, definition default =
InfoNCE / full-rank / τ 0.07) reproduced the off-server driver to 4 decimals:
R@10 **0.21174**, MRR **0.12199**, artist@10 0.3326, genre@10 0.4011, over 1,431
test sessions — clearing the accept gate (≥0.205 and >0.19) decisively. It was
promoted via `POST /api/models` as **`blend-gru-markov-content-proj`** (the
projection matrix is baked into the model dir for serving). `GET /api/best-models`
confirms it as **rank 1** at 0.21174, above the prior servable champion
`blend-gru-markov-content` (rank 2, 0.18588). The whole sequence family's ranking
serving contract (2026-07-18) means this is a genuinely deployable model, servable
via `POST /api/models/blend-gru-markov-content-proj/predict`, not just a board row.

## Step 4–5 — objective / rank / τ ablation scan (6 cells, sorted by recall@10)

All overlays on the promoted definition (definition unmodified); the InfoNCE /
full / τ 0.07 cell is the promoted verify run (reused, not re-run). `hit@10`
equals `recall@10` by construction on the next-distinct single-target eval.

| config | recall@10 | MRR | artist@10 | genre@10 | hit@10 | run_id |
|---|---|---|---|---|---|---|
| S2 · infonce / full / τ **0.10** | **0.2229** | **0.1273** | **0.3424** | **0.4032** | **0.2229** | `run-20260718-193111-a781c-seq-blend` |
| **★ infonce / full / τ 0.07 (PROMOTED CHAMPION — default, HELD)** | **0.2117** | **0.1220** | 0.3326 | 0.4011 | **0.2117** | `run-20260718-191539-1b89c-seq-blend` |
| S3 · infonce / **rank 64** / τ 0.07 | 0.2055 | 0.1169 | 0.3326 | 0.3990 | 0.2055 | `run-20260718-193650-66c96-seq-blend` |
| S1 · infonce / full / τ **0.05** | 0.1929 | 0.1111 | 0.3354 | 0.3997 | 0.1929 | `run-20260718-192443-57432-seq-blend` |
| S4 · **bpr** / full / τ 0.07 | 0.1579 | 0.0939 | 0.3187 | 0.3913 | 0.1579 | `run-20260718-194228-4cc06-seq-blend` |
| S5 · **bpr** / rank 64 / τ 0.07 | 0.1523 | 0.0909 | 0.3194 | 0.3906 | 0.1523 | `run-20260718-194832-d070f-seq-blend` |

Bold row = the promoted champion (winner per the decision rule); bold cells = the
best value in each metric column (all held by S2, τ 0.10). No run failed.

**Decision rule applied.** The rule: re-promote only a config that beats the
promoted default (InfoNCE / full / τ 0.07) on BOTH recall@10 AND MRR by MORE than
the noise band (~±0.015 on recall@10). The only config that beats the default at
all is **S2 (τ 0.10): recall Δ = +0.0112, MRR Δ = +0.0054** — the recall margin is
**inside** the ±0.015 band, so S2 is a **tie**, not a re-promotion candidate.
**Decision: the default InfoNCE / full-rank / τ 0.07 is CONFIRMED as the robust
promoted champion `blend-gru-markov-content-proj`. No re-promotion.**

## Findings

- **Objective — InfoNCE ≫ BPR, the sharpest axis.** Both BPR cells (S4 0.1579,
  S5 0.1523) fall ~0.055 below the InfoNCE default and land **below even the old
  R+M+C champion (0.1859)** — a swing far beyond noise. The pairwise BPR metric
  map produces a materially worse geometry for both shared-latent legs than the
  softmax/contrastive InfoNCE map. InfoNCE is the correct default and BPR is a
  documented dead end for this projection.
- **Rank — full ≥ rank-64, a tie.** Dropping the map to rank 64 (S3 0.2055) costs
  −0.006 vs full-rank (0.2117) under InfoNCE — inside the noise band. This
  reconfirms the lit-fit finding that *the gain is the supervision, not the
  capacity*; a low-rank map is essentially as good, but full-rank is nominally
  ahead and is what is promoted. (Under BPR, rank makes no difference — both
  variants are equally bad.)
- **τ — mildly sensitive, monotone up over the tested range.** InfoNCE full-rank:
  τ 0.05 → 0.1929, τ 0.07 → 0.2117, τ 0.10 → 0.2229. τ 0.05 (sharper contrast) is
  ~0.019 *below* the default (just over the band — a real, if small, degradation);
  τ 0.10 (softer) is +0.0112 above (inside the band — a nominal, sub-noise
  improvement). So there is a mild "softer-is-a-bit-better" trend, but nothing
  clears the re-promotion bar on a single split. τ 0.10 is the one config worth a
  seed + second-split check before any swap (see follow-ups).
- **Graded metrics track recall here** — artist@10 and genre@10 move in the same
  direction as recall across all six cells (both best at S2, both worst at the BPR
  cells), unlike the reweighting axis in the lit-fit campaign where graded and
  exact recall could be traded. This scan moves the *signal quality*, so all
  metrics rise/fall together.
- **On-server reproduction is exact.** The registered predictor path reproduces
  the sanctioned off-server driver to 4 decimals (0.21174 vs 0.2117), retiring the
  `scratchpad/litfit/` projection driver the same way the R+M+C registration
  retired `analyze_rmc.py`. The projection is now a reproducible, servable,
  best-models-promotable capability.

## Best on record after this work (RANKING board — PCA-192 champion space, single leak-free split)

| model | recall@10 [95% CI] | MRR | artist@10 / genre@10 | status | run / def |
|---|---|---|---|---|---|
| **★ R′+M+C′ learned content projection (infonce/full/τ0.07) — CHAMPION, now PROMOTED + SERVABLE** | **0.2117** [0.191, 0.233] | **0.1220** | 0.333 / 0.401 | **registered + promoted `blend-gru-markov-content-proj`; best-models rank 1** | `run-20260718-191539-1b89c-seq-blend` |
| — R+M+C z-blend (GRU×Markov×content-kNN) | 0.1859 [0.166, 0.207] | 0.1025 | 0.331 / 0.394 | promoted+servable `blend-gru-markov-content` (best-models rank 2) | `run-20260718-123738-38725-seq-blend` |
| — R+M z-blend α=0.5 on AE-64 (productization anchor) | 0.173 [0.154, 0.192] | 0.096 | 0.322 / 0.397 | prior champion; the 2-leg app slot | `blend-gru-markov` |
| B2 first-order Markov (bar) | 0.107 [0.092, 0.124] | 0.069 | 0.298 / 0.389 | app incumbent | `transit.rs::affinity` |

The 0.212 [0.191, 0.233] CI and paired-Δ vs the R+M+C crown (+0.0259 [+0.012,
+0.041], entirely >0), seed-robustness (×3) and cold-reach are carried from the
lit-fit campaign; this campaign contributes the on-server reproduction,
registration, promotion, the objective/rank/τ robustness confirmation, and (below)
the second-split hardening confirm.

## Second-split hardening

_Added 2026-07-18 (evening). Purpose: the projection crown was measured on a
single leak-free split (`seq-20260715-131139`, 80–100% chronological tail, cold
0.55). This pass re-runs it — plus the two reference blends — on a **SECOND,
disjoint, leak-free split** to settle whether the win is split-specific. This is a
**VALIDATION-ONLY** pass: nothing is promoted or re-promoted; the promoted champion
remains the first-split model._

- **Dataset:** `seq-20260718-211238` — an EARLIER-HOLDOUT 60–80% chronological
  band on the same PCA-192 space (`spotify_tracks_song_pca192`; 5,723 sessions /
  19,402 items; train 4,292 [earliest 60%] / test 1,431 [60–80% band] @ cuts
  2023-04-17 → 2024-08-24). Item space is **byte-identical** to the champion split
  (item_latents.f32 + items.json identical); the **test window is DISJOINT in both
  index and time** from the champion split's 80–100% test (max new-test start
  2024-08-24T00:08 < 2024-08-24T14:28 = min original-test start), and leak-free
  (max train start ≤ min test start). 17/17 parity/quality checks pass.
- **Cold-item rate 0.64** (vs 0.55 on the first split): the earlier window has a
  smaller accumulated single-user catalog → a stronger, independent cold-reaching
  stress test. **By construction, absolute R@10 lands BELOW the first-split 0.212
  on this split — that is EXPECTED and is not a failure.** The hardening verdict is
  the **paired per-session Δ (projection − reference) ON THIS split**, not absolute
  reproduction of 0.212.
- **Serialization:** three runs, strictly one at a time (torch
  thread-oversubscription pitfall). All succeeded, zero failures, zero relaunches;
  n_test uniformly 1,431. Walltime ≈ 6–10 min/run.

**Point metrics (3 runs on `seq-20260718-211238`, sorted by recall@10):**

| config | recall@10 | MRR | artist@10 | genre@10 | hit@10 | n_test | run_id |
|---|---|---|---|---|---|---|---|
| **★ R′+M+C′ projection champion (`blend-gru-markov-content-proj`)** | **0.1705** | **0.1050** | **0.2991** | **0.3417** | **0.1705** | 1,431 | `run-20260718-212639-50e1d-seq-blend` |
| R+M+C crown (`blend-gru-markov-content`) — PRIMARY paired-Δ reference | 0.1279 | 0.0816 | 0.2956 | 0.3410 | 0.1279 | 1,431 | `run-20260718-213156-cf1dd-seq-blend` |
| R+M 2-leg (`blend-gru-markov`) — secondary context reference | 0.1174 | 0.0791 | 0.2844 | 0.3368 | 0.1174 | 1,431 | `run-20260718-214114-93bad-seq-blend` |

Bold row = the projection champion (best per column, all metrics). All three
absolute R@10 land below the first-split values, as expected on the 0.64-cold
split; the ordering (projection > R+M+C > R+M) is preserved and the projection's
lead is *wider* here than on the first split.

**Paired per-session Δ (2000-resample bootstrap, seed 1337 — the campaign
standard; predictions aligned by identical row_id order, verified n=1,431):**

| comparison | PROJ R@10 | REF R@10 | paired-Δ R@10 [95% CI] | CI vs 0 |
|---|---|---|---|---|
| **PROJ vs R+M+C (PRIMARY)** | 0.1705 | 0.1279 | **+0.0426 [+0.0280, +0.0573]** | **entirely > 0** |
| PROJ vs R+M (context) | 0.1705 | 0.1174 | +0.0531 [+0.0391, +0.0678] | entirely > 0 |

**MRR check (no-regression gate):** PROJ MRR 0.1050 vs R+M+C MRR 0.0816 → **+0.0234,
no regression** (PROJ also leads R+M on MRR, +0.0259).

**Verdict — CROWN HARDENED (CONFIRMED).** The decision rule (paired-Δ vs R+M+C on
R@10 with 95% CI entirely > 0 AND MRR not regressing) is met on the second,
disjoint, 0.64-cold leak-free split: paired-Δ **+0.0426 [+0.0280, +0.0573]**, MRR
non-regressing. The projection's advantage is not split-specific — in fact the
margin is **larger** here (+0.0426) than on the first split (+0.0259), consistent
with the mechanism: on a colder split the R+M+C crown's Markov/content-kNN legs
reach fewer targets, and the supervised projection's cold-reaching content leg (C′)
carries proportionally more of the win. Note the correct reading of the absolute
numbers: the champion's on-this-split 0.1705 is BELOW its first-split 0.212 purely
because 0.64 of these targets are cold (vs 0.55) — this is the by-construction
harder split, and the hardening claim rests on the paired-Δ, which is unambiguous.

_This pass is VALIDATION-ONLY: no promotion or re-promotion. The promoted champion
remains `blend-gru-markov-content-proj` trained on the first/canonical split
`seq-20260715-131139`. (The server auto-recompute did slot these three second-split
validation runs into the best-models auto top-12 by their raw recall@10; because
they are cross-split — a different, colder dataset — they warrant exclusion by the
best-model-selector, which does not change the pinned rank-1 champion.)_

## Follow-ups

- ~~**Harden the crown on a second leak-free test split**~~ **DONE — CONFIRMED
  (2026-07-18 evening).** On the disjoint 0.64-cold earlier-holdout
  `seq-20260718-211238`, the projection beats the R+M+C crown by paired-Δ
  **+0.0426 [+0.0280, +0.0573] (CI entirely >0)**, no MRR regression (and beats
  R+M by +0.0531 [+0.0391, +0.0678]). Combined with the prior seed-robustness (×3)
  and cold-reach, the crown is now hardened on two disjoint leak-free splits. See
  "## Second-split hardening".
- **Chase the τ 0.10 nominal edge** — τ 0.10 beat the default by +0.0112 recall /
  +0.0054 MRR (sub-noise on one split). A 3-seed + second-split check would settle
  whether it is a real small gain worth swapping the promoted default, or noise.
  A finer τ sweep (0.12, 0.15) could locate the softness optimum. BPR and rank-64
  are closed — do not revisit.
- **Graded-relevance projection objective** — fit the map toward artist/genre
  adjacency for a graded-preferred variant (from the lit-fit follow-up list; this
  scan showed the graded metrics track the signal-quality axis, so a graded map is
  the natural lever for artist@10/genre@10).
- **Promote R′ (projected single GRU ≈0.176)?** Still the cleanest servable
  *single* next-track model (nearly the old champion blend as one model) — a
  possible lighter-weight deployable alongside the blend champion.
- **Curate the best-models group / re-run report synthesis** — the champion
  changed (new rank-1 servable model), so the best-model-selector should re-curate
  (family diversity, fluke/overfit exclusion — including the three second-split
  validation runs now in the auto slots) and the report-curator should fold this
  campaign into `docs/experiments.tex`.

## Provenance

All Step 1–5 runs on `seq-20260715-131139` (PCA-192) via `POST /api/runs`
overlaying the `blend-gru-markov-content-proj` definition, predictor venv (torch
2.12.0+cpu), one at a time. Verify run reproduces the lit-fit off-server projection
driver (`scratchpad/litfit/`, R′+M+C′ 0.2117) exactly. The **second-split
hardening** runs are on `seq-20260718-211238` (PCA-192, earlier-holdout 60–80%,
cold 0.64) — three definitions (`blend-gru-markov-content-proj`,
`blend-gru-markov-content`, `blend-gru-markov`) launched one at a time, paired-Δ
computed off the on-server `data/runs/<id>/predictions.json` (aligned by row_id,
2000-resample bootstrap seed 1337) via `predictors/.venv/bin/python`. Promotion via
`POST /api/models` (Step 1–3 only); the hardening pass promotes NOTHING.
best-models group auto-recomputed on every completion (deterministic top-12 by
recall@10). No server restart, no `models.toml` / `data/` hand-edits — all
mutations through the API. Prior report:
`2026-07-18-nexttrack-literature-fit-campaign.md`.
