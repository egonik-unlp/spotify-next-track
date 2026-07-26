# PROJECT FACTS — Spotify Next-Track (session recommendation)

Living, agent-maintained roll-up of empirical knowledge for the **next-track /
session-recommendation** instance. The `experiments/*.md` campaign reports are
PRIMARY; this file indexes them. Newest report wins.

_Last updated: 2026-07-26 (latest) - PRE-ENCODER SCAN — **REFUTED, and HARMFUL in the single tower** (`2026-07-26-nexttrack-pre-encoder-scan.md`, NOTHING PROMOTED / NOTHING REGISTERED, champion + leaderboard + pins UNCHANGED). Tested the axis the 2026-07-25 view-split null could not: a **LEARNED nonlinear per-step re-embedding in front of the recurrence** (`[Linear(d_in,width), ReLU(), Dropout] × pre_layers` applied timestep-wise, GRU `input_size = pre_hidden`). NEW predictor params this session: `pre_hidden`/`pre_layers` on **`seq-nexttrack`** (conditionally constructed — no `nn.Module`, no RNG draw at width 0), mirroring `seq-dualgru`'s `pre_hidden_a/b`. 10 runs on `seq-20260715-131139`, ALL on remote worker `snappler:4101029` via `queue:true` serialized (54 min wall clock, 0 failed / 0 interrupted); all 11 param counts re-measured and matching the design exactly. **TWO BIT-IDENTICAL REPRODUCTION GATES BOTH PASSED** — C0 h256 returned recall@10 `0.12299091544374564` = **176/1431**, identical to the last digit (so the `pre_hidden` edit did NOT move the family's on-record baseline; 7th reproduction), and C3 bare-dual returned `0.10272536687631029` = **147/1431**, exactly on-record → all dual verdicts valid. VERDICT — **Rule 7 REFUTED: not ONE of the 6 pre-MLP arms beats C0 by paired-Δ CI>0** (5 lose CI<0, 1 ties). **The decisive single-tower ablation S1 (`MLP256→GRU256`, the one arm unambiguously attributable to the pre-encoder) does not merely tie — it LOSES: 0.10901 vs 0.12299, Δ −0.0140 [−0.0266,−0.0014] CI<0, 156 hits vs 176.** Because a *linear* pre-encoder at `pre_hidden ≥ latent_dim` is provably expressivity-neutral (`W_i·(Vx) = (W_iV)x`, no rank constraint), **the ReLU is the ONLY expressivity change → the rectifier itself is destroying prefix information the GRU's own gated LINEAR input transform preserved**; likely mechanism = PCA-192 latents are zero-centred and near-symmetric, so half-wave rectification zeroes ~half of every input vector's coordinates and `pre_hidden=256` (1.33× input dim) is not over-complete enough to re-encode the discarded negative half-space. Best pre arm P* = T1-f0 (`pre_hidden_b=256`, `fusion_layers=0`) only TIES C0 (0.12369 = 177/1431, Δ +0.0007 straddles) and also ties the on-record h425 and last campaign's bare D-LL-f0 → **the dual side does not move either**. SUB-FINDINGS: (1) **Rule 5 DOSE CURVE = FLAT** — all three adjacent pairs C3(0)→T2(128)→T1(256)→T3(384) straddle 0 AND the endpoints do not separate, with every dosed arm nominally BELOW the bare dual (never above) → no interior peak to chase, and the pre-registered FLAT clause would independently have BLOCKED registration even had a win rule fired; (2) **Rule 4 ASYMMETRY REFUTED as irrelevant** — T1 (A0/B256) vs T4 (A256/B256) straddles on recall (−0.0007), so keeping tower A bare is NOT load-bearing; (3) **Rule 2 INVERTED, and the inversion IS the attribution** — T1-f0 BEATS S1 CI>0 (+0.0147), i.e. the second tower is not redundant over a single pre-encoded tower, but since T1-f0 only TIES plain h256 the bare tower A is **RESCUING** what the rectified tower B destroyed (restoring a direct un-rectified path to the readout), not complementing it — a repair, not co-adaptation; (4) **the ONE solid positive: `fusion_layers=1` regression PERSISTS with a pre-encoder and gets BIGGER** — T1-f0 beats T1 by **+0.0266 [+0.0140,+0.0391] CI>0** vs the previously measured +0.016/+0.020 (3rd confirmation; the more input-side nonlinearity you add, the more the output-side warp costs) → `fusion_layers=0` is the family default, full stop; (5) **Rule 8 did NOT fire for ANY arm — the FIRST campaign to yield ZERO holisticness candidates**, because no arm reaches ΔH CI>0 vs C0 (closest S1 +0.00005, straddles; 4 arms CI<0). Counterfactual: under the OLD 3-factor H, FIVE of these arms would have shown H 'wins' over C0 (0.1444–0.1481 vs 0.14172), every one bought by de-eagering at a recall loss — **the 4-factor `music@10` grounding factor cancels them because music@10 falls in lockstep with recall (0.4558→0.4343 across the batch). The metric cut earned its keep.** It remains INCOMPLETE though: the recomputed all-time 4-factor board (32 facet-carrying runs, re-derived by the runner; every designer-precomputed value reproduced) is STILL topped by the DEGENERATE `cummean/cummean` arm `88a4c` at **H 0.050436 on recall@10 0.06639 = 95/1431** → **the ±0.015 recall floor stays MANDATORY**. (Grounding DID demote `seq-mood` from rank 1 to rank 24, H 0.23319→0.043881.) NEW CAVEAT ON PARAM-MATCHED CONTROLS: the h297 capacity control **C1 itself LOST to C0 CI<0** (0.11321 = 162/1431, Δ −0.0098) — the batch's width points are h256→176, h297→162, h425→173, h454→179 hits, **non-monotone spanning 0.0119 recall with no trend**, so 'single-GRU width is flat on PCA-192' survives as a TREND claim (h454 ties h256) but per-width single-split jitter is ~±0.01 ≈ ⅔ of the noise band → **a width-matched control that is itself a single unreplicated run is NOT a reliable capacity anchor at this effect size** (Rule 1's 'beats C1' leg was consequently nearly free; S1's verdict rests on the reproduction-verified C0 leg). This is the **FIFTH consecutive architecture null** and together with the view-split null it CLOSES the input-side asymmetry direction: adding representation machinery AROUND a single GRU over a good latent space does not help on this corpus — the wins on record came from improving the LATENT SPACE ITSELF (InfoNCE content projection, PCA-192) and from the fixed z-blend. Phase B: B1/B2/B3/B5 did NOT fire (no Rule 1/3/8 win; Rule 5 FLAT); **B4 FIRED** via Rule 7's tie clause (S1 ties C1 within ±0.015; T1-f0 ties C0/h425/D-LL-f0) → SAE read at topk=32 on **T1-f0** `run-20260726-161801-a6423` (taps `pre_b`/`tower_a`/`tower_b`) and **S1** `run-20260726-161801-ad53f` (`model-sae` now taps `pre` ahead of `recurrent`), asking whether the rectified representation is measurably LOSSIER than the bare tower — delegated to information-capture-analyst, analysis-only. B5 NOT fired but the **dropout confound remains un-disentangled and now cuts the other way**: the pre-MLP ends in `Dropout`, so pre arms get GRU-INPUT dropout the bare arms lack (`seq_nexttrack` drops the RNN OUTPUT only) → S1's loss could be partly over-regularization; a `pre_dropout` param (default = `dropout`) is RECOMMENDED over the matched `dropout=0.0` pair, which confounds input- and output-side dropout — an orchestrator change, deliberately NOT improvised. best-models was again STALE after the remote batch (the known `spawn_recompute` trigger gap) — a manual `POST /api/best-models/recompute` absorbed it (5/12 → 12/12 entries) and **FLOODED the group with 7 of this campaign's 10 refuted runs, 3 of them below 0.10 recall** → best-model-selector needed. Prior 2026-07-25 - DUAL-TOWER FUSION SCAN — **REFUTED** (`2026-07-25-nexttrack-dual-tower-fusion-scan.md`, NOTHING PROMOTED / NOTHING REGISTERED, champion + leaderboard + pins UNCHANGED). First real-data outing of the NEW `seq-dualgru` predictor (two INDEPENDENT recurrent towers over configurable causal views {latent, delta, cummean}, per-step hidden states concatenated + fused by one MLP head; reuses `seq_nexttrack.make_batches`/`step_loss` verbatim; strictly causal / no bidirectional mode by design). 10 runs on the canonical PCA-192 split `seq-20260715-131139`, ALL on the SAME remote worker `snappler:4101029` via `queue:true` serialized (~59 min wall clock, 0 failed / 0 interrupted). Design = the complete upper triangle of the {latent,delta,cummean}² view matrix at `fusion_layers=1` (6 cells) + a fusion-depth axis (2 cells at `fusion_layers=0`) + TWO single-GRU controls: C0 h256 (baseline) and **C1 h425 = a 0.098% PARAM-MATCHED capacity control** (871,017 vs the dual arms' 871,872 params, re-measured by instantiating the real classes at latent_dim=192). VERDICT — **Rule 3 REFUTED: not ONE of the 8 dual arms beats C0 by paired-Δ CI>0** (6 arms CI<0 outright; the 2 `fusion_layers=0` arms only TIE: best dual arm D-LL-f0 0.11880 vs C0 0.12299, Δ −0.0042 [−0.0126,+0.0042] straddles 0) **AND C1 h425 also TIES C0** (0.12089, Δ −0.0021 [−0.0112,+0.0063] straddles 0) → capacity is NOT the missed lever either and **single-GRU WIDTH IS NOW CLOSED ON PCA-192** (previously only closed on AE-64 by the 2026-07-13 topology scan). This is the **FOURTH independent learned-combiner null** on this corpus (XGB stacker 0.08–0.09 below its own base leg / RRF −0.029 / ORACLE test-fit blend weights +0.004 straddling 0 / now representation-level JOINT end-to-end fusion) and it closes the standing "of course a post-hoc combiner fails, train the fusion end-to-end" rebuttal → materially CONSOLIDATES "the fixed equal-thirds z-blend is the combiner of record". THREE sub-findings survive: (1) **the MLP fusion head is ACTIVELY HARMFUL, not merely useless** — both f0 arms beat their f1 twins by paired-Δ CI>0 (+0.01607 latent/latent, +0.02027 latent/cummean) at 10% FEWER params → **default `fusion_layers=0`**; (2) **the raw `latent` view SUBSUMES `delta` and `cummean` for a recurrent tower** — Rule-4 view attribution: `latent/cummean` (the design's "best hope", the pooled view `seq-ann` was built from) and `latent/delta` both TIE their latent/latent parent and beat only the weaker parent (raw tower carries the cell, second view is dead weight), BUT **`delta`+`cummean` IS genuinely complementary** (D-DC beats BOTH parents CI>0, +0.0287 vs D-DD / +0.0203 vs D-CC) just at a hopeless absolute level (0.0867) → the co-adaptation mechanism is REAL, the raw latent view simply already integrates movement + running mean internally; (3) **holisticness@10 and recall@10 are ANTI-CORRELATED across the batch** — 5 arms lift H by paired-ΔH CI>0 topping out at D-CC **H 0.19715 (+0.0554)** which would rank #2 on the whole crown board, but at recall@10 0.06639 = **95 of 1,431 vs the baseline's 176**, and **EVERY H-winning arm FAILS Rule 6's ±0.015 recall floor → Rule 6 did NOT fire, crown UNTOUCHED**; the floor caught exactly the `mood-session` degenerate mode (H 0.2332 at recall 0.0238) it was written for → any future holisticness objective on this corpus MUST be recall-constrained. Phase B: B1/B2/B3 did NOT fire (no Rule 1, no Rule 6, and C1 does not beat C0 so the width curve is flat — do not spend a run on h512); **B4 FIRED** (mixed-view D-LC-f0 TIES C1 within the band) → SAE tower-specialization read on `run-20260725-143502-3e5d4-seq-dualgru` (topk=32) delegated to information-capture-analyst, analysis-only. **PRIMARY-METRIC CONTRADICTION RECONCILED** (explicit campaign deliverable — see §Target & task and §Best-models group): the SERVER-EFFECTIVE primary metric is **holisticness@10** (`GET /api/best-models` returns `"primary_metric":"holisticness@10"` and ranks `seq-mood` H 0.2332 at rank 1 ABOVE a 0.212-recall blend; `domain.toml` `[metrics].primary` = `holisticness@10` since 2026-07-23) — this file's earlier "recall@10 stays primary" statements were TRUE WHEN WRITTEN and are now SUPERSEDED; the experimental record's comparability currency remains recall@10 (the whole leaderboard is denominated in it). ALSO WITHDRAWN: the "cross-host reduction-order offset −0.0028" — C0 on a FRESHENED worker checkout returns **0.12299, bit-identical to the hub**, so the prior remote 0.1201957 (`run-20260723-002044-af11a`/`-17cf8`) was a **CODE difference (stale `~/lensing-worker/predictors/`), NOT a host effect**; those two runs are not code-comparable to current batches. ALSO FLAGGED: the best-models group is **STALE**, not flooded (`updated_at` 2026-07-23T01:54, 2/12 entries filled, despite 10 new holisticness-carrying runs) → best-model-selector needed. Prior 2026-07-23 - MODEL-SAE INTERPRETABILITY NOTE (analysis-only, NOTHING PROMOTED, champion/leaderboard UNCHANGED, recall@10 stays primary): the per-model SAE over the GRU's recurrent hidden state does NOT sparsify under L1 — an L1 sweep 0.0015→0.05→0.5 (330×) only moved l0 82%→35% at var_explained ~0.99, so `n_interpretable_concepts` saturates near the full dictionary (treat it as saturated; trust ranked `atoms_by_concept` + `next_item_decodability`). Added a TOP-K SAE mode (`SAE(topk=)` in `seq_model_sae.py`, `--topk` on seq_nexttrack/seq_ann `model-sae`, AND wired into the server API `ModelSaeRequest.topk` — server rebuilt + gated-restarted): k=32 gives l0=32 exactly with selective atoms (freq ~2% vs ~50% under L1) at reconstruction cost var_explained 0.90. Also detail cap `SHOWN_N` 12→40 and `CONCEPT_SEP` 0.5→0.35 (that's why runs "detail" 40 not 8). See `docs/sae-interpretability-note.{md,pdf}`. Prior 2026-07-22 - SESSION-HOLISTICNESS METRICS + ANTI-EAGER LEVERS CAMPAIGN (`2026-07-22-session-holisticness-and-antieager-levers.md`, MEASUREMENT + exploratory, NOTHING PROMOTED, champion/leaderboard UNCHANGED, recall@10 stays primary). New DISPLAY-ONLY holisticness columns wired into `seq_common.py::eval_from_scores` + run.rs/domain.rs/domain.toml/UI like music@10: artist_adj@10 (top-10 fraction sharing the SEED artist, LOWER=less eager), mood_coh@10 (top-10 mean cosine to prefix mood centroid, higher=better), ild@10 (intra-list sonic diversity, higher=less duplicative); album_adj@10 omitted on `seq-20260715-131139` (no album field, EXPECTED); suffix_recall@10/cont_prec@10 via the on-demand `seq_continuation_eval.py` harness only. Sanity gate PASSED (music/mood/ild populate → content-metric Qdrant loaded). FINGERPRINT (all on the canonical PCA-192 split, seed 1337, n_test 1431): eagerness TRACKS recall INSIDE the blend family — champion `blend-gru-markov-content-proj` (0.2117) is the MOST artist-eager (artist_adj 0.610), R+M 0.579 — BUT the eagerness is the MARKOV bigram leg, not the RNN: standalone markov artist_adj 0.543 vs single GRU/LSTM ≈0.345 (the RNNs are the LEAST-eager, HIGHEST mood_coh 0.479/0.458 models). GRU≈LSTM on PCA-192 (recall 0.123 vs 0.119 TIE within ±0.015; artist_adj 0.347 vs 0.344) → the GRU-beats-LSTM-BECAUSE-more-eager hypothesis is NOT supported. artist_adj/ild are NOT trivially gamed (popularity maxes ild 0.898 / min artist_adj 0.003 but ≈0 recall, NEGATIVE mood_coh −0.041). LEVERS: MMR eval-re-rank λ (mmr_lambda) is the STRONG knob — offline sweep on the champion checkpoint (validated: λ=off reproduces 0.2117): λ0.9 essentially FREE (recall 0.2103 TIE, ild 0.406→0.436, eagerness 0.610→0.603, music@10 0.454→0.462), λ0.7 ~2 recall pts (0.1936) for ild→0.520 + de-eager→0.571, λ0.3 crushes eagerness→0.293/ild→0.815 at recall 0.110; music@10 RISES as you diversify. Anti-eager TRAIN regularizer β (eager_beta) is a WEAK knob (β0→0.2 moves GRU artist_adj only 0.347→0.340 at flat recall). Multi-step continuation: GRU>markov (suffix_recall@10 0.096 vs 0.087, cont_prec 0.027 vs 0.024). FLAGGED for a future USER promotion decision (promoted nothing): MMR λ≈0.9 as a near-free holisticness upgrade (needs a native run + 3-seed confirm first); β not recommended. Prior 2026-07-18 (night) - REPRESENTATION-EXPLORATION VERDICT IN (`2026-07-18c-nexttrack-representation-exploration.md`, VALIDATION-ONLY, NOTHING PROMOTED): tested the 8 alt item-latent spaces + baseline (18 runs, SERIALIZED, 0 failures, all n_test 1431) under BOTH the champion projection `blend-gru-markov-content-proj` (readout A = the space's ceiling under the winning recipe) and the frozen `gru-infonce-h256` single (readout C = raw representation), paired-Δ on exact hit@10 (2000 boot, rng 1337) vs the same-readout baseline (baseline re-run in-batch, reproduced 0.21174 / 0.12299). TWO VERDICTS: (1) EXACT — on readout A NO variant beats the PCA-192 champion with paired-Δ CI>0 (best nominal V3 std-noaco +0.0049 / V5 balanced +0.0042, both straddle 0; V4 textcat −0.0168 and V6 sonic-64 −0.0182 both CI<0 = REGRESS), so the item representation is NOT hindering the DEPLOYED champion — the InfoNCE projection already re-weights away the loud acoustic/numeric directions; on readout C SIX of 8 variants BEAT the frozen GRU by paired-Δ CI>0 (+0.025…+0.036; biggest V5 balanced +0.0356 [+0.0189,+0.0517] and V3 std-noaco +0.0342 [+0.0154,+0.0545]), so at the RAW level the full-matrix PCA-192 geometry IS a handicap and whitening / acoustic-drop is the fix — BUT the gain is fully ABSORBED by the projection (lifts C, not A → the pre-registered 'projection already compensates for the loud directions', NOT the strong both-readouts result). (2) GRADED — whitening lifts frozen-GRU genre@10 to ~0.47 (V7 std-txtcatup 0.4717, V3 std-noaco 0.4710 vs base 0.4500), and V10 text-only is the readout-A graded-preferred space (artist@10 0.3466 / genre@10 0.4144, both readout-A maxima, at −0.0119 exact). NO promotion candidate (nothing beats the champion CI>0 on readout A) — CHAMPION ROW UNCHANGED; best open direction = fit the projection ON a whitened/std-noaco/metric space (compose the two fixes). See §'Representation / dataset lineage'. Prior 2026-07-18 (evening) - REPRESENTATION-EXPLORATION DATASETS BUILT (item-vector block composition, DATASETS ONLY - nothing trained/promoted). 8 dim-192 sequence item-space variants on the CHAMPION split (17/17 parity checks PASS on every one: sessions/offsets/train/test/items.json byte-identical to seq-20260715-131139, only item_latents.f32 differs, manifest 7154/19402/5723/1431 @ cut 2024-08-24T14:28:02, cold 0.5507), testing whether the equal-weighted 4-block PCA item vector hinders next-track prediction. EVR preflight confirmed the hypothesis DECISIVELY: the RAW input variance budget is 44% acoustic + 43% numeric vs only 2.5% text (text is L2-normalized -> tiny per-dim scale), so full-matrix PCA-192 spends 87% of its budget reconstructing numeric+acoustic to R2=1.000 while text reaches only R2=0.72, and the top-10 PCs (65% of variance) are all popularity/era/loudness directions with ~0 text loading-mass; per-column standardize flips the captured budget to 73% text / 21% categorical. Variants -> seq ids: V1 noaco seq-20260718-222742, V2 std -222754, V3 std-noaco -222807, V4 textcat -222820, V5 balanced-pca192 -222834, V7 std-txtcatup -222848, V10 text-only -222901, V6 sonic-64 -222905 (64-d). Backward-compatible builder edits: build_song_pca.py --blocks/--block-weights/--standardize (DEFAULT output byte-identical, verified cosine 1.0 vs the live champion collection), new reduce_named_vector.py, sequences.py --vector-name. Handed to experiment-runner for the representation verdict (each vs the PCA-192 champion under proj-blend / content-kNN / frozen gru-infonce-h256, reporting exact recall@10 AND graded artist@10/genre@10). See 'Representation / dataset lineage'. Prior 2026-07-18 (evening, later) — R′+M+C′ CROWN HARDENED ON THE SECOND SPLIT (CONFIRMED): re-run on the disjoint 0.64-cold earlier-holdout `seq-20260718-211238`, the projection champion beats the R+M+C crown by paired-Δ on R@10 **+0.0426 [+0.0280, +0.0573] (CI entirely >0)** with NO MRR regression (point metrics — PROJ 0.1705/0.1050, R+M+C 0.1279/0.0816, R+M 0.1174/0.0791; all n_test 1431; runs `run-20260718-212639-50e1d` / `-213156-cf1dd` / `-214114-93bad`). Absolute R@10 lands below the first-split 0.212 BY CONSTRUCTION on the colder split — the verdict is the paired-Δ, not absolute reproduction. The margin is WIDER than on the first split (+0.0259) — the cold-reaching projection carries proportionally more of the win on the colder split. VALIDATION-ONLY: nothing promoted/re-promoted; the champion stays `blend-gru-markov-content-proj` trained on the first split. (Server auto-recompute slotted the three second-split validation runs into the best-models auto top-12 by raw recall@10 — they are cross-split and warrant exclusion by best-model-selector; pinned rank-1 champion unchanged.) See `2026-07-18b-...projection-registration-and-scan.md` §Second-split hardening. Prior same-evening — SECOND LEAK-FREE SPLIT BUILT (`seq-20260718-211238`): an EARLIER-HOLDOUT 60–80% chronological split on PCA-192 (5723 sessions / 19402 items, train 4292 / test 1431 @ cuts 2023-04-17T19:26:24 → 2024-08-24T14:28:02, cold-item rate 0.64; item space byte-identical to the champion split), TEST WINDOW DISJOINT (index + time) from the champion's `seq-20260715-131139` (80–100%) — 17/17 parity checks pass, READY for the standing R′+M+C′ second-split hardening (experiment-runner re-runs the champion + R+M+C crown on it, verdict = paired-Δ ON THIS split). Prior same-day — PROJECTION REGISTERED + PROMOTED + SCANNED (`2026-07-18b-nexttrack-projection-registration-and-scan.md`): the learned content projection reproduced ON-SERVER EXACTLY (R@10 0.21174 / MRR 0.12199, matches the off-server 0.2117) and is now REGISTERED + PROMOTED + SERVABLE as `blend-gru-markov-content-proj` (best-models rank 1, above the R+M+C model 0.186 rank 2); an objective/rank/τ ablation scan CONFIRMS the InfoNCE/full-rank/τ0.07 default (InfoNCE ≫ BPR — BPR collapses to ~0.15, below even the R+M+C crown; full-rank ≥ rank-64; τ 0.05→0.193 / 0.07→0.212 / 0.10→0.223, the τ0.10 top-cell a sub-noise +0.011 nominal edge → default HELD). Prior — LITERATURE-FIT CAMPAIGN (`2026-07-18-nexttrack-literature-fit-campaign.md`): a supervised **learned content projection** is a NEW best-on-record ROW — **R′+M+C′ z-blend on PCA-192, R@10 0.212 [0.191, 0.233], MRR 0.122**, paired-Δ vs the R+M+C champion **+0.0259 [+0.012, +0.041] (>0)**, and — unlike the champion — SEED-ROBUST (3 seeds, all CI>0) and COLD-REACHING (content leg C′ scores cold 0.152 ≈ warm 0.151). Off-server driver (not yet registered → same productization gap the R+M+C row had before registration). Every borrowed-method lane CLOSED: session-kNN/V-SKNN (drags blend), rank-fusion/RRF (−0.029), ranking-aware combiner (even ORACLE test-fit weights +0.004, CI straddles 0), Markov KN-smoothing (bar 0.107→0.110, washes out), GRU importance negatives (blend tie). Earlier 2026-07-18 (am): (1) registered R+M+C as `seq-blend` leg + definition `blend-gru-markov-content` (`run-20260718-123738-38725-seq-blend` reproduces 0.1859/0.1025 EXACTLY); (2) added the RANKING SERVING CONTRACT — the whole sequence family is promotable + servable via `POST /api/models/{name}/predict`; champion + best-single LSTM promoted, best-models group populated. Prior crown: `2026-07-15c-nexttrack-phase2-model-sweep.md`._

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

> **PRIMARY-METRIC RECONCILIATION (2026-07-25, `2026-07-25-nexttrack-dual-tower-fusion-scan.md`).**
> The `[metrics].primary = "recall@10"` statement above, and every
> "recall@10 stays primary" line elsewhere in this file, were TRUE WHEN WRITTEN and
> are now **SUPERSEDED**. As of 2026-07-23 the **SERVER-EFFECTIVE primary metric is
> `holisticness@10`**: `domain.toml` `[metrics].primary` was switched to it, and
> `GET /api/best-models` returns `"primary_metric": "holisticness@10"` — verifiably
> so, since the live group ranks `seq-mood` (H 0.2332, recall@10 0.0238) at rank 1
> ABOVE a `seq-blend` at recall@10 ≈0.212, i.e. the ordering is INVERTED relative to
> recall. `holisticness@10` = `clamp(mood_coh@10, 0) × ild@10 × (1 − artist_adj@10)`
> (product of the three run-aggregate means — NOT a mean of per-row products;
> verified to 6 decimals against stored `holisticness_at_k`).
> **The experimental record's comparability currency remains recall@10** — the whole
> "Best on record" leaderboard is denominated in it, and cross-campaign comparisons
> must stay on it. Treat the two boards SEPARATELY and pre-register both, as the
> dual-tower campaign did (its Rule 3 judged the science on recall@10, its Rule 6
> judged the board event on holisticness@10 **with a hard ±0.015 recall floor**).
> ⚠ **Unconstrained holisticness@10 is maximized by predicting WORSE** — across the
> dual-tower batch the H ordering is close to the reverse of the recall ordering, and
> the group's own rank-1 `mood-session` is a degenerate near-zero-recall model. Never
> read a holisticness win without its recall floor.

**Graded-relevance metrics (first-class since 2026-07-14):** alongside exact
recall@k / mrr / hit@k, the eval credits landing the right band/vibe —
**artist@k** (some top-k candidate shares the truth's artist), **genre@k**, and
**artist_mrr** — over the same prefix-excluded ranking. `[metrics].columns =
[recall@10, artist@10, genre@10, music@10, mrr, hit@10]` (music@10 added 2026-07-20,
see below); wired through
`seq_common.eval_from_scores` → Rust `Metrics`/`MetricsSpec` → `domain.toml` → UI.
These are the HONEST scorecard: exact recall is modest by construction (55% cold
items), but the champion lands the right **artist ~32%** / **genre ~40%** of the
time in the top-10 (artist_mrr ~0.30) — far more useful than the 17% exact number.
Source: `2026-07-14-nexttrack-graded-relevance-and-new-legs.md`.

**`music@10` — continuous "sounds-alike" relevance (2026-07-20).** A 4th
graded-relevance column: mean over test cases of the best (max) cosine, in the
`balanced` intrinsic MUSICAL-DISTANCE space, between the true next track and the
top-10 candidates. A smooth SUPERSET of recall@10 (exact hit → 1.0); credits a
prediction that *sounds like* the truth on an exact/artist/genre miss. 0..1,
higher-better, DISPLAY-ONLY (primary stays recall@10). The distance is a proper
metric (whitened-Euclidean over the Song-AE latent + text + metadata + genre
prototype; Qdrant `spotify_tracks_content_metric`, built by
`pipeline/build_content_metric.py`, dials balanced/tight/sonic). Wired
`seq_common.eval_from_scores` (best-effort — omitted if the index is absent, never
breaks eval) → `Metrics.music_at_k` → `domain.rs::extract` → `domain.toml`
columns → UI. `[metrics].columns` now `[recall@10, artist@10, genre@10, music@10,
mrr, hit@10]`. Companion `GET /api/pathfinder/similar` serves "songs like this".
**Finding:** the recall@10 champion (`seq-blend`, 0.223) is NOT the music@10
champion — `seq-nexttrack` (predicts the content latent directly) lands the
musically-closest misses (best music@10 0.482, family mean 0.448); corr(recall@10,
music@10)=0.609 over the 53 backfilled runs; `seq-popularity` worst (0.185).
Source: `2026-07-20-musical-distance-metric.md` (full 53-run backfill table).

## Framework capability added here (build-concrete-first; upstream later)

Additive sequence/ranking support in this instance's `crates/` (Phase 2):
ranking metrics (`Metrics.recall_at_k/mrr/hit_rate` + `MetricsSpec` maps +
`compute_ranking_metrics`), `SequenceDataset` (parallel artifact family, pointwise
`Dataset` untouched), `Task::Ranking` + `[sequence]` domain block, the
`pipeline/corpus/sequences.py` offline producer, and the `seq-nexttrack` torch GRU
predictor + baselines. To be generalized + upstreamed to the mother lensing
framework (Phase 5).

**Ranking serving contract (2026-07-18).** The whole sequence family
(`seq-blend`, `seq-nexttrack`, `seq-ann`, `seq-markov`, `seq-popularity`,
`seq-stacker`) now satisfies the model interface — promotable + servable, not
just train-only. Mechanism: `promote()` detects a sequence dataset
(`sequence-manifest.json`) and bakes the whole artifact (item_latents + items +
session arrays) alongside the run checkpoint into the model dir
(`models.rs::promote_ranking`); `POST /api/models/{name}/predict` takes an ordered
`prefix` (Spotify URIs / ids / vocab indices), forwards it to the predictor's new
`predict` subcommand (skipping the pointwise Featurizer), and the predictor
`load_artifact(model_dir)`s to rebuild its exact vocab + train-only legs and rank
via the shared `seq_common.predict_ranking` (same next-distinct rule as eval →
byte-identical to R@10). The response enriches `top_k_ids` → track identity
(uri/name/artist/genre) from the baked items.json. The stacker snapshots its
neural base legs (`base_gru.pt`/`base_ann.pt`) so it serves without retraining.
Registry gains `predict_args` per predictor (baselines carry `--mode`). Blends are
NOT ONNX-exportable (export cleanly refuses). With the family now promotable the
best-models group + auto-promotion populate for ranking (previously empty).
Vocab is frozen at promote; new corpus tracks ride the existing refresh→rebuild→
retrain path. To be generalized + upstreamed (Phase 5).

## Best on record (RANKING board — next-distinct, leak-free session split)

Artifact `data/seq/seq-20260713-014226` (7,154 sessions / 19,402 items;
chronological split 5,723 train / 1,431 test @ cut 2024-08-24; cold-item rate
0.55). Source: `experiments/2026-07-13-nexttrack-sequence-sweep.md`. The CHAMPION ROW
(best on record) is now **R′+M+C′ (learned content projection) on PCA-192**
(2026-07-18 lit-fit campaign; NOW REGISTERED + PROMOTED + SERVABLE as `blend-gru-markov-content-proj`, best-models rank 1 — see `2026-07-18b-...projection-registration-and-scan`); the prior
crown R+M+C on PCA-192 is the promoted+servable model (`blend-gru-markov-content`);
the productization anchor is R+M on AE-64. Sources:
`2026-07-18-...literature-fit-campaign`, `2026-07-15c-...phase2-model-sweep`.

| model | Recall@10 [95% CI] | MRR | artist@10 / genre@10 | note |
|---|---|---|---|---|
| **★ R′+M+C′ learned content projection on PCA-192 (CHAMPION ROW — best on record)** | **0.212** [0.191, 0.233] | **0.122** | 0.333 / 0.401 | GRU′×Markov×content′ thirds, where ′ = retrieved in a supervised LINEAR PROJECTION of the PCA-192 content space (InfoNCE metric-learning over train consecutive pairs; symmetric map, rank 192; leak-free). paired-Δ vs the R+M+C crown **+0.0259 [+0.012, +0.041] (entirely >0)**, MRR lead. **SEED-ROBUST** (3 seeds 1337/7/99: +0.026/+0.025/+0.029, all CI>0 — a genuine seed-varied confirm, STRONGER than the deterministic champion's) and **COLD-REACHING** (the projected content leg C′ scores cold targets 0.152 ≈ warm 0.151, vs Markov cold 0.010 — genuine content generalization, not memorization). Components: C′ standalone 0.152 (vs unsup content-kNN 0.069); projected single GRU R′ ≈0.176 (vs base GRU 0.123 — nearly the OLD champion blend as a single model); adding C′ as a 4th leg to the promoted champion (R+M+C+C′) already gives 0.206 [+0.020 CI>0]. NOW REGISTERED + PROMOTED + SERVABLE as `blend-gru-markov-content-proj` (2026-07-18b): the projection is a first-class `seq-blend` capability (hyperparams `projection`/`projection_rank`/`projection_objective`/`projection_tau`/`projection_epochs`/`projection_lr`; `projection=true`⇒`content=true`; map baked into the model dir as `projection.npz` for serving). An on-server run reproduces the off-server driver EXACTLY — R@10 0.21174, MRR 0.12199, artist@10 0.3326, genre@10 0.4011 (`run-20260718-191539-1b89c-seq-blend`) — and tops the best-models group (rank 1, vs R+M+C 0.186 rank 2). SCAN (2026-07-18b, default InfoNCE/full/τ0.07): InfoNCE ≫ BPR (BPR 0.152–0.158, below even the R+M+C crown); full-rank ≥ rank-64 (−0.006, tie); τ 0.05→0.193 / 0.07→0.212 / 0.10→0.223 (τ0.10 +0.011 nominal, sub-noise → default HELD). **SECOND-SPLIT HARDENED — CONFIRMED (2026-07-18 evening):** on the disjoint 0.64-cold earlier-holdout `seq-20260718-211238`, paired-Δ vs the R+M+C crown **+0.0426 [+0.0280, +0.0573] (CI entirely >0)**, no MRR regression (PROJ 0.1705/0.1050 vs R+M+C 0.1279/0.0816; vs R+M +0.0531 [+0.0391, +0.0678]); absolute R@10 below 0.212 by construction on the colder split (verdict is the paired-Δ), margin WIDER than the first split (+0.0259) → crown now hardened on TWO disjoint leak-free splits (validation-only, nothing promoted). `seq-20260715-131139` (train) + `seq-20260718-211238` (2nd split); `scratchpad/litfit/`; `2026-07-18-literature-fit-campaign`, `2026-07-18b-projection-registration-and-scan` §Second-split hardening |
| **— R+M+C z-blend on PCA-192 (prior crown — promoted+servable)** | **0.186** [0.166, 0.207] | **0.103** | 0.331 / 0.394 | GRU×Markov×content-kNN thirds; CONFIRMED 5× (GRU-leg reproductions, spread 0), paired-Δ vs prior champion **+0.013 [+0.0007,+0.026] every rerun** + MRR lead. **NOW REGISTERED + SERVABLE (2026-07-18): the content leg is a first-class `seq-blend` hyperparam (`content=true`/`content_agg=max`, reuses `seq_models.content_knn_scorer`), definition `blend-gru-markov-content` on PCA-192; an on-server run reproduces it EXACTLY — R@10 0.18588, MRR 0.10247, artist@10 0.3305, genre@10 0.3941 (`run-20260718-123738-38725-seq-blend`), retiring the off-server `analyze_rmc.py` path. PROMOTED as `blend-gru-markov-content` and served via `POST /api/models/{name}/predict` (see "Ranking serving contract" below); it now tops the best-models group.** On the 2nd split (`seq-20260718-211238`, cold 0.64) it scores 0.1279/0.0816 — the PRIMARY reference the projection hardening beat by paired-Δ +0.0426 (CI>0). Caveats: reproduction spread 0 (seq-nexttrack deterministic → tested determinism not seed-robustness); paired-Δ lower bound +0.0007 (thin, single split). `seq-20260715-131139`; `2026-07-15c` |
| — R+M+C z-blend on PCA-128 | 0.182 [0.162, 0.203] | 0.099 | 0.332 / 0.396 | content additive on the rich space (+0.015 [+0.004,+0.028] over R+M-128); paired-Δ vs champion straddles 0 |
| — R+M z-blend α=0.5 on AE-64 (prior champion / PRODUCTIZATION ANCHOR) | 0.173 [0.154, 0.192] | 0.096 | 0.322 / 0.397 | GRU-infonce × train-only Markov, 3-seed confirmed; the 2-leg blend that maps onto the app's transition slot. The 3-leg R+M+C is now registered AND servable as promoted model `blend-gru-markov-content` (ranking serving contract, 2026-07-18), so it — not just the 2-leg — is now a deployable next-track model. On the 2nd split (`seq-20260718-211238`) the R+M 2-leg scores 0.1174/0.0791 (secondary hardening reference; projection beats it by paired-Δ +0.0531 CI>0). `seq-blend` / `blend-gru-markov` |
| — R+M+C (+content-kNN) on AE-64 (graded-preferred) | 0.174 [0.154, 0.194] | 0.101 | **0.330** / 0.400 | on AE-64 the content leg only TIED the champion on exact recall (paired-Δ +0.0014) — the rich PCA space makes it pay off (see PCA-192 R+M+C) |
| — R+M z-blend on PCA-192 | 0.171 [0.151, 0.192] | 0.095 | 0.329 / 0.398 | ≈ champion (tie, paired-Δ −0.0014); rich-space R+M. `run-...96f53` |
| — R+M z-blend on PCA-128 | 0.167 [0.148, 0.187] | 0.092 | 0.329 / 0.398 | ≈ champion (tie); the single-GRU gain does NOT propagate to the R+M blend |
| B2 first-order Markov (bar) | 0.107 [0.092, 0.124] | 0.069 | 0.298 / 0.389 | `transit.rs::affinity`, train-only; the app's incumbent |
| — R+A+M z-blend | 0.168 | 0.094 | — | adding the ANN leg (redundant w/ GRU) slightly HURTS the champion |
| — R+M+I (+item2vec) | 0.135 | 0.087 | 0.327 / 0.399 | co-listening item2vec is a noisy low-ceiling leg — DRAGS the blend down |
| — R+M2 / R+M+M2 (2nd-order Markov) | 0.173 | 0.099 / 0.098 | 0.322 / 0.321 | trigram ≈ neutral |
| **GRU LSTM on PCA-128 (best SINGLE model)** | **0.127** [0.110, 0.145] | 0.050 | 0.335 / 0.458 | LSTM > GRU on PCA-128 (paired-Δ +0.013 [+0.0007,+0.024] CI>0); new best single, still ≪ the blend. `run-...546d0` |
| GRU infonce h256 on PCA-192 | 0.123 [0.107, 0.140] | 0.053 | 0.338 / 0.450 | reproduced twice (15b + 15c confirm); paired-Δ vs PCA-128 grazes 0 |
| — single GRU **h425** on PCA-192 (CAPACITY CONTROL, 2026-07-25) | 0.1209 (173) | 0.052 | 0.341 / 0.458 | **TIE with h256** — paired-Δ vs C0 −0.0021 [−0.0112, +0.0063] straddles 0, at 871,017 params (2.2× h256's 394,944). **Single-GRU WIDTH is now CLOSED on PCA-192**, as the 2026-07-13 topology scan had closed it on AE-64; the width curve is FLAT at the h256→h425 step, so h512 is not worth a run. **ADDENDUM 2026-07-26:** the flat-TREND conclusion stands (h454 also ties h256, +0.0021 straddling), but the curve is NOT smooth — h256→176 hits, h297→**162**, h425→173, h454→179, i.e. non-monotone over 0.0119 recall, and the h297 point LOSES to h256 CI<0. Per-width single-split jitter ≈±0.01 ⇒ a lone width-matched run is a weak capacity anchor (see §Pitfalls). holisticness@10 0.1385. `run-20260725-143502-be11b`; `2026-07-25-...dual-tower-fusion-scan` |
| — dual-tower GRU 2×256 `fusion_layers=0` latent/latent (BEST DUAL ARM, 2026-07-25) | 0.1188 (170) | **0.056** | 0.337 / 0.456 | **TIE with BOTH controls** (vs C0 −0.0042 [−0.0126,+0.0042]; vs the param-matched C1 −0.0021 [−0.0112,+0.0077]) — a tie inside the noise band is NOT a win, so **NOTHING was registered or promoted**. Notable only in that its MRR 0.0563 is the batch maximum, above C0's 0.0534 and C1's 0.0522 (the sole metric on which a dual arm leads; no paired MRR CI is computable — `predictions.json` carries no rank field and MRR is NOT reconstructible from `top_k_ids` order, 0.0461 vs stored 0.0534). 789,696 params. holisticness@10 0.1377. `run-20260725-143502-18b4a` |
| — dual-tower GRU 2×256 `fusion_layers=1`, all 6 view cells (2026-07-25) | 0.058–0.1027 | 0.027–0.042 | 0.246–0.309 / 0.406–0.462 | **ALL LOSE to the h256 baseline, paired-Δ CI<0 every cell.** Ranked: latent/latent 0.1027, latent/cummean 0.0971, latent/delta 0.0915, delta/cummean 0.0867, cummean/cummean 0.0664, delta/delta 0.0580. The MLP fusion head is the main culprit (removing it recovers +0.016…+0.020, CI>0 both regimes). holisticness@10 runs the OTHER way, 0.1334–0.1972 — the highest-H cell (cummean/cummean 0.1972) is the WORST-recall real arm (95 of 1,431). `run-20260725-143502-{12fe6,a95c5,8d214,e9cfb,88a4c,4324f}` |
| — single GRU **h454** on PCA-192 (CAPACITY CONTROL, 2026-07-26) | 0.1251 (179) | 0.054 | 0.338 / 0.454 | **TIE with h256** — paired-Δ +0.0021 [−0.0070, +0.0112] straddles 0, at 969,936 params (2.5× h256). Batch-max nominal recall but a CONTROL, not a candidate: authorizes nothing. Extends the width curve to a 4th point and confirms the flat TREND while exposing the ±0.01 per-width jitter. 4-factor holisticness 0.045765. `run-20260726-161801-7646d`; `2026-07-26-...pre-encoder-scan` |
| — single GRU **h297** on PCA-192 (CAPACITY CONTROL, 2026-07-26) | 0.1132 (162) | 0.052 | 0.328 / 0.450 | **LOSES to h256, paired-Δ −0.0098 [−0.0175, −0.0028] CI<0** at 494,697 params. The batch's low outlier and the reason param-matched width controls are now flagged as a weak instrument. 4-factor holisticness 0.045285. `run-20260726-161801-7bab3` |
| — single **`MLP256→GRU256`** pre-encoder on PCA-192 (DECISIVE PRE-MLP ABLATION, 2026-07-26) | 0.1090 (156) | 0.045 | 0.324 / **0.469** | **LOSES to plain h256, paired-Δ −0.0140 [−0.0266, −0.0014] CI<0** (156 vs 176 hits) at 493,504 params — a nonlinear per-step pre-encoder is ACTIVELY HARMFUL in a single tower. Since a *linear* pre-encoder is provably expressivity-neutral here, the ReLU is the only expressivity change ⇒ the rectifier destroys signed prefix information the GRU's gated linear input transform preserves. Highest genre@10 in the batch (0.4689) and the batch-max 4-factor holisticness **0.047132** — but ΔH vs h256 STRADDLES (+0.00005), so no crown. `run-20260726-161801-ad53f`; `2026-07-26-...pre-encoder-scan` |
| — dual-tower A-bare / B-preMLP256, `fusion_layers=0` (BEST PRE ARM, 2026-07-26) | 0.1237 (177) | **0.055** | 0.333 / 0.465 | **TIE with h256** (paired-Δ +0.0007 [−0.0119, +0.0133]), TIE with the h425 control (+0.0028) and TIE with the 2026-07-25 bare dual f0 (+0.0049) → **nothing registered or promoted.** It DOES beat the bare f1 dual +0.0210 CI>0 and beats the single pre-MLP tower +0.0147 CI>0 — the bare tower A **rescues** what the rectified tower B destroyed (a repair, not co-adaptation). Batch-max MRR 0.0547, exactly echoing 2026-07-25's f0 arm. 888,256 params. 4-factor holisticness 0.044541. `run-20260726-161801-a6423` |
| — dual-tower pre-MLP arms at `fusion_layers=1` (dose 128/256/384 + symmetric, 2026-07-26) | 0.0971–0.0999 | 0.038–0.045 | 0.294–0.306 / 0.439–0.457 | **ALL LOSE to h256, paired-Δ CI<0 every arm** (−0.023…−0.026). Dose is FLAT (B=0→128→256→384: 147/143/139/143 hits, every adjacent pair straddles, endpoints do not separate) and every dosed arm is nominally BELOW the bare dual; symmetric A256/B256 ties asymmetric A0/B256 (−0.0007) ⇒ placement irrelevant. `run-20260726-161801-{dc7d2,9450b,ca0b8,bd5b5}` |
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

Read: **A supervised LEARNED CONTENT PROJECTION (R′+M+C′) on PCA-192 is the new
CHAMPION ROW / best on record** (R@10 0.212, MRR 0.122, paired-Δ vs the R+M+C
crown +0.0259 CI>0, seed-robust ×3, cold-reaching, and — as of 2026-07-18 evening —
SECOND-SPLIT HARDENED on the disjoint 0.64-cold `seq-20260718-211238` with paired-Δ
vs the R+M+C crown +0.0426 [+0.0280, +0.0573] CI>0) — the 2026-07-18 lit-fit
campaign's one win, and the strongest signal since content itself. Mechanism:
replace the *unsupervised* PCA-192 geometry (that both the content-kNN leg and the
GRU retrieve over) with a *supervised* linear map fit toward next-track adjacency
(InfoNCE over train consecutive pairs); it lifts BOTH shared-latent legs and is
the only channel that reaches the 55% cold targets. NOW REGISTERED + PROMOTED + SERVABLE as `blend-gru-markov-content-proj` (2026-07-18b; on-server R@10 0.21174, best-models rank 1; the InfoNCE/full/τ0.07 default confirmed best in the objective/rank/τ scan — InfoNCE ≫ BPR). **The prior crown R+M+C (GRU×Markov×content-kNN) on
PCA-192** (R@10 0.186, MRR 0.103, confirmed 5×) is the promoted+servable model
(`blend-gru-markov-content`). **The prior champion
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
- **Item-VECTOR composition is a RAW-model handicap but NOT a champion bottleneck**
  (2026-07-18c representation exploration). The equal-weighted 4-block PCA-192 spends
  87% of its variance budget on numeric+acoustic (EVR preflight) — a real handicap for
  the FROZEN GRU: whitening / dropping acoustic / the musical-distance metric geometry
  lifts the frozen `gru-infonce-h256` by paired-Δ +0.025…+0.036 CI>0 (6/8 alt spaces;
  best V5 balanced-pca192 +0.0356, V3 std-noaco +0.0342). BUT the 2026-07-18 learned
  InfoNCE projection is exactly this fix, LEARNED — so NO alt space beats the champion
  `blend-gru-markov-content-proj` on readout A (paired-Δ CI>0): the gain lifts the raw
  readout (C) and NOT the champion (A) → the projection already compensates for the loud
  directions. Keep NUMERIC (V4 textcat = drop numeric+acoustic REGRESSED the champion
  −0.0168 CI<0); whiten/drop only ACOUSTIC. Graded axis moves more than exact: whitening
  lifts frozen-GRU genre@10 to ~0.47, text-only maximizes the projection's artist@10/
  genre@10 (the sonic-metric ⟂ behavioral pattern). Open: fit the projection ON a
  whitened/metric space (compose the two fixes). `2026-07-18c-...representation-exploration`.

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
- SECOND LEAK-FREE SPLIT (2026-07-18): `seq-20260718-211238` — an EARLIER-HOLDOUT 60–80% chronological split on PCA-192, built to harden the R′+M+C′ champion on a test window DISJOINT from the champion split. Same corpus/sessionization + BYTE-IDENTICAL item space as `seq-20260715-131139` (item_latents.f32 + items.json identical; full 19,402-item vocab / PCA-192). n_sessions 5723 (latest-20% tail order[5723:] DISCARDED — physically absent), train 4292 (earliest 60%, order[:4292]) / test 1431 (60–80% band, order[4292:5723]) @ cuts 2023-04-17T19:26:24 (train|test) → 2024-08-24T14:28:02 (test|discard, == the original 80% cut). Test window DISJOINT index+time from the champion split's 80–100% test (max new-test start 2024-08-24T00:08:21 < 2024-08-24T14:28:02 = min orig-test start); leak-free (max train start ≤ min test start). Cold-item rate 0.64 (vs the original 0.55 — the earlier window has a smaller accumulated single-user catalog → a stronger, independent cold-reaching stress test; absolute R@10 expected below 0.212 by construction). Built by a standalone producer that reuses `sequences.py` helpers (default 80/20 path untouched), then registered into `data/datasets/`; 17/17 parity/quality checks pass (load_artifact, item-space byte-parity, sizes, cold rate, leak-free + disjointness). **USED (2026-07-18 evening) for the R′+M+C′ second-split hardening — verdict CONFIRMED** (projection paired-Δ vs R+M+C crown +0.0426 [+0.0280,+0.0573] CI>0; runs `run-20260718-212639-50e1d` PROJ / `-213156-cf1dd` R+M+C / `-214114-93bad` R+M).

- ITEM-VECTOR BLOCK-COMPOSITION EXPLORATION (2026-07-18, evening) - 8 dim-192 sequence item spaces built from block subsets / weightings / whitenings of the same leak-safe `spotify_tracks_content` source, ALL on the CHAMPION split (parity gate PASSED: vocab/split byte-identical to `seq-20260715-131139`, only item_latents.f32 differs; 7154/19402/5723/1431 @ cut 2024-08-24T14:28:02; cold-item 0.5507 == the 0.55 canonical). Motivated by the EVR preflight (full PCA-192 spends 87% of its variance budget on numeric+acoustic at R2=1.000 while text - the artist/genre-signal block - is 2.5% of input variance / R2=0.72). Builders: extended `build_song_pca.py` (backward-compatible `--blocks`/`--block-weights`/`--standardize`; DEFAULT byte-identical, verified cosine 1.0 vs the live champion collection), new `pipeline/reduce_named_vector.py`, and `sequences.py --vector-name` (named-vector metric collection). `std` = per-column diagonal whiten (full ZCA is the obvious follow-up if V2/V3 win). Variant -> seq id (latent_dim) - recipe [cumEVR@192], collection:
  - full (baseline, existing) -> seq-20260715-131139 (192) - all 4 blocks, whiten off [0.9922], `spotify_tracks_song_pca192`.
  - V1 noaco -> seq-20260718-222742 (192) - text+numeric+categorical (drop acoustic), whiten off [0.9884], `spotify_tracks_song_pca192_noaco`.
  - V2 std -> seq-20260718-222754 (192) - all 4 blocks, per-column standardize [0.8806], `spotify_tracks_song_pca192_std`.
  - V3 std-noaco -> seq-20260718-222807 (192) - text+numeric+categorical, standardize [0.8885], `spotify_tracks_song_pca192_std_noaco`.
  - V4 textcat -> seq-20260718-222820 (192) - text+categorical only (drop numeric+acoustic), whiten off [0.9579], `spotify_tracks_song_pca192_textcat`.
  - V5 balanced-pca192 -> seq-20260718-222834 (192) - musical-distance `balanced` 517-d metric vector reduced -> PCA-192 [0.9561], `spotify_tracks_song_pca192_balanced`.
  - V7 std-txtcatup -> seq-20260718-222848 (192) - all 4, standardize + weights text2/cat2/num1/aco0.25 (acoustic soft-drop) [0.8917], `spotify_tracks_song_pca192_std_txtcatup`.
  - V10 text-only -> seq-20260718-222901 (192) - text block only [0.9758], `spotify_tracks_song_pca192_textonly`.
  - V6 sonic-64 -> seq-20260718-222905 (64) - musical-distance `sonic` 64-d whitened-AE named vector, direct (no PCA; dim<192-peak caveat), reuses `spotify_tracks_content_metric`.
  EVALUATED 2026-07-18 (night) — `2026-07-18c-nexttrack-representation-exploration.md` (validation-only, nothing promoted). The approved design ran readouts A (champion projection `blend-gru-markov-content-proj`) + C (frozen `gru-infonce-h256`) ONLY (not the content-kNN standalone); 18 runs serialized, 0 failures, all n_test 1431. VERDICT — readout A: NO variant beats the PCA-192 champion by paired-Δ CI>0 (V4 textcat −0.0168 & V6 sonic-64 −0.0182 CI<0 REGRESS; V3 std-noaco +0.0049 / V5 balanced +0.0042 sub-noise) ⇒ the InfoNCE projection already fixes the loud-direction problem, the item space is NOT a champion bottleneck. readout C: 6/8 variants BEAT the frozen GRU by paired-Δ CI>0 (+0.025…+0.036; best V5 balanced +0.0356, V3 std-noaco +0.0342, V2 std +0.0300) ⇒ the raw PCA-192 geometry IS hindered by its acoustic/numeric-dominated variance budget, whitening/acoustic-drop lifts the raw model, but the projection ABSORBS the gain (C only, not A). GRADED: whitening lifts frozen-GRU genre@10 to ~0.47 (V7 0.4717 / V3 0.4710); V10 text-only is the readout-A graded-preferred (artist@10 0.3466 / genre@10 0.4144). NO promotion candidate; champion row UNCHANGED. Readout-A runs (proj) d593e/fb487/f455d/55314/1a295/4fd6c/08f1c/2c5e1/4520d; readout-C runs (frozen GRU) 49721/9a37f/23b6c/d3d4a/c1b8e/61137/212f7/141a6/31702 (all dated 2026-07-18).

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

**Reproduction spread ±0.0035, and the "cross-host offset" is WITHDRAWN
(2026-07-25).** The observed same-config reproduction spread on the canonical split
is **±0.0035**; any nominal gap ≤0.0035 is reproduction noise, and any single-split
gap inside ±0.015 without a paired-Δ CI>0 is "at least equal, likely better — needs
the seed check", never "beats". A **−0.0028 cross-host reduction-order offset** was
briefly on the books (the two remote `seq-nexttrack` h256 runs
`run-20260723-002044-af11a` / `-17cf8` both returned 0.1201957 vs the hub's
0.12299/0.12369). **That inference is WITHDRAWN: it was a CODE difference, not a
host effect.** Those runs executed on a STALE `~/lensing-worker/predictors/`
checkout (md5-divergent `seq_nexttrack.py`; `seq_common.py` lacking
`holisticness_at_k` entirely); on a freshened, md5-verified checkout the same config
on the same remote host returns **0.12299 — bit-identical to the hub**
(`run-20260725-143143-97c1e`, the dual-tower campaign's C0 gate). **Consequences:**
(a) there is NO measured host effect on this corpus — hub and `snappler` agree
bit-for-bit given identical code; (b) `af11a`/`17cf8` are **not code-comparable** to
current batches (they remain valid as the within-batch β0.1/β0.2 eagerness
measurements they were taken for); (c) the same-host discipline is still worth
keeping for paired statistics, but it is cheap insurance, not a correction for a
known bias. See `2026-07-25-nexttrack-dual-tower-fusion-scan.md` §Provenance.

**Reproduction spread REAFFIRMED and sharpened to BIT-IDENTITY across a code edit
(2026-07-26).** After `pre_hidden`/`pre_layers` were added to `seq-nexttrack` (built
conditionally: no `nn.Module`, no RNG draw at width 0), the h256 baseline returned
recall@10 `0.12299091544374564` — **identical to the last digit** to
`run-20260725-143143-97c1e` — and the bare `seq-dualgru` path likewise returned
`0.10272536687631029` exactly. **So for these two deterministic paths the
reproduction spread is 0, not ±0.0035, and a bit-identity gate is the right
preflight for any batch that follows a predictor edit** (it distinguishes a
genuine code-path change from run noise, which a ±0.0035 tolerance cannot). Note
the stored aggregates sit 1 ULP off the exact quotient (176/1431, 147/1431)
because of how the run accumulates them — **judge such gates on the HIT COUNT,
not the last digit of the aggregate.** Distinguish this from **BETWEEN-CONFIG
jitter**, which is much larger than the reproduction spread: adjacent single-GRU
widths differ by up to 0.0119 recall with no trend (see §Pitfalls, param-matched
width controls). Same config = 0 spread; neighbouring configs ≈ ±0.01.

## Pitfalls / infra

- **`seq-dualgru` (dual-tower) — REFUTED FAMILY; `fusion_layers=1` is a documented
  REGRESSION MODE (2026-07-25).** Joint end-to-end dual-tower fusion does not beat a
  single GRU on this corpus (best arm ties, 6/8 lose CI<0) and a 0.098%-param-matched
  single GRU h425 also ties, so it is not a capacity shortfall. If the family is ever
  revisited: **start at `fusion_layers=0`** — the 256-wide MLP fusion head costs
  0.016–0.020 recall@10 (paired-Δ CI>0 against it in BOTH the redundant-view and
  complementary-view regimes) for 10% MORE params. **THIRD CONFIRMATION, and the
  penalty GROWS with input-side nonlinearity (2026-07-26):** with a pre-MLP on
  tower B, `fusion_layers=0` beats `fusion_layers=1` by **+0.0266
  [+0.0140,+0.0391] CI>0** — larger than either 2026-07-25 measurement. Mechanism
  sharpened: the more nonlinearity you have already added on the INPUT side, the
  more the output-side warp costs the InfoNCE metric read-out. Treat
  `fusion_layers=0` as the family default, full stop. Mechanism: the objective is
  cosine/InfoNCE retrieval IN the item-latent space, so an extra non-linear layer
  between the recurrent state and the metric read-out just adds a warp the loss must
  undo. Do NOT re-spend runs on: `fusion_layers=1`; `arch_a`/`arch_b`=lstm (LSTM
  0.1188 < GRU 0.1230 on PCA-192); `layers_*`>1 (the depth cliff); single-GRU width
  >h425 on PCA-192 (flat). `2026-07-25-...dual-tower-fusion-scan`.
- **A NONLINEAR per-step PRE-ENCODER in front of the recurrence is REFUTED — and
  in a single tower it is ACTIVELY HARMFUL (2026-07-26).** `pre_hidden` on
  `seq-nexttrack` / `pre_hidden_a|b` on `seq-dualgru`
  (`[Linear, ReLU, Dropout] × pre_layers`, timestep-wise). The decisive
  single-tower ablation `MLP256→GRU256` **LOSES** to plain h256: 0.10901 vs
  0.12299, paired-Δ **−0.0140 [−0.0266,−0.0014] CI<0** (156 vs 176 hits). This is
  identifiable, not just a null: a *linear* pre-encoder at
  `pre_hidden ≥ latent_dim` is provably expressivity-neutral
  (`W_i·(Vx) = (W_iV)x`, no rank constraint), so **the ReLU is the only
  expressivity change and the rectifier is the culprit** — PCA latents are
  zero-centred and near-symmetric, so half-wave rectification zeroes ~half of
  every input vector's coordinates, and 1.33–2× the input dim is not
  over-complete enough to re-encode the discarded negative half-space. The GRU's
  own gated LINEAR input transform preserves the full signed latent. **Do NOT
  re-spend runs on** `pre_hidden` ≤ 384 with `pre_layers=1` on PCA-192 (bounded
  at 0.12369 = a TIE with baseline, and negative in the single tower); dose is
  FLAT (all adjacent pairs 0→128→256→384 straddle, endpoints do not separate,
  every dosed arm nominally BELOW the bare dual); placement is irrelevant
  (asymmetric A0/B256 vs symmetric A256/B256 straddles). **If anyone revisits
  this axis, the mechanism predicts the only sensible probes:** `pre_hidden` ≥ 3–4×
  input dim, and/or a sign-preserving activation (tanh/GELU) or linear pre-encoder
  — the latter needs an `activation` param that does not exist. Corollary on
  attribution: a second BARE tower alongside a pre-encoded one beats the single
  pre-encoded tower CI>0 (+0.0147) yet only TIES plain h256, i.e. the bare tower
  **RESCUES** what the rectifier destroyed (restores a direct un-rectified path to
  the readout) — a repair, not co-adaptation. `2026-07-26-nexttrack-pre-encoder-scan`.
- **CONFOUND STILL OPEN on the pre-MLP arms: they get GRU-INPUT dropout the bare
  arms lack (2026-07-26).** The pre-MLP ends in `Dropout` (`seq_dualgru.py:236`),
  whereas `seq_nexttrack` applies dropout to the RNN *output* only. So a pre-arm
  regression could be partly OVER-REGULARIZATION rather than the rectifier. The
  clean disambiguation is a **`pre_dropout` parameter** (default = `dropout`, so
  existing behaviour is unchanged), NOT a matched `dropout=0.0` pair — the pair
  confounds input- and output-side dropout in one comparison. Not yet implemented;
  carry this caveat on any pre-MLP verdict.
- **PARAM-MATCHED WIDTH CONTROLS ARE A WEAK INSTRUMENT at this effect size
  (2026-07-26).** The h297 control **LOST to h256 CI<0** (0.11321, Δ −0.0098).
  Single-GRU width points on PCA-192 now read h256→176, h297→162, h425→173,
  h454→179 hits: **non-monotone, spanning 0.0119 recall with no trend.** "Width is
  flat on PCA-192" survives as a TREND claim (h454 ties h256, so the 2026-07-25
  closure stands), but **per-width single-split jitter is ~±0.01 ≈ ⅔ of the noise
  band**, so a width-matched capacity control that is itself a single unreplicated
  run cannot carry a verdict alone. Design any capacity-controlled rule to require
  the comparison against the REPRODUCTION-VERIFIED baseline too (as the
  pre-encoder design's Rule 1 correctly did).
- **The raw `latent` view SUBSUMES `delta` and `cummean` for a recurrent tower
  (2026-07-25).** A GRU over raw prefix latents already integrates step-to-step
  movement and the running mean internally, so pairing it with an explicit
  computation of either adds NOTHING (both mixed cells tie their latent/latent
  parent and beat only the weaker parent). `delta`+`cummean` ARE mutually
  complementary (beats BOTH parents, CI>0 twice) but hopeless in absolute terms
  (0.0867) — so the co-adaptation mechanism is real, it just has nothing left to
  contribute once a raw-latent tower is present. **Corollary: a SINGLE-tower
  `seq-nexttrack` on `delta` or `cummean` is not worth a run** — the ablation arms
  bound it at 0.058–0.066 (vs the pooled-MLP `seq-ann` prior of 0.090).
- **FOUR independent learned-combiner nulls — the fixed z-blend is the combiner of
  record (2026-07-25 consolidation).** (1) XGB stacker 0.080–0.087, below its own GRU
  base leg (`2026-07-15c`); (2) RRF rank-fusion −0.029 (`2026-07-18`); (3) ORACLE
  test-fit blend weights +0.004, CI straddles 0 (`2026-07-18`); (4) **representation-level
  JOINT end-to-end dual-tower fusion, param-matched — refuted** (`2026-07-25`). (4)
  specifically closes the standing rebuttal to (1)–(3) ("of course a post-hoc,
  score-level combiner fails — train the fusion end-to-end so the parts co-adapt"):
  it was trained jointly from scratch at the representation level and still fails.
  **Do not propose another learned combiner without a mechanism that is not
  capacity, not co-adaptation, and not score-level fusion.**
- **UNCONSTRAINED holisticness@10 is maximized by predicting WORSE — always carry a
  recall floor (2026-07-25).** Across the 10-arm dual-tower batch the holisticness
  ordering is close to the REVERSE of the recall ordering: 5 arms lift H by paired-ΔH
  CI>0, topping out at cummean/cummean **H 0.19715 (+0.0554 CI>0)** with the batch's
  lowest artist_adj (0.190) — at recall@10 0.06639 = **95 of 1,431 vs the baseline's
  176**. That is the same pathology as the best-models group's incumbent rank-1
  `mood-session` (H 0.2332 at recall@10 0.0238). The dual-tower design's
  pre-registered **±0.015 recall floor** on its crown rule blocked it correctly and
  is the pattern to reuse: an H win without a recall floor is not a board event.
  **UPDATE (2026-07-26) — the 4-factor grounding factor helps but does NOT remove
  the need for the floor.** The post-2026-07-25 composite multiplies in
  `clamp((music@10 − 0.185)/0.815, 0, 1)`, and it works because `music@10` falls in
  lockstep with recall: the pre-encoder batch is the **FIRST to yield ZERO H
  candidates** (no arm reaches ΔH CI>0 vs the h256 baseline), whereas under the OLD
  3-factor H **five** of those arms would have shown H "wins", every one bought by
  de-eagering at a recall loss. It also demoted `seq-mood` from rank 1 to rank 24
  (H 0.23319 → 0.043881). **BUT the recomputed all-time 4-factor board is STILL
  topped by the degenerate `cummean/cummean` arm `run-20260725-143502-88a4c`
  (H 0.050436 at recall@10 0.06639 = 95/1431 vs a baseline 176)** → **keep the
  ±0.015 recall floor on every H claim.** Also note 4-factor H is exactly
  recomputable for PRE-cut runs from the per-row facets stored in
  `predictions.json` (`music_sim`/`mood_coh`/`ild`/`artist_adj`,
  `seq_common.py:507-517`), so the metric cut is NOT a comparability wall —
  verified to 7 significant figures against the stored value.
  Paired ΔH must be computed as a **product-of-means recomputed per bootstrap
  resample** (resample session ids, recompute each run's mean mood_coh / mean ild /
  mean artist_adj, then form H) — a mean of per-row products is a DIFFERENT
  statistic and will not reproduce the stored `holisticness_at_k`.
- **Remote-worker checkout can silently DIVERGE from the hub — the symptom is a WRONG
  NUMBER, not a crash (2026-07-25).** `~/lensing-worker/run_worker.sh` `cd`s to
  `~/lensing-worker`, so **`~/lensing-worker/predictors/` is the LIVE directory** —
  NOT the instance-path checkout under `~/Documents/git/.../spotify-next-track/`. It
  was found stale: md5-divergent `seq_nexttrack.py`, `seq_common.py` with ZERO
  occurrences of `holisticness_at_k`, and `seq_model_sae.py`/`seq_mood.py`/
  `seq_continuation_eval.py` MISSING (which would have crashed every `seq_dualgru`
  run, since it imports `make_batches`/`step_loss` from `seq_nexttrack`). It silently
  produced a −0.0028 recall@10 shift that was very nearly written into the record as
  a "cross-host reduction-order effect". **Before trusting any remote number: md5 the
  whole `predictors/seq_*.py` family against the hub, and gate the batch on the
  control run reproducing its on-record value** (the dual-tower campaign's Gate 0 +
  G-b/G-c/G-d preflight is the pattern).
- **`GET /api/best-models` can be STALE, not just polluted (2026-07-25).** After 10
  metric-carrying runs completed, the group still reported `updated_at`/`selected_at`
  **2026-07-23T01:54** with only **2 of 12** auto entries filled — the deterministic
  recompute did not absorb the batch. Do NOT assume the group reflects the newest
  runs: compare its `updated_at` against the newest run's `finished_at`. (Separately,
  when it does recompute under `primary_metric = holisticness@10`, a degenerate
  high-H/low-recall arm will slot near the top — flag any entry with recall@10 <0.10.)
- **MRR is NOT paired-testable from `predictions.json`, and NOT reconstructible from
  `top_k_ids` (2026-07-25).** The artifact carries `top_k_ids` but no rank field;
  reconstructing MRR from the top-10 ordering gives 0.04605 against a stored 0.05341,
  so the stored MRR uses a deeper/differently-ordered candidate list. Use `metrics.mrr`
  as an **aggregate no-regression check only** and never claim a paired MRR CI.
- **Model-SAE interpretability (2026-07-23):** on the GRU's dense recurrent
  hidden state the L1-SAE **won't sparsify** — an L1 sweep 0.0015→0.5 (330×) only
  moved l0 82%→35% at var_explained ~0.99, so `n_interpretable_concepts` saturates
  near the full dictionary (read it as saturated, not "N clean concepts"; trust
  ranked `atoms_by_concept` + `next_item_decodability`). Fix added: a **top-k SAE**
  (`--topk` on `seq_nexttrack.py model-sae`, `SAE(topk=)` in `seq_model_sae.py`) —
  k=32 gives l0=32 exactly, selective atoms (freq ~2% vs ~50%), at reconstruction
  cost var_explained 0.90. Detail cap raised 12→40 (`SHOWN_N`), `CONCEPT_SEP`
  0.5→0.35. `topk` is wired end-to-end (predictor `--topk` + server `ModelSaeRequest.topk`,
  `POST /api/interp/model-sae {..,"topk":32}`; server rebuilt + gated-restarted). Taste read
  from the atom labels in `docs/taste-from-sae-atoms.{md,pdf}`. See
  `docs/sae-interpretability-note.{md,pdf}`.
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
- **Second-split hardening reads on the PAIRED-Δ, not absolute R@10 (2026-07-18).**
  A colder holdout window drops every model's absolute recall by construction
  (cold-item rate 0.64 vs 0.55 → projection 0.212→0.171, R+M+C 0.186→0.128, R+M
  →0.117 on `seq-20260718-211238`), so "absolute R@10 fell" is EXPECTED and NOT a
  refutation. Judge the crown by the paired per-session Δ vs the reference ON THE
  SAME split — which HARDENED (projection vs R+M+C +0.0426 CI>0, in fact WIDER than
  the first split's +0.0259 because the cold-reaching projection leg carries more
  when more targets are cold). Build the second split to be leak-free AND
  index+time-DISJOINT from the first (byte-identical item space) so the check is
  independent, not a re-measurement.
- **Literature-fit map (2026-07-18 campaign):** the session-rec bibliography's
  FAMOUS answers do NOT fit this single-user, 55%-cold corpus: session-kNN /
  V-SKNN / STAN drag the blend (single-user strips the cross-user neighborhood;
  warm-only vs cold targets — the item2vec pattern), rank-fusion (RRF) hurts
  recall, and NO combiner beats the fixed equal-thirds z-blend on recall — even
  ORACLE test-fit weights give only +0.004 (CI straddles 0). The ONE additive
  direction is IMPROVING THE CONTENT SIGNAL: a supervised **learned content
  projection** (linear metric map of the content latent, InfoNCE over train
  consecutive pairs) is the new champion — it's the only channel that reaches
  cold items (verified: content leg scores cold≈warm≈0.15) and it lifts BOTH the
  content-kNN leg and the GRU that share the latent. Kneser-Ney/abs-disc Markov
  smoothing is a real but tiny standalone bar bump (0.107→0.110) that washes out
  in the blend. GRU importance negatives: null (InfoNCE in-batch negs already
  ~popularity^1).
- **Combiner-weight fitting is leak-prone:** you CANNOT fit z-blend weights on a
  held-out slice of TRAIN sessions — the count legs (Markov) memorize their own
  training data, so the fit collapses to pure-Markov. Use an oracle-on-test bound
  to gauge headroom, or rebuild the legs on train-minus-val. (The 3-leg R+M+C
  equal-thirds has essentially zero recall headroom regardless — oracle bound +0.004.)
- **Sequence-dataset registration:** `sequences.py` writes to `data/seq/`; the
  server reads datasets from `data/datasets/`. A new seq artifact must be copied
  into `data/datasets/<id>/` to be runnable (there is no ingest API).
- **Two execution slots / thread oversubscription:** running >1 torch predictor
  concurrently on a 12-core box oversubscribes threads (11/proc) and tanks
  throughput ~85× (~6 min/epoch vs ~4.5 s solo). Serialize (`--max-runs 1`).
  (Server restart is an ORCHESTRATOR action with user approval, never a runner
  action.) RE-CONFIRMED 2026-07-22 (holisticness campaign): launching 13 runs at
  once against the server ran 2 torch predictors concurrently and produced exactly
  this ~6.7 min/epoch collapse (a local-only batch would have been ~10–20 h) — the
  fix used was to offload the tail runs (β0.1/β0.2) to the remote "snappler" worker.
- **MMR λ is an EVAL-ONLY re-rank — never RETRAIN per λ (2026-07-22).** `mmr_lambda`
  re-orders the top-`mmr_pool` at eval from a SINGLE trained model; sweeping it by
  launching a training run per λ is pure waste. Sweep it OFFLINE from one champion
  checkpoint (validated: λ=off reproduces recall@10 0.2117 exactly;
  `scratchpad/mmr_sweep_results.json`). The 4 Phase-C training runs in that campaign
  were CANCELLED for this reason (cancelled, not failed experiments).
- **Blend eagerness is sourced from the MARKOV bigram leg, NOT the RNN (2026-07-22).**
  On the canonical split the standalone first-order markov has artist_adj@10 0.543
  and the champion blend 0.610, while the single GRU/LSTM sit at ≈0.345 (the
  LEAST-eager, HIGHEST-mood_coh models). The blends buy their recall@10 lead by being
  artist-adjacent (bigram-like) — so any future de-eagering should target the Markov
  leg / the fusion weights, not the neural model. MMR eval-re-rank (λ) is a strong
  post-hoc de-eager knob; the anti-eager TRAIN regularizer (eager_beta) is weak
  (β0→0.2 moves GRU artist_adj only 0.347→0.340). Holisticness metrics are
  DISPLAY-ONLY (primary stays recall@10) and NOT trivially gamed (popularity maxes
  ild 0.898 / min artist_adj 0.003 at ≈0 recall + negative mood_coh).
- **Cold items dominate forward eval:** 55% of test targets unseen in train —
  absolute recall is modest by construction; judge vs the B2 bar.
- **Runtime:** own port **8100** + own Postgres DB **`nexttrack`** in the shared
  container; `LENSING_DISABLE_PATHFINDER=1`; reads the SHARED corpus Qdrant.
  Never restart the parent's 8096.

## Best-models group (curation) — server top-12 by **holisticness@10** (was recall@10 until 2026-07-23), curated via `PUT /api/best-models`

_**STATE AFTER 2026-07-26 (pre-encoder campaign) — RE-FLOODED, curation OWED.** The
known `spawn_recompute` trigger gap struck again (all 10 runs were remote): the group
sat at `updated_at` 2026-07-25T15:48 with **5 of 12** entries until a manual
`POST /api/best-models/recompute` (→ `2026-07-26T17:11:52`, 12/12). That recompute
then absorbed **7 of the campaign's 10 REFUTED runs** — `bdcbc` (C0 baseline, fine),
`7646d` (h454 control), `ad53f` (S1, the HARMFUL pre-MLP arm), `0df6f` (C3 bare
dual), and **three arms below the 0.10 recall sanity line**: `dc7d2` 0.0999, `ca0b8`
0.0999, `9450b` 0.0971. Note the deterministic recompute ranks by holisticness@10, so
low-recall arms slot in on merit under the current primary metric — exactly the
degenerate-flood mode. **best-model-selector must exclude the refuted arms and make
`88a4c`'s exclusion permanent.** Pins were NOT touched by the runner (curation is not
a runner action); the pinned champion is unchanged._

_Header corrected 2026-07-25: the group's `primary_metric` is now `holisticness@10`
(confirmed live via `GET /api/best-models`). The curation notes BELOW are dated and
were written under the recall@10 regime — their reasoning is preserved as history;
read the ranking metric they refer to as recall@10 unless the note says otherwise._

_2026-07-18 (evening) — CURATED after the champion change (projection registration + scan, `2026-07-18b`). The deterministic recompute had the group dominated by ~6 near-identical `best-seq-blend-*` auto-promotions of the SAME projection model (the objective/rank/τ scan cells), with the UNCONFIRMED τ0.10 scan run masquerading as rank 1. Applied in ONE `PUT` (pins + excludes); no suspicious-metric/leakage flags anywhere in the candidate field (n_test uniformly 1,431; top recall 0.223 is only +0.011 over the champion, i.e. sub-noise, not a breakthrough)._

_**Pins (all promoted; durable across recomputes):**_
- _`blend-gru-markov-content-proj` (R′+M+C′, InfoNCE / full-rank / τ0.07, R@10 0.21174, `run-20260718-191539-1b89c`) — **ANCHOR**: the confirmed, seed-robust (×3), cold-reaching champion. Pinned rank 1._
- _`blend-gru-markov-content` (R+M+C content-kNN z-blend, 0.18588, `run-20260718-123738-38725`) — **DIVERSITY**: the prior crown; a genuinely distinct blend family (content-kNN leg, not a projection variant)._
- _`gru-lstm-pca128` (LSTM single, 0.12718, `run-20260715-142440-546d0`) — **DIVERSITY**: best SINGLE model; a distinct non-blend family. Pinned to protect it from being flooded out of the top-12 by the imminent projection-scan follow-ups (finer τ 0.12/0.15, graded-objective map, R′ promotion)._

_**Exclusions (by run_id — the 5 redundant / dominated / fluke projection scan cells; only the promoted τ0.07 champion is kept as the projection representative):**_
- _`run-20260718-193111-a781c` (proj InfoNCE/full/τ**0.10**, 0.22292) — **UNCONFIRMED SINGLE-SPLIT FLUKE**: +0.0112 over the champion but INSIDE the ±0.015 noise band, not seed-confirmed; the experiment-runner explicitly flagged "sub-noise, do NOT auto-swap." Excluded so it does not masquerade as a confirmed crown. **CANDIDATE** pending the standing seed + second-split hardening — re-include / re-promote only if it confirms._
- _`run-20260718-193650-66c96` (proj InfoNCE/**rank-64**/τ0.07, 0.20545) — **REDUNDANT**: ties full-rank (−0.006, inside noise); the gain is the supervision, not the capacity — adds no diversity._
- _`run-20260718-192443-57432` (proj InfoNCE/full/τ**0.05**, 0.19287) — **REDUNDANT**: below the τ0.07 default; a dominated scan cell, adds no diversity._
- _`run-20260718-194228-4cc06` (proj **BPR**/full/τ0.07, 0.15793) — **DOMINATED**: BPR collapses below even the old R+M+C crown (0.186); documented dead end (do not revisit)._
- _`run-20260718-194832-d070f` (proj **BPR**/rank-64/τ0.07, 0.15234) — **DOMINATED**: same BPR collapse._

_Result: champion pinned rank 1; the group now spans 4 families (projection blend / content-kNN blend / R+M 2-leg blend / single GRU-LSTM) instead of a projection monoculture. Left ALONE (no action needed, well-represented via auto slots): the R+M 2-leg blends (0.167–0.173) and the GRU-infonce singles (0.115–0.123). NB: with 3 pins overlaid on the 12 auto slots, `GET /api/best-models` returns 14 entries — expected pin semantics (pins are additive to the top-12)._

_**2026-07-18 (evening, later) — NEW AUTO-SLOT ENTRIES from the second-split hardening pass (PENDING RE-CURATION, not yet actioned by this runner):** the deterministic recompute slotted the three second-split VALIDATION runs on `seq-20260718-211238` into the auto top-12 by their raw recall@10 — `run-20260718-212639-50e1d` (PROJ, 0.1705), `run-20260718-213156-cf1dd` (R+M+C, 0.1279), `run-20260718-214114-93bad` (R+M, 0.1174). These are CROSS-SPLIT (a different, colder 0.64-cold dataset), not comparable to the first-split board and not new champions — they should be EXCLUDED by best-model-selector. The pinned rank-1 champion `blend-gru-markov-content-proj` is unchanged; the champion did NOT change (this pass CONFIRMED the standing champion). Runner did not promote/curate — validation-only pass._

_**2026-07-18 (evening, later still) — RESOLVED: the three cross-split auto-slot entries above are NOW EXCLUDED (best-model-selector re-curation).** In one `PUT /api/best-models`, excluded by run_id the three second-split VALIDATION runs that the deterministic recompute had auto-slotted by raw recall@10: `run-20260718-212639-50e1d-seq-blend` (PROJ, R@10 0.1705), `run-20260718-213156-cf1dd-seq-blend` (R+M+C, 0.1279), `run-20260718-214114-93bad-seq-blend` (R+M, 0.1174) — ALL evaluated on the 0.64-cold earlier-holdout `seq-20260718-211238`, NOT the canonical 0.55-cold split; their raw recall@10 is not comparable to the rest of the board (the board is all on the 0.55-cold canonical/phase-2-base family), and they are validation-only from the second-split hardening confirm, so they do not belong in the group. Rationale per run: cross-split — different, harder, disjoint dataset; not comparable; validation-only. Excluded list is now 8 (the prior 5 projection scan-fluke/dominated/BPR cells + these 3 cross-split). Pins UNCHANGED and preserved (`blend-gru-markov-content-proj` anchored rank 1, plus diversity pins `blend-gru-markov-content` and `gru-lstm-pca128`); the champion did NOT change. The freed slots backfilled with next-best CANONICAL-split candidates (PCA-256 / PCA-128 GRU-infonce singles, R@10 0.117 / 0.115). Verified: 0 entries on `seq-20260718-211238` remain; every group entry is on a first-split-family dataset (`seq-20260713-014226` / `seq-20260715-030509` / `seq-20260715-131139` / `seq-20260715-131145`)._

_**2026-07-18 (night) — RE-CURATED after the representation-exploration campaign (`2026-07-18c-nexttrack-representation-exploration.md`, VALIDATION-ONLY, nothing promoted, champion UNCHANGED).** The deterministic recompute had auto-slotted the campaign's readout-A exploration runs into the top-12 by raw recall@10 — two SUB-NOISE cross-representation cells (V3 std-noaco `run-...230418-55314` 0.2166 and V5 balanced `run-...232119-4fd6c` 0.2159) even outranked the pinned champion (0.21174, shoved to rank 3). These are validation-only runs trained on ALTERNATIVE item-latent spaces (whitened / acoustic-dropped / metric-geometry variants on datasets `seq-20260718-2227*`/`2228*`); their raw recall@10 is NOT comparable to the canonical PCA-192 board, they are sub-noise vs the champion (readout-A paired-Δ straddles 0 for every variant — NO CI>0 win), and none is a promotion candidate — exactly the second-split validation-run pollution pattern handled on 2026-07-18 evening. In ONE `PUT /api/best-models` (excludes only; pins and the prior 8 exclusions untouched), excluded 17 new run_ids:_
- _**9 readout-A `seq-blend` exploration runs** (champion projection re-fit on each alt space): `run-20260718-223848-d593e` (baseline champion RE-RUN on the canonical split — a redundant duplicate of the pinned champion, same 0.21174), `...224718-fb487` (V1 noaco), `...225548-f455d` (V2 std), `...230418-55314` (V3 std-noaco), `...231318-1a295` (V4 textcat), `...232119-4fd6c` (V5 balanced), `...233019-08f1c` (V7 std-txtcatup), `...234110-2c5e1` (V10 text-only), `...235010-4520d` (V6 sonic-64). Rationale: cross-representation validation-only runs on non-canonical item spaces (or, for d593e, a redundant duplicate of the pinned champion); sub-noise, unconfirmed, not comparable to the board._
- _**8 readout-C `seq-nexttrack` (frozen `gru-infonce-h256`) cells on the 2227*/2228* datasets** (recall ~0.12–0.16): `run-20260718-225218-9a37f`, `...230048-23b6c`, `...230948-d3d4a`, `...231818-c1b8e`, `...232649-61137`, `...233549-212f7`, `...234640-141a6`, `...235540-31702`. Rationale: same cross-representation validation-only pollution — raw frozen-GRU readouts on non-canonical alt item spaces; they would have slotted into the freed top-12 once the readout-A runs were excluded._
- _**LEFT ALONE (not pollution):** the readout-C BASELINE `run-20260718-224518-49721-seq-nexttrack` (0.12299) is on the CANONICAL split `seq-20260715-131139` (not a 2227*/2228* alt space); it reproduces the on-record GRU-infonce-h256 / PCA-192 number 0.123 already represented by 5 identical canonical runs, so it stays as a legitimate canonical-split single._
- _Result: champion `blend-gru-markov-content-proj` (0.21174) RESTORED to anchored rank 1; group spans the same 4 families as before (projection blend / content-kNN blend / R+M 2-leg blend / single GRU-LSTM) with every entry on a CANONICAL-family dataset (`seq-20260713-014226` / `seq-20260715-030509` / `seq-20260715-131139` / `seq-20260715-131145`) — zero entries on any `seq-20260718-2227*`/`2228*` alt-representation dataset. Pins UNCHANGED (`blend-gru-markov-content-proj` rank 1, `blend-gru-markov-content`, `gru-lstm-pca128`); prior 8 exclusions preserved. Excluded list now 25 (5 projection scan-fluke/BPR cells + 3 second-split + 9 readout-A + 8 readout-C). Champion did NOT change (the campaign confirmed it)._


_**2026-07-20 — music@10 confirmation/curation pass (NO ACTION; group CLEAN, champion UNCHANGED).** `music@10` (continuous "sounds-alike" mean best-cosine in the balanced `spotify_tracks_content_metric` space) is now a first-class metric in `domain.toml` `[metrics].columns`, but VERIFIED display-only: `[metrics].primary` stays `recall@10`, the best-models group's `primary_metric` is `recall@10`, and music@10 is NOT persisted in any run's stored metrics (the server ranks off `recall@10`/`hit_rate` only) — so it is structurally impossible for music@10 to reorder the group. The recall@10 ordering/membership is UNCHANGED by music@10. The pinned champion `blend-gru-markov-content-proj` LEADS on the new axis too (off-server analysis on existing runs: champion music@10 0.4538 vs R+M+C `blend-gru-markov-content` 0.4491 — the projection wins on sonic-closeness as well), so there is NO music@10-diversity argument to add a different-family model over the current pins; the group already spans 4 families. No new training runs since the 2026-07-18 night re-curation (latest run 2026-07-18 23:55; group `updated_at` 2026-07-19T00:10:48), no live/running runs (the 6 interrupted 2026-07-15 runs carry null metrics and are not selected), and no suspicious metrics anywhere (all group entries n_test 1,431; top recall 0.21174 is the champion, +0.026 over R+M+C, nothing near-perfect). Pins (3) and exclusions (25) preserved exactly; ZERO deltas applied.


_**2026-07-20 (later) — music@10 NATIVE-BACKFILL de-duplication (best-model-selector; 3 excludes, pins UNCHANGED, champion UNCHANGED).** Three champion definitions were re-run on a remote worker to backfill `music@10` NATIVELY (the older promoted runs pre-date the metric → music@10 blank), then a `POST /api/best-models/recompute` auto-slotted the reproductions into the top-12, creating exact-recall DUPLICATES of already-pinned models (champion pinned rank 1 vs its new run rank 2; R+M+C pinned rank 4 vs its new run rank 3). In ONE `PUT /api/best-models` (excludes only) excluded the 3 backfill runs by run_id:_
- _`run-20260720-160424-e0cdd-seq-blend` (from `blend-gru-markov-content-proj`, canonical `seq-20260715-131139`, recall@10 0.21174 — byte-identical to the pinned champion's run `run-20260718-191539-1b89c`; music@10 0.4538) — **REDUNDANT REPRODUCTION of the pinned champion** (same def+dataset+deterministic seed). Direct precedent: the already-excluded readout-A duplicate `run-20260718-223848-d593e` (also a 0.21174 reproduction of the champion)._
- _`run-20260720-160424-ebb46-seq-blend` (from `blend-gru-markov-content`, canonical split, recall@10 0.18588 — byte-identical to the pinned R+M+C run `run-20260718-123738-38725`; music@10 0.4491) — **REDUNDANT REPRODUCTION of the pinned R+M+C**._
- _`run-20260720-160424-a6c17-seq-nexttrack` (`gru-infonce-h256` on `seq-20260718-222807` = the V3 std-noaco ALT-representation, recall@10 0.157, music@10 0.482 = the music@10 leader) — **CROSS-REPRESENTATION / NON-COMPARABLE**: on a whitened/acoustic-dropped alt item space, NOT the canonical PCA-192 board, so its recall@10 is not comparable to the rest of the group — it reproduces the previously-EXCLUDED readout-C pollution cell `run-20260718-230948-d3d4a` (same dataset, same 0.15723 recall). The seq-nexttrack single family is ALREADY represented (LSTM `gru-lstm-pca128` + the canonical gru-infonce-h256 singles), and music@10 is DISPLAY-ONLY (never reorders the group), so its music@10 leadership does not justify membership. Consistent with the 2026-07-18-night exclusion of all `seq-20260718-2227*/2228*` alt-rep runs._

_DE-DUP RATIONALE (why the canonical pins were KEPT over the music-carrying reproductions): the group entries do NOT surface music@10 (`primary_metric` recall@10; music@10 display-only, lives on the run), so the music-carrying run confers ZERO functional advantage AS A GROUP MEMBER — the only difference is which run-provenance the group references. Keeping the canonical, richly-annotated, servable `blend-gru-markov-content-proj` / `blend-gru-markov-content` preserves the recognizable rank-1/2 anchors the whole record refers to; the alternative (unpin champion + pin the anonymous `best-seq-blend-20260720-*` auto-label) would erase the champion name from rank 1. PROPER path to attach music@10 UNDER the canonical names = RE-PROMOTE each definition onto its new run (`blend-gru-markov-content-proj`→`run-...-e0cdd`, `blend-gru-markov-content`→`run-...-ebb46`) via model-definitions (a PROMOTION action, outside best-models curation); afterward exclude the superseded old runs `1b89c`/`38725` and un-exclude e0cdd/ebb46 — the canonical pins then carry music@10 at the same rank-1/2 ordering. RESULT: champion `blend-gru-markov-content-proj` pinned rank 1 (UNCHANGED, 0.21174), R+M+C `blend-gru-markov-content` rank 2 (0.18588), `gru-lstm-pca128` rank 7 (0.12718); freed auto slots backfilled with canonical R+M 2-leg blends (0.167–0.173) + canonical gru-infonce-h256 singles. Pins (3) UNCHANGED; excluded list now 28 (prior 25 + these 3). FLAGGED-BUT-LEFT-ALONE: the canonical gru-infonce-h256 single reproduces 0.12299 SIX times in the tail (ranks 8-13) — a pre-existing identical-reproduction cluster the 2026-07-18 curation consciously kept as auto-slot filler; de-duping the tail was not requested and would be curation for its own sake (noted only for consensus-weighting awareness)._

_RE-PROMOTION EXECUTED (2026-07-20, later; superseded by the per-row round below): the canonical pins were re-promoted onto remote-.12 reproductions carrying music@10 + query-track `prefix_ids` (runs 7f358 / 23271), superseding 1b89c / 38725._

_PER-ROW music@10 ROUND (2026-07-20, final): the eval also now records a PER-ROW `music_sim` (each query's best musical-distance cosine of the top-10 to the truth — hit→1.0, sonically-close miss→high) so the runs view can show that an exact-track "miss" was a sonic near-hit. The two pinned blend champions were re-run a final time on .12 with the complete eval (recall + music@10 + per-row music_sim + prefix) and RE-PROMOTED: `blend-gru-markov-content-proj`→`run-20260720-185329-29036-seq-blend` (recall@10 0.21174, music@10 0.4538) and `blend-gru-markov-content`→`run-20260720-185329-f38a0-seq-blend` (recall@10 0.18588, music@10 0.4491); music_sim + prefix on all 1431 predictions. Superseded runs 7f358 / 23271 EXCLUDED. Pins (3) UNCHANGED, recall@10 identical (faithful reproductions), no dupes. The 3rd pin `gru-lstm-pca128` (rank 7) still points to its pre-music run — not re-run (no from_definition). The runs view (`RankedPredictionsTable`) now shows: the QUERY current track (was a bare session id), a sortable per-row `music@10` column (miss ≥0.4 accent-colored), and an All/Hits/Misses outcome filter. EXPANDED DETAIL: each ranked suggestion is annotated with its `music@10` similarity to the TRUE next track (decomposes the row's music@10 = the max of these), fetched live via a new sidecar endpoint `GET /api/pathfinder/music-scores?ref=&ids=&dial=` (Python `api_music_scores` over `spotify_tracks_content_metric` + Rust `proxy_music_scores` route) — so it works for ANY run with no re-training. Verified: on champion miss #6627 (true next Juan Campodónico/candombe), suggestion #8 (same artist) scores 0.98 and equals the row's music@10._

_**2026-07-22 (later) — SESSION-HOLISTICNESS + ANTI-EAGER CAMPAIGN de-duplication (best-model-selector; 7 excludes, pins UNCHANGED, champion UNCHANGED).** The campaign (`2026-07-22-session-holisticness-and-antieager-levers.md`) was MEASUREMENT / DISPLAY-ONLY — nothing was intended for promotion — but its succeeded canonical-split (`seq-20260715-131139`) runs from the batches `20260722-165345` (local) and `20260723-002044` (remote `worker:1432649`, the β0.1/β0.2 offload) were auto-slotted into the top-12 by the deterministic recompute, the documented DUPLICATE-REPRODUCTION pattern (precedent: 2026-07-18-night representation runs, 2026-07-20 music-backfill runs). Excluded 7 campaign runs by run_id in ONE `PUT /api/best-models` (excludes only; the 3 pins + prior 32 exclusions untouched):_
- _`run-20260722-165345-82a8f-seq-blend` (A1, from `blend-gru-markov-content-proj`, recall@10 0.21174 — BYTE-IDENTICAL to the pinned champion run `run-20260720-185329-29036`) — **REDUNDANT REPRODUCTION of the pinned champion** (auto-slotted rank 2, ahead of the R+M+C pin). Direct precedent: the excluded d593e / e0cdd champion dups._
- _`run-20260722-165345-28dbb-seq-blend` (A2 R+M, from `blend-gru-markov`, recall@10 0.17121 — ties the already-present canonical R+M PCA-192 run `96f53`) — **REDUNDANT: R+M family already represented** (ranks 3-6)._
- _`run-20260722-165345-353b5-seq-nexttrack` (β0.05 GRU, from `gru-infonce-h256`, 0.12369) — **GRU-single family already represented** (LSTM pin `gru-lstm-pca128` + the canonical 0.12299 gru-infonce cluster); eager_beta is a documented WEAK knob (β0→0.2 flat recall) so this is a within-noise GRU reproduction, no diversity added._
- _`run-20260722-165345-e39c9-seq-nexttrack` (A3 plain GRU, from `gru-infonce-h256`, 0.12299 — byte-identical to the canonical gru-infonce reproductions) — **REDUNDANT GRU reproduction.**_
- _`run-20260723-002044-af11a-seq-nexttrack` (β0.1 GRU, remote `worker:1432649`, from `gru-infonce-h256`, 0.12020) — **would BACKFILL a freed tail slot; GRU-single family reproduction, β weak-knob, no diversity.** Treated like any other GRU measurement run (per the campaign runner).
- _`run-20260723-002044-17cf8-seq-nexttrack` (β0.2 GRU, remote `worker:1432649`, 0.12020) — same: would-backfill GRU reproduction._
- _`run-20260722-165345-bb387-seq-nexttrack` (campaign GRU measurement, 0.11880) — would BACKFILL the next freed slot; same GRU-reproduction pollution._
_WHY exclude the would-backfill runs too (af11a/17cf8/bb387), not just the 4 auto-slotted ones: excluding only the top 4 frees tail slots that the deterministic recompute immediately re-fills with the NEXT campaign GRU reproductions (verified from the candidate field: af11a/17cf8 at 0.12020, bb387 at 0.11880 sit directly below the canonical 0.12299 cluster) — so a clean, campaign-free group requires excluding all seven. LEFT ALONE (campaign runs that fall below the top-12 cut and CANNOT backfill — exclusion would be a no-op): `run-20260722-165345-047cc-seq-markov` (0.10692, dup of existing markov runs) and `run-20260722-165345-34614-seq-popularity` (0.00070, floor). The 4 Phase-C `seq-blend` runs (`5ff57`/`5c208`/`39848`/`0b796`) are `failed` (intentionally cancelled — MMR λ is an eval-only re-rank, never retrain-per-λ) and the old `c3419`/`13035` are killed → status-filtered, never selected. RESULT: champion `blend-gru-markov-content-proj` pinned rank 1 (0.21174, UNCHANGED), R+M+C `blend-gru-markov-content` rank 2 (0.18588), `gru-lstm-pca128` rank 7 (0.12718); the 12 auto slots are now ALL canonical-family (4 R+M blends `seq-20260713-014226`/`131139`/`030509`, then 6 canonical gru-infonce-h256 0.12299 reproductions + b4d6e 0.11670 + a1d9d 0.11461). Group spans the same 4 families as prior curations (projection blend / content-kNN blend / R+M 2-leg blend / single GRU-LSTM); ZERO entries on any `20260722`/`20260723` campaign run. No suspicious metrics anywhere (all group entries n_test 1,431; top 0.21174 is the champion, nothing near-perfect). Pins (3) UNCHANGED; excluded list now 39 (prior 32 + these 7)._

_**2026-07-25 — DUAL-TOWER CAMPAIGN: group is STALE, NOT flooded — PENDING best-model-selector (runner did NOT curate; nothing promoted).** The dual-tower fusion scan (`2026-07-25-nexttrack-dual-tower-fusion-scan.md`) completed 10 runs on the canonical split, ALL carrying `holisticness_at_k`, and — contrary to the expected auto-flood — `GET /api/best-models` still reports `updated_at`/`selected_at` **2026-07-23T01:54:06** with only **2 of 12** auto entries filled (`best-seq-mood-20260723-014711-b9172`, H 0.23319, rank 1; `best-seq-blend-20260723-014711-1c1a3`, H 0.06316, rank 2). **The deterministic recompute did not absorb the batch**, so the group is stale rather than polluted. Pins (3) and the 39-entry exclusion list are intact. TWO things need best-model-selector judgment: (a) the group is stale relative to 10 completed metric-carrying runs; (b) **when it does recompute under `primary_metric = holisticness@10`, `run-20260725-143502-88a4c-seq-dualgru` (cummean/cummean, H 0.19715) would slot straight in at rank 2 and MUST be excluded as a DEGENERATE-MODE entry — its recall@10 is 0.06639 = 95 of 1,431 vs the canonical single-GRU baseline's 176** (the same pathology as the incumbent rank-1 `mood-session`, H 0.2332 at recall@10 0.0238, which predates this campaign). The other 9 campaign runs are validation-only ablations / control reproductions on the canonical split (C0 `run-20260725-143143-97c1e` reproduces the on-record 0.12299 EXACTLY — a redundant reproduction of the already-represented gru-infonce-h256 cluster) and fall under the standing campaign-run de-duplication precedent (2026-07-18-night representation runs, 2026-07-20 music-backfill, 2026-07-22 holisticness campaign). Suggested standing rule for this metric regime: **flag any group entry with recall@10 < 0.10 as a degenerate-mode exclusion candidate.** Champion `blend-gru-markov-content-proj` UNCHANGED; this campaign registered and promoted NOTHING._


_**2026-07-25 (later) — DUAL-TOWER CAMPAIGN CURATED (best-model-selector; 7 excludes, pins UNCHANGED, champion UNCHANGED). PLUS a CONFIRMED SERVER DEFECT and a STRUCTURAL consequence of the crown flip — read both, they matter more than the deltas.**_

_**(1) THE STALENESS WAS A REAL SERVER DEFECT, NOT a promoted-only gating condition.** Diagnosed by reading `crates/lensing-server/src/best_models.rs`: `recompute_locked`'s candidate filter (`add_run`, lines 136-151) requires only `status ∈ {Succeeded, Stopped}` + the primary metric present + `predictor.supports_predict()` — it explicitly does NOT require promotion (unpromoted selections are AUTO-promoted at lines 233-245, per the module doc lines 5-8). So "only promoted models are eligible" is FALSE and cannot explain the frozen group. The actual cause is a **missing recompute trigger on the remote-worker completion path**: `spawn_recompute` is called from exactly four places — `runs.rs:282` (HUB-orchestrated run finalization), `api.rs:1593` (manual model promotion), `api.rs:1650` (model deletion) and `api.rs:1732` (`POST /api/best-models/recompute`). A **remote training worker** (`lensing-server worker`) finalizes its runs by writing straight to Postgres (`worker.rs` ~lines 127-150: `put_run_artifacts` + `upsert_run` + a `run_event` status line) and **never notifies the hub**, so no recompute is spawned. ALL 10 dual-tower runs ran on remote worker `snappler:4101029` → the group sat frozen at `updated_at` 2026-07-23T01:54 with 2/12 entries. **PROOF the recompute pathway itself is healthy:** the single curation `PUT` below (which recomputes) absorbed the whole batch immediately. **CONSEQUENCE for the record: the best-models group is silently stale after ANY remote/queued batch.** Until the trigger gap is fixed (candidate fix: have the hub spawn a recompute when a DB-backed run transitions to a terminal status, e.g. in the run-status poll at `api.rs` ~1428, or have the worker POST `/api/best-models/recompute` to its `--hub-url`), **every campaign that offloads to a worker MUST `POST /api/best-models/recompute` (or `PUT`) before the group can be judged.** This also retro-explains the 2026-07-22 note's remote `worker:1432649` batch. FRAMEWORK-LEVEL bug (not domain) → belongs upstream._

_**(2) STRUCTURAL: the 2026-07-23 crown flip silently emptied the group and DISABLED the 3 pins.** Only **12 of 95** runs carry `holisticness_at_k` at all (the metric was added with the 2026-07-22/23 holisticness suite in `predictors/seq_common.py`), and 10 of those 12 are this campaign's. Runs that pre-date the metric are NOT RANKABLE (`add_run` returns early when `metrics.extract(primary, ...)` is None), so **the entire recall@10-era board — including the champion — is structurally invisible to the group.** The 3 pins (`blend-gru-markov-content-proj`, `blend-gru-markov-content`, `gru-lstm-pca128`) are therefore **PRESENT in `pinned` but ABSENT from `entries`**: pin wins over exclusion, but a pin CANNOT rescue an unrankable candidate, and the `prev_value` carry-over fallback (lines 203-217) was voided by the `regime_changed` guard (line 184) at the recall@10→holisticness@10 flip. **Pins deliberately LEFT IN PLACE** (not unpinned): they are the correct standing instruction and will auto-restore those models to the group the moment each is re-promoted onto a holisticness-carrying run — unpinning would destroy that intent for zero benefit (they are inert, not harmful). **The group is under-filled at 5/12 for the same reason — that is metric coverage, not over-curation; no amount of curation can fill it.**_

_**(2b) ACTIONABLE FOLLOW-UP (promotion, NOT curation — outside this agent's remit):** `run-20260723-014711-1c1a3-seq-blend` (group rank 5, H 0.06316) is a **bit-identical reproduction of the pinned champion** `blend-gru-markov-content-proj` — same `seq-blend` hyperparams (projection/InfoNCE/full-rank/τ0.07, seed 1337) on the same canonical split, recall@10 `0.21174004192872117` matching the champion run `run-20260720-185329-29036` to 14 decimals, mrr agreeing to 9 — but WITH holisticness. So the champion is already in the group under the anonymous auto-promoted alias `best-seq-blend-20260723-014711-1c1a3`. **Re-promote `blend-gru-markov-content-proj` onto `run-20260723-014711-1c1a3-seq-blend`** (via model-definitions) and the pin re-anchors the champion in the group UNDER ITS OWN NAME — this is exactly the documented "PROPER path" from the 2026-07-20 music@10 de-dup note. Likewise `gru-lstm-pca128` and `blend-gru-markov-content` need a holisticness-carrying re-run before their pins can bite._

_**(3) THE DEGENERATE-MODE FLOOR (the judgment call). Rule adopted: a group member must have recall@10 ≥ 0.108**, i.e. not materially below the canonical single-GRU baseline (0.12299 − the ±0.015 CI half-width). **This is STRICTER than the campaign's suggested "flag recall@10 < 0.10"** and is preferred because it is DERIVED from the noise band rather than being a round number: it makes the floor mean "recall statistically indistinguishable from, or better than, the canonical baseline", which is the exact property the floor is supposed to guarantee. It changes exactly ONE verdict vs the 0.10 suggestion (`12fe6`, recall@10 0.10273 = 0.0203 BELOW baseline, i.e. outside the noise band → excluded). **WHY a recall floor is needed at all:** the crown `holisticness@10 = clamp(mood_coh,0) × ild × (1−artist_adj)` (`predictors/seq_common.py:538`) **contains no correctness term**, so a model can win it by declining to predict the true next track — holisticness and recall are ANTI-CORRELATED across this batch (Spearman-negative by inspection: the H-ranked order 0.197→0.133 runs recall 0.066→0.118 upward). A group member exists to serve `POST /api/best-models/predict`; a member returning 34-139 correct next-tracks out of 1,431 poisons that consensus. **HONEST CAVEAT: the floor is an operational bright line, not a statistical boundary** — the excluded `a95c5` (0.09713) and the excluded `12fe6` (0.10273) differ by 0.0056, well INSIDE the ±0.015 band, so they are statistically TIED with each other; any threshold in this region necessarily cuts between indistinguishable runs. Applied in ONE `PUT /api/best-models` (excludes only; the 3 pins + prior 39 exclusions untouched), excluded 7 run_ids:_
- _`run-20260723-014711-b9172-seq-mood` (H **0.23318**, recall@10 0.02376 = **34 of 1,431**) — **DEGENERATE, and this is the INCUMBENT rank-1 being removed.** The `seq-mood` session generator is working AS DESIGNED (its own promoted-model note says it "does NOT target true-next-track (recall 0.024, by design)"), and it is precisely the pathology that motivated Rule 6's anti-degenerate recall floor (see §"Best on record" and the 2026-07-25 campaign report). Winning a correctness-free composite by not attempting correctness must not buy a seat in a next-track consensus group. **NB: this excludes the auto-promoted run alias from the GROUP only — the `mood-session` model stays promoted and directly servable by name for its intended mood-session use case. Nothing was deleted.**_
- _`run-20260725-143502-88a4c-seq-dualgru` (D-CC cummean/cummean, H **0.19715**, recall@10 0.06639 = **95 of 1,431** vs the baseline's 176) — **DEGENERATE; the PRE-REGISTERED Risk-8 exclusion.** Would have slotted straight in at rank 2. Same pathology as the mood incumbent, from a REFUTED family._
- _`run-20260725-143502-e9cfb-seq-dualgru` (D-DC delta/cummean, H 0.16340, recall@10 0.08665 = 124/1,431) — **DEGENERATE** (below the 0.108 floor)._
- _`run-20260725-143502-a95c5-seq-dualgru` (D-LC f1 latent/cummean, H 0.15000, recall@10 0.09713 = 139/1,431) — **DEGENERATE** (below floor)._
- _`run-20260725-143502-8d214-seq-dualgru` (D-LD latent/delta, H 0.14673, recall@10 0.09154 = 131/1,431) — **DEGENERATE** (below floor)._
- _`run-20260725-143502-12fe6-seq-dualgru` (D-LL f1 latent/latent, H 0.14635, recall@10 0.10273 = 147/1,431) — **DEGENERATE by the noise-band floor** (0.0203 below baseline, outside ±0.015); this is the one case where the 0.108 floor is stricter than the suggested 0.10. See the caveat above — it is statistically tied with the excluded a95c5._
- _`run-20260725-143502-4324f-seq-dualgru` (D-DD delta/delta, H 0.13335, recall@10 0.05800 = 83/1,431) — **DEGENERATE AND DOMINATED**: lowest H *and* near-lowest recall of the batch, i.e. excluded on both axes. Consistent with the report's finding that a single tower on `delta`/`cummean` is bounded at 0.058-0.066._

_**RESULTING GROUP (5 of 12 slots; every member recall-healthy):** rank 1 `best-seq-nexttrack-20260725-143143-97c1e` (`seq-nexttrack`, H 0.14172, recall@10 0.12299 — the C0 baseline, bit-identical to the on-record gru-infonce-h256 number); rank 2 `best-seq-nexttrack-20260725-143502-be11b` (H 0.13850, recall 0.12089 — the C1 h425 param-matched control); rank 3 `best-seq-dualgru-20260725-143502-18b4a` (H 0.13774, recall 0.11880 — D-LL-f0); rank 4 `best-seq-dualgru-20260725-143502-3e5d4` (H 0.13648, recall 0.11740 — D-LC-f0); rank 5 `best-seq-blend-20260723-014711-1c1a3` (`seq-blend`, H 0.06316, recall@10 0.21174 — the champion reproduction, see (2b)). All 5 on the canonical split `seq-20260715-131139`, all n_test 1,431. Every member's recall@10 is within ±0.015 of (or far above) the canonical baseline 0.12299. No suspicious metrics anywhere in the candidate field: n_test uniformly 1,431, top H 0.197 (excluded) is nowhere near 1.0, no leakage signature. Pins 3 (UNCHANGED, inert — see (2)); excluded list now **46** (prior 39 + these 7)._

_**FLAGGED BUT DELIBERATELY LEFT ALONE:**_
- _**The 5 remaining campaign runs were NOT excluded, DEPARTING from the standing campaign-run de-duplication precedent (2026-07-18-night / 2026-07-20 / 2026-07-22) — deliberately.** 4 of the 5 group members are this campaign's negative-result/control runs, and the precedent would exclude them as validation-only. But that precedent always rested on an unstated premise: **that exclusion BACKFILLS from a richer candidate field.** Here it cannot — only 12 runs in the entire 95-run history carry the primary metric (see (2)), so excluding the recall-healthy campaign runs would shrink the group to **ONE** member and collapse `POST /api/best-models/predict` into a single-model passthrough. That is a strictly worse outcome than seating 4 recall-healthy, noise-equivalent-to-baseline models. **Their membership is PLACEHOLDER, not endorsement** — the dual-tower family remains REFUTED (no arm beat C0 by paired-Δ CI>0) and nothing here is a champion or a promotion candidate. Re-visit and apply the normal de-dup precedent once the champions are re-run under holisticness-carrying code (see (2b)); the group should then be re-curated to a genuine top-12._
- _**Family monoculture (2 of 5 = 40% `seq-dualgru`, a REFUTED family) — accepted, not fixed.** No diversity pin is possible: the only unrepresented families (`seq-markov`, `seq-popularity`, `seq-mood`) have either NO holisticness-carrying run, or only the degenerate mood run just excluded. Pinning a clearly-worse or degenerate model for diversity's sake is not warranted, so the trade-off is stated rather than acted on._
- _**`test-r` (promoted `seq-nexttrack` model, no notes, points at `run-20260723-002044-af11a-seq-nexttrack`) — a stray test artifact.** Recommend the USER delete it (deletion is not this agent's call). **No curation delta needed: it is already doubly blocked** from the group — its run carries no `holisticness_at_k` (unrankable) AND `af11a` has been in the exclusion list since 2026-07-22._
- _**The prior 39 exclusions were reviewed for staleness and ALL PRESERVED — every one is currently INERT.** None of the 39 excluded runs carries `holisticness_at_k`, so under the current crown they are unrankable regardless; un-excluding any would be a pure no-op that discards audit history. Notably `run-20260718-193111-a781c` (the τ0.10 sub-noise fluke) remains unconfirmed → its original reason still holds. Re-review if/when any of those runs is re-evaluated under the holisticness suite._

_**2026-07-26 (post-campaign) — PRE-ENCODER FLOOD CURATED + THE METRIC-GENERATION CUT MADE EXPLICIT (best-model-selector; 13 run_id excludes + 1 model-name exclude, ZERO pins/unpins/unexcludes; champion UNCHANGED, nothing promoted, nothing deleted). Resulting group = TWO members. Read (2) first — the ordering defect is bigger than the flood.**_

_**(1) THE FLOOD, EVICTED.** The runner's mandatory manual recompute (12/12 at 17:11:52) had seated 7 of the pre-encoder campaign's 10 REFUTED runs. Applied in ONE `PUT /api/best-models`; every remaining rankable run was adjudicated (not just the seated ones), because freeing slots backfills from the same field (standing lesson from the 2026-07-22 note). **Below the standing 0.108 recall floor** (adopted 2026-07-25 = canonical baseline 0.12299 − the ±0.015 band): `run-20260726-161801-0df6f-seq-dualgru` (0.10273 — the C3 bare-dual reproduction, EXACTLY the value that got `12fe6` excluded yesterday, so this is precedent-identical), `-dc7d2` (0.09993), `-ca0b8` (0.09993), `-bd5b5` (0.09783), `-9450b` (0.09713). **Demonstrated WORSE than the canonical baseline by paired-Δ CI<0** — a strictly stronger disqualifier than the floor, which both of these squeak past: `-ad53f` (S1 `MLP256→GRU256`, 0.10901, Δ −0.01398 [−0.02657, −0.00140]; the campaign's decisive LOSS and the one arm the report calls actively harmful) and `-7bab3` (h297 capacity control, 0.11321, Δ −0.0098 [−0.0175, −0.0028]). **Tie-only CONTROLS / redundant near-duplicates from the two refuted campaigns** (each ties the baseline with paired-Δ straddling 0, i.e. "equal, likely" and never "better"; keeping them would seat several noise-equivalent copies of one architecture): `-7646d` (h454, 0.12509 — nominally the batch-max single-GRU recall, but its own leaderboard row states it "authorizes nothing", 2.5× params, Δ +0.0021 straddles), `run-20260725-143502-be11b` (h425, 0.12089, Δ −0.0021 straddles), `run-20260725-143502-18b4a` (D-LL-f0, 0.11880) and `-3e5d4` (D-LC-f0, 0.11740) and `run-20260726-161801-a6423` (T1-f0, 0.12369, was at rank 13 and would have backfilled) — three `fusion_layers=0` cells of a REFUTED family all returning the same verdict, so at most one could be a representative and none is needed (see (3) on the diversity trade-off). **Redundant reproduction:** `run-20260725-143143-97c1e` (C0 h256, recall@10 0.12299 = 176/1431) is bit-identical to `run-20260726-161429-bdcbc`; the later copy is KEPT and the older excluded, because `97c1e`'s stored holisticness is from the retired 3-factor generation and was sitting at **rank 1 above the champion** purely for that reason (see (2))._

_**(2) THE GROUP RANKS ON TWO DEAD METRIC GENERATIONS — the real defect, and curation CANNOT fix it.** Of 105 runs, **22 carry `holisticness_at_k` at all**, and they split cleanly by code generation: **12 runs (2026-07-23/25 batches) hold 3-FACTOR values (0.063–0.233)** and **10 runs (2026-07-26 batch) hold GROUNDED 4-FACTOR values with `MUSIC_FLOOR=0.185` (0.0440–0.0471)**. **ZERO runs carry the CURRENT (2026-07-26, third) generation** — `eager = max(artist_adj, artist_conc)` with `MUSIC_FLOOR = 0.34`. Verified per run by re-deriving each stored value from its own facets. Consequence: **the server's ordering is dominated by which code generation wrote the value, not by model quality** — every 3-factor run mechanically outranks every 4-factor run, because the grounding factor multiplies by ~0.3. That is what put a capacity control and a refuted dual-tower arm above the 0.212-recall champion. **Recomputed on the LIVE definition** (all facets are stored per row; `artist_conc` is reconstructible from `top_k_ids` + `items.json` as the normalized Herfindahl, which is how this was done — method verified by reproducing the orchestrator's own numbers exactly: C0 → **0.02454** vs its reported 0.0245, `mood-session` → **0.00000**), the healthy field is FLAT and re-ordered: C0 h256 `bdcbc`/`97c1e` **0.02454** (field leader), h425 `be11b` 0.02487, S1 `ad53f` 0.02410, h454 `7646d` 0.02399, D-LL-f0 `18b4a` 0.02378, `0df6f` 0.02284, `ca0b8` 0.02205, `dc7d2` 0.02188, D-LC-f0 `3e5d4` 0.02152, `9450b` 0.02127, **champion blend `1c1a3` 0.00860** (last — it is the most artist-eager model on record: artist_adj 0.610 / artist_conc 0.692). The whole spread is 0.0036 against a holisticness CI half-width of 0.0018 (measured today, n=1431) → **under the live metric these models are statistically indistinguishable from each other, and the group's rank order carries no information.** `run-20260726-161801-{a6423,7bab3,bd5b5}` and the six 2026-07-25 excludes could not be re-derived at all: their run dirs were never materialized locally (DB-only remote runs, no `predictions.json`) — a second reason not to trust stored cross-generation values. **FORWARD HAZARD:** because generation-3 values (~0.024) are uniformly BELOW the stale 3-/4-factor values (0.044–0.233), every genuinely good FUTURE run will rank at the BOTTOM of this group and will be locked out once the 12 slots fill. The 10 free slots left by this curation are deliberate headroom, not under-curation. **OWED (orchestrator-level, not curation):** rescore `holisticness_at_k` for the facet-carrying runs under the live definition (every input is stored per row — keep it that way), or the group will keep ranking on retired formulas._

_**(3) RESULTING GROUP — 2 of 12 slots, and each seat is earned by evidence that SURVIVES the metric cut:** rank 1 `best-seq-blend-20260723-014711-1c1a3` (`seq-blend`, stored H 0.063159 [3-factor], **recall@10 0.21174**) — the bit-identical reproduction of the pinned champion `blend-gru-markov-content-proj`; it earns its seat on the record's comparability currency (best on record, seed-robust ×3, second-split hardened), NOT on holisticness, where it ranks last. Rank 2 `best-seq-nexttrack-20260726-161429-bdcbc` (`seq-nexttrack`, stored H 0.047086 [4-factor], **recall@10 0.12299 = 176/1431**) — the canonical single-GRU h256 baseline, reproduced bit-identically 7×, AND the leader of the whole field under the live generation-3 metric (0.02454). Both on the canonical split `seq-20260715-131139`, both n_test 1,431. **No suspicious metrics anywhere in the candidate field:** n_test uniformly 1,431, no near-perfect score (top stored H 0.233 belongs to the excluded degenerate mood run), no leakage signature, no tiny-split artifact. **NO PINS ADDED — deliberately.** (a) Neither member can be displaced: both stored values (0.063 / 0.047) sit above anything generation-3 can produce (~0.024), so a pin would be inert insurance. (b) The right instrument for anchoring the h256 baseline is a PROMOTION, not a pin: promote `gru-infonce-h256` onto `run-20260726-161429-bdcbc` and the anchor appears under its own name instead of the anonymous `best-seq-nexttrack-…-bdcbc` alias (same "PROPER path" as the 2026-07-20 de-dup note; likewise re-promote `blend-gru-markov-content-proj` onto `1c1a3` per the 2026-07-25 note (2b)). (c) **No diversity pin is possible or warranted:** `seq-markov` / `seq-popularity` carry no holisticness-scored run, `seq-mood`'s only one is the degenerate excluded run, and the sole remaining family — `seq-dualgru` — is REFUTED with its best arm merely TYING the baseline while the record shows its towers largely DUPLICATE the raw-latent tower (correlated, not complementary, failure modes). **The trade-off taken explicitly: a distinct-family seat for `18b4a` was available at no measurable recall cost, and was declined** because its rank-1 position came entirely from a retired-generation value (stored 0.1377 vs live 0.0238, a 5.8× inflation) and seating a refuted-family arm ABOVE the champion is exactly the misleading state this pass was called to fix. Nothing was deleted; every excluded model stays promoted and directly servable by name._

_**(4) `88a4c` — the degenerate H leader — exclusion CONFIRMED PERMANENT, and the metric revision has now independently defanged it.** `run-20260725-143502-88a4c-seq-dualgru` (cummean/cummean, recall@10 0.06639 = **95 of 1,431**) was already excluded by run_id on 2026-07-25 and that exclusion HELD through today's flood — it is absent from `entries` and survives every recompute (exclusions are persisted in `data/best-models.json` and re-applied in `rank()`). Its apparent "return" is on the all-time analysis board the runner recomputes off-record, not in the group. Belt-and-braces added anyway: its auto-promoted **model name** `best-seq-dualgru-20260725-143502-88a4c` is now excluded too, so it is barred by BOTH keys that `rank()` checks. **There is no pin-based alternative** — pins only ADD members and cannot demote an auto entry, so exclusion is the only instrument; the durable protection against a RE-RUN of the same config (a fresh run_id cannot be pre-excluded) is the standing **recall@10 ≥ 0.108 floor**, which rejects 0.066 on sight. Note also that the 2026-07-26 crown revision removes its metric claim at the source: under `eager = max(artist_adj, artist_conc)` + floor 0.34 it scores **0.0141**, i.e. BELOW the healthy field (0.021–0.025) instead of leading it — the blind spot it exploited (artist_adj 0.19 vs artist_conc 0.29) is now closed. Original reason (degenerate: wins a correctness-free composite by not attempting correctness) still holds regardless._

_**(5) FLAGGED, LEFT ALONE.** (a) **The 3 pins stay in place and stay INERT** (`blend-gru-markov-content-proj`, `blend-gru-markov-content`, `gru-lstm-pca128`): their runs pre-date the holisticness suite, so they are unrankable and a pin cannot rescue an unrankable candidate (2026-07-25 note (2)); they remain the correct standing instruction and will bite the moment each is re-promoted onto a scored run. (b) **The prior 46 exclusions were re-reviewed and ALL PRESERVED** — every one is still inert or still correct; notably `run-20260718-193111-a781c` (τ0.10 sub-noise fluke) is still unconfirmed. Zero unexcludes. (c) **`POST /api/best-models/predict` is a NO-OP for this domain** — `predict_group` (`best_models.rs`) clears `prefix` and the `Task::Ranking` branch is documented as "not used for ranking", so consensus predict cannot serve next-track; the group's real function here is the SHORTLIST OF RECORD (UI + agents). This weakens the 2026-07-25 argument for seating placeholder members to avoid "collapsing consensus to a passthrough", which is why this pass applied the normal de-duplication precedent instead of deviating from it. (d) **`test-r`** (stray promoted `seq-nexttrack` on `af11a`) — still recommended for USER deletion; no delta needed (unrankable AND already excluded). (e) **Is this a defensible top-N? It is a defensible top-2 and NOT a top-12** — the group is still shaped by metric COVERAGE (22 of 105 runs scored, none on the live definition), and no amount of curation can fix that: the champions of record (R+M+C 0.186, R+M 0.173, LSTM single 0.127) remain structurally invisible. Every seat that COULD be filled honestly is filled._

## Follow-ups
- **NEW (2026-07-26) — report-curator (sync mode) for the pre-encoder campaign:**
  fold `2026-07-26-nexttrack-pre-encoder-scan.md` into `docs/experiments.tex` (+
  `docs/experiments.es.tex`) and rebuild the PDFs. Natural home = alongside the
  dual-tower null as the closing of the **input-side** asymmetry direction (the
  two together close deterministic view-splits AND learned re-embedding); the
  `fusion_layers=0` default now has THREE confirmations with a growing effect
  (+0.016 / +0.020 / +0.0266) and deserves a stated default, not a caveat. NOT
  auto-updated.
- **NEW (2026-07-26) — best-model-selector (do first):** after the mandatory manual
  `POST /api/best-models/recompute`, the group holds **7 of the pre-encoder
  campaign's 10 refuted runs**, three of them below 0.10 recall (`dc7d2` 0.0999,
  `ca0b8` 0.0999, `9450b` 0.0971). Exclude the refuted arms, restore family
  diversity, consider pinning the h256 baseline as the single-model anchor, and
  make the exclusion of the degenerate H leader `88a4c` PERMANENT so it stops
  being re-litigated every campaign. See §Best-models group.
- **NEW (2026-07-26) — B4 SAE read on the pre-encoder (trigger FIRED via the tie
  clause, analysis-only):** `POST /api/interp/model-sae` `topk=32` on
  **`run-20260726-161801-a6423-seq-dualgru`** (T1-f0; taps `pre_b`, `tower_a`,
  `tower_b`) and **`run-20260726-161801-ad53f-seq-nexttrack`** (S1; `model-sae` now
  taps `pre` ahead of `recurrent`). Both `has_checkpoint=true`. For
  information-capture-analyst. **The question, sharpened by the harm finding:** is
  the RECTIFIED representation measurably LOSSIER than the bare recurrent state —
  i.e. lower `next_item_decodability` on `pre`/`pre_b` than on `tower_a` — which
  would confirm the half-wave-rectification mechanism directly? The view-split
  variant's answer was +0 new next-item concepts at the worst genre AUC measured
  (0.764 vs 0.801). Carry the L1-won't-sparsify pitfall (read
  `n_interpretable_concepts` as saturated; trust ranked `atoms_by_concept` +
  `next_item_decodability`).
- **NEW (2026-07-26) — two orchestrator-level code changes the campaign identified
  but deliberately did NOT improvise:** (a) a **`pre_dropout`** param (default =
  `dropout`) to disentangle the pre-MLP's trailing `Dropout` — pre arms currently
  get GRU-INPUT dropout the bare arms lack, so part of the measured harm may be
  over-regularization; (b) an **`activation`** param on the pre-encoder, needed for
  the only probe the harm mechanism makes worth running (sign-preserving
  tanh/GELU/linear, and/or `pre_hidden` ≥ 3–4× input dim). Both need a fresh
  design + approval; neither is authorized by the pre-encoder campaign.
- **NEW (2026-07-26) — one cheap replication worth buying: single-GRU width jitter.**
  Adjacent widths differ by up to 0.0119 recall with no trend (h297 LOSES to h256
  CI<0 while h454 ties it), which weakens every param-matched control this project
  uses. `seq-nexttrack` is deterministic, so this needs a second SPLIT, not a
  second seed: re-run h297 + h454 on `seq-20260718-211238`. 2 runs.
- **NEW (2026-07-25) — report-curator (sync mode) for the dual-tower campaign:** fold
  `2026-07-25-nexttrack-dual-tower-fusion-scan.md` into `docs/experiments.tex` (+
  `docs/experiments.es.tex`) and rebuild the PDFs. Natural home = the combiner-null
  arc (this is the 4th and strongest null: representation-level, jointly trained,
  param-matched — it closes the "post-hoc combiner" rebuttal); the 2026-07-13
  topology-scan section should also gain "width is now closed on PCA-192, not just
  AE-64". NOT auto-updated.
- **NEW (2026-07-25) — best-model-selector:** the group is STALE (`updated_at`
  2026-07-23, 2/12 entries) after 10 metric-carrying runs, and on recompute the
  degenerate `run-20260725-143502-88a4c-seq-dualgru` (H 0.19715 at recall@10 0.0664)
  must be excluded. See §Best-models group.
- **NEW (2026-07-25) — B4 SAE tower-specialization read (trigger FIRED, analysis-only):**
  `POST /api/interp/model-sae` `topk=32` on `run-20260725-143502-3e5d4-seq-dualgru`
  (D-LC-f0, `has_checkpoint=true`), for information-capture-analyst. `seq_dualgru`'s
  `model-sae` taps `tower_a`, `tower_b` and each `fusion_*` layer SEPARATELY, so it
  can answer directly what the recall numbers only imply: do the two towers capture
  different next-item concepts or duplicate each other? (Prediction from the view
  attribution: duplication in every cell containing a raw-`latent` tower.) Carry the
  registered pitfall: the L1 SAE will NOT sparsify on these dense recurrent states —
  read `n_interpretable_concepts` as saturated, trust ranked `atoms_by_concept` +
  `next_item_decodability`.
- **ARCHITECTURE IS NOW CLOSED for the next-track family (2026-07-25, REINFORCED
  2026-07-26).** Depth / width / residual / direction closed on AE-64 (2026-07-13);
  single-GRU width closed on PCA-192 and dual-tower fusion + fusion depth refuted
  (2026-07-25); **the INPUT side is now closed too — a learned nonlinear per-step
  pre-encoder is refuted and is actively harmful in a single tower (2026-07-26),
  which is the FIFTH consecutive architecture null.** The two 2026-07-25/26 nulls
  are complementary: deterministic view-splits AND learned re-embedding both fail,
  so "asymmetric encoder" is closed in both its variants. Every real
  gain on this corpus has come from the REPRESENTATION / SUPERVISION side, so the
  best-motivated open direction remains **fitting the InfoNCE projection ON a
  whitened / `std-noaco` / metric space** (compose the two known-good fixes,
  `2026-07-18c`). De-prioritize dual-tower-as-the-R′-leg inside the champion blend:
  it needs a `seq-blend` code change AND the standalone model does not work, and the
  record's lesson is that single-model gains do not propagate to the blend (content
  the sole exception).
- **A CONSTRAINED holisticness objective**, if that board is pursued: the dual-tower
  batch's H gains are all recall-destroying. The MMR λ≈0.9 eval-time re-rank
  (`2026-07-22`) remains the only near-free holisticness lever on record.
- ~~Blend GRU-infonce with the Markov bigram~~ DONE — the champion.
- ~~Item-representation bake-off + dim extension~~ DONE — PCA-128 base, dim peaks ~192.
- ~~Phase-2 model-family sweep (LSTM/ANN/stacker) + R+M+C on rich space~~ DONE —
  combiner = fixed z-blend; LSTM best single; R+M+C-192 crowned (see below).
- ~~CONFIRM R+M+C on PCA-192~~ DONE — confirmed 5× (spread 0, paired-Δ vs prior
  champion +0.013 >0 every rerun) → **CROWNED as the new champion ROW** (best on
  record; train-only + off-server, NOT promoted). Base stays PCA-128.
- ~~Literature-fit exploration (session-kNN / rules / attention / fusion / cold-start)~~
  DONE — `2026-07-18-literature-fit-campaign.md`: learned content projection is the
  new best-on-record ROW; all borrowed-method lanes closed (see the lit-fit map in
  Pitfalls).
- ~~Register the learned content projection (R′+M+C′) as a leg / predictor~~ DONE
  (2026-07-18b) — implemented as first-class `seq-blend` hyperparams (`projection`/
  `projection_rank`/`projection_objective`/`projection_tau`/`projection_epochs`/
  `projection_lr`; `projection=true`⇒`content=true`; map baked into the model dir as
  `projection.npz` for serving). Definition `blend-gru-markov-content-proj`
  reproduces the off-server driver EXACTLY (R@10 0.21174 / MRR 0.12199,
  `run-20260718-191539-1b89c-seq-blend`) and is PROMOTED + SERVABLE, best-models
  rank 1. The objective/rank/τ scan confirmed the InfoNCE/full/τ0.07 default.
  `2026-07-18b-nexttrack-projection-registration-and-scan.md`.
- ~~**Harden the R′+M+C′ crown on a second leak-free test split**~~ **DONE —
  CONFIRMED (2026-07-18 evening).** Re-ran `blend-gru-markov-content-proj` + the
  R+M+C crown (`blend-gru-markov-content`) + R+M (`blend-gru-markov`) on the
  disjoint 0.64-cold earlier-holdout `seq-20260718-211238`
  (`run-20260718-212639-50e1d` PROJ / `-213156-cf1dd` R+M+C / `-214114-93bad` R+M;
  all n_test 1431). PAIRED-Δ vs the R+M+C crown **+0.0426 [+0.0280, +0.0573] (CI
  entirely >0)**, no MRR regression (PROJ 0.1705/0.1050 vs R+M+C 0.1279/0.0816); vs
  R+M +0.0531 [+0.0391, +0.0678]. Absolute R@10 below 0.212 by construction (0.64
  cold) — the verdict is the paired-Δ. Margin WIDER than the first split (+0.0259).
  With seed-robust ×3 + cold-reaching, the crown is now hardened on TWO disjoint
  leak-free splits. VALIDATION-ONLY (nothing promoted/re-promoted; champion stays
  the first-split model). `2026-07-18b-...projection-registration-and-scan.md`
  §Second-split hardening.
- **Sweep the projection** — objective (BPR vs InfoNCE) + rank + coarse τ DONE
  (2026-07-18b: InfoNCE ≫ BPR, full-rank ≥ rank-64, τ mildly sensitive — softer
  slightly better, τ0.10 a sub-noise +0.011 nominal edge over τ0.07). REMAINING:
  epochs, a finer τ sweep (0.12/0.15) to chase the τ0.10 edge with a seed +
  second-split confirm, and a GRADED-relevance projection objective (fit toward
  artist/genre adjacency; the graded axis is also reweight-movable — oracle/RRF
  lift artist@10/genre@10).
- **Promote R′ (projected single GRU ≈0.176)?** nearly the OLD champion blend as a
  SINGLE model — the cleanest servable next-track model if pursued.
- **Register R+M+C as a 3-leg `seq-blend`** — prerequisite to making the champion
  row promotable / app-exportable (currently off-server driver only).
- **Generalize `seq-blend` to 3 legs** so R+M+C is a first-class registered
  predictor (currently off-server) — prerequisite to a promotable champion row.
- Confirm PCA-192 vs PCA-128 as base (paired-Δ still grazes 0) — folds into the
  R+M+C confirm above (same GRU reruns).
- Phase 4: "extend-a-session" showcase demo in `../app`.
- Phase 5: generalize + upstream the sequence/ranking framework support.
- `predict` subcommand → make ranking models best-models-promotable + app-exportable.
