# PROJECT FACTS — Spotify Next-Track (session recommendation)

Living, agent-maintained roll-up of empirical knowledge for the **next-track /
session-recommendation** instance. The `experiments/*.md` campaign reports are
PRIMARY; this file indexes them. Newest report wins.

_Last updated: 2026-08-02c (latest) after `2026-08-02c-dropout-seed-grid-and-the-walk-read.md` - **THE f0 DROPOUT LINE IS CLOSED: A POSITIVE RECALL EFFECT ON THE DEPLOYED `latent/cummean` PAIR IS **EXCLUDED**, AND THE CAMPAIGN'S BY-PRODUCT - THE FAMILY'S **FIRST SEED-VARIANCE MEASUREMENT** - RE-FLAGS THE D2-R2 WIN THAT LAUNCHED THE LINE AS **UNCONFIRMED**** (`2026-08-02c-dropout-seed-grid-and-the-walk-read.md`, 4 new runs + 2 REUSED as the seed-1337 cell, all remote on `snappler:879965` via `queue:true`, 27.5 min new compute, 0 failed, 2 chunks of 2, plus 3 walk invocations on the hub; NOTHING DEPLOYED, NO champion registration, NO definition saved, crown metric + `domain.toml` UNCHANGED, orphan row NOT touched, NO server restart; 2 measurement-only promotions, both correctly EXCLUDED from best-models by the concurrent selector pass). **PHASE A - seed grid `dropout` {0.0,0.1} x `seed` {1337,7,42} on `seq-bank` `views=latent+cummean` h256.** Per-seed paired Drecall@10 (d0.1-d0.0): seed 1337 **+4 hits** +0.002795 [-0.002795,+0.007704], seed 7 **-4 hits** -0.002795 [-0.008386,+0.002795], seed 42 **-13 hits** -0.009085 [-0.018868,+0.000699] - **all straddle, SIGNS DISAGREE (+ - -)**. Seed-averaged per-session instrument: **-0.003028 [-0.006988,+0.000932], hw 0.003960, straddles, leaning NEGATIVE**. Verdict by the pre-registered rule: **SETTLED-POSITIVE EXCLUDED; primary branch STILL UNRESOLVED in SIGN** (signs disagree AND the per-seed spread sd 0.005943 is ~2x the |across-seed mean| 0.003028). **BUT the motivating question IS settled: the resolvable floor 0.003960 is FINER than the D2-R2 effect hunted (+0.004892), so had that effect existed here it would have shown CI>0 - it came back the other sign. There is NO free recall in regularizing the deployed pair.** Floors pass (worst abs recall 0.111810 >= 0.108; every Dmusic@10 straddles). **PHASE B - walk, 40x20, artist cap 1, base = the DEPLOYED engine, both cells.** E1 @ a0.8/s0.5 (PRIMARY): **Dvibe +0.0024 [-0.0044,+0.0082] straddles -> NOT CI<0, so E1 PASSES the candidacy test** and does NOT repeat the tower-bank collapse; Dstride_err -0.0129 (better, straddles) so the vibe reading is not the tiny-steps artifact; Dground -0.0042 straddles. E1 @ a0.4/s0.3 (matched): Dvibe +0.0007 straddles, Dstride_err +0.0036 straddles, **Ddrift -0.0226 [-0.0401,-0.0056] CI<0** - a real late-walk mood-holding regression at the matched cell that does NOT appear at the deployed cell. **But Phase A removed the REASON to redeploy, so the campaign ends with no ship recommendation.** 

_Previously: 2026-08-02b after `2026-08-02b-dropout-on-the-deployed-view-pair.md` - **THE f0 DROPOUT WIN DID NOT REPLICATE ON THE DEPLOYED `latent/cummean` VIEW PAIR, BUT THE TEST WAS UNDERPOWERED RATHER THAN NEGATIVE, SO THE DEPLOYED ENGINE'S REGULARIZATION STATUS IS **OPEN**, NOT SETTLED** (`2026-08-02b-dropout-on-the-deployed-view-pair.md`, 2 runs both remote on `snappler:879965` via `queue:true`, 13.3 min compute / 13.4 min wall, 0 failed, no contingency runs authorized or spent; NOTHING DEPLOYED, NOTHING PROMOTED, NO DEFINITION SAVED, no champion registration, crown metric + `domain.toml` UNCHANGED, orphan row NOT touched, NO server restart). S1 (paired, 2000 resamples, rng 1337, n=1431, E1 dropout 0.1 vs E0 dropout 0.0 at byte-identical params on `views=latent+cummean` h256): **Drecall@10 +0.002795 [-0.002795, +0.007704] STRADDLES, 172 vs 168 hits, half-width 0.005250** -> **NOT REPLICATED** by the rule. But the half-width is LARGER than the whole D2-R2 effect it was sent to reproduce (+0.004892), and the two intervals are mutually compatible in BOTH directions (D2-R2's point estimate sits inside this CI; this point estimate sits inside D2-R2's CI; same sign, ~57% of the magnitude) - only **16 of 1431 rows flip at all** (10 gained, 6 lost), so these runs cannot distinguish "no effect on this view pair" from "the same effect, unresolved". Do NOT read this as localizing the dropout win to `latent+latent`. Floors both pass (abs recall 0.120196 >= 0.108; Dmusic@10 +0.001068 [-0.002067, +0.004160] straddles). **The f0-inertness DEFECT is now DEMONSTRATED, not just inferred:** the deployed engine (nominally `dropout=0.1`) and a `seq-bank` twin at EXPLICIT `dropout=0.0` produce **the same 1431 ranked lists, byte for byte** (G-E0). S2/walk NOT entered (gated on S1 replicating). 

_Previously: 2026-08-02 after `2026-08-02-tower-bank-parallel-towers.md` - **THE PARALLEL TOWER-BANK AXIS (N>=3 bare-GRU towers at `fusion_layers=0`) IS CLOSED ON BOTH SURFACES, AND EVERY `fusion_layers=0` ARM ON RECORD — INCLUDING THE DEPLOYED ENGINE — IS NOW KNOWN TO HAVE TRAINED UNDER-REGULARIZED** (`2026-08-02-tower-bank-parallel-towers.md`, 10 runs all remote on `snappler:879965` via `queue:true`, 84.2 min compute / 86.5 min wall, 0 failed, **0 of 3 contingency runs spent**; NEW PREDICTOR `seq-bank` REGISTERED + server restarted at the zero-live-runs gate; 8 arms promoted **FOR MEASUREMENT ONLY** (every `notes` = "promoted to enable walk evaluation; not a crown claim"); **NO DEFINITION SAVED**, nothing promoted as champion, NOT deployed, crown metric + `domain.toml` UNCHANGED, orphan row NOT touched). Tested N (2/3/4) and view-multiset as ONE hyperparameter (`views`) on ONE module, capacity equalised to ~1,185,000 params (mutual spread **0.405%**, all counts verified from the artifacts' own `n_params`). **VERDICT — THE DOSE HYPOTHESIS IS REFUTED; V3 CLOSURE SATISFIED.** S1 (paired vs in-batch C0, 2000 resamples, rng 1337, n=1431): **every one of the nine arms STRADDLES 0** — no win, no refutation, all S1-TIE; but five arms LOSE to the param-matched control C1 h512 CI<0 (V1 −0.01677, V2 −0.01398, R2/B2 −0.01118, B3 −0.00978). **B4 (four h211 towers, 1,187,700 params) and C1 (one h512 GRU, 1,182,912 params) return the SAME 186/1431, Δ = 0.000000 [−0.00839, +0.00839]** — topology and width are interchangeable at this budget, which is exactly what the mandatory C1 leg exists to catch. **ADDING `cummean` TOWERS MONOTONICALLY DESTROYS THE WALK** (S2 primary a0.4/s0.3, base C0, G3 green): V1 `l+l+cummean` Δvibe −0.0174 CI<0 (and Δgenres −1.725 CI<0) and V2 `l+cummean+delta` −0.0144 CI<0 both REFUTED, V3 −0.0094 straddles, all-`latent` arms ≈0 — so the "fixed reference never overwritten by recurrence" mechanism does NOT survive being dosed. B2 and V3 each earned a real S2-TRADE at the primary cell (Δdrift +0.0339 / +0.0286 CI>0, vibe held) and **BOTH FLIP TO Δvibe CI<0 AT THE DEPLOYED CELL a0.8/s0.5** (−0.0064, −0.0112) with the drift gain evaporating — **the secondary-cell clause is what prevented two false saves; keep it mandatory.** Stride probe s∈{0.2,0.3,0.4} on V1/V2/V3: **all three PEAK at s0.3**, the cell they were already in (3rd campaign running), and at s0.2 vibe only goes positive-straddling by taking visibly shorter steps (Δstride_err CI>0) and losing genres (CI<0) — a clean illustration of the coupling caveat. **THE ONE POSITIVE FINDING IS A BY-PRODUCT: D2 − R2 at BYTE-IDENTICAL params (789,696 both), only `dropout` differing, = +0.004892 [+0.000699, +0.009783] CI>0 (177 vs 170 hits) — dropout 0.1 BEATS dropout 0.0.** Since `DualTowerNextLatent` has NO active dropout at f0/layers=1/pre_hidden=0, every f0 arm on record trained unregularized and left ~+0.005 recall on the table; the three recorded f0-vs-f1 gaps (+0.016/+0.020/+0.0266) are NOT like-regularized and the true f0 advantage is if anything LARGER. Cheapest item on the board: retrain the deployed dual `l/cummean` at dropout 0.1. **4th confirmation of the val-loss dissociation, now quantified: Spearman(best val InfoNCE, hits) = −0.2954, p=0.407** (B4 6.5657/186 vs B2 6.5660/170 — 0.0003 of val loss, 16 hits; V1 has the 4th-best val loss and the WORST hits; C0 has the WORST val loss and beats six of eight bank arms). Gates: P1 md5 closure fired again (`seq_bank.py` ABSENT on the worker, `registry.toml` STALE pre-edit), P2 + a new P2c training-stream identity check both 0.00e+00, **G2a C0 = 176/1431 EXACTLY (10th reproduction)**, **G2b R2 = 170/1431 EXACTLY (hard gate, band unused)**, G1 13/13 keys on all 10, G3 straddles both axes, P5 reproduced all ten 2026-07-31b means. Closure scope: **"N>=3 parallel bare-GRU tower banks at `fusion_layers=0`" ONLY — the dual-tower family stays REOPENED and remains the deployed engine.** — 2026-08-01 - **THE STAGE-STACK AXIS IS CLOSED ON BOTH SURFACES, AND THE TRAINING OBJECTIVE IS NOW PROVABLY MISALIGNED WITH WHAT WE RANK ON** (`2026-08-01-stage-stack-topology.md`, 7 runs all remote on `snappler:1914390` via `queue:true`, 90.0 min compute / ~91 min wall, 0 failed, **0 of 3 contingency runs spent**; NEW PREDICTOR `seq-stack` REGISTERED + server restarted at 0 live runs; 7 arms promoted **FOR MEASUREMENT ONLY** (every `notes` = "promoted to enable walk evaluation; not a crown claim"); NOTHING PROMOTED AS CHAMPION, no champion registration, NOT deployed, crown metric + `domain.toml` UNCHANGED, orphan row NOT touched). Tested the user's four `ann`/`gru` arrangements as ONE hyperparameter (`stages`) on ONE module so topology is not confounded with implementation, capacity equalised to ~1,282,000 params (mutual spread **0.445%**; A/B IDENTICAL by formula; C/D matched to 0.015%; all 7 counts re-verified by instantiating the module). **VERDICT — ALL FIVE 4-STAGE ARMS REFUTED ON BOTH SURFACES; V3 CLOSURE SATISFIED.** V1 (paired vs the in-batch C0, 2000 resamples, rng 1337, n=1431): Δrecall@10 CI<0 for every arm — B `ann+gru+gru+ann` −0.03843 [−0.05381, −0.02306] (123/1431), E `gru+ann+gru+gru` −0.04193 (118), C `ann+gru+gru+gru` −0.04472 (114), D `gru+gru+gru+gru` −0.04752 (110), A `ann+gru+ann+gru` −0.05381 [−0.06848, −0.03913] (101) vs **C0 0.12439 = 178/1431** — with ΔH CI<0 too, losses of **2.5–3.7 measured half-widths**, and all five **below the 0.108 absolute recall floor**. V2 (walk, primary cell a0.4/s0.3, base C0, 40x20, artist cap 1): every arm fires REFUTED (Δvibe CI<0 with Δstride_err not CI<0) — A −0.0375, E −0.0316, B −0.0258, C −0.0231, D −0.0140; **D fires TWICE** (Δground −0.0250 CI<0) and A also loses genre variety (−1.200 CI<0). A stride probe s∈{0.2,0.3,0.4} on the only two arms with directional movement (C, E) shows **both PEAK at s0.3, the cell they were already measured in** — no arm is rescued by re-tuning stride — and C loses at the deployed a0.8/s0.5 cell too (Δvibe −0.0158 CI<0). **THE HEADLINE MECHANISM RESULT: the family's one-step teacher-forced InfoNCE and top-10 retrieval DISSOCIATE AT DEPTH.** B reaches the **LOWEST validation loss in the whole batch (6.5378 vs C0's 6.5832)** while retrieving **55 FEWER HITS** — so any future architecture search on this family that selects on val loss will select the WRONG ARM; select on recall@10 + the walk. **THE DROPOUT-COUNT CONTINGENCY DID NOT FIRE, and its failure is itself the control:** it required every 4-stage arm to stop EARLIER with higher val loss than C0, but **not one stops earlier** (21–40 epochs vs C0's 14) and **B and E reach LOWER best val loss** — so 4x dropout application is exonerated and the arms are not early-stopped out of the race. **THE USER'S ARM-E MECHANISM HYPOTHESIS IS REFUTED, the most reusable fact in the batch:** the N=1 record says a leading MLP loses −0.0140 CI<0 because "the ReLU destroys signed prefix information the GRU's gated linear input transform preserves", so E put the MLP on a **GATED HIDDEN STATE** instead of raw signed PCA — and E (118) merely TIES B and C, beats only the worst arrangement A, and comes nowhere near C0's 178. Rectifier POSITION is not the mechanism; **depth of the recurrent chain is**, and three controls close the alternatives simultaneously (capacity: C1 h537 at 3.24x params TIES C0 on both surfaces, Δrecall −0.00280 straddles — **6th confirmation single-GRU width is flat on PCA-192**; regularization: see the epoch table; the MLP itself: C vs D is recall-neutral). **V4 CONTRASTS, the durable topology facts:** (1) **BRACKETING BEATS INTERLEAVING at BYTE-IDENTICAL capacity** — `ann+gru+gru+ann` beats `ann+gru+ann+gru` **+0.01537 [+0.00280, +0.02797] CI>0**, i.e. interposing a per-step MLP BETWEEN two recurrences costs more than putting MLPs on the outside; (2) **a leading `ann` is NOT harmful at depth 4** (C vs D, params matched 0.015%: Δrecall straddles, **ΔH +0.00291 CI>0**) → **SCOPE THE PRE-ENCODER PITFALL TO THE SHALLOW CASE**, "leading ann is harmful" is an N=1 result; (3) recurrence COUNT is flat (mean(A,B) vs C, Δrecall +0.00140 straddles). **PHASE 0 — the N=2 `ann+gru` relative was WALK-EVALUATED FOR THE FIRST TIME AND IS ALSO REFUTED:** `dual-premlp256-f0` (`run-20260726-161801-a6423`, tower A bare / tower B pre-MLP 256, `fusion_layers=0`, recall 0.12369), closed in 2026-07-26 on a one-shot TIE which under the standing rule said nothing about the walk — at the matched cell a0.4/s0.3 vs the shipped GRU **Δvibe −0.0255 [−0.0432, −0.0096] CI<0** with Δstride_err straddling ⇒ REFUTED; its one real gain is **LESS DRIFT** (Δdrift +0.0403 [+0.0086, +0.0719] CI>0, −0.080 → −0.039). This is the **SAME SIGNATURE THE LSTM SHOWED on 2026-07-31b** (better stride/drift, worse vibe), so that now looks like a GENERAL property of adding pre-recurrence machinery, not anything specific to either arm. ⚠ its stored H 0.04454 lacks `artist_conc_at_k` (dead metric generation) and was NOT compared to the crown board. **ALL FOUR GATES GREEN, and two earned their keep:** the functional-equivalence gate (`seq_stack.py verify`) passed with max|Δ| **0.00e+00** at T=1,2,5,17,40 and 394,944 params both sides — run BEFORE the registry edit and restart so a fail would have cost nothing; **G2 in-batch control PASSED at 178/1431, only 2 hits off the historical 176** exactly as the re-seeding argument predicts; **G3 walk instrument PASSED** — the shipped `gru.onnx` and a freshly-trained C0 are INTERCHANGEABLE at a0.4/s0.3 (Δvibe −0.0033, Δstride_err +0.0137, both straddle), so V2 verdicts vs C0 are not a base artefact; G1 metric completeness green on all 7. **THE WALK HARNESS IS NOW A VALIDATED INSTRUMENT** — it reproduced all TEN recorded 2026-07-31b means EXACTLY and reproduced them AGAIN under 2 and then 7 added arms (arm addition does not perturb the shared session sample or RNG), which is what licenses cross-invocation pairing and made the stride probe + secondary cell affordable. `ground` again discriminated almost nothing (1 of 9 arms) — floor, never a ranking. Every 4-stage arm DE-EAGERS substantially (conc 0.359→0.263–0.318, ild 0.455→0.511–0.537) and would look interesting under an H-primary read; the **0.108 recall floor + a per-baseline relevance floor derived from C0's own `music@10` 0.45519** (every arm 5.0–8.4% below it) killed them — 5th campaign the floors have saved, and the canonical absolute 0.43 music floor would have PASSED several of these arms. best-models auto-recomputed at 02:07:10Z and the 7 measurement promotions entered the top-12 (ranks 9, 11, 13, 14+), including refuted arms at recall 0.07–0.08, plus a server-created `best-seq-stack-20260801-003536-d612e` from arm D → **best-model-selector needed**. Follow-ups: the **3-seed confirm was NOT pre-authorized and needs the user** (6 runs); **B is EPOCH-CAPPED** (ran to 40, patience never fired, best epoch 36) so its recall is a LOWER BOUND — an `epochs=120` re-run is the clean test of the dissociation; the untested axis remains the TRAINING REGIME (scheduled sampling), now sharpened by finding (6). — 2026-07-31b - **THE SHOWCASE NOW RUNS A DUAL-TOWER, AND THE CROWN METRIC WAS MEASURING THE WRONG SURFACE.** The user restated the objective explicitly: models are **CREATORS OF SEQUENCES** that carry a mood / playlist-geist, enabling **cohesive GENRE EXPLORATION** (different genres are welcome if they preserve the vibe). `domain.toml [metrics]` already says this in prose — "the goal is mood-coherent, non-eager SESSIONS, not guessing the literal next track" — but `holisticness@10` is computed on a **ONE-SHOT top-10** and its `mood_coh` is anchored to the **PREFIX** centroid, which on a walk GROWS: a journey that drifts by a thousand individually-legal steps scores well at every step while ending in the opposite region. **Drift is invisible by construction.** Also: ∂H/∂music@10 = +0.206 vs ∂H/∂eager = −0.038, so H is 5.5× more sensitive to relevance than to eagerness. NEW WALK-LEVEL EVAL (`tools/walk_eval.py`, `walk_frontier.py`, `walk_headtohead.py`; the showcase is `clients/infinite-playlist`, NOT `../app`, which belongs to a different instance): journeys from held-out real sessions scored on **transition-distribution match** (per playlist-lab: measure TRANSITIONS not averages, calibrated to the user's own sessions — real median step cosine **0.261**), **seed-anchored vibe**, **variety as credit**, and **grounding** to what the user actually played next. FINDINGS: the shipped GRU walked **2.5× tighter than real listening** (median step 0.661 vs 0.261) — "lots of artists but flat" was many small safe moves; the blend champion matched the real stride (0.317) but **could not hold a mood** (vibe 0.145–0.229, **−0.204** on one playlist) because its InfoNCE projection encodes "what follows this", not "what sounds like this" — anchoring it in PROJECTED space caps at 0.44 vibe, in RAW space it reaches 0.58. A new deterministic **STRIDE** term (subtract `s·cos(cand, prev_track)`; NOT sampling, which would break the permalink's same-recipe-same-journey guarantee) is orthogonal to the anchor and makes the conjunction reachable. **MATCHED-SAMPLE HEAD-TO-HEAD** (40 identical held-out sessions, paired bootstrap, `stride_err` replaces the unpairable W1): `dual latent/cummean f0 @ anchor 0.8 stride 0.5` vs the shipped GRU → **stride_err −0.255 CI<0, genres +1.50 CI>0, drift +0.035 CI>0, vibe +0.001 straddles (HELD, not traded)**, ground straddles; `dual latent/latent f0 @ 0.8/0.4` → **vibe +0.030 CI>0** too. Pushing the single GRU to a real stride instead **COSTS vibe (−0.041 CI<0)** and the champion **−0.133 CI<0**. ⚠ **THE DUAL-TOWER FAMILY WAS CLOSED BY THREE CAMPAIGNS ON RECALL GROUNDS AND IS THE WINNER ON THE STATED OBJECTIVE** — and the evidence was already in the record, filed as a negative: `seq_dualgru.py:113` notes the cummean tower "added real linear directions (CKA 0.458) but **ZERO next-item concepts** tower A lacked". Dead weight for next-track; apparently the useful part for mood-holding. (My prediction that cummean holds a vibe NATIVELY is refuted — native vibe 0.236 ≈ latent/latent's 0.234; it only earns its place once anchor+stride are applied.) **DEPLOYED** to the showcase (version `4a7054eb`): `dualgru.onnx` 3.16 MB (parity vs torch 1e-7 across T=1..40 — the `cummean` cumsum was the trace risk), dual-tower is now the DEFAULT engine, GRU kept for comparison, tuned anchor/stride live in `manifest.dualgru` so re-tuning is a re-bake not a code edit. The champion engine was REMOVED end-to-end (−22.5 MB): its Worker path cost ~0.94 s CPU/batch, ~100× the Cloudflare FREE plan's 10 ms cap. Live artifacts verified **byte-identical (md5)** to the measured ones. Still open: `ground` straddles for every arm (no arm is established as better at predicting the real continuation), one split / one seed, and the crown metric itself is UNCHANGED pending a decision on whether to make this the primary. — 2026-07-31 - **THE ARTIST CAP IS HARDENED ON TWO DISJOINT SPLITS AND PORTED TO THE APP** (`2026-07-30c-artist-cap-measurement.md` §Second-split hardening + §Ported to the app; 4 runs on the disjoint 0.64-cold `seq-20260718-211238`, all `queue:true`; NOTHING PROMOTED/REGISTERED, NOT deployed, NOT committed). Pre-registered two legs, BOTH HELD: Δ`artist_conc` **−0.52699 [−0.54546, −0.50825] CI<0** and Δ`music@10` **+0.02644 [+0.01919, +0.03368] CI>0** at cap 3 vs the in-batch gated control (Δhit-rate −0.04542 CI<0). **The unexpected `music@10` GAIN REPRODUCES on a colder window** (+0.026 vs +0.013 canonical), so "the cap trades LITERAL for GRADED relevance" is replicated, not a one-split quirk. Cross-split: every effect same sign/magnitude, absolute conc at cap 3 within 0.008 of canonical (0.121 vs 0.113), recall cost SMALLER and relevance gain LARGER on split 2. Bonus **3rd independent `markov_gate` reproduction** — S0/S1 hit 244/272 exactly and the gate's paired Δrecall came out **+0.01957 [+0.01048, +0.03005]**, matching the record digit-for-digit; note the gate LOWERS conc on this split (−0.071 CI<0) while it RAISED it on french touch, so its diversity effect is **query-dependent**. ⚠ **the canonical `music@10 ≥ 0.43` absolute floor does NOT transfer** (split-2 control sits at 0.4295 legitimately) — pre-registering on the paired Δ is the only reason this didn't read as a false failure; **derive floors per-baseline**. **PORT:** both levers now in `../app/worker-core/src/champion.rs` — `markov_n_u()` recovers `n_u` from the RAW baked bigram counts (no `champion.bin` rebuild, no vectors, no retraining); `apply_artist_cap()` mirrors `artist_cap_order` incl. the best-effort backfill and treats `"?"`/empty artist as UNKNOWN→never capped to match `_relevance_ids`' −1 class (0 such items today; guard is for a corpus refresh); `blend_rank()` stays ungated/uncapped so the OLD parity test still pins the original arithmetic, and `Champion::extend` (**the app's ONLY recommendation surface — nothing downstream diversifies**) now serves `RankOpts::deployed(k)`. **k-SCALING SETTLED BY MEASUREMENT:** the app allows k≤50; a FIXED cap 3 at k=50 drives conc to 0.021 (5× past the hardened point) reaching base rank p95 **278**, vs scaled p95 **156** holding conc 0.118 → cap is **30% of slots, floor 3** (`effective_artist_cap` in Rust + `predict_ranking` server-side, identical at k=10 so the hardened result is untouched). **PARITY 4/4:** `topk_matches_python_reference` passes UNCHANGED (ungated path byte-identical after the refactor), plus new `deployed_policy_matches_python_reference` (exact on all 5 fixture prefixes) and `artist_cap_scales_with_k` (3/6/15 at k=10/20/50). WASM rebuilt, full `npm run build` green. — 2026-07-30c - **THE HARD PER-ARTIST CAP IS THE FIRST DE-EAGERING LEVER THAT WORKS** (`2026-07-30c-artist-cap-measurement.md`; off-run measurement + serving implementation, NO training runs, NOTHING PROMOTED/REGISTERED/TAGGED, no champion moved). New `seq_common.artist_cap_order()` + `artist_cap` hyperparam, applied AFTER MMR in **both** `eval_from_scores` and `rank_topk`/`predict_ranking` across all 5 declaring families — eval and serving agree BY CONSTRUCTION. Default `None` ⇒ every existing run/model byte-identical; needs **NO sonic vectors**, so unlike MMR it ports to a thin client unchanged. Measured on the canonical split, paired on **bit-identical cached scores** (2000 resamples, rng1337, n=1431): on the champion, `artist_conc` **0.6917 → 0.1230** at cap 3 and H@10 **0.008603 → 0.033862**; on champion+`markov_gate`, `artist_conc` **0.5982 → 0.1134** and H@10 **0.018699 → 0.048063** — which would **top the crown board by ~40% at 1.5× rank 1's recall**. **THE UNEXPECTED RESULT: `music@10` goes UP, CI>0, at EVERY cap on BOTH models** (+0.011…+0.023) — the pre-registered "de-eagers by degrading into filler" failure mode does NOT occur; items promoted from rank 75+ are MORE musically apt to the true continuation than the same-artist tracks they displace, i.e. the cap trades LITERAL for GRADED relevance. Every arm clears the 0.43 music floor (0.474–0.506) and the 0.108 recall floor (0.151–0.198), so none is degenerate. COST: exact recall is the ONLY regressing metric — Δhit-rate CI<0 at every cap; `gate+cap3` is **−52/1431 hits vs the deployed status quo** (the gate's +29 does NOT pay for the cap's −81). MRR, ild, mood_coh and music@10 all end ABOVE status quo. WHY IT WORKS WHERE THE OTHERS FAILED: `eager_beta` (train-time) and MMR (score-space) are both SOFT and lose to a steep relevance gradient — on `rock chabon ahre` 59 of the top 60 candidates are one artist and the next artist is at **rank 75**, while MMR's whole diversity budget there is worth ≈0.078 of normalized score. A cap **cannot be outscored** (the property `seq_extend`'s `artist_cooldown` already relies on). SPOT-CHECK on the 3 real playlists: **`rock chabon ahre` FIXED** — 10 consecutive Cerati → at cap 3 Cerati×3 / Soda Stereo×3 / Babasonicos×2 / Patricio Rey / La Portuaria (1 → 5 artists); the cap also **exactly repairs `markov_gate`'s only regression** (french touch 8→4 artists under the gate, back to 7 at cap 3 and 8 at cap 2); and it is a literal **no-op at every cap on an already-varied list**. Cap is BEST-EFFORT not absolute (backfills with deferred items when a prefix's candidate space is exhausted). **CONFIRMED BY REGISTERED RUNS, SAME DAY — EXACT REPRODUCTION TO EVERY DIGIT** (3 runs, all `queue:true` on `snappler:434418`): C0r `run-20260730-235524-0ec1d` cap 0 → **332 hits**, H `0.018699174464215984` (**bit-identical** to the original gated `507bb`, so `markov_gate` reproduces too); A1r `run-20260730-235524-93d38` cap **3** → **251 hits**, conc `0.11344048450966691`, H `0.048062885636393195`, music `0.5057618871551192`; A2 `run-20260730-234635-a6252` cap 5 → **284 hits**, conc `0.23133783678857056`, H `0.040055574262463825`. So the off-run measurement transfers exactly and the plumbing is verified end-to-end (registry → API validation → remote worker → train-time eval → stored metrics). **A1r would rank 1 on the crown board by ~40% over the incumbent 0.034171, at 1.5× its recall** — but best-models stayed **stale** at `2026-07-28T01:39:56` and auto-promoted nothing (**5th** `spawn_recompute` confirmation); no recompute was forced (it auto-promotes). ⚠ **THE FIRST LAUNCH WAS INVALID AND ONLY THE BIT-IDENTITY GATE CAUGHT IT** — the worker's `seq_blend.py`/`seq_common.py` had ZERO occurrences of `markov_gate` OR `artist_cap` (stale checkout), so both hyperparams were silently ignored and C0 returned the UNGATED champion's numbers exactly (303 hits / conc 0.6917 / H 0.008603). Fixed by rsyncing all predictor `.py` + `registry.toml` (md5 parity verified after). Two damaged rows remain: `run-20260730-234635-505ae` **succeeded with `markov_gate=true` in its hyperparams but ungated metrics** (misleading to anyone mining the record), and `run-20260730-234635-7a297` is **orphaned at `running`** — `/stop` 409s ("is not running": it tracks only LOCAL runs) and `DELETE` refuses ("running on a worker"), so only a direct Postgres `UPDATE` can clear it. **RULE: md5 the predictor files for THE FAMILY BEING LAUNCHED immediately before POSTing** — the preceding campaign verified `seq_dualgru.py`/`seq_nexttrack.py`, which said nothing about `seq_blend.py`. RECOMMENDED next deployment candidate: **`markov_gate=true` + `artist_cap=3`** (or cap 5 conservatively, −19 hits vs status quo, MRR still above today's); remaining gaps are second-split hardening and the `champion.rs` port. — 2026-07-30b - **MMR NOW REACHES THE SERVING PATH — a metric-integrity defect is fixed** (orchestrator change, predictor code only, NO server restart needed since `predict` spawns a fresh Python process per call). BEFORE: `mmr_lambda` entered ONLY via `eval_from_scores`, so it shaped every leaderboard/holisticness number but was **silently dropped at inference** — `predict_ranking` delegates to `rank_topk`, which did its own bare `argsort`. Consequence measured on 3 real user playlists: two promoted models differing ONLY by `markov_gate`+`mmr_lambda=0.7` returned **BYTE-IDENTICAL top-10s on 2 of 3**, i.e. best-models ranked and auto-promoted on gains production could not reproduce. FIX: `mmr_lambda`/`mmr_pool` threaded through `rank_topk` + `predict_ranking` (`seq_common.py`) and wired at the 5 call sites that declare them (`seq_nexttrack`, `seq_blend`, `seq_dualgru`, `seq_embed`, `seq_mood`; `seq_ann`/`seq_stacker`/`seq_baselines` neither declare nor eval with it, untouched). Music vectors are fetched ONLY when λ<1 so MMR-off models stay **byte-identical** (verified: deployed champion returned the SAME ids on all 3 playlists), and an unavailable Qdrant index now logs LOUDLY instead of silently serving a pure-score ranking. EFFECT: λ=0.7 models changed on all 3 playlists — artist_conc 0.467→**0.133** (french touch) and 0.622→**0.467** (PREVIA) on the gated blend; the GRU arm hit **0.022 / 9 distinct artists in 10 slots** on PREVIA. `zig build py-check` 0. ⚠ **A SEPARATE, PRE-EXISTING eval-vs-serving DIVERGENCE REMAINS and is NOT closed by this fix** — see the pitfall entry. Caveat: `rock chabon ahre` still returns artist_conc 1.000 on the blend (its candidate pool is near-uniformly one artist, so MMR has nothing to trade into). New cost: an MMR predict now fetches ~19,402 music vectors from Qdrant per call — baking them into the model dir at promotion time is the obvious follow-up. — 2026-07-30 - **`eager_beta` REFUTED on `seq-dualgru`: the anti-eager TRAIN regularizer is MIS-SPECIFIED, not weak** (`2026-07-30-eager-margin-calibrated-refutation.md`, 8 runs all remote on `snappler:3207287`, ~68m wall, 0 failed; NOTHING PROMOTED / NOTHING REGISTERED / NOTHING TAGGED; no champion moved). The penalty hinges on `cos(pred, x_current)`, but margins were calibrated from the **item→item** distribution `cos(x_t, x_{t+1})` (p75 0.584 / p90 0.830) — a DIFFERENT quantity. A hedged InfoNCE prediction's absolute cosine to the current track never exceeds ~0.6, so B3 (m=0.70/β=35) and B4 (m=0.8304/β=92) reproduced the control's loss trace to **16 significant digits at every one of 14 epochs** — provably inert at β up to the schema cap of 100. At m=0 the knob fires but eagerness and relevance are the same mechanism (β=3 → music@10 0.400, floor breached; β=12 → conc 0.371→0.134 but 20/1431 hits). G1 bit-identity PASSED (control returned exactly 170 hits, conc 0.370588 == 18b4a to 15 dp). **`fusion_layers=1` (`0df6f`, Δconc −0.051 at −0.016 recall) remains the cheapest real de-eagering on record — no arm beat it inside the relevance floor.** Demoting H@10 to secondary was load-bearing: the best-facet arm (A2) had the WORST H (0.003569), so an H-primary design would have read this as a flat null instead of a mapped frontier. Only unexplored band: m=0, β ∈ [0.2, 3]. **By elimination, making MMR reach the serving path is now the strongest remaining de-eagering lever.** — 2026-07-27c - **`markov_gate` — CONFIRMED ON TWO SPLITS: the Markov leg must ABSTAIN where it has no evidence, and it is worth +0.020 recall@10 on the CHAMPION** (`2026-07-27c-nexttrack-markov-gate-confirm.md`, NOTHING PROMOTED / NOTHING REGISTERED / NOTHING TAGGED by the campaign — promotion RECOMMENDATION only, per the design boundary; champion ROW is now OUT-RANKED but not yet replaced). NEW hyperparam **`markov_gate`** on `seq-blend` (default `False`, implemented+verified by the orchestrator before the campaign; `registry.toml` block updated; server restarted at 0 live runs, all 117 prior runs intact): when true the Markov leg ABSTAINS on queries whose last prefix item `u` has ZERO train-level bigram support (`n_u == 0`) and the blend divisor renormalizes `(zg + w_m*zm + zc)/(2.0 + w_m)` from three legs to two. MECHANISM: `markov_scorer` mixes its bigram row with an artist/genre back-off via `trust = n_u/(n_u+8)` (`seq_baselines.py:182`), so at `n_u=0` the leg is **PURE back-off** — carrying no track-level evidence at all — and `_zcand` (`seq_blend.py:169-172`) is **scale-invariant**, re-inflating that evidence-free vector to unit variance so it votes at a full one-third weight on **795/1431 = 55.6%** of canonical test queries. 6 runs, ALL on the hub (never `queue:true`), serialized, ~1h05m wall, 0 failed / 0 interrupted, base definition `blend-gru-markov-content-proj`. **PREFLIGHT BIT-IDENTITY GATE PASSED** — the ungated control C1 returned recall@10 `0.21174004192872117` = **303/1431** exactly (10th reproduction), proving the `markov_gate=false` default left the champion bit-identical. **VERDICT — CONFIRMED, both pre-registered legs met:** canonical `seq-20260715-131139` paired Δrecall@10 **+0.02027 [+0.01048, +0.03075] CI>0** (0.21174 → **0.23201**, 303 → **332** hits) and disjoint 0.64-cold `seq-20260718-211238` **+0.01957 [+0.01048, +0.03005] CI>0** (0.17051 → 0.19008, 244 → 272), with **MRR UP on both** (0.12199→0.13352; 0.10498→0.11801) — no regression anywhere. **A1 `run-20260728-005349-507bb-seq-blend` (0.23201) IS THE NEW BEST RECALL ON RECORD**, ahead of the champion row by more than its own coronation margin, from a nine-line change with NO new parameters and NO extra training cost. **THE BUCKET DECOMPOSITION IS THE WHOLE ARGUMENT AND IT REPRODUCED NATIVELY** (through the real `build_train_transitions`, on all THREE control/arm pairs): `n_u=0` 0.1660→0.2025 (795 q, +0.0365), `n_u 1-10` 0.3116→**0.3116**, `n_u>10` 0.2326→**0.2326** — and the warm buckets' **top-k lists are BYTE-FOR-BYTE IDENTICAL** (n_disc 0), so the gate is a surgically pure `n_u=0` intervention with literally zero warm-side cost. Same shape on the 2nd split (`n_u=0` 892 q = 62.3%, +0.0314) and under MMR λ0.7 (+0.0352). **THIS RECONCILES B3 RATHER THAN CONTRADICTING IT:** the 2026-07-18 ORACLE test-fit search over GLOBAL STATIC SCALAR weights returned +0.004 straddling 0 *because the buckets cancel exactly* (dropping M globally is a −0.0070 TIE) — no constant can express a gate whose sign flips with `n_u`. → **the learned-combiner pitfall is NARROWED to forbid LEARNED SCORE-LEVEL FUSION, not PER-QUERY EVIDENCE GATING** (no parameters fit, conditioned on a TRAIN-SIDE DATA STATISTIC, not on a score). **UNPREDICTED SECOND RESULT — the gate is a CROWN lever too, and the first one that de-eagers the champion WITHOUT paying in mood coherence:** ΔH CI>0 in all three pairs (+0.010096 canonical / +0.005022 2nd split / +0.015691 under λ0.7) because all four facets move the right way at once (artist_conc 0.69165→0.59821, artist_adj 0.60971→0.57477, mood_coh 0.39822→**0.46226 UP**, music@10 0.45379→0.49319) — the exact double-payment the λ frontier called structural, avoided because this is not a re-rank fighting the model's own scores but the removal of a vote that had no business being cast. **HEADLINE FOR THE CROWN BOARD — A3 (gate + MMR λ0.7/pool200) IS THE FIRST TIER-1 FREE CROWN ARM AT CHAMPION-LEVEL RECALL:** H **0.030980** at recall **0.21314 (305 hits)**, i.e. recall TIE vs the champion (+0.00140 [−0.01188, +0.01468] STRADDLES) for ΔH **+0.022377 [+0.019941, +0.024923] CI>0** = 3.6× the champion's H — crown-board **rank 2**, behind only the λ0.7-pool200 single GRU (0.034171) which scores it at recall 0.11461, so A3 delivers 96% of the crown leader's H at **1.86× its recall**. Standing alone λ0.7 on the champion is PRICED (C3−C1 −0.01817 CI<0 for +0.006686 H); **the gate pays that cost back exactly and flips MMR-on-the-blend from TIER-2 PRICED to TIER-1 FREE.** GATE AND MMR ARE **ADDITIVE, NOT SUBSTITUTIVE** (+0.02027 alone vs +0.01957 under λ0.7 — indistinguishable; they act on disjoint pipeline stages). MMR-validity gate PASSED (C3 ild 0.52010 / A3 0.56770 vs control 0.40637) and C3 reproduces the 2026-07-26 λ0.7-on-champion recall **0.19357 exactly**. **HONEST CAVEAT CARRIED FROM THE MEMO:** the 10 gating variants were screened ON THE TEST SET so the banked +0.02027 was selection-optimistic — what redeems it is that the effect reproduces on a second, disjoint, unscreened 0.64-cold split to within 0.0007, reproduces under an unscreened third condition (λ0.7), sat on a PLATEAU in the screen (4 related variants all +0.019–0.020), and the zero warm-bucket movement is a STRUCTURAL prediction rather than a fitted outcome; the one residual selection concern is the `n_u=0` BOUNDARY, whose boundary-free sequel (soft gate: scale the Markov leg by `trust` BEFORE `_zcand`) is the top follow-up. **NEW GENERAL DEFECT LOGGED:** `_zcand`'s scale-invariance ERASES PER-LEG CONFIDENCE — the Markov `n_u=0` back-off is one instance; any leg with a query-dependent "I have no evidence here" is mis-weighted the same way, and the content/GRU legs are UNAUDITED. **INFRA (blocking, resolved by routing, NOT remediated):** the remote worker `snappler` pid 573145 is LIVE and its `~/lensing-worker/predictors/seq_blend.py` contains **ZERO occurrences of `markov_gate`** — a stale checkout; any `queue:true` `seq-blend` run would silently execute the UNGATED path and return a null. It could not happen here because remote dispatch is OPT-IN via `"queue": true` and every run was posted without it (`claimed_by: null` on all six); freshening the checkout is an ORCHESTRATOR action and was deliberately not performed. The server's deterministic top-12-by-H recompute auto-promoted four of this batch into best-models (A3 rank 2, A1 rank 7, C3 rank 9, C1 rank 12) → **best-model-selector needed**. Follow-ups: register+promote A1 (and A3 for the crown) on user approval; the SOFT GATE; audit the other two legs for the same confidence defect; 3-seed confirm of A1 (`seq-blend` exposes `seed` and the projection IS seed-varied); freshen the `snappler` checkout; re-screen the whole λ curve OFFLINE from the GATED champion. Prior 2026-07-27 - **ROLLOUT-HOLD METRIC + FIVE SWEEPS — REFUTED: rollout fade does NOT separate this family** (`2026-07-27-rollout-hold-metric.md`, NOTHING PROMOTED / NOTHING REGISTERED / NO METRIC WIRED, champion + leaderboard + pins UNCHANGED). Built the measurement every prior null was accused of lacking — an **AUTOREGRESSIVE** one. Every metric on record (`recall@10`, the four crown facets, `music@10`, even multi-step `seq_continuation_eval`) scores ONE list from a REAL prefix; a 20-step lab journey feeds the model its OWN output, so trajectory decay is structurally invisible. NEW: **`hold@N`** (`tools/rollout_hold.py`) = `1 − clamp(mean_a fade_a + stasis, 0, 1)` where `fade_a = |mean(2nd half) − mean(1st half)| / band_a` over energy/valence/acousticness/tempo, bands calibrated on **2,135 real TRAIN sessions** (energy fade p10 .0141 / p50 .0769 / p90 .1963 — real sessions are NOT flat, so rewarding zero fade would rank a rigid generator above its own listener), and `stasis = max(0, jump_p10 − mean step jump) / jump_p10` is the ANTI-GAMING term that puts `holisticness`'s external ±0.015 recall floor INSIDE the metric (synthetic flatlined rollout: 0.048; **first implementation normalised by band WIDTH and scored it 0.740, ABOVE a real generator's 0.477** — a guard beatable by its target launders degeneracy as a good score). NEW: `predictors/seq_continuation_eval.py --rollout` offline eval over sampled TEST sessions, which **imports `seq_extend.rollout()`** (factored out this session, verified BYTE-IDENTICAL) rather than reimplementing the retrieval policy. ORIGIN was a USER PREFERENCE, not a metric: on playlist `cenamos` the user strongly preferred `seq-dualgru` over `seq-nexttrack` where `holisticness@10` differed by 0.003. VERDICT after **5 sweeps / 120 paired rollouts / 240 journeys**, paired bootstrap over seeds + leave-one-out: **at MATCHED SPLIT the architectures TIE TWICE** (18b4a vs C0 +0.033 [−0.030,+0.093] 0/39 LOO; dc7d2 vs C0 +0.051 [−0.012,+0.118] 0/39), **on the user's OWN 11 playlists the preferred pair TIES** (+0.004 [−0.088,+0.104] 0/11), and the ONLY surviving effect is **CHECKPOINT-level** (`a1d9d` vs `C0`, SAME architecture, −0.072 [−0.141,−0.002]) which does not transfer to playlists. **TWO POSITIVES WERE RETRACTED BY CONTROLS:** a mood-extreme 8-seed pilot gave +0.314 [+0.107,+0.529] 8/8 LOO, a RANDOM 40-seed pre-registered replication gave +0.122 [+0.044,+0.201] 39/39 LOO and MET its pre-registered rule — then the same-split control showed it decomposes into a non-significant architecture component (+0.05) and a significant CHECKPOINT one (+0.072). **NEW PITFALL, the important one: the lab's two default preselects are trained on DIFFERENT SPLITS** (`dc7d2` on `seq-20260715-131139`, `a1d9d` on `seq-20260715-030509`; identical vocab + item latents, different train/test assignment) — comparing them attributes checkpoint+split differences to architecture; the campaign's own C0/D-LL-f0 pair (same batch, same worker, same seed, and a recall TIE so neither is the better predictor) existed the whole time. Also NEW: **mood-extreme seed selection inflates effects ~2.5×**; a `None`-scoring rollout must drop the PAIR and report the smaller n; and **per-seed variance DOMINATES model identity** — individual gaps on ONE pair span +0.39 to −0.27 while the model-level mean difference on the user's playlists is +0.004, so **budget ≥40 seeds for any rollout comparison** and prefer serving an ENSEMBLE (`POST /api/best-models/predict`) over hunting a single better generator. Per-axis at n=39: only ENERGY fade resolves (−0.039 [−0.069,−0.009]); valence / acousticness / tempo TIE (they had looked significant at n=8 — noise, and the jackknife said so: 4/8, 4/8); and a **pre-registered NEGATIVE CONTROL FAILED** — energy LEVEL |mean−seed| went the OTHER way (+0.018 [+0.005,+0.032], nexttrack closer to the seed), re-scoping the claim from 'holds the seed better' to 'is more stationary' and showing `hold@N`'s deliberate exclusion of the level term CHANGES the ranking rather than simplifying neutrally. RECOMMENDATION: keep `hold@N` as a per-journey LAB readout, do NOT wire it as a leaderboard column — its ground truth is listener preference and that stands at **n=1 blind vote** (the lab now records them: `localStorage lab.votes.v1`, shuffled sides, wins over appearances); promoting an unvalidated composite to primary is how the crown ended up needing an external recall floor. **CORRECTION TO THIS FILE: the §2026-07-18c line naming 'fit the projection ON a whitened/std-noaco/metric space (compose the two fixes)' as the best open direction is STALE — that report's readout A ALREADY RE-FITS the projection per space** (its §Design says so), V3 `std-noaco` scored +0.0049 straddling 0, and it concludes "whitening and the learned projection are the SAME fix". Do not re-run it. **THE UNTESTED AXIS IS THE TRAINING REGIME, NOT THE TOPOLOGY:** all five architecture nulls held the objective/batching/early-stopping byte-for-byte the family's (one-step teacher forcing) while the product rolls 20 steps on its own output — exposure bias is the one thing those nulls could not detect, and scheduled sampling on the single GRU at `fusion_layers=0` is now MEASURABLE via `--rollout`. Live-crown standing on the dual-tower question, for the record: best legitimate dualgru arm **0.02378** vs single-GRU C0 **0.02454**, a 0.0008 gap against the stated ~0.0018 crown CI half-width = **a TIE, not a loss** — `seq-dualgru` was refuted on `recall@10` (now "no longer load-bearing" per domain.toml), ties on the live primary, and shows nothing on the rollout axis. Follow-ups: blind votes (binding constraint), scheduled sampling, **change the lab's slot-B preselect off `a1d9d`** (worst holder of the four measured, 0.5266 vs C0 0.5983), attention encoder for the LONG-prefix regime only. Prior 2026-07-26 (latest) - **MMR λ FRONTIER — CONFIRMED: the FIRST GENUINE CROWN LEVER ON RECORD, and the `artist_conc` mechanism resolved** (`2026-07-26-nexttrack-mmr-lambda-frontier.md`, **6 DEFINITIONS REGISTERED, NOTHING PROMOTED**, champion + recall leaderboard + pins UNCHANGED). Native λ sweep of the EVAL-TIME MMR re-rank (`mmr_lambda`, `mmr_pool`) on both anchors, 10 runs on `seq-20260715-131139`, ALL on remote worker `snappler:573145` via `queue:true` serialized (37.7 min wall clock, 0 failed / 0 interrupted). **ALL THREE PRE-REGISTERED GATES PASSED** — G1 bit-identity (A0 → **303/1431** exactly + mrr to 3.1e-11; B0 → **176/1431** exactly, 9th reproduction), G2 metric-index (`artist_conc_at_k` on all 10 arms — only after an infra fix, see §Pitfalls), G3 MMR-active (every λ<1 arm's `ild_at_k` differs from its own control by ≫1e-6, so no arm silently no-opped). Every arm's H **recomputed from `predictions.json`** under the live 4-factor definition by ONE code path (validated on both controls: |Δ| 7.4e-8 / 2.1e-7 vs stored). VERDICT — **EVERY ONE of the 8 λ<1 arms lifts holisticness@10 with paired ΔH CI entirely >0 (8/8)**, because three of four crown factors improve together (ild ↑, eagerness ↓, **music@10 ↑** 0.4558→0.4747 on the GRU) and only recall pays — categorically unlike the recall-destroying `cummean/cummean` + `seq-mood` pathologies. **THREE TIER-1 'FREE' ARMS ON THE SINGLE GRU BEAT THE ALL-TIME BAR `H_bar` 0.024872 AT A RECALL THAT IS A STATISTICAL TIE:** best **B4 λ0.7/pool50 H 0.030355** (+22.1%, ΔH +0.005811 [+0.004819,+0.006846], Δrecall −0.002908 [−0.008386,+0.002795] STRADDLES, recall 0.12020 = 172/1431), then B2 λ0.8 H 0.029458, then B1 λ0.9 H 0.026761; B4 beats B2 on H by +0.000889 **CI>0** so its rank as best TIER-1 is RESOLVED. TIER-2 PRICED (registered, promotion NOT recommended): B3 λ0.7/pool200 **H 0.034171 = the highest H measured anywhere** but Δrecall CI<0, and A2 champion λ0.7 (H 0.015289). TIER-3 REJECTED (frontier mapping only, Δrecall < −0.03): A3 λ0.5, A4 λ0.3. **R2 MECHANISM VERDICT — H1 (RESPONSIVE) CONFIRMED UNANIMOUSLY, 8/8 arms:** `Δartist_conc ≤ Δartist_adj` everywhere with ratio |Δconc|/|Δadj| = **1.59×–3.06×**, so H2 (sticky, which required <0.5×) is refuted by a factor of 3–6 in the WRONG direction → **same-artist tracks ARE sonic near-neighbours, and purely sonic MMR IS a de-eagering lever with no explicit notion of artist**. Consequence: the pre-registered H2 sequel (porting `seq-mood`'s explicit `artist_penalty`) is **NOT** the indicated next move; pushing λ and testing λ on the blend's Markov leg is. The ratio SHRINKS monotonically as λ falls (3.06→1.59) — easy artist de-concentration is bought first, then saturates. **THE BINDING FACET SWITCHES, and that locates the end of the lever:** `artist_conc` binds at λ1.0 on both models but `artist_adj` takes over by λ0.5 on the champion and already by λ0.9 on the GRU; since `artist_adj` falls ~2.5× SLOWER, once it binds the `(1−eager)` factor is the stiff one and each further recall point buys less crown → pushing λ below ~0.7 on the GRU is poor value. **THE SHARED-WEIGHTS INSTRUMENT IS 3–33× SHARPER THAN THE FAMILY NOISE FLOOR** — measured hw(ΔH) **0.000382–0.002476** and hw(Δrecall) **0.004193–0.018169** vs the conservative 0.0018 / 0.0124, with `n_disc` as low as **9** discordant sessions; A1 was pre-registered as 'likely UNRESOLVABLE' and in fact resolved at **4.8 measured half-widths** → budget MDE for eval-time levers from the shared-weights pairing, NOT the architecture-family figure. **The 2026-07-22 offline sweep transferred EXACTLY** (projected recall 0.2103/0.1936/0.1642/0.1097 vs native 0.21034/0.19357/0.16422/0.10971, 4-for-4 to ~4 dp) → the offline MMR driver is validated as a SCREENING instrument, which STRENGTHENS the eval-only pitfall. **THE 2026-07-22 STANDING FLAG IS CLOSED POSITIVE, NOT REFUTED:** λ≈0.9 IS near-free on both models (A1 301/303 and B1 173/176, both recall ties, both ΔH CI>0) — but the VALUABLE operating points are λ0.7–0.8 **on the single GRU**, not λ0.9 on the champion. **CHAMPION-vs-GRU CONTRAST RESOLVED IN FAVOUR OF FACTOR HEALTH, not de-eagering headroom:** at λ0.7 the GRU reaches 0.034171 vs the champion's 0.015289 (2.24× gap, widening at every λ) because as the champion diversifies its `mood_coh` COLLAPSES (0.3982→0.2378 at λ0.3, −40%) while the GRU's barely moves (−4%) — the champion's recall comes from artist-adjacency (the Markov bigram leg), so walking away from the seed artist also walks it out of the prefix's mood neighbourhood; it pays twice. No λ closes the champion's crown gap. **`mmr_pool` CLOSES AS INTERMEDIATE, not a second axis (R3):** pool=50 does recover recall at fixed λ (Δrecall(B4−B3) +0.005584 CI>0) but loses H (ΔH −0.003838 CI<0), so it re-parameterizes the same frontier rather than winning — yet B4 beats the nearest pure-λ point B2 on H CI>0 at a recall tie, which is why B4 not B2 is the best TIER-1 arm. Treat pool as a fine-tuning knob at a chosen λ; do NOT open it as a full axis. **RECALL@10 IS UNTOUCHED** — no arm improves it; the recall champion stays `blend-gru-markov-content-proj` 0.21174 (303/1431). Also: **`seq-blend` reproduces BIT-IDENTICALLY on `snappler`** (A0, the first `seq-blend` run ever on the worker) → the withdrawn cross-host offset is now confirmed absent for a SECOND predictor family and the documented hub-fallback was not needed; and `tools/rescore_holisticness.py`'s RECONSTRUCTED `artist_conc` is independently VALIDATED (native A0/B0 emissions match its reconstructions to the last digit). best-models was STALE after the remote batch again (3rd confirmation of the `spawn_recompute` trigger gap) and the mandatory manual recompute **FLOODED all 12 auto slots with 2026-07-26 runs, 10 of them this campaign's**, including BOTH TIER-3 rejects (A4 at rank 2) and both G2-defective discarded controls → best-model-selector needed. Prior 2026-07-26 - PRE-ENCODER SCAN — **REFUTED, and HARMFUL in the single tower** (`2026-07-26-nexttrack-pre-encoder-scan.md`, NOTHING PROMOTED / NOTHING REGISTERED, champion + leaderboard + pins UNCHANGED). Tested the axis the 2026-07-25 view-split null could not: a **LEARNED nonlinear per-step re-embedding in front of the recurrence** (`[Linear(d_in,width), ReLU(), Dropout] × pre_layers` applied timestep-wise, GRU `input_size = pre_hidden`). NEW predictor params this session: `pre_hidden`/`pre_layers` on **`seq-nexttrack`** (conditionally constructed — no `nn.Module`, no RNG draw at width 0), mirroring `seq-dualgru`'s `pre_hidden_a/b`. 10 runs on `seq-20260715-131139`, ALL on remote worker `snappler:4101029` via `queue:true` serialized (54 min wall clock, 0 failed / 0 interrupted); all 11 param counts re-measured and matching the design exactly. **TWO BIT-IDENTICAL REPRODUCTION GATES BOTH PASSED** — C0 h256 returned recall@10 `0.12299091544374564` = **176/1431**, identical to the last digit (so the `pre_hidden` edit did NOT move the family's on-record baseline; 7th reproduction), and C3 bare-dual returned `0.10272536687631029` = **147/1431**, exactly on-record → all dual verdicts valid. VERDICT — **Rule 7 REFUTED: not ONE of the 6 pre-MLP arms beats C0 by paired-Δ CI>0** (5 lose CI<0, 1 ties). **The decisive single-tower ablation S1 (`MLP256→GRU256`, the one arm unambiguously attributable to the pre-encoder) does not merely tie — it LOSES: 0.10901 vs 0.12299, Δ −0.0140 [−0.0266,−0.0014] CI<0, 156 hits vs 176.** Because a *linear* pre-encoder at `pre_hidden ≥ latent_dim` is provably expressivity-neutral (`W_i·(Vx) = (W_iV)x`, no rank constraint), **the ReLU is the ONLY expressivity change → the rectifier itself is destroying prefix information the GRU's own gated LINEAR input transform preserved**; likely mechanism = PCA-192 latents are zero-centred and near-symmetric, so half-wave rectification zeroes ~half of every input vector's coordinates and `pre_hidden=256` (1.33× input dim) is not over-complete enough to re-encode the discarded negative half-space. Best pre arm P* = T1-f0 (`pre_hidden_b=256`, `fusion_layers=0`) only TIES C0 (0.12369 = 177/1431, Δ +0.0007 straddles) and also ties the on-record h425 and last campaign's bare D-LL-f0 → **the dual side does not move either**. SUB-FINDINGS: (1) **Rule 5 DOSE CURVE = FLAT** — all three adjacent pairs C3(0)→T2(128)→T1(256)→T3(384) straddle 0 AND the endpoints do not separate, with every dosed arm nominally BELOW the bare dual (never above) → no interior peak to chase, and the pre-registered FLAT clause would independently have BLOCKED registration even had a win rule fired; (2) **Rule 4 ASYMMETRY REFUTED as irrelevant** — T1 (A0/B256) vs T4 (A256/B256) straddles on recall (−0.0007), so keeping tower A bare is NOT load-bearing; (3) **Rule 2 INVERTED, and the inversion IS the attribution** — T1-f0 BEATS S1 CI>0 (+0.0147), i.e. the second tower is not redundant over a single pre-encoded tower, but since T1-f0 only TIES plain h256 the bare tower A is **RESCUING** what the rectified tower B destroyed (restoring a direct un-rectified path to the readout), not complementing it — a repair, not co-adaptation; (4) **the ONE solid positive: `fusion_layers=1` regression PERSISTS with a pre-encoder and gets BIGGER** — T1-f0 beats T1 by **+0.0266 [+0.0140,+0.0391] CI>0** vs the previously measured +0.016/+0.020 (3rd confirmation; the more input-side nonlinearity you add, the more the output-side warp costs) → `fusion_layers=0` is the family default, full stop; (5) **Rule 8 did NOT fire for ANY arm — the FIRST campaign to yield ZERO holisticness candidates**, because no arm reaches ΔH CI>0 vs C0 (closest S1 +0.00005, straddles; 4 arms CI<0). Counterfactual: under the OLD 3-factor H, FIVE of these arms would have shown H 'wins' over C0 (0.1444–0.1481 vs 0.14172), every one bought by de-eagering at a recall loss — **the 4-factor `music@10` grounding factor cancels them because music@10 falls in lockstep with recall (0.4558→0.4343 across the batch). The metric cut earned its keep.** It remains INCOMPLETE though: the recomputed all-time 4-factor board (32 facet-carrying runs, re-derived by the runner; every designer-precomputed value reproduced) is STILL topped by the DEGENERATE `cummean/cummean` arm `88a4c` at **H 0.050436 on recall@10 0.06639 = 95/1431** → **the ±0.015 recall floor stays MANDATORY**. (Grounding DID demote `seq-mood` from rank 1 to rank 24, H 0.23319→0.043881.) NEW CAVEAT ON PARAM-MATCHED CONTROLS: the h297 capacity control **C1 itself LOST to C0 CI<0** (0.11321 = 162/1431, Δ −0.0098) — the batch's width points are h256→176, h297→162, h425→173, h454→179, **h512→186 (2026-08-02, Δ +0.00699 STRADDLES — 7th width tie)**, h537→174 hits, **non-monotone spanning 0.0119 recall with no trend**, so 'single-GRU width is flat on PCA-192' survives as a TREND claim (h454 ties h256) but per-width single-split jitter is ~±0.01 ≈ ⅔ of the noise band → **a width-matched control that is itself a single unreplicated run is NOT a reliable capacity anchor at this effect size** (Rule 1's 'beats C1' leg was consequently nearly free; S1's verdict rests on the reproduction-verified C0 leg). This is the **FIFTH consecutive architecture null** and together with the view-split null it CLOSES the input-side asymmetry direction: adding representation machinery AROUND a single GRU over a good latent space does not help on this corpus — the wins on record came from improving the LATENT SPACE ITSELF (InfoNCE content projection, PCA-192) and from the fixed z-blend. Phase B: B1/B2/B3/B5 did NOT fire (no Rule 1/3/8 win; Rule 5 FLAT); **B4 FIRED** via Rule 7's tie clause (S1 ties C1 within ±0.015; T1-f0 ties C0/h425/D-LL-f0) → SAE read at topk=32 on **T1-f0** `run-20260726-161801-a6423` (taps `pre_b`/`tower_a`/`tower_b`) and **S1** `run-20260726-161801-ad53f` (`model-sae` now taps `pre` ahead of `recurrent`), asking whether the rectified representation is measurably LOSSIER than the bare tower — delegated to information-capture-analyst, analysis-only. B5 NOT fired but the **dropout confound remains un-disentangled and now cuts the other way**: the pre-MLP ends in `Dropout`, so pre arms get GRU-INPUT dropout the bare arms lack (`seq_nexttrack` drops the RNN OUTPUT only) → S1's loss could be partly over-regularization; a `pre_dropout` param (default = `dropout`) is RECOMMENDED over the matched `dropout=0.0` pair, which confounds input- and output-side dropout — an orchestrator change, deliberately NOT improvised. best-models was again STALE after the remote batch (the known `spawn_recompute` trigger gap) — a manual `POST /api/best-models/recompute` absorbed it (5/12 → 12/12 entries) and **FLOODED the group with 7 of this campaign's 10 refuted runs, 3 of them below 0.10 recall** → best-model-selector needed. Prior 2026-07-25 - DUAL-TOWER FUSION SCAN — **REFUTED** (`2026-07-25-nexttrack-dual-tower-fusion-scan.md`, NOTHING PROMOTED / NOTHING REGISTERED, champion + leaderboard + pins UNCHANGED). First real-data outing of the NEW `seq-dualgru` predictor (two INDEPENDENT recurrent towers over configurable causal views {latent, delta, cummean}, per-step hidden states concatenated + fused by one MLP head; reuses `seq_nexttrack.make_batches`/`step_loss` verbatim; strictly causal / no bidirectional mode by design). 10 runs on the canonical PCA-192 split `seq-20260715-131139`, ALL on the SAME remote worker `snappler:4101029` via `queue:true` serialized (~59 min wall clock, 0 failed / 0 interrupted). Design = the complete upper triangle of the {latent,delta,cummean}² view matrix at `fusion_layers=1` (6 cells) + a fusion-depth axis (2 cells at `fusion_layers=0`) + TWO single-GRU controls: C0 h256 (baseline) and **C1 h425 = a 0.098% PARAM-MATCHED capacity control** (871,017 vs the dual arms' 871,872 params, re-measured by instantiating the real classes at latent_dim=192). VERDICT — **Rule 3 REFUTED: not ONE of the 8 dual arms beats C0 by paired-Δ CI>0** (6 arms CI<0 outright; the 2 `fusion_layers=0` arms only TIE: best dual arm D-LL-f0 0.11880 vs C0 0.12299, Δ −0.0042 [−0.0126,+0.0042] straddles 0) **AND C1 h425 also TIES C0** (0.12089, Δ −0.0021 [−0.0112,+0.0063] straddles 0) → capacity is NOT the missed lever either and **single-GRU WIDTH IS NOW CLOSED ON PCA-192** (previously only closed on AE-64 by the 2026-07-13 topology scan). This is the **FOURTH independent learned-combiner null** on this corpus (XGB stacker 0.08–0.09 below its own base leg / RRF −0.029 / ORACLE test-fit blend weights +0.004 straddling 0 / now representation-level JOINT end-to-end fusion) and it closes the standing "of course a post-hoc combiner fails, train the fusion end-to-end" rebuttal → materially CONSOLIDATES "the fixed equal-thirds z-blend is the combiner of record". THREE sub-findings survive: (1) **the MLP fusion head is ACTIVELY HARMFUL, not merely useless** — both f0 arms beat their f1 twins by paired-Δ CI>0 (+0.01607 latent/latent, +0.02027 latent/cummean) at 10% FEWER params → **default `fusion_layers=0`**; (2) **the raw `latent` view SUBSUMES `delta` and `cummean` for a recurrent tower** — Rule-4 view attribution: `latent/cummean` (the design's "best hope", the pooled view `seq-ann` was built from) and `latent/delta` both TIE their latent/latent parent and beat only the weaker parent (raw tower carries the cell, second view is dead weight), BUT **`delta`+`cummean` IS genuinely complementary** (D-DC beats BOTH parents CI>0, +0.0287 vs D-DD / +0.0203 vs D-CC) just at a hopeless absolute level (0.0867) → the co-adaptation mechanism is REAL, the raw latent view simply already integrates movement + running mean internally; (3) **holisticness@10 and recall@10 are ANTI-CORRELATED across the batch** — 5 arms lift H by paired-ΔH CI>0 topping out at D-CC **H 0.19715 (+0.0554)** which would rank #2 on the whole crown board, but at recall@10 0.06639 = **95 of 1,431 vs the baseline's 176**, and **EVERY H-winning arm FAILS Rule 6's ±0.015 recall floor → Rule 6 did NOT fire, crown UNTOUCHED**; the floor caught exactly the `mood-session` degenerate mode (H 0.2332 at recall 0.0238) it was written for → any future holisticness objective on this corpus MUST be recall-constrained. Phase B: B1/B2/B3 did NOT fire (no Rule 1, no Rule 6, and C1 does not beat C0 so the width curve is flat — do not spend a run on h512); **B4 FIRED** (mixed-view D-LC-f0 TIES C1 within the band) → SAE tower-specialization read on `run-20260725-143502-3e5d4-seq-dualgru` (topk=32) delegated to information-capture-analyst, analysis-only. **PRIMARY-METRIC CONTRADICTION RECONCILED** (explicit campaign deliverable — see §Target & task and §Best-models group): the SERVER-EFFECTIVE primary metric is **holisticness@10** (`GET /api/best-models` returns `"primary_metric":"holisticness@10"` and ranks `seq-mood` H 0.2332 at rank 1 ABOVE a 0.212-recall blend; `domain.toml` `[metrics].primary` = `holisticness@10` since 2026-07-23) — this file's earlier "recall@10 stays primary" statements were TRUE WHEN WRITTEN and are now SUPERSEDED; the experimental record's comparability currency remains recall@10 (the whole leaderboard is denominated in it). ALSO WITHDRAWN: the "cross-host reduction-order offset −0.0028" — C0 on a FRESHENED worker checkout returns **0.12299, bit-identical to the hub**, so the prior remote 0.1201957 (`run-20260723-002044-af11a`/`-17cf8`) was a **CODE difference (stale `~/lensing-worker/predictors/`), NOT a host effect**; those two runs are not code-comparable to current batches. ALSO FLAGGED: the best-models group is **STALE**, not flooded (`updated_at` 2026-07-23T01:54, 2/12 entries filled, despite 10 new holisticness-carrying runs) → best-model-selector needed. Prior 2026-07-23 - MODEL-SAE INTERPRETABILITY NOTE (analysis-only, NOTHING PROMOTED, champion/leaderboard UNCHANGED, recall@10 stays primary): the per-model SAE over the GRU's recurrent hidden state does NOT sparsify under L1 — an L1 sweep 0.0015→0.05→0.5 (330×) only moved l0 82%→35% at var_explained ~0.99, so `n_interpretable_concepts` saturates near the full dictionary (treat it as saturated; trust ranked `atoms_by_concept` + `next_item_decodability`). Added a TOP-K SAE mode (`SAE(topk=)` in `seq_model_sae.py`, `--topk` on seq_nexttrack/seq_ann `model-sae`, AND wired into the server API `ModelSaeRequest.topk` — server rebuilt + gated-restarted): k=32 gives l0=32 exactly with selective atoms (freq ~2% vs ~50% under L1) at reconstruction cost var_explained 0.90. Also detail cap `SHOWN_N` 12→40 and `CONCEPT_SEP` 0.5→0.35 (that's why runs "detail" 40 not 8). See `docs/sae-interpretability-note.{md,pdf}`. Prior 2026-07-22 - SESSION-HOLISTICNESS METRICS + ANTI-EAGER LEVERS CAMPAIGN (`2026-07-22-session-holisticness-and-antieager-levers.md`, MEASUREMENT + exploratory, NOTHING PROMOTED, champion/leaderboard UNCHANGED, recall@10 stays primary). New DISPLAY-ONLY holisticness columns wired into `seq_common.py::eval_from_scores` + run.rs/domain.rs/domain.toml/UI like music@10: artist_adj@10 (top-10 fraction sharing the SEED artist, LOWER=less eager), mood_coh@10 (top-10 mean cosine to prefix mood centroid, higher=better), ild@10 (intra-list sonic diversity, higher=less duplicative); album_adj@10 omitted on `seq-20260715-131139` (no album field, EXPECTED); suffix_recall@10/cont_prec@10 via the on-demand `seq_continuation_eval.py` harness only. Sanity gate PASSED (music/mood/ild populate → content-metric Qdrant loaded). FINGERPRINT (all on the canonical PCA-192 split, seed 1337, n_test 1431): eagerness TRACKS recall INSIDE the blend family — champion `blend-gru-markov-content-proj` (0.2117) is the MOST artist-eager (artist_adj 0.610), R+M 0.579 — BUT the eagerness is the MARKOV bigram leg, not the RNN: standalone markov artist_adj 0.543 vs single GRU/LSTM ≈0.345 (the RNNs are the LEAST-eager, HIGHEST mood_coh 0.479/0.458 models). GRU≈LSTM on PCA-192 (recall 0.123 vs 0.119 TIE within ±0.015; artist_adj 0.347 vs 0.344) → the GRU-beats-LSTM-BECAUSE-more-eager hypothesis is NOT supported. artist_adj/ild are NOT trivially gamed (popularity maxes ild 0.898 / min artist_adj 0.003 but ≈0 recall, NEGATIVE mood_coh −0.041). LEVERS: MMR eval-re-rank λ (mmr_lambda) is the STRONG knob — offline sweep on the champion checkpoint (validated: λ=off reproduces 0.2117): λ0.9 essentially FREE (recall 0.2103 TIE, ild 0.406→0.436, eagerness 0.610→0.603, music@10 0.454→0.462), λ0.7 ~2 recall pts (0.1936) for ild→0.520 + de-eager→0.571, λ0.3 crushes eagerness→0.293/ild→0.815 at recall 0.110; music@10 RISES as you diversify. Anti-eager TRAIN regularizer β (eager_beta) is a WEAK knob (β0→0.2 moves GRU artist_adj only 0.347→0.340 at flat recall). Multi-step continuation: GRU>markov (suffix_recall@10 0.096 vs 0.087, cont_prec 0.027 vs 0.024). FLAGGED for a future USER promotion decision (promoted nothing): MMR λ≈0.9 as a near-free holisticness upgrade (needs a native run + 3-seed confirm first); β not recommended. Prior 2026-07-18 (night) - REPRESENTATION-EXPLORATION VERDICT IN (`2026-07-18c-nexttrack-representation-exploration.md`, VALIDATION-ONLY, NOTHING PROMOTED): tested the 8 alt item-latent spaces + baseline (18 runs, SERIALIZED, 0 failures, all n_test 1431) under BOTH the champion projection `blend-gru-markov-content-proj` (readout A = the space's ceiling under the winning recipe) and the frozen `gru-infonce-h256` single (readout C = raw representation), paired-Δ on exact hit@10 (2000 boot, rng 1337) vs the same-readout baseline (baseline re-run in-batch, reproduced 0.21174 / 0.12299). TWO VERDICTS: (1) EXACT — on readout A NO variant beats the PCA-192 champion with paired-Δ CI>0 (best nominal V3 std-noaco +0.0049 / V5 balanced +0.0042, both straddle 0; V4 textcat −0.0168 and V6 sonic-64 −0.0182 both CI<0 = REGRESS), so the item representation is NOT hindering the DEPLOYED champion — the InfoNCE projection already re-weights away the loud acoustic/numeric directions; on readout C SIX of 8 variants BEAT the frozen GRU by paired-Δ CI>0 (+0.025…+0.036; biggest V5 balanced +0.0356 [+0.0189,+0.0517] and V3 std-noaco +0.0342 [+0.0154,+0.0545]), so at the RAW level the full-matrix PCA-192 geometry IS a handicap and whitening / acoustic-drop is the fix — BUT the gain is fully ABSORBED by the projection (lifts C, not A → the pre-registered 'projection already compensates for the loud directions', NOT the strong both-readouts result). (2) GRADED — whitening lifts frozen-GRU genre@10 to ~0.47 (V7 std-txtcatup 0.4717, V3 std-noaco 0.4710 vs base 0.4500), and V10 text-only is the readout-A graded-preferred space (artist@10 0.3466 / genre@10 0.4144, both readout-A maxima, at −0.0119 exact). NO promotion candidate (nothing beats the champion CI>0 on readout A) — CHAMPION ROW UNCHANGED; best open direction = fit the projection ON a whitened/std-noaco/metric space (compose the two fixes) **[SUPERSEDED 2026-07-27: readout A already re-fits the projection per space; V3 std-noaco +0.0049 straddles 0 and that report concludes whitening and the projection are the SAME fix — this direction is CLOSED]**. See §'Representation / dataset lineage'. Prior 2026-07-18 (evening) - REPRESENTATION-EXPLORATION DATASETS BUILT (item-vector block composition, DATASETS ONLY - nothing trained/promoted). 8 dim-192 sequence item-space variants on the CHAMPION split (17/17 parity checks PASS on every one: sessions/offsets/train/test/items.json byte-identical to seq-20260715-131139, only item_latents.f32 differs, manifest 7154/19402/5723/1431 @ cut 2024-08-24T14:28:02, cold 0.5507), testing whether the equal-weighted 4-block PCA item vector hinders next-track prediction. EVR preflight confirmed the hypothesis DECISIVELY: the RAW input variance budget is 44% acoustic + 43% numeric vs only 2.5% text (text is L2-normalized -> tiny per-dim scale), so full-matrix PCA-192 spends 87% of its budget reconstructing numeric+acoustic to R2=1.000 while text reaches only R2=0.72, and the top-10 PCs (65% of variance) are all popularity/era/loudness directions with ~0 text loading-mass; per-column standardize flips the captured budget to 73% text / 21% categorical. Variants -> seq ids: V1 noaco seq-20260718-222742, V2 std -222754, V3 std-noaco -222807, V4 textcat -222820, V5 balanced-pca192 -222834, V7 std-txtcatup -222848, V10 text-only -222901, V6 sonic-64 -222905 (64-d). Backward-compatible builder edits: build_song_pca.py --blocks/--block-weights/--standardize (DEFAULT output byte-identical, verified cosine 1.0 vs the live champion collection), new reduce_named_vector.py, sequences.py --vector-name. Handed to experiment-runner for the representation verdict (each vs the PCA-192 champion under proj-blend / content-kNN / frozen gru-infonce-h256, reporting exact recall@10 AND graded artist@10/genre@10). See 'Representation / dataset lineage'. Prior 2026-07-18 (evening, later) — R′+M+C′ CROWN HARDENED ON THE SECOND SPLIT (CONFIRMED): re-run on the disjoint 0.64-cold earlier-holdout `seq-20260718-211238`, the projection champion beats the R+M+C crown by paired-Δ on R@10 **+0.0426 [+0.0280, +0.0573] (CI entirely >0)** with NO MRR regression (point metrics — PROJ 0.1705/0.1050, R+M+C 0.1279/0.0816, R+M 0.1174/0.0791; all n_test 1431; runs `run-20260718-212639-50e1d` / `-213156-cf1dd` / `-214114-93bad`). Absolute R@10 lands below the first-split 0.212 BY CONSTRUCTION on the colder split — the verdict is the paired-Δ, not absolute reproduction. The margin is WIDER than on the first split (+0.0259) — the cold-reaching projection carries proportionally more of the win on the colder split. VALIDATION-ONLY: nothing promoted/re-promoted; the champion stays `blend-gru-markov-content-proj` trained on the first split. (Server auto-recompute slotted the three second-split validation runs into the best-models auto top-12 by raw recall@10 — they are cross-split and warrant exclusion by best-model-selector; pinned rank-1 champion unchanged.) See `2026-07-18b-...projection-registration-and-scan.md` §Second-split hardening. Prior same-evening — SECOND LEAK-FREE SPLIT BUILT (`seq-20260718-211238`): an EARLIER-HOLDOUT 60–80% chronological split on PCA-192 (5723 sessions / 19402 items, train 4292 / test 1431 @ cuts 2023-04-17T19:26:24 → 2024-08-24T14:28:02, cold-item rate 0.64; item space byte-identical to the champion split), TEST WINDOW DISJOINT (index + time) from the champion's `seq-20260715-131139` (80–100%) — 17/17 parity checks pass, READY for the standing R′+M+C′ second-split hardening (experiment-runner re-runs the champion + R+M+C crown on it, verdict = paired-Δ ON THIS split). Prior same-day — PROJECTION REGISTERED + PROMOTED + SCANNED (`2026-07-18b-nexttrack-projection-registration-and-scan.md`): the learned content projection reproduced ON-SERVER EXACTLY (R@10 0.21174 / MRR 0.12199, matches the off-server 0.2117) and is now REGISTERED + PROMOTED + SERVABLE as `blend-gru-markov-content-proj` (best-models rank 1, above the R+M+C model 0.186 rank 2); an objective/rank/τ ablation scan CONFIRMS the InfoNCE/full-rank/τ0.07 default (InfoNCE ≫ BPR — BPR collapses to ~0.15, below even the R+M+C crown; full-rank ≥ rank-64; τ 0.05→0.193 / 0.07→0.212 / 0.10→0.223, the τ0.10 top-cell a sub-noise +0.011 nominal edge → default HELD). Prior — LITERATURE-FIT CAMPAIGN (`2026-07-18-nexttrack-literature-fit-campaign.md`): a supervised **learned content projection** is a NEW best-on-record ROW — **R′+M+C′ z-blend on PCA-192, R@10 0.212 [0.191, 0.233], MRR 0.122**, paired-Δ vs the R+M+C champion **+0.0259 [+0.012, +0.041] (>0)**, and — unlike the champion — SEED-ROBUST (3 seeds, all CI>0) and COLD-REACHING (content leg C′ scores cold 0.152 ≈ warm 0.151). Off-server driver (not yet registered → same productization gap the R+M+C row had before registration). Every borrowed-method lane CLOSED: session-kNN/V-SKNN (drags blend), rank-fusion/RRF (−0.029), ranking-aware combiner (even ORACLE test-fit weights +0.004, CI straddles 0), Markov KN-smoothing (bar 0.107→0.110, washes out), GRU importance negatives (blend tie). Earlier 2026-07-18 (am): (1) registered R+M+C as `seq-blend` leg + definition `blend-gru-markov-content` (`run-20260718-123738-38725-seq-blend` reproduces 0.1859/0.1025 EXACTLY); (2) added the RANKING SERVING CONTRACT — the whole sequence family is promotable + servable via `POST /api/models/{name}/predict`; champion + best-single LSTM promoted, best-models group populated. Prior crown: `2026-07-15c-nexttrack-phase2-model-sweep.md`._

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
| **★★ R′+M+C′ + `markov_gate` on PCA-192 (NEW BEST RECALL ON RECORD, 2026-07-27c — NOT registered, NOT promoted)** | **0.232** (0.23201, 332/1431) | **0.134** (0.13352) | 0.367 / 0.463 | The champion definition `blend-gru-markov-content-proj` plus ONE hyperparam, `markov_gate=true`: the Markov leg ABSTAINS where the last prefix item has ZERO train-level bigram support (`n_u == 0`) and the blend divisor renormalizes from three legs to two. paired-Δ vs a FRESH in-batch champion control **+0.02027 [+0.01048, +0.03075] CI>0** on the canonical split and **+0.01957 [+0.01048, +0.03005] CI>0** on the disjoint 0.64-cold `seq-20260718-211238` (0.17051→0.19008, 244→272 hits) → **hardened on TWO splits at once**, MRR up on both, no regression anywhere. holisticness@10 0.018699 (ΔH +0.010096 CI>0 — it is a crown lever too). MECHANISM, reproduced natively via the real `build_train_transitions`: ALL of the movement is in the `n_u=0` bucket (795 q, 0.1660→0.2025) and the warm buckets' top-k lists are **BYTE-FOR-BYTE IDENTICAL** (`n_u 1-10` 0.3116→0.3116, `n_u>10` 0.2326→0.2326, n_disc 0) — zero warm-side cost. NOT a learned combiner: nothing is fit, the condition is a train-side data statistic. Caveat: the gating variants were screened on TEST, so the OUT-OF-SAMPLE evidence is the second split + the λ0.7 condition (both unscreened, both reproduce to within 0.0007). **Registration/promotion deliberately NOT performed** — recommendation only, awaiting user approval; a 3-seed confirm is the cheap next check. `run-20260728-005349-507bb-seq-blend`; `2026-07-27c-nexttrack-markov-gate-confirm.md` |
| — R′+M+C′ + `markov_gate` + MMR λ0.7/pool200 (CROWN CANDIDATE, 2026-07-27c) | 0.213 (0.21314, 305/1431) | 0.129 (0.12852) | 0.382 / 0.497 | Recall **TIE** with the champion (+0.00140 [−0.01188, +0.01468] straddles) at holisticness@10 **0.030980** (ΔH vs champion **+0.022377 [+0.019941, +0.024923] CI>0**) → the FIRST TIER-1 FREE crown arm at champion-level recall, crown-board rank 2. Without the gate, λ0.7 on the champion is TIER-2 PRICED (−0.01817 recall CI<0); the gate pays the cost back exactly. Gate and MMR are ADDITIVE (gate Δ +0.01957 under λ0.7 ≈ +0.02027 alone). `run-20260728-012030-6d1b7-seq-blend`; `2026-07-27c-...markov-gate-confirm` |
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

### CROWN board — holisticness@10, and the MMR λ frontier (NEW 2026-07-26)

The **crown** (server-effective primary metric) is a DIFFERENT board from the
recall table above, and as of 2026-07-26 it has its own best-on-record row. All
values below are on the live 4-factor definition
(`max(mood_coh,0) × ild × (1−max(artist_adj,artist_conc)) ×
clamp((music@10−0.34)/0.66,0,1)`), recomputed from per-row facets by one code
path, on the canonical split `seq-20260715-131139`, n_test 1,431. **The healthy
recall floor 0.108 (= baseline 0.12299 − the ±0.015 band) is MANDATORY on this
board** — see the "unconstrained holisticness is maximized by predicting worse"
pitfall.

| rank | arm / model | H@10 | recall@10 | hits | run id | status |
|---|---|---|---|---|---|---|
| **2 (NEW 2026-07-27c)** | **champion + `markov_gate` + MMR λ0.7/pool200** | **0.030980** | **0.21314** | 305 | `run-20260728-012030-6d1b7-seq-blend` | **TIER-1 FREE at CHAMPION-LEVEL RECALL** (Δrecall vs champion +0.00140 straddles; ΔH +0.022377 CI>0) — 96% of rank 1's H at 1.86× its recall. NOT registered, NOT promoted; crown candidate pending user approval |
| (NEW 2026-07-27c) | champion + `markov_gate` (the new RECALL best-on-record) | 0.018699 | **0.23201** | **332** | `run-20260728-005349-507bb-seq-blend` | would rank 7; ΔH vs champion +0.010096 CI>0 AT a recall WIN, not a recall price |
| (NEW 2026-07-27c) | champion + MMR λ0.7/pool200, UNGATED (control) | 0.015289 | 0.19357 | 277 | `run-20260728-011245-bfc67-seq-blend` | would rank 8; TIER-2 PRICED without the gate (Δrecall −0.01817 CI<0). Reproduces the 2026-07-26 λ0.7-on-champion recall exactly |
| 1 | λ0.7 pool200 `gru-infonce-h256-mmr-l07` | **0.034171** | 0.11461 | 164 | `run-20260726-184115-bf395-seq-nexttrack` | registered; **TIER-2 PRICED** (Δrecall CI<0) — promotion NOT recommended |
| 2 | **λ0.7 pool50 `gru-infonce-h256-mmr-l07-pool50`** | **0.030355** | 0.12020 | 172 | `run-20260726-184116-20d9b-seq-nexttrack` | registered; **TIER-1 FREE, best TIER-1 arm, crown candidate** — promotion pending R4 + user approval |
| 3 | λ0.8 `gru-infonce-h256-mmr-l08` | 0.029458 | 0.11950 | 171 | `run-20260726-184115-147a6-seq-nexttrack` | registered; TIER-1 FREE, crown candidate |
| 4 | λ0.9 `gru-infonce-h256-mmr-l09` | 0.026761 | 0.12089 | 173 | `run-20260726-184115-e1539-seq-nexttrack` | registered; TIER-1 FREE, crown candidate |
| 5 | single GRU h425 (prior bar `H_bar`) | 0.024872 | 0.12089 | 173 | `run-20260725-143502-be11b-seq-nexttrack` | unchanged |
| 6 | `gru-infonce-h256` (λ=1.0 baseline) | 0.024539 | 0.12299 | 176 | `run-20260726-183333-9a0b2-seq-nexttrack` | unchanged; 9th bit-identical reproduction |
| 7 | λ0.9 champion `blend-gru-markov-content-proj-mmr-l09` | 0.010435 | 0.21034 | 301 | `run-20260726-184115-6cb38-seq-blend` | registered; TIER-1 FREE (far below the crown) |
| 8 | `blend-gru-markov-content-proj` (RECALL champion, λ=1.0) | 0.008603 | 0.21174 | 303 | `run-20260726-183333-1c361-seq-blend` | unchanged |

Excluded as **degenerate / TIER-3** (Δrecall < −0.03, frontier mapping only, must
stay out of best-models by BOTH keys): A4 λ0.3 H 0.031626 @ recall 0.10971
(`run-20260726-184115-1dc77-seq-blend`) and A3 λ0.5 H 0.023576 @ recall 0.16422
(`run-20260726-184115-7a2f7-seq-blend`).

Read: **the crown and the recall board are now optimized by DIFFERENT models, and
the gap is structural, not incidental.** The single GRU is 2.85× ahead of the
recall champion on the crown before any re-ranking, and MMR widens it — because
the champion's recall comes from artist-adjacency (the Markov bigram leg), so
de-eagering it also destroys its `mood_coh` (−40% at λ0.3 vs −4% on the GRU): it
pays twice. **MMR's crown value scales with the health of the other three
factors, not with headroom to de-eager.** λ is EVAL-ONLY, so every row above
shares its model's trained weights — these are re-ranking policies, not new
models. `2026-07-26-nexttrack-mmr-lambda-frontier.md`.

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

- **TRANSITION-COSINE DISTRIBUTION of `seq-20260715-131139` (measured 2026-07-30, durable — reusable by any future de-eagering / curriculum work on this corpus).** Over the **73,632 consecutive train pairs**, `cos(x_t, x_{t+1})` in the PCA-192 item-latent space: p25 −0.022, **p50 0.262**, **p75 0.584**, **p90 0.830**, p95 0.961, p99 0.993; mean 0.289, sd 0.375, and **27.0% of real transitions are NEGATIVE-cosine** (random item pairs: mean 0.0014, median −0.044). Coupling to artist eagerness: `P(same artist | cos > 0.830) = 0.939`, `| cos > 0.584) = 0.764`, `| cos > 0)= 0.427`, vs a base rate `P(same artist | consecutive) = 0.316` — i.e. **the high-cosine tail IS the artist-repetition signal**, so a sonic criterion in that tail de-eagers without needing artist labels (independently corroborating the MMR campaign's 8/8 H1 verdict). ⚠ **CAVEAT — THIS IS THE item→item DISTRIBUTION AND MUST NOT BE USED TO CALIBRATE THRESHOLDS ON prediction→item QUANTITIES.** Doing exactly that produced **4 provably inert arms** on 2026-07-30: a hedged InfoNCE prediction's `cos(pred, x_t)` never exceeds ~0.6, so p75/p90 margins drawn from this table sit OUTSIDE the support of the penalized quantity (`2026-07-30-eager-margin-calibrated-refutation.md`).

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

- 🚨 **THE FAMILY'S FIRST SEED-VARIANCE MEASUREMENT, AND IT IS BIG ENOUGH TO INVALIDATE
  NARROW SINGLE-SEED WINS (2026-08-02c).** `seq-nexttrack` does **not declare `seed`**, so
  every noise figure previously on record for this family is a between-config or
  reproduction spread — seed variance had never been measurable. `seq-bank` declares it.
  Measured on `views=latent+cummean` h256, `seq-20260715-131139`, n=1431, seeds {1337, 7, 42}:
  `dropout=0.0` hits 168 / 166 / 173, **sd 0.002520, range 0.004892 (7 hits)**;
  `dropout=0.1` hits 172 / 162 / 160, **sd 0.004493, range 0.008386 (12 hits)**;
  pooled over the six arms sd 0.003656, range 0.009085 (13 hits).
  All six same-config pairwise seed deltas **straddle 0** — ordinary seed noise, which is
  exactly what makes it dangerous. **At `dropout=0.1` the seed RANGE (0.008386) EXCEEDS the
  sharpest single-hyperparameter half-width on record (0.005250).** Consequences: (a) the
  D2−R2 dropout win (+0.004892 CI>0) was **single-seed** and its effect size equals this seed
  range — **now UNCONFIRMED**; (b) narrow wins are exposed, TIES are not (a tie plus noise is
  still a tie), so the tower-bank −0.0098…−0.0168 losses to C1 survive; (c) `dropout=0.1`
  appears to INCREASE seed variance (sd 0.004493 vs 0.002520) — suggestive at n=3, and the
  opposite of the usual intuition. Carrying this figure to `latent+latent` is an inference
  from an otherwise-identical config, not a measurement.
  `2026-08-02c-dropout-seed-grid-and-the-walk-read.md`.

- **VALIDATION LOSS DOES NOT RANK RETRIEVAL — 4th confirmation, now quantified (2026-08-02).**
  Across the 10-run tower-bank batch, **Spearman(best val InfoNCE, hits) = −0.2954, p = 0.407**
  (bank arms only, n=8: −0.2196, p = 0.601). Sharpest pairs: B4 6.5657 → 186 hits vs B2
  6.5660 → 170 hits (**0.0003 of val loss, 16 hits**); V1 has the 4th-best val loss in the
  batch and the **worst** hit count (162); C0 has the **worst** val loss and beats six of the
  eight bank arms. Never select on val loss; keep the epoch/val table diagnostic-only.
  `2026-08-02-tower-bank-parallel-towers.md`. **5th confirmation, and adversarial
  (2026-08-02b):** at byte-identical params and the same early-stop epoch (24 both), the
  dropout-0.0 arm has the BETTER best-val InfoNCE (6.5763 vs 6.5767) and the WORSE hit
  count (168 vs 172) — 0.0004 of val loss, 4 hits, opposite direction.
  `2026-08-02b-dropout-on-the-deployed-view-pair.md`.
- **A 10-hit swing at n=1431 is INSIDE the noise band.** The tower-bank C1 h512 control took
  **186 vs C0's 176** and the paired Δ still **straddles** (+0.006988 [−0.00210, +0.01607]).
  A ~10-hit / ~0.007-recall single-split lead is "at least equal, likely better — needs the
  3-seed check", never "beats". `2026-08-02-tower-bank-parallel-towers.md`.

- **MEASURED HALF-WIDTH for the SHARPEST possible one-shot contrast — SAME family, SAME
  split, BYTE-IDENTICAL params, ONE hyperparameter differing (2026-08-02b):** hw(Δrecall@10)
  **0.005250**, with only **16 of 1431 rows discordant**. This is the floor of what this
  n=1431 split can resolve on a fresh-weights single-hyperparameter contrast — and it is
  **wider than the ~0.005 effect sizes the dropout question turns on**, which is why that
  question needs seeds rather than one more split. Ladder for orientation: sharp
  shared-weights eval-time instrument 0.0042–0.0182 < this single-hyperparameter contrast
  0.00525 < fresh-weights architecture contrast 0.00874–0.01537 < conservative family figure
  0.0124. `2026-08-02b-dropout-on-the-deployed-view-pair.md`.
- **MEASURED HALF-WIDTHS for `seq-stack` vs an in-batch control (2026-08-01, n=1431, 2000
  resamples, rng 1337, SHARED SPLIT but INDEPENDENTLY TRAINED weights):** hw(Δrecall@10)
  **0.00874–0.01537**, hw(ΔH) **0.00116–0.00261**, hw(ΔMRR@10) 0.00383–0.00753. So a
  fresh-weights architecture contrast on this split resolves at ~±0.009–0.015 recall —
  BETWEEN the sharp shared-weights eval-time instrument (hw 0.0042–0.0182 with n_disc as
  low as 9) and the conservative 0.0124 family figure. The stage-stack losses were
  2.5–3.7 half-widths, i.e. far outside any noise reading.
- **WALK-SURFACE half-widths at n=40 sessions x 20 steps (paired):** hw(Δvibe)
  ~0.006–0.019, hw(Δstride_err) ~0.019–0.038, hw(Δgenres) ~0.8–1.2, hw(Δground)
  ~0.024–0.031. `vibe` is the sharpest facet by far and `ground` the bluntest — its
  between-arm range (0.0060) is ~1/24 of its per-session sd (0.1417), needing ~2,140
  sessions, so **`ground` is a degeneracy FLOOR and must never be ranked or multiplied**.
- **A run-to-run offset of ~2 hits/1431 is the floor for a "reproduction" on this family
  when weights are retrained** — the `seq-stack` `gru` h256 control returned 178 vs the
  historical 176 because `fit()` re-seeds on entry (init weights match, dropout stream
  does not). Do NOT pre-register bit-identity for any predictor that builds its model
  before calling `fit`.

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

**MEASURED PAIRED HALF-WIDTHS FOR A SHARED-WEIGHTS (EVAL-TIME) COMPARISON — 3–33×
SHARPER THAN THE ARCHITECTURE-FAMILY FLOOR (2026-07-26, MMR λ frontier).** When two
arms differ ONLY in an eval-time re-rank, the trained weights are bit-identical and the
comparison is maximally paired. Measured over the 8 λ arms (2,000 resamples, rng 1337,
n=1431, ALL five crown facets + H recomputed per resample for both arms on the SAME
index set): **hw(ΔH) 0.000382–0.002476** and **hw(Δrecall@10) 0.004193–0.018169**,
against the conservative architecture-family figures of 0.0018 (H) and 0.0124 (recall).
The reason is visible in `n_disc`, the number of sessions whose hit@10 changed at all:
as low as **9** for a λ0.9 arm (vs 188 at λ0.3), because a re-rank of the top-`pool`
only flips sessions whose truth crosses the k=10 boundary. **CONSEQUENCE: the
conservative family half-width materially UNDER-POWERS eval-time levers.** The λ0.9
champion arm was pre-registered as 'likely UNRESOLVABLE' at 0.6–0.8 conservative
half-widths and in fact resolved at **4.8 measured half-widths** (ΔH +0.001835
[+0.001458,+0.002223], hw 0.000382). Budget MDE for any eval-time-lever campaign from
the shared-weights pairing, and always report the MEASURED per-arm half-width rather
than a family estimate. Note the half-widths GROW with distance from the control
(0.000382 at λ0.9 → 0.002476 at λ0.3), so the sharpness is greatest exactly where the
effects are smallest. Conversely `H`'s ABSOLUTE scale differs 3× between models: 0.0018
is 21% of the champion's H (0.0086) but only 7% of the GRU's (0.0245), so the crown
instrument is intrinsically ~3× more precise on the single GRU.
`2026-07-26-nexttrack-mmr-lambda-frontier.md`.

## Walk-surface reads of architectures the one-shot metric closed (2026-07-31b)

- **REGULARIZING THE DEPLOYED ENGINE HOLDS THE WALK AT THE DEPLOYED CELL BUT COSTS LATE-WALK
  MOOD-HOLDING AT THE MATCHED CELL (2026-08-02c).** E1 (`seq-bank` `latent+cummean` h256
  `dropout=0.1`, `run-20260802-214618-a38be-seq-bank`) vs the DEPLOYED engine, 40×20, artist
  cap 1, paired, G3 green. At **a0.8/s0.5** (deployed cell): **Δvibe +0.0024
  [−0.0044, +0.0082] straddles → PASSES the candidacy test**, and it is not the tiny-steps
  artifact because Δstride_err is −0.0129 (better) at the same time; Δdrift −0.0046,
  Δgenres −0.275, Δground −0.0042, all straddle. At **a0.4/s0.3** (matched): Δvibe +0.0007
  and Δstride_err +0.0036 straddle, but **Δdrift −0.0226 [−0.0401, −0.0056] CI<0** — its
  end-quarter vibe falls 0.084 below start vs the deployed engine's 0.061. So dropout on the
  concatenated hidden state appears to cost mood-holding LATE in the walk, at one cell only,
  which no one-shot metric can see. **E1 was NOT recommended for deployment** — Phase A of the
  same campaign removed the recall reason to redeploy.
  `2026-08-02c-dropout-seed-grid-and-the-walk-read.md`.

- **LSTM long memory does NOT hold a mood — hypothesis REFUTED on its own decisive
  test.** The user's hypothesis was that a one-step metric cannot reward long memory,
  and that an LSTM's cell state should carry "we are in this vibe" across a walk.
  Tested `run-20260722-165345-bb387-seq-nexttrack` (arch=lstm h256, **canonical
  PCA-192**, local weights) against `run-20260726-183333-9a0b2` (gru h256) — paired,
  40 identical held-out sessions, 20 steps. **With anchor ZERO** (the decisive cell,
  no external help): GRU vibe 0.310 vs LSTM 0.254, Δ **−0.0557 [−0.1097, −0.0064]
  CI<0**. Δvibe is CI<0 in ALL FOUR cells (−0.056 / −0.025 / −0.031 / −0.011). The
  extra 256 numbers of cell state (LSTM state 512 = h+c vs GRU 256) do not hold the
  seed's mood.
- **But two walk properties the one-shot metric could not see DO favour the LSTM.**
  `stride_err` is lower in all four cells (CI<0 in two): it moves closer to the user's
  real rhythm than the GRU, which is the exact defect diagnosed in the shipped engine.
  And at anchor 0 its **grounding is better: Δ +0.0354 [+0.0049, +0.0664] CI>0** — the
  ONLY place grounding discriminated between two reasonable arms in the whole
  2026-07-31 effort. Both effects vanish once the anchor is on. Its one-shot recall is
  0.1188 vs the GRU's 0.12299 (slightly worse), i.e. invisible-to-harmful on the old
  metric. **LSTM's best (anchor, stride) cell has NOT been swept** — it already moves,
  so its optimum is plausibly more anchor / less stride than the GRU's.
- ⚠ **METRIC CAVEAT found by this comparison: `vibe` and `stride` are NOT independent.**
  A model taking tiny steps scores high vibe trivially, because it never went anywhere.
  At anchor 0 the GRU has WORSE stride_err (0.328 vs 0.293) and BETTER vibe (0.310 vs
  0.254) — the two are coupled. Disentangle by comparing at MATCHED stride: at
  a0.4/s0.3 the LSTM has better stride (0.098 vs 0.129) and vibe 0.534 vs 0.565, so a
  small real vibe cost survives. Never read `vibe` without `stride_err` beside it.
- **Interpretation that survives:** holding a mood appears to need a FIXED reference
  (the seed centroid, never overwritten). A recurrent cell is rewritten every step by
  design, so the external anchor is not substituting for recurrent memory — it does
  something recurrence structurally does not. This reframes the anchor from "a crutch
  for a weak model" to "a different mechanism".
- **`N x (ANN + GRU)` (pre-MLP before the recurrence) ALREADY EXISTS and is
  un-walk-evaluated.** 6 runs used `pre_hidden*`; the record's verdict was one-shot
  only. **N=2 best arm `run-20260726-161801-a6423-seq-dualgru`** (tower A bare / tower
  B pre-MLP 256, `fusion_layers=0`, recall 0.1237) **TIED h256** (+0.0007, straddles) —
  under the standing rule above, a one-shot TIE says nothing about the walk, so this is
  exactly the position the dual-tower was in before 2026-07-31. It is NOT promoted and
  has no local weights (remote run) but **has 4 artifacts in Postgres, so it can be
  promoted without retraining**. Loadable pre-MLP arms today: `ca0b8` (pre_b=384) and
  `dc7d2` (pre_b=128), both `fusion_layers=1` — the topology the record shows LOSES, so
  they are a weak proxy for the idea. `seq-nexttrack` N=1 (`ad53f`, pre_hidden 256,
  recall 0.109) likewise has no local weights.
- **RESOLVED 2026-08-01 — `N x (ANN + GRU)` IS NOW WALK-EVALUATED AT N=2 AND AT DEPTH 4,
  AND IT LOSES EVERYWHERE.** `dual-premlp256-f0` was promoted from its Postgres artifacts
  (no retraining) and walk-evaluated: at the matched cell a0.4/s0.3 vs the shipped GRU,
  **Δvibe −0.0255 [−0.0432, −0.0096] CI<0** with Δstride_err straddling ⇒ REFUTED under
  the V2 rule. Its one real gain is **less drift** (Δdrift +0.0403 [+0.0086, +0.0719]
  CI>0, −0.080 → −0.039); its best cell is a0.8/s0.5 (vibe 0.602 / stride_err 0.146 /
  ground 0.364). Then five 4-stage `seq-stack` arms were refuted on BOTH surfaces (see the
  2026-08-01 roll-up), so **the stage-stack axis is CLOSED under V3** — the first
  architecture closure on this corpus that satisfies the standing rule, because it carries
  a walk read rather than resting on one-shot metrics.
- ⚠ **THE "better stride/drift, worse vibe" SIGNATURE IS GENERAL, not LSTM-specific.**
  The LSTM (2026-07-31b) and the N=2 pre-MLP dual (2026-08-01) show the SAME shape: they
  move closer to the user's real rhythm and/or drift less, and pay for it in seed-anchored
  vibe. Two independent arms now ⇒ treat it as a property of adding pre-recurrence
  machinery in front of / around the recurrence, and expect any new such arm to trade the
  same way rather than to escape the trade.
- **The walk harness is a VALIDATED instrument as of 2026-08-01.** `walk_headtohead.py`
  reproduced all ten recorded 2026-07-31b means exactly, and reproduced them AGAIN with 2
  and then 7 extra arms present — arm addition does not perturb the shared session sample
  or the RNG. Cross-invocation pairing is therefore licensed (the session set is a
  deterministic rng-1337 permutation), which is what makes multi-cell sweeps affordable.
  It also gained `--include-stack` / `--arm family:model:anchor:stride[:label]` /
  `--defaults all|gru|none` / `--base-label`, all defaulting to the historical behaviour,
  plus `promoted_score()` which loads the ARTIFACT from the dataset and only the WEIGHTS
  from the model dir after asserting `item_latents.f32` is byte-identical.
- **G3 IS THE CHEAP INSTRUMENT CHECK EVERY WALK CAMPAIGN SHOULD RUN.** Δ(shipped
  `gru.onnx` @ a0.4/s0.3 vs a freshly-trained h256 control @ the same cell) straddles 0 on
  both `vibe` (−0.0033) and `stride_err` (+0.0137) ⇒ the deployed ONNX engine and an
  in-batch control are interchangeable on the walk's two primary axes, so walk verdicts
  stated vs a fresh control are not an artefact of the base. One extra arm, no training.

## Standing rules on how verdicts may be reached (2026-07-31b)

- 🚨 **A NARROW SINGLE-SEED WIN IS NOT A WIN ON A `seed`-DECLARING PREDICTOR (2026-08-02c).**
  Measured seed range inside ONE fixed config is up to **12 hits / 0.008386 recall** (see
  §Noise), which SWALLOWS the record's only measured dropout win (+0.004892). So any win
  claim with **|Δrecall@10| < ~0.009** on a predictor that declares `seed` (`seq-bank`,
  `seq-blend`, `seq-stack`) requires a **3-seed check before it is quoted as a win**. Ties
  and larger losses are unaffected. This is the rule the D2−R2 result would have failed.
  `2026-08-02c-dropout-seed-grid-and-the-walk-read.md`.
- ✅ **THE WALK HARNESS + PROMOTION PATH ARE VALIDATED TO EXACTLY ZERO, NOT MERELY "GREEN"
  (2026-08-02c).** A `seq-bank` twin that is bit-identical to the deployed `seq-dualgru`
  engine on all 1,431 one-shot rows was walk-evaluated against it at BOTH cells: Δ =
  **+0.0000 [+0.0000, +0.0000]** on all five axes (vibe, stride_err, drift, genres, ground),
  per session, zero-width intervals. This is stronger than G3 (same model, not merely an
  interchangeable one) and simultaneously validates `walk_headtohead.py`, the
  `promoted_score` path, the Postgres→`data/models` materialization, and the cross-family
  `seq_bank` ↔ `seq_dualgru` substitution. No prior walk verdict needs re-examination.
  `2026-08-02c-dropout-seed-grid-and-the-walk-read.md`.

- **A WALK RESULT AT ONE CELL IS NOT A WALK RESULT (2026-08-02).** Any arm that reaches
  S2-WIN or S2-TRADE at the primary cell MUST be re-measured at the deployed cell
  (a0.8/s0.5) and must hold the same sign there before anything is saved. Measured: B2 and
  V3 both earned a genuine Δdrift CI>0 with vibe held at a0.4/s0.3, and **both flipped to
  Δvibe CI<0 at a0.8/s0.5** with the drift gain gone. Without this clause the tower-bank
  campaign would have reported two positive walk results and saved two definitions.
  `2026-08-02-tower-bank-parallel-towers.md`.
- **A PARAM-MATCHED CONTROL LEG IS MANDATORY FOR ANY TOPOLOGY WIN CLAIM (2026-08-02).**
  B4 (four h211 GRU towers) and C1 (one h512 GRU) returned the **same 186/1431** at the same
  param budget, Δ = 0.000000 [−0.00839, +0.00839]. B4's +10 hits over the h256 baseline
  would have read as a topology win against C0 alone; the C1 leg shows it is capacity.

- **NO FAMILY MAY BE CLOSED ON ONE-SHOT METRICS ALONE.** `recall@10` and
  `holisticness@10` are computed on a single top-10; the product serves an
  autoregressive WALK. The dual-tower family was closed by THREE campaigns on recall
  grounds and is the winner on the stated objective (see the 2026-07-31b roll-up) — and
  the corroboration was already in the record as a negative (`seq_dualgru.py:113`: the
  cummean tower "added real linear directions (CKA 0.458) but ZERO next-item concepts").
  A campaign may still REJECT an arm on one-shot grounds, but "this axis is closed" now
  requires a walk-surface read (`tools/walk_eval.py` / `walk_headtohead.py`).
  **The dual-tower family is hereby REOPENED.**
- **A design's decision rule must name the SURFACE its verdict rests on** — one-shot
  top-10, or generated walk. They disagree: on the walk surface every model ties once a
  policy is applied, and the champion blend (best one-shot recall) is the WORST arm on
  mood-holding (Δvibe −0.133 CI<0).
- **⚠ `seq-20260718-211238` CANNOT harden a model trained on `seq-20260715-131139`.**
  Measured 2026-07-31: **all 1431** of split-2's test sessions are inside split-1's
  train set (split 1 train = earliest 80%, split 2 test = the 60–80% band). It is a
  valid disjoint split only for arms RETRAINED on split-2's train — which is what the
  earlier champion/gate hardening did. Re-using promoted split-1 models on split-2 test
  sessions is a total leak. For walk-eval stability without retraining, resample over
  disjoint halves of the SAME test set and label it replication, not hardening.

## Pitfalls / infra

- 🚨 **`fusion_layers=0` DUAL/BANK ARMS ON RECORD TRAINED WITH NO ACTIVE DROPOUT — INCLUDING
  THE DEPLOYED ENGINE (2026-08-02).** In `DualTowerNextLatent`, dropout enters ONLY via a
  pre-MLP (`pre_hidden=0` ⇒ absent), an inter-layer gap (`layers=1` ⇒ 0.0), or the fusion MLP
  (`fusion_layers=0` ⇒ a bare `Linear`). So at f0/layers=1/pre_hidden=0 the `dropout`
  hyperparameter is INERT, while the h256 `seq-nexttrack` anchor applies `self.drop` before
  its head. Consequences, both measured: (a) **dropout 0.1 BEATS dropout 0.0** at
  byte-identical params — `seq-bank` D2 vs R2, Δrecall +0.004892 [+0.000699, +0.009783]
  **CI>0**, 177 vs 170 hits — so the f0 arms left ~+0.005 recall on the table and the
  deployed dual `l/cummean` is **under-regularized**; (b) the three recorded "f0 beats f1"
  gaps (+0.016 / +0.020 / +0.0266) are **NOT like-regularized** and must not be quoted as
  clean topology measurements — the true f0 advantage is if anything LARGER, not smaller.
  Retraining the deployed arm at `dropout=0.1` is a 1-run free-improvement follow-up.
  `2026-08-02-tower-bank-parallel-towers.md`.
- 🚨 **…AND THE FOLLOW-UP DID NOT SETTLE IT — THE DEPLOYED PAIR IS *UNRESOLVED*, NOT FIXED
  AND NOT REFUTED (2026-08-02b).** The 1-run follow-up above was run (as 2 runs: a control
  plus the twin) and **the D2−R2 win did NOT replicate on the deployed `latent/cummean`
  pair**: Δrecall@10 **+0.002795 [−0.002795, +0.007704] STRADDLES**, 172 vs 168 hits,
  half-width **0.005250**. Read this correctly, because the naive read is wrong in a
  costly way: the half-width is **LARGER than the entire D2−R2 effect** (+0.004892) it was
  sent to reproduce, the sign agrees (~57% of the magnitude), and the two CIs are mutually
  compatible in BOTH directions (each campaign's point estimate lies inside the other's
  interval). Only **16 of 1431 rows flip at all** (10 gained / 6 lost) — the whole contrast
  is a net of 4 sessions. So the evidence does **NOT** localize the dropout win to
  `latent+latent`, and does **NOT** license 'the deployed engine is fine': the deployed
  engine's regularization status is **OPEN**. Settling it needs the **3-seed check**, which
  is NOT authorized. Two by-products worth keeping: (a) **the f0 inertness is now
  DEMONSTRATED end-to-end, not inferred from the constructor** — the deployed engine
  (nominally `dropout=0.1`) and a `seq-bank` twin at EXPLICIT `dropout=0.0` return **the
  same 1431 ranked lists, byte for byte** (all `top_k_ids` identical, not merely the same
  hit count); (b) regularizing showed **no facet cost** (H@10 +0.00044, music@10 +0.0011,
  mrr +0.00098, mood_coh +0.0053; artist_conc −0.0010, ild −0.0014), so a better-powered
  test risks finding *no gain*, not a penalty. **Do not redeploy on this evidence** — no
  walk measurement was taken (S2 is gated on S1 replicating, and did not run).
  `2026-08-02b-dropout-on-the-deployed-view-pair.md`.
- ✅ **…AND IT IS NOW CLOSED: A POSITIVE EFFECT ON THE DEPLOYED PAIR IS EXCLUDED, AND THE
  ORIGINAL D2−R2 WIN IS RE-FLAGGED AS UNCONFIRMED (2026-08-02c).** The 3-seed grid was run
  (4 new runs; the 2026-08-02b pair WAS the seed-1337 cell and was reused after a key-by-key
  hyperparam diff). Per-seed Δrecall@10 (d0.1−d0.0): **+4 / −4 / −13 hits** at seeds
  1337 / 7 / 42, all straddling, **signs disagree**; seed-averaged **−0.003028
  [−0.006988, +0.000932]**, leaning NEGATIVE. By the pre-registered rule: SETTLED-POSITIVE
  **excluded**; sign STILL UNRESOLVED (spread swamps the effect: per-seed sd 0.005943 vs
  |mean| 0.003028). **The decisive detail is the resolvable floor: hw 0.003960 is FINER than
  the +0.004892 effect being hunted, so this is a resolution-sufficient non-finding, NOT
  another underpowered one** — categorically unlike `2026-08-02b`. Practical upshot: **there
  is no free recall in regularizing the deployed engine; stop spending runs on this axis.**
  The code defect is untouched and now DOUBLY demonstrated (the `dropout=0.0` twin reproduces
  the deployed engine's 1,431 ranked lists byte for byte AND reproduces its walk to EXACTLY
  0.0000 on all five axes at both cells) — it is simply an EMPTY opportunity. Do NOT quote
  the D2−R2 +0.0049 CI>0 result as a win without re-seeding it (2 runs; optional).
  `2026-08-02c-dropout-seed-grid-and-the-walk-read.md`.
- ⚠ **WALL-CLOCK ON THE REMOTE WORKER IS NOT A PROXY FOR EPOCHS (2026-08-02).** D2 took
  15.7 min and R2 4.7 min at identical params and identical early-stop epoch (**e14 both**).
  The 3.3× gap is shared-CPU noise, not a training signal. Read `run_events` for
  "early stop at epoch N (best val X)" before inferring anything from duration.
- ⚠ **`seq-nexttrack` DOES NOT DECLARE `tau` / `val_fraction` / `seed` (2026-08-02).**
  Sending them in a launch body trips the `unknown hyperparameter` rejection. `tau` is
  hardcoded `0.07` at `seq_nexttrack.py:86` and consumed as `hp["tau"]`, so parity with an
  explicit `tau:0.07` on another family is exact. The safe pattern for a reproduction
  control is to POST the reference run's **stored hyperparams verbatim** rather than
  re-deriving them from a design's "held constant" list.
- ⚠ **THE HUB LAPTOP SUSPENDS MID-CAMPAIGN AND REAPS THE SERVER (2026-08-02).** The server
  clock jumped 00:36Z → 18:17Z, the detached process was reaped, and `/tmp` was cleared with
  it (taking a scratchpad log and a registry backup). Launching in **small chunks** bounds
  the loss — the tower-bank batch lost no run across four chunks. The remote worker daemon
  does NOT survive it either and must be relaunched; verify with `pgrep -af "[l]ensing-server
  worker"` (**the bracket form** — a plain `pgrep -af "lensing-server worker"` matches its own
  ssh command line and reports a dead daemon as RUNNING).

- **`seq-stack` HAS NO BIT-IDENTITY GATE, BY CONSTRUCTION — gate on FUNCTIONAL EQUIVALENCE
  PLUS AN IN-BATCH CONTROL INSTEAD (2026-08-01).** `seq_stack.train()` builds the model and
  hands it to `seq_nexttrack.fit(..., model=...)`, but **`fit()` re-seeds on entry**, so
  passing a pre-built model shifts the training RNG stream by the init draws: the init
  WEIGHTS match the historical h256 GRU, the DROPOUT stream does not. Consequence measured:
  `stages=gru` h256 returns **178/1431, not the on-record 176** — a 2-hit offset that is
  CORRECT BEHAVIOUR, not a defect, and pre-registering bit-identity would have read it as a
  failure and burned a relaunch. The two gates that DO work: (1) `seq_stack.py verify`
  proves `stages=["gru"]` computes the SAME FUNCTION as `SeqNextLatent` by copying weights
  across — max|Δ| **0.00e+00** on `forward` and `predict_next` at T=1,2,5,17,40, params
  394,944 both sides, RNG-independent; (2) an in-batch `stages=gru` control in every batch,
  with a pre-registered BAND (used: hits ∈ [155, 198] = 0.12299 ± 0.015). **Generalize:
  any predictor that constructs its module before calling a re-seeding `fit` cannot be
  bit-identity gated; band the control instead.** Also run `verify` BEFORE the registry
  edit and server restart — it is free and RNG-independent, so a fail costs nothing.
- **MD5 THE WHOLE IMPORT CLOSURE OF THE FAMILY BEING LAUNCHED, NOT JUST THE FILES YOU
  COPIED (2026-08-01).** A campaign copied `seq_stack.py` + `registry.toml` to the worker
  and the parity check then failed on a file it had NOT copied: `seq_common.py`, hub
  `09e7aef666455d49e33f187d73615bda` vs worker `03e14e7c246ba6521ea315805d002157`. The
  divergence was the 2026-07-31 `artist_cap` k-scaling block in `predict_ranking` (+
  `import math`), 19 lines, touching no metric key and inert at `artist_cap=0` / k=10 — so
  that batch was valid either way, but establishing THAT required reading the diff, which
  is exactly the work the gate exists to force. This is the SECOND time a stale worker copy
  of `seq_common.py`/`seq_blend.py` was caught by an md5 gate (cf. 2026-07-30b, where a
  stale checkout silently ignored `markov_gate` AND `artist_cap` and returned a null).
  **A stale worker checkout yields a WRONG NUMBER, not a crash.** Re-verify immediately
  before EACH chunk POST, not once per campaign.
- **A DESIGN'S LAUNCH BODY CAN CONTRADICT ITS OWN REGISTRY BLOCK — the API catches it, and
  the fix must be PROVED inert (2026-08-01).** The approved `seq-stack` body sent `arch`,
  `bidirectional`, `residual`, `num_layers`, `eager_beta`, `eager_margin`; the approved
  registry block declares none of them, so `POST /api/runs` rejected every post with
  `unknown hyperparameter arch` (the server validates hyperparams against the registry's
  declared params). Resolved by dropping the six keys **after proving `seq_stack.load_hp()`
  returns an IDENTICAL resolved dict either way** (`DEFAULTS` supplies exactly
  `False/False/1/"gru"/0.0/0.0`), leaving the registry block as approved. **Rule: when a
  launch body is rejected, prove the repair is semantics-preserving on the predictor's own
  `load_hp` before re-posting — do not silently re-shape the arm.**
- **VALIDATION LOSS IS AN ACTIVELY MISLEADING SELECTION CRITERION FOR THIS FAMILY
  (2026-08-01).** `ann+gru+gru+ann` reached the **LOWEST val InfoNCE in its batch (6.5378
  vs the h256 control's 6.5832)** while retrieving **55 FEWER hits (123 vs 178/1431)**. The
  one-step teacher-forced objective and top-10 retrieval **dissociate at depth**, so any
  architecture search that early-stops or model-selects on val loss will pick the wrong
  arm. Select on `recall@10` + the walk surface. Corollary for depth experiments: check
  `epochs run` / `best epoch` before blaming regularization — the 4-stage arms all trained
  LONGER than the control (21–40 epochs vs 14), which is what exonerated the 4x-dropout
  confound without spending the contingency runs.
- **DEPTH IS THE DAMAGE, AND THREE CONTROLS ARE NEEDED TO SAY SO (2026-08-01).** To
  attribute a depth loss you must simultaneously close: capacity (a param-matched wide
  control — C1 h537 at 3.24x params TIED, Δrecall −0.00280 straddles), regularization (the
  epoch table, above), and the added module type (C vs D showed a leading per-step MLP at
  depth 4 is recall-NEUTRAL and ΔH +0.00291 CI>0). Only with all three closed does "the
  depth of the recurrent chain itself" survive. ⚠ Related SCOPE CORRECTION: the 2026-07-26
  pre-encoder pitfall "a leading per-step MLP is HARMFUL" is an **N=1** result (−0.0140
  CI<0 vs a single GRU); **at depth 4 the same addition is free-to-slightly-good**. Scope
  it to the shallow case.
- **TOPOLOGY FACT WORTH REUSING: BRACKETING BEATS INTERLEAVING (2026-08-01).**
  `ann+gru+gru+ann` beats `ann+gru+ann+gru` by **+0.01537 [+0.00280, +0.02797] CI>0** at
  **byte-identical parameter count by formula** — interposing a per-step MLP BETWEEN two
  recurrences costs more than putting MLPs on the outside of the stack. Recurrence COUNT,
  by contrast, is flat (mean(A,B) vs C: Δrecall +0.00140 straddles).
- **MEASUREMENT PROMOTIONS FLOOD THE AUTO best-models GROUP IMMEDIATELY (2026-08-01).**
  Promoting arms solely to walk-evaluate them puts them straight into the server's
  deterministic top-12-by-H recompute — 7 promotions entered at ranks 9, 11, 13, 14 and
  below (recompute 2026-08-01T02:07:10Z), including arms at recall 0.07–0.08, and the
  server additionally auto-created `best-seq-stack-20260801-003536-d612e` from the WORST
  pure-depth arm. Unlike the older `spawn_recompute` STALENESS gap, the recompute now fires
  correctly — the problem is the opposite one. **Any campaign that promotes for measurement
  must hand off to best-model-selector**, and should consider deleting the measurement
  promotions once their walk numbers are banked.

- **A `(deleted)` REMOTE-WORKER BINARY IMAGE STRIPS UNKNOWN METRIC KEYS SILENTLY —
  rsyncing the binary is NOT enough, the daemon must be CYCLED (2026-07-26).** After
  `run.rs` gained `pub artist_conc_at_k` (14:23) and both `lensing-server` binaries were
  rebuilt + rsynced (14:25), **remote runs still arrived at the hub with no
  `artist_conc_at_k`** — even though `~/lensing-worker/predictors/` was md5-identical to
  the hub and the worker's OWN `data/runs/<id>/metrics.json` carried the key and
  `predictions.json` carried it per row. Cause: the worker DAEMON process had started
  2026-07-25 14:00, so `/proc/<pid>/exe` pointed at `…/lensing-server (deleted)` and the
  RUNNING image contained **0** occurrences of `artist_conc_at_k` (vs 1 in the on-disk
  replacement). The old image deserialized the predictor's metrics into a `Metrics`
  struct lacking the field, dropped it, and POSTed the filtered set **without erroring**.
  `holisticness_at_k` stayed CORRECT throughout — the PREDICTOR computes H itself with
  `eager = max(artist_adj, artist_conc)` before the Rust layer sees it — which is exactly
  what made the defect invisible. **Every remote run between 14:23 and 18:13 on
  2026-07-26 silently lost `artist_conc_at_k`** (incl. the two discarded controls
  `run-20260726-175258-c4968-seq-blend` / `-6f5fb-seq-nexttrack`, both `succeeded` and
  both G2-defective). **DIAGNOSTIC RECIPE:** compare the worker's own
  `data/runs/<id>/metrics.json` against the hub's DB row — if the worker has the key and
  the hub does not, it is the daemon, not the predictors; confirm with `readlink
  /proc/<pid>/exe` (a `(deleted)` suffix is the tell) and by grepping the running image.
  **RULE: cycle the remote worker daemon whenever a metric field is added to `run.rs`,
  and never infer 'stale predictors' from a missing metric without checking the daemon.**
  Corollary: NO run in the DB had ever carried a NATIVELY-ingested `artist_conc_at_k`
  before 18:13 — the 22 runs that had one were the orchestrator's rescore set, and
  `tools/rescore_holisticness.py:101-118` RECONSTRUCTS the facet from `top_k_ids` +
  `items.json`. That reconstruction is now VALIDATED: native A0/B0 emissions
  (0.6916530786551751 / 0.35591272614333413) match it to the last digit.
  `2026-07-26-nexttrack-mmr-lambda-frontier.md` §Findings 7.
- **MMR SILENTLY NO-OPS when the content-metric index is unreachable — ALWAYS gate on
  `ild` (2026-07-26).** `mmr_rerank_order` returns the input order unchanged when
  `music_vecs is None` (`seq_common.py:306`; the known `QDRANT_URL` requirement) and it
  fails **without any error**, so an affected arm looks exactly like 'λ has no effect'.
  **GATE: for every λ<1 arm, `ild_at_k` must differ from its OWN model's λ=1.0 control by
  > 1e-6.** Exact equality ⇒ the arm is INVALID (discard, fix the env, relaunch), NOT a
  null result. Measured healthy separations for calibration: 0.018–0.048 on the single
  GRU (λ0.9→0.7) and 0.030–0.408 on the champion blend (λ0.9→0.3).
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
- **THE RECTIFIER MECHANISM IS NOW MEASURED, not inferred (B4 SAE read,
  2026-07-26).** The decisive control was a counterfactual tap the SAE engine
  cannot express — the model's OWN affine map with the ReLU deleted — now kept as
  `tools/rectifier_control.py`. Next-item genre decodability at the same step, on
  `seq-20260715-131139` (11,184 probe test rows, 25 classes, paired SE ≲0.0065):
  `raw_latent` 0.4033/AUC 0.7609 → `pre_linear` 0.4084/0.7667 (**the affine map is
  neutral, ~0.8σ**, exactly as `W_i·(Vx)=(W_iV)x` predicts) → `pre_relu`
  0.3700/0.7508 (**the ReLU is the whole loss, ~5.9σ**). `frac_exact_zero` on the
  rectified tap = **0.5327**, matching `pre_linear`'s `frac_negative_coords`
  = 0.5327 to four digits — the rectifier deletes precisely the negative half,
  leaving ~120 live units for 192 signed directions, i.e. FEWER live coordinates
  than the input has dimensions. The damage survives the recurrence: within one
  model, `tower_b` (fed rectified) 0.3929/0.7845 vs its identical twin `tower_a`
  (fed raw) 0.4183/0.8004, ~3.9σ — and `tower_a` reproduces the standalone plain
  GRU (0.4235/0.8006), which is what "the bare tower RESCUES" requires. Two
  caveats: the SAE's OWN metrics were non-discriminating here (at `topk=32`
  utilization pins ~100%, `n_interpretable_concepts` saturates at 428–474 of 512,
  classes-represented 25/25 and 0/50 dropped for EVERY layer of every model — the
  signal was `next_item_decodability` alone); and one unexplained counter-signal
  runs the other way, top-atom concept PURITY being higher on rectified-fed states
  (S1 `recurrent` 0.394 vs REF 0.343) — do not build on it. Operational: the
  row-building phase thrashes if several `model-sae` runs go concurrently (3 at
  once did not finish in 12 min; alone, under 60 s) — run them one at a time.
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
  **NARROWED 2026-07-27c — this pitfall forbids LEARNED SCORE-LEVEL FUSION, NOT
  PER-QUERY EVIDENCE GATING.** `markov_gate` is exactly the "third thing" the
  clause above demanded and it WON (+0.02027 / +0.01957 paired-Δ CI>0 on two
  splits): no parameters are learned, no weights are fit, and the intervention is
  conditioned on a **train-side DATA STATISTIC** (`n_u`, the bigram support of the
  last prefix item), not on a score. Null (3) — the ORACLE test-fit blend weights
  at +0.004 straddling 0 — is **not contradicted, it is EXPLAINED**: that search
  ranged over GLOBAL STATIC SCALARS, and the bucket decomposition shows the two
  regimes cancel exactly (dropping M globally is a −0.0070 TIE; at `n_u=0` the
  gain is +0.0365 over 795 queries and in the warm buckets the top-k lists are
  BYTE-IDENTICAL). No constant can express a gate whose sign flips with `n_u`.
  The right reading of all five results together: **fusion WEIGHTS are closed;
  fusion ABSTENTION is open.** `2026-07-27c-nexttrack-markov-gate-confirm.md`.
- **`_zcand`'s SCALE-INVARIANCE ERASES PER-LEG CONFIDENCE — a GENERAL defect of the
  z-blend, of which the Markov leg is only the first instance found (2026-07-27c).**
  `seq_blend.py:169-172` maps every leg's score vector to unit variance so the legs
  are commensurable — which is exactly what makes the blend robust to their wildly
  different score scales, and exactly what destroys their **confidence**. A leg that
  knows nothing about the current query still gets re-inflated to the same footprint
  as a leg that does, and votes at full weight. MEASURED INSTANCE: at `n_u = 0` the
  Markov leg is pure artist/genre back-off (`trust = n_u/(n_u+8) = 0`), yet spoke at a
  full one-third on **55.6% of canonical test queries**; silencing it there is worth
  **+29 net hits (+0.02027 recall@10, CI>0, hardened on a 2nd split)** on the best
  model on record. **CONSEQUENCE: any leg with a query-dependent notion of "I have no
  evidence here" is mis-weighted by this blend.** The content leg (cold-item
  neighbourhood density?) and the GRU leg are **UNAUDITED** — same mechanism,
  possibly the same size of prize. Also note the corollary for design: the fix is a
  GATE (abstain / renormalize the divisor), not a re-weight, because a re-weight is
  the refuted lane. `2026-07-27c-nexttrack-markov-gate-confirm.md`.
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
- **STANDING (2026-07-27c, UNREMEDIATED): the `snappler` worker checkout is STALE for
  `seq-blend` — `~/lensing-worker/predictors/seq_blend.py` contains ZERO occurrences of
  `markov_gate`.** The daemon (`lensing-server worker --hub-url http://192.168.0.7:8096
  --poll-secs 5`, pid 573145) is LIVE. **Any `queue:true` `seq-blend` run dispatched
  there will silently execute the UNGATED path and return a null** — the divergence
  pitfall below, live right now. The 2026-07-27c campaign avoided it by routing: remote
  dispatch is **OPT-IN** via `"queue": true` on `POST /api/runs`
  (`api.rs::StartRunRequest`, `runs.rs::enqueue_run`); a bare POST orchestrates on the
  hub and never reaches the queue, confirmed by `claimed_by: null` on all six runs.
  Freshening/cycling the worker is an ORCHESTRATOR action; until it is done, run
  `seq-blend` on the hub only (and note the hub default is `--max-runs 2`, so a
  serialized batch must be POSTed one at a time — see the oversubscription pitfall).
- **EVAL AND SERVING DO NOT AGREE ROW-FOR-ROW, EVEN WITH MMR NOW WIRED (measured
  2026-07-30b — OPEN, unexplained).** Replaying stored eval rows through
  `POST /api/models/{name}/predict` using each run's own `prefix_ids` (12 rows/model,
  comparing to the run's stored `top_k_ids`):

  | model | eval trained on | exact | same-set reordered | **different item set** |
  |---|---|---|---|---|
  | `best-seq-blend-20260728-012030-6d1b7` (λ0.7) | **LOCAL-HUB** | **8/12** | 1 | 3 |
  | `blend-gru-markov-content-proj` (**MMR OFF**) | `worker:2348266` | 6/12 | 4 | 2 |
  | `best-seq-nexttrack-20260726-184115-bf395` (λ0.7) | `snappler:573145` | 6/12 | 1 | 5 |

  **This is NOT caused by the MMR fix**: the MMR-OFF champion diverges too (6/12), and
  its serving output is byte-identical before/after that change. If anything the λ0.7
  hub-trained model has the BEST parity. Two contributions are visible and neither is
  confirmed: (a) the two worse models were **evaluated on the remote worker** while
  serving runs on the hub → cross-machine float/BLAS differences reordering near-ties
  (consistent with the 4/12 pure reorderings on the champion), and (b) hub-trained B
  still diverges on 4/12, so hardware is NOT the whole story — suspect the batched-eval
  vs single-query scoring path. **The `different item set` rows are the concerning
  class** — a genuinely different item entering the top-10 is not tie-breaking. NET: a
  promoted model's leaderboard row is still not exactly reproducible at inference. Do
  not treat serving output as a proxy for a stored eval metric, in either direction,
  until this is diagnosed.
- **THE REMOTE WORKER HAS NO POSTGRES RECONNECT — ANY TRANSIENT DROP KILLS IT PERMANENTLY,
  SILENTLY (2026-07-30, observed TWICE in one day).** Symptom in
  `~/lensing-worker/worker.log`: `Error: claim queued run` / `Caused by: connection
  closed`, after which the process is **gone** and queued runs simply sit unclaimed —
  nothing in the UI, no alert, no failed run row. The second occurrence was NOT a
  database outage (the `spotify-nexttrack-postgres` container had been up 33h and the hub
  was serving 200s), so a transient tailscale/idle-timeout blip is enough. **Consequence
  for campaign discipline: `GET /api/runs` showing runs stuck at `queued` means CHECK THE
  WORKER PROCESS FIRST** (`ssh 100.107.96.30 'pgrep -af "lensing-server worker"'`), and
  re-verify it is alive immediately BEFORE POSTing a batch, not just at design time.
  Relaunch: `HOSTNAME=snappler nohup bash ~/lensing-worker/run_worker.sh
  >>~/lensing-worker/worker.log 2>&1 </dev/null &` in one ssh, then verify in a SEPARATE
  ssh (look for the `connected (...); hub ...` line — note the old `connection closed`
  lines sit ABOVE it and are not a current fault). Proper fix: run it under systemd with
  `Restart=always`, or add reconnect/retry around `claim_queued_run`.
- **REGISTRY VALIDATION REJECTS UNDECLARED HYPERPARAMS, AND THE CHAMPION'S STORED SET IS
  NOT POSTABLE VERBATIM (2026-07-30c).** `lensing-core/src/registry.rs:162` enforces
  `unknown hyperparameter {k}`, so (a) a NEW hyperparam needs a `[[predictors.params]]`
  block in `registry.toml` **plus a gated server restart** before any run can set it, and
  (b) round-tripping a promoted model's `hyperparams.json` straight into `POST /api/runs`
  **FAILS** — `seq-blend` stores six GRU knobs that are deliberately not registry-exposed
  (`bidirectional`, `dropout`, `num_layers`, `residual`, `tau`, `val_fraction`). Filter to
  the declared set; those six sit at `seq_blend.DEFAULTS` on the champion, so omitting
  them preserves the config (verify this before claiming bit-identity).
- **THE LAN WAS RENUMBERED AND IT SILENTLY KILLED THE REMOTE WORKER (2026-07-30).** The
  hub's LAN IP is now **192.168.88.49** (was 192.168.0.7) and the worker's old LAN
  address **192.168.0.12 is DEAD** (100% packet loss). `~/lensing-worker/run_worker.sh`
  hardcoded `192.168.0.7` for `DATABASE_URL` / `QDRANT_URL` / `PATHFINDER_QDRANT_URL` /
  `--hub-url`, so after the renumber the daemon died with `Error: claim queued run /
  Caused by: connection closed` and **simply stayed down since the 2026-07-26 batch** —
  no alert, nothing in the UI; a campaign's prerequisite check is the only thing that
  catches it. **FIXED PERMANENTLY: the script now uses the TAILSCALE hub `100.107.86.38`
  for all four** (backup `run_worker.sh.bak`), since tailscale survives LAN renumbering;
  reach the worker at **100.107.96.30**, never a LAN address. Verified from the worker:
  PG 5437 OPEN, Qdrant 6337 → 200, hub API 8096 → 200. **The hub `lensing-server` was
  ALSO found down** (nothing listening on 8096; it binds `0.0.0.0:8096` when alive) and
  was restarted at 0 live runs, all 123 prior runs intact. Two further drift items found
  the same day: the worker's **`lensing-server` BINARY** had diverged from the hub's
  (`0880457…` vs `3b7c3172…`) — a check DISTINCT from the predictor-script md5, and a
  stale binary risks a DB/API mismatch rather than a clean failure — and
  `~/lensing-worker/registry.toml` is stale (`b4ef03b6…` vs hub `33b482cd…`), lacking
  `seq-embed`'s eager params: harmless for `seq-dualgru`/`seq-nexttrack`, a live risk for
  any `seq-embed` remote campaign. Note `seq-20260715-131139` IS cached under
  `~/lensing-worker/data/datasets/`, so runs on it do not need the hub archive endpoint.
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
  **REFINED, NOT CONTRADICTED, by the 2026-07-26 λ frontier.** That campaign DID launch
  one training run per λ (10 runs) and the pitfall still stands, sharpened into two parts:
  (a) **for LEARNING the λ curve, retraining is pure waste** — and this campaign PROVED it,
  since both λ=1.0 controls reproduced BIT-IDENTICALLY and the 2026-07-22 offline sweep's
  recall projections matched the native runs 4-for-4 to ~4 dp (0.2103/0.1936/0.1642/0.1097
  → 0.21034/0.19357/0.16422/0.10971), so the cheap offline driver is a VALIDATED screening
  instrument; (b) **native runs are nonetheless REQUIRED to bank a result** — only the
  native eval emits per-row `artist_conc` (the 2026-07-22 offline driver predates it) and
  only on-record runs with stored live-definition metrics can carry a registration or a
  promotion decision. **RULE: SCREEN λ offline from one checkpoint, then spend native runs
  ONLY on the 2–4 points you intend to bank.** The 2026-07-26 batch could have been ~5 runs
  instead of 10 under that rule.
- **Blend eagerness is sourced from the MARKOV bigram leg, NOT the RNN (2026-07-22).**
  On the canonical split the standalone first-order markov has artist_adj@10 0.543
  and the champion blend 0.610, while the single GRU/LSTM sit at ≈0.345 (the
  LEAST-eager, HIGHEST-mood_coh models). The blends buy their recall@10 lead by being
  artist-adjacent (bigram-like) — so any future de-eagering should target the Markov
  leg / the fusion weights, not the neural model. MMR eval-re-rank (λ) is a strong
  post-hoc de-eager knob; the anti-eager TRAIN regularizer (`eager_beta`) is
  **REFUTED on `seq-dualgru` across 5 margins × a 60× dose range (8 runs,
  `2026-07-30-eager-margin-calibrated-refutation.md`)** — not a weak knob, a
  **MIS-SPECIFIED** one. It hinges on `cos(pred, x_current)`, but a hedged
  InfoNCE prediction's absolute cosine to the current track **never exceeds
  ~0.6** (bit-identical loss traces to 16 sig-digits across all 14 epochs at
  m=0.70/β=35 and m=0.8304/β=92 ⇒ hinge exactly 0), so every margin calibrated
  from the real-transition tail (p75 0.584, p90 0.830) is **INERT at any β up to
  the schema cap of 100**. At m=0 it fires, but eagerness and relevance are then
  the SAME mechanism: β=3 → music@10 0.400 (floor 0.430 breached), β=12 → conc
  0.134 but 20/1431 hits and music@10 0.358. Only unexplored band: **m=0,
  β ∈ [0.2, 3]** (a factor of 15, bounded above by degeneration and below by no
  effect). LESSON: eagerness is a **RANKING** property — an absolute-cosine
  hinge cannot express it; penalize `cos(pred,x_t) − cos(pred,x_{t+1})` or a
  same-artist top-k mass term instead. Holisticness metrics are
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

_**2026-07-26 (post λ-frontier) — MMR FLOOD CURATED; the group is now a 9-member, THREE-WEIGHT-SET shortlist on ONE live metric generation (best-model-selector; 16 exclude keys covering 9 runs, 1 UNEXCLUDE, ZERO pin/unpin changes; nothing promoted, nothing deleted). The "ranks on two dead metric generations" defect recorded this morning is CLOSED — verified independently below.**_

_**(1) METRIC-GENERATION AUDIT — CLOSED, VERIFIED.** Re-derived the live 4-factor crown (`max(mood_coh,0) × ild × (1−max(artist_adj,artist_conc)) × clamp((music@10−0.34)/0.66,0,1)`) from each run's own stored facets for all 43 runs carrying `holisticness_at_k`: **32 reproduce their stored value to the last digit** (the orchestrator's rescore set) and the **9 that cannot be re-derived are ALL already excluded** (`88a4c`, `e9cfb`, `a95c5`, `8d214`, `12fe6`, `4324f`, `7bab3`, `a6423`, `bd5b5` — DB-only remote runs with no `artist_conc_at_k`). So for the first time the rankable field is generation-uniform and the group's rank ORDER carries information. Note the whole seated spread is 0.0342→0.0083; the shared-weights (eval-time) half-widths measured today (0.000382–0.002476) DO resolve the top of it, unlike the ±0.0018 family floor. Two exceptions to the "uniform generation" claim are worth naming: the two G2-defective controls (`6f5fb`/`c4968`) also lack `artist_conc_at_k` and were seated at ranks 8/12 — both are now excluded (see (2))._

_**(2) EXCLUDES APPLIED — 9 runs, each by BOTH keys (run_id + auto-promoted model name) per the `88a4c` precedent, in ONE `PUT`:**_
- _`run-20260726-184115-1dc77-seq-blend` / `best-seq-blend-20260726-184115-1dc77` (A4 λ0.3, H 0.031626 — **was rank 2**) — **TIER-3 REJECTED**: Δrecall **−0.102** CI<0 (157/1431 hits vs its own λ=1.0 control's 303). Its 0.10971 squeaks past the 0.108 floor, so the disqualifier is the paired loss, not the floor: it is a deliberately crippled operating point of the champion, frontier-mapping only._
- _`run-20260726-184115-7a2f7-seq-blend` / `best-seq-blend-20260726-184115-7a2f7` (A3 λ0.5, H 0.023576, recall 0.16422 — was rank 9) — **TIER-3 REJECTED**: Δrecall −0.047, outside the pre-registered −0.03 gate._
- _`run-20260726-175258-6f5fb-seq-nexttrack` / `best-seq-nexttrack-20260726-175258-6f5fb` (was rank 8) and `run-20260726-175258-c4968-seq-blend` / `best-seq-blend-20260726-175258-c4968` (was rank 12) — **G2-DEFECTIVE, NOT CODE-COMPARABLE**: hub-stored metrics lack `artist_conc_at_k` (the `(deleted)`-image stale-binary defect), so their H cannot be re-derived or audited; both were discarded by the campaign and superseded by the re-gated `9a0b2` / `1c361` at identical values. Incomplete metrics must not hold seats._
- _`run-20260726-161429-bdcbc-seq-nexttrack` / `best-seq-nexttrack-20260726-161429-bdcbc` (was rank 6) — **REDUNDANT BIT-IDENTICAL REPRODUCTION** of the λ=1.0 h256 baseline (recall 0.12299 = 176/1431, live H 0.024538, identical to `9a0b2`). Same-day precedent: this morning's pass excluded `97c1e` and kept ONE representative. `9a0b2` is kept as that representative because it is the λ-frontier batch's own gate run (same host `snappler:573145` as every seated λ arm → the correctly paired control) and is the row the CROWN board cites._
- _`run-20260723-014711-1c1a3-seq-blend` / `best-seq-blend-20260723-014711-1c1a3` (this morning's rank-1 seat) — **REDUNDANT BIT-IDENTICAL REPRODUCTION** of the champion (recall 0.21174, live H 0.008603, identical to `1c361`). Superseded by `1c361`, the λ-frontier's A0 control on the same host as the seated λ arms and the champion row the crown board now cites. **CONSEQUENCE for the standing re-promotion advice (2026-07-25 note (2b), 2026-07-26 note (3)): the target is now `run-20260726-183333-1c361-seq-blend`, not `1c1a3`.** (Harmless if the user promotes onto `1c1a3` anyway — pin beats exclude.)_
- _`run-20260723-012311-79c7b-seq-mood` / **`mood-session`** — **DEGENERATE**: recall 0.02376 = **34 of 1,431**, live H **0.000000**. `seq-mood` does not target next-track by design. It was NOT seated before (the slots were full) but sits directly in the backfill path of the four evictions above, and its sibling `b9172` has been excluded on exactly this ground since 2026-07-25. **The model NAME is excluded too** — deliberately stronger than the `b9172` precedent, so the same generator cannot re-enter through a future re-promotion. `mood-session` remains promoted and directly servable by name for its intended mood-session use case; nothing was deleted._
- _`run-20260722-165345-34614-seq-popularity` — **DEGENERATE FLOOR**: recall 0.00070 = **1 of 1,431**, live H 0.000000. Left alone on 2026-07-22 only because it could not backfill; after the evictions it could (it would have taken a tail seat). No auto-promoted alias exists, so the run_id key is sufficient and also prevents the auto-promotion._
- _`run-20260722-165345-047cc-seq-markov` — **BELOW THE STANDING 0.108 RECALL FLOOR** (0.10692 = 153/1,431, i.e. 0.0161 below the canonical baseline 0.12299, just outside the ±0.015 band); live H 0.004409. **This is the pass's costliest call and it is stated as a trade-off, not a clean win:** `047cc` is the ONLY live-scored run from an otherwise unrepresented family (`seq-markov`) and it is NOT degenerate — it ranks second-from-bottom on H, so it cannot be gaming the correctness-free composite the floor was written to defend against. It was excluded anyway because (a) it misses the derived bright line, by 0.0011 — a margin inside the line's own noise, exactly the "cuts between indistinguishable runs" caveat logged on 2026-07-25; (b) the floor has been applied without exception (`12fe6` 0.10273, `0df6f` 0.10273), and a one-off waiver for diversity would convert a rule into a preference; (c) `POST /api/best-models/predict` is a no-op for this domain, so the correlated-failure argument for a diversity seat buys nothing operationally here. **The honest fix is a stronger markov-family run, not a weaker seat.** Un-exclude on sight if `seq-markov` is ever re-run at ≥0.108._

_**(3) ONE STALE EXCLUSION REVERTED (the only one that had gone stale).** **UN-EXCLUDED `run-20260722-165345-28dbb-seq-blend`** (R+M 2-leg `blend-gru-markov`, recall@10 **0.17121** = 245/1,431, live H 0.008259) — its 2026-07-22 reason was verbatim "REDUNDANT: R+M family already represented (ranks 3-6)", and that premise is now **void**: the recall-era R+M entries it deferred to are all structurally gone, so the exclusion was suppressing the family it was written to de-duplicate. It re-enters at rank 9 on merit, displacing nobody, comfortably above the recall floor, on the canonical split and on the live metric generation. This is what makes the group **three distinct sets of trained weights instead of two** (see (5)). All other 46+ exclusions were re-reviewed and PRESERVED — each original reason still holds (notably `run-20260718-193111-a781c`, the τ0.10 sub-noise fluke, still unconfirmed; `be11b` h425 / `7646d` h454 / `353b5` β0.05 are still tie-only controls whose live H sits inside the baseline cluster; `ad53f` / `7bab3` are still paired-Δ CI<0 WORSE than baseline)._

_**(4) THE B3 JUDGMENT CALL — B3 (`bf395`) is LEFT AT RANK 1, deliberately.** TIER-2 PRICED (H 0.034171, the highest measured anywhere; recall 0.11461, Δrecall −0.008492 [−0.016073,−0.001398] CI<0) vs the best TIER-1 FREE arm B4 (`20d9b`, H 0.030355, recall 0.12020, Δrecall straddles 0), now rank 2. Reasoning: (a) B3 is a legitimate operating point — above the mandatory 0.108 floor, not degenerate, the report's own crown-board rank 1; (b) the group's declared `primary_metric` IS holisticness@10, and ΔH(B3,B4) = +0.003816 resolves against the measured shared-weights half-widths (≈1.5–10×) as well as the ±0.0018 family floor, so B3 genuinely leads the board it is ranked on; (c) **the only instrument that could demote it is EXCLUSION** (pins add, they never reorder — 2026-07-26 note (4)), and excluding the metric leader to express a different objective would silently substitute the curator's utility function for the server's ranking metric; (d) the correct instrument for "which arm should we actually run" is PROMOTION, which is the user's call and blocked on R4. **READ THE GROUP ACCORDINGLY: rank 1 here means highest holisticness@10, NOT "recommended". The promotion-recommended arm is rank 2 (`20d9b`, B4), and the report's TIER-2/TIER-1 split is the authority on that, not the rank order.**_

_**(5) RESULTING GROUP — 9 of 12 seats, every seat auditable, all on canonical `seq-20260715-131139`, all n_test 1,431, ALL on the live metric generation.** Ranks: 1 `best-seq-nexttrack-20260726-184115-bf395` (H 0.034171, recall 0.11461, TIER-2 PRICED); 2 `…-20260726-184116-20d9b` (0.030355, 0.12020, **TIER-1 FREE crown candidate**); 3 `…-184115-147a6` (0.029458, 0.11950, TIER-1); 4 `…-184115-e1539` (0.026761, 0.12089, TIER-1); 5 `…-20260726-183333-9a0b2` (0.024538, 0.12299 = 176/1431, the λ=1.0 h256 baseline); 6 `best-seq-blend-20260726-184115-36edb` (0.015289, 0.19357, TIER-2 on the champion); 7 `…-184115-6cb38` (0.010435, 0.21034, TIER-1 on the champion); 8 `…-20260726-183333-1c361` (0.008603, **0.21174 — the RECALL CHAMPION is back in the auto group**, on merit, no pin needed); 9 `best-seq-blend-20260722-165345-28dbb` (0.008259, 0.17121, R+M 2-leg). Verified stable across a forced `POST /api/best-models/recompute` (identical membership and order). **No suspicious metrics anywhere in the candidate field:** n_test uniformly 1,431, top H 0.0342 is nowhere near 1.0, no leakage signature, no tiny-split artifact, and every seated H is reproducible from its own stored facets. **Monoculture, stated precisely:** by predictor the group is 5 `seq-nexttrack` / 4 `seq-blend`, but by WEIGHTS it is only **THREE distinct trained models** — `gru-infonce-h256` (ranks 1-5, five eval-time λ/pool policies over ONE weight set), `blend-gru-markov-content-proj` (ranks 6-8, three λ policies over ONE weight set) and `blend-gru-markov` (rank 9) — because λ and pool are EVAL-ONLY knobs. That is inherent to a campaign whose whole finding is an eval-time lever, and it is the sharper way to read the diversity of this group. **NO PINS ADDED and none removed:** the recall champion no longer needs one (it seats itself at rank 8), a pin cannot demote B3 or reorder anything, and no additional family can be seated without breaching the recall floor (see (2), `047cc`). The 3 existing pins (`blend-gru-markov-content-proj`, `blend-gru-markov-content`, `gru-lstm-pca128`) stay in place and stay INERT — their promoted runs pre-date the holisticness suite, so they remain unrankable, and they will bite the moment each is re-promoted onto a scored run. **Excluded list now 75 keys** (prior 60 + 16 new keys − 1 unexclude). **Is this a defensible top-12? It is a defensible top-9 and not a top-12** — the remaining 3 seats cannot be filled honestly: every other live-scored candidate is either below the recall floor, a paired-Δ-CI<0 loser, a bit-identical duplicate of a seated model, or unauditable. The under-fill is metric coverage plus discipline, not over-curation._

_**(6) FLAGGED, LEFT ALONE.** (a) **`test-r`** (stray promoted `seq-nexttrack` on `af11a`, no notes) — still recommended for USER deletion; still doubly blocked (its run is excluded and its stored H is not on the live generation). (b) **The 3 inert pins** — see (5); unpinning would destroy a correct standing instruction for zero benefit. (c) **Promotion is OWED, not done** (out of remit, and R4 has not run): promote `gru-infonce-h256-mmr-l07-pool50` onto `run-20260726-184116-20d9b` after R4, and re-promote `blend-gru-markov-content-proj` onto `run-20260726-183333-1c361` — the group's anchors then appear under their own names instead of `best-seq-*` aliases. (d) **The forward hazard logged this morning is GONE**: with the field generation-uniform, a good future run no longer ranks below stale inflated values. The NEW hazard is the opposite one — with 5 of 9 seats held by eval-time policies over a single weight set, a genuinely new ARCHITECTURE will enter at the bottom of the board unless its crown gain clears the λ-policy stack. (e) This pass discharges the "best-model-selector (DO FIRST, group is FLOODED)" follow-up below._

_**2026-07-28 — `markov_gate` CONFIRM-CAMPAIGN CURATED (best-model-selector; 8 exclude keys covering 4 runs, ZERO pins/unpins/unexcludes; nothing promoted, nothing deleted; champion UNCHANGED). Resulting group = 11 of 12 seats, all canonical split, all on the live metric generation.**_

_**(1) THE FOUR EXCLUDES — the batch's CONTROLS and its SECOND-SPLIT arms, each by BOTH keys (run_id + auto-promoted alias) per the `88a4c` precedent.** The six-run confirm campaign (`2026-07-27c-nexttrack-markov-gate-confirm.md`) ran entirely on the hub, so the recompute triggers fired normally and the group was FRESH (`updated_at` 01:27:28 = the last run's finish) — no staleness this time. It had auto-seated 4 of the 6 runs. Applied in ONE `PUT /api/best-models`:_
- _`run-20260728-004444-b32ea-seq-blend` / `best-seq-blend-20260728-004444-b32ea` (C1, the preflight bit-identity control; was rank 12) — **REDUNDANT BIT-IDENTICAL REPRODUCTION** of the seated λ=1.0 champion row `run-20260726-183333-1c361-seq-blend`: EVERY stored metric agrees to the last digit (recall 0.211740042 = 303/1431, H 0.008603039046494065, mrr, all five crown facets). Note it is a CROSS-HOST reproduction (`1c361` on remote `snappler:573145`, `b32ea` on the hub) and still bit-identical — a clean datapoint for the withdrawn cross-host-offset question. Same precedent line as `1c1a3` / `bdcbc` / `97c1e` / `82a8f` / `d593e`: keep ONE representative of a weight-set+policy, and keep the one the crown board cites._
- _`run-20260728-011245-bfc67-seq-blend` / `best-seq-blend-20260728-011245-bfc67` (C3, the λ0.7/pool200 ungated control; was rank 9) — **REDUNDANT REPRODUCTION** of the seated `run-20260726-184115-36edb-seq-blend` (same weights, same eval policy λ0.7/pool200). Correction to the "identical H" reading: it is identical to 6 s.f. but NOT bit-identical — H 0.015288558 vs 0.015289259, a 7.0e-7 gap arising ONLY in `ild_at_k` (1.1e-5) and `mood_coh_at_k` (8.8e-6); recall, mrr, artist_adj, artist_conc, artist@10, genre@10 and music@10 all agree to the last digit, i.e. **the top-k lists are the same and the residual is float-aggregation noise in two content-space facets**. That gap is ~1/500 of the SHARPEST measured shared-weights hw(ΔH) (0.000382), so the duplicate verdict is unaffected._
- _`run-20260728-010658-39e20-seq-blend` / `best-seq-blend-20260728-010658-39e20` (A2, gate arm) and `run-20260728-010113-e2599-seq-blend` / `best-seq-blend-20260728-010113-e2599` (C2, its control) — **CROSS-SPLIT, VALIDATION-ONLY.** Both are on the 0.64-cold second split `seq-20260718-211238`, not the canonical 0.55-cold `seq-20260715-131139` that every other seat is on; the campaign itself says ds2 is to be read "only through the paired Δ" because its absolute numbers are lower BY CONSTRUCTION. Their raw H (0.008414 / 0.003392) is therefore not comparable to the board. Neither was seated, but BOTH sat directly in the backfill path of the two control evictions above — `39e20` at 0.008414 was the single highest un-excluded free candidate and would have taken a freed seat immediately (the standing "exclusion backfills from the same field" lesson, 2026-07-22). Exact precedent: the three second-split validation runs excluded on 2026-07-18 evening (`50e1d`/`cf1dd`/`93bad`). Both already carried auto-promoted aliases from a mid-batch recompute, so both keys were needed. **Nothing about the gate's ds2 CONFIRMATION is weakened by this** — the second split's job is the paired Δ (+0.01957 [+0.01048, +0.03005] CI>0), which lives in the report, not in the shortlist._

_**(2) THE TWO GENUINE ENTRANTS ARE KEPT ON MERIT, AND NEITHER IS PINNED.** A3 `run-20260728-012030-6d1b7-seq-blend` (gate + MMR λ0.7/pool200) seats itself at **rank 2, H 0.030980, recall@10 0.21314** — a recall TIE with the champion (paired +0.00140 [−0.01188, +0.01468]), 3.6× its crown, above the mandatory 0.108 floor, and it PASSES the registered MMR-validity gate (ild 0.56770 vs the ungated 0.40637). A1 `run-20260728-005349-507bb-seq-blend` (gate alone) seats itself at **rank 7, H 0.018699, recall@10 0.23201 = 332/1431 — the best recall on record**, paired CI>0 on two disjoint splits with MRR up on both. **No suspicious-metric flag on either:** n_test 1,431 like every other seat, nothing near-perfect (A3's H is BELOW the incumbent crown leader's 0.034171, i.e. inside the established band rather than a breakout), the gate reads `build_train_transitions` (train-only, leak-free by construction — verified in `predictors/seq_blend.py:206`), and the mechanism is corroborated by warm buckets whose top-k lists are bit-identical. **NO PINS ADDED, deliberately:** (a) both seat themselves with margin — A3's nearest un-excluded competitor is 0.030355 and A1 has 4 empty/low seats beneath it, so a pin is inert insurance (2026-07-26 precedent); (b) the only pinnable name today is the anonymous auto alias `best-seq-blend-20260728-{012030-6d1b7,005349-507bb}`, and pinning it would entrench, under a machine name, arms whose REGISTRATION+PROMOTION is an explicit pending user decision (suggested names `blend-gru-markov-content-proj-gated` / `-gated-mmr-l07`); (c) a pin survives recomputes and would keep A1 seated even if the OWED 3-seed confirm (follow-up 4) failed — the record's own bar for a champion row is seed-confirmation, and A1 is single-seed (1337) with a bucket boundary that was chosen with test knowledge (the report's own §Findings-7 caveat). **STANDING TRIGGER, recorded so the next curator does not have to re-derive it:** A1 is rank 7 with SIX arms above it; one more crown batch of the 2026-07-26 shape (which produced 4 arms above H 0.0187 in a single launch) could evict the best-recall model on record from the shortlist, because the crown contains no correctness term. **Before the next crown campaign, either promote A1 under a real name (the proper anchor) or pin it — re-curate on sight if it drops out.**_

_**(3) DIVERSITY, STATED PRECISELY — nothing changed, and the sharper read is WEIGHT SETS, not predictors.** By predictor the group is 5 `seq-nexttrack` / 6 `seq-blend`. By trained weights it is still only **THREE distinct models**: `gru-infonce-h256` (ranks 1,3,4,5,6 — five eval-time λ/pool policies over ONE weight set), `blend-gru-markov-content-proj` (ranks 2,7,8,9,10 — now FIVE eval-time policies over ONE weight set: λ1.0, λ0.9, λ0.7, gate+λ1.0, gate+λ0.7) and `blend-gru-markov` (rank 11). **`markov_gate` is an EVAL-TIME knob exactly like `mmr_lambda`** — verified in `predictors/seq_blend.py`: it only sets `w_m = 0.0` inside `blend_score_fn` and the training path is untouched, which is why C1 reproduced the champion bit-identically. So A1/A3 are two new OPERATING POINTS of the champion, not two new models; the group did not gain a family. **No diversity pin is possible or warranted:** `seq-markov`'s only live-scored run (`047cc`, 0.10692) is still below the 0.108 floor, `seq-dualgru` is refuted with every live-scored arm at or below the baseline, `seq-mood`/`seq-popularity` are degenerate and name-excluded, and the LSTM pin `gru-lstm-pca128` is still inert (no holisticness-carrying run). Pinning a clearly worse model for diversity's sake remains unjustified._

_**(4) RESULTING GROUP — 11 of 12 seats, verified stable across a forced `POST /api/best-models/recompute` (identical membership and order).** 1 `best-seq-nexttrack-20260726-184115-bf395` (H 0.034171, recall 0.11461); 2 `best-seq-blend-20260728-012030-6d1b7` (**0.030980, recall 0.21314 — A3, gate+λ0.7**); 3 `…-20260726-184116-20d9b` (0.030355, 0.12020); 4 `…-184115-147a6` (0.029458, 0.11950); 5 `…-184115-e1539` (0.026761, 0.12089); 6 `…-20260726-183333-9a0b2` (0.024538, 0.12299); 7 `best-seq-blend-20260728-005349-507bb` (**0.018699, recall 0.23201 = 332/1431 — A1, gate alone, best recall on record**); 8 `…-20260726-184115-36edb` (0.015289, 0.19357); 9 `…-184115-6cb38` (0.010435, 0.21034); 10 `…-20260726-183333-1c361` (0.008603, 0.21174 — the champion row); 11 `best-seq-blend-20260722-165345-28dbb` (0.008259, 0.17121, R+M 2-leg — it BACKFILLED into the seat freed by the C1 eviction, which is the 2026-07-26 un-exclusion continuing to pay off). All 11 on canonical `seq-20260715-131139`, all n_test 1,431, all on the live 4-factor metric generation. **The 12th seat cannot be filled honestly** — the only remaining un-excluded rankable candidates are the two ds2 cross-split runs just excluded; every other candidate is below the recall floor, a paired-Δ CI<0 loser, a duplicate, or unauditable. **Excluded list now 83 keys** (prior 75 + 8). Rank 1 still means highest holisticness@10, NOT "recommended" — see the 2026-07-26 note (4)._

_**(5) FLAGGED, LEFT ALONE.** (a) **The 3 inert pins** (`blend-gru-markov-content-proj`, `blend-gru-markov-content`, `gru-lstm-pca128`) — unchanged; still correct standing instructions that will bite when each is re-promoted onto a holisticness-carrying run. (b) **The prior 75 exclusion keys were re-reviewed and ALL PRESERVED** — zero had gone stale; notably `047cc` (`seq-markov`, still 0.0011 below the floor, un-exclude on sight if the family is re-run at ≥0.108) and `a781c` (τ0.10, still unconfirmed). (c) **PROMOTION IS OWED AND WAS NOT DONE** (explicitly out of remit and pending the user's decision): A1 → `blend-gru-markov-content-proj-gated`, A3 → `blend-gru-markov-content-proj-gated-mmr-l07`; also still owed from 2026-07-26, `gru-infonce-h256-mmr-l07-pool50` → `run-20260726-184116-20d9b` and `blend-gru-markov-content-proj` → `run-20260726-183333-1c361`. Until then the group's five most interesting seats are held under anonymous `best-seq-*` aliases. (d) **`test-r`** — stray promoted `seq-nexttrack` on `af11a`, still recommended for USER deletion, still doubly blocked. (e) **The stale remote worker `snappler:573145`** (no `markov_gate` in its checkout) is an orchestrator action, not curation, but it is a live curation hazard: a `queue:true` re-run of A1/A3 would silently return UNGATED numbers under a gated name and could seat a mislabelled duplicate._


_**2026-07-31 — WALK-SURFACE RE-CURATION: the DEPLOYED dual-tower is now IN the group, the two dual-tower exclusions are REVERTED AS STALE, and two campaign floods are evicted (best-model-selector; 14 new excludes, 2 UNEXCLUDES, 3 NEW PINS; nothing promoted by hand, nothing deleted, no restart). Resulting group = 15 entries (12 auto + 3 pinned), verified IDENTICAL across a forced `POST /api/best-models/recompute`.**_

_**(1) STALENESS AND THE AUTO-PROMOTION IT FORCED — DISCLOSED, NOT AVOIDABLE.** The group was frozen at `updated_at` **2026-07-28T01:39:56** (the 6th confirmation of the `spawn_recompute` remote-batch trigger gap), so it had absorbed NONE of the 2026-07-30 artist-cap campaign, NONE of the 2026-07-30 `eager_beta` batch and NONE of the 2026-07-31 split-2 batch — 14 rankable runs invisible to it. Any curation `PUT` recomputes, and a recompute auto-promotes unpromoted selections, so **two auto-promotions were unavoidable and are recorded here: `best-seq-blend-20260730-235524-93d38` (A1r, `markov_gate`+`artist_cap=3`) and `best-seq-blend-20260730-234635-a6252` (A2, cap 5)** — exactly the two runs the 2026-07-30c campaign nominates as deployment candidates, and the only two created (promoted-model count 83 → 85). No other run was promoted; the pending promotion decisions listed in the 2026-07-28 note are still OWED and still the user's call._

_**(2) THE TWO STALE EXCLUSIONS REVERTED, AND THE DEPLOYED MODEL PINNED — this is the substance of the pass.** `run-20260725-143502-3e5d4-seq-dualgru` (D-LC-f0, latent/cummean, `fusion_layers=0`) and `run-20260725-143502-18b4a-seq-dualgru` (D-LL-f0, latent/latent) were excluded on 2026-07-26 with the verbatim reason "three `fusion_layers=0` cells of a REFUTED family all returning the same verdict… none is needed", and the pass explicitly recorded "a distinct-family seat for `18b4a` was available at no measurable recall cost, and was declined". **That premise is now VOID by the 2026-07-31b standing rule ("NO FAMILY MAY BE CLOSED ON ONE-SHOT METRICS ALONE… The dual-tower family is hereby REOPENED")**, and the matched-sample walk head-to-head (40 identical held-out sessions, paired bootstrap 2000/rng1337, `tools/walk_headtohead.py`) makes these two arms the WINNERS on the stated objective: `3e5d4` Δstride_err **−0.255 CI<0**, Δgenres **+1.50 CI>0**, Δdrift **+0.035 CI>0**, Δvibe +0.001 (straddles = HELD, not traded); `18b4a` Δvibe **+0.030 CI>0**. So: **UNEXCLUDED both**, and **PINNED both by model name** (`best-seq-dualgru-20260725-143502-3e5d4`, `best-seq-dualgru-20260725-143502-18b4a`). The pin — not merely the un-exclusion — is the correct instrument for three reasons: (a) **`3e5d4` IS THE MODEL THE SHOWCASE SERVES** (`clients/infinite-playlist`, exported `dualgru.onnx`, version `4a7054eb`, torch-parity 1e-7 across T=1..40), and a shortlist of record that omits what is actually deployed is misleading by construction; (b) their seats rest on WALK evidence, which `holisticness@10` cannot see — at stored H 0.021520 / 0.023785 they sit at ranks 10 / 9 and the very next campaign flood would evict them (five consecutive floods are on record); (c) pins are additive and never reorder, so this adds the two arms WITHOUT displacing any auto seat and without expressing an opinion the declared metric does not hold. **NB the family reopening does NOT rehabilitate the six 2026-07-25 degenerate arms** (`88a4c` 0.06639, `e9cfb` 0.08665, `a95c5` 0.09713, `8d214` 0.09154, `12fe6` 0.10273, `4324f` 0.05800): all are below the standing 0.108 recall floor AND their stored H (0.133–0.197) is on the RETIRED 3-factor generation with no `artist_conc_at_k` (re-verified today) — un-excluding any would seat cross-generation inflation at ranks 1–6. They stay excluded, and the walk evidence covers only the two `f0` arms above._

_**(2b) THIRD PIN — `best-seq-blend-20260722-165345-28dbb` (R+M 2-leg `blend-gru-markov`, H 0.008259, recall@10 0.17121 = 245/1,431), a WEIGHT-SET diversity pin, stated as a trade-off.** Measured, not assumed: with the two cap entrants seated, the 12 auto slots end exactly at `1c361` (0.008603) and `28dbb` fell out at 13th. It is the ONLY remaining seat whose trained weights are neither the champion projection (6 seats), the h256 GRU (5 seats), nor the dual-tower (2 pins) — the same family it was UN-EXCLUDED for on 2026-07-26. **The honest margin: it is 0.007030 of H below the last auto seat, which is resolvable (≈3–18× the measured shared-weights hw(ΔH) 0.000382–0.002476), so this is NOT a within-noise pin** — it is bought with the fact that it beats five seated arms on recall by ~+0.05 and is not degenerate (comfortably above the 0.108 floor, and it ranks near-LAST on H, so it cannot be gaming the correctness-free composite the floor defends against). **Deliberately NOT pinned by contrast: the champion row `1c361`** (0.008603, recall 0.21174) — it survives at auto rank 14 anyway, and its weight set already holds 6 seats, so no pin is warranted; the RIGHT fix for the champion's naming is the OWED re-promotion of `blend-gru-markov-content-proj` onto a holisticness-carrying run, which would make the long-inert 2026-07-18 pin bite at last._

_**(3) THE 14 EXCLUDES — two campaign floods plus two damaged/redundant controls. None had an auto-promoted alias yet** (the staleness meant no recompute had ever seen them), so the run_id key alone is sufficient AND also prevents their auto-promotion — verified after the fact: only `93d38`/`a6252` gained aliases._
- _**Cross-split, validation-only (4)** — all on the 0.64-cold second split `seq-20260718-211238`, not the canonical 0.55-cold `seq-20260715-131139` every other seat is on; their absolute H is lower BY CONSTRUCTION and is to be read "only through the paired Δ": `run-20260731-001833-37102-seq-blend` (cap 3, H 0.030529 — would have seated at rank 5), `-758dd` (cap 5, 0.023929), `-8a90d` (cap 0 gated, 0.008414), `-23767` (ungated control, 0.003392). Exact precedent: `39e20`/`e2599` (2026-07-28) and `50e1d`/`cf1dd`/`93bad` (2026-07-18). **Nothing about the artist cap's second-split hardening is weakened by this** — ds2's job is the paired Δ (both pre-registered legs held), not the board. Note these four ARE legitimately retrained on split-2's train, so they are not touched by the 2026-07-31 leak warning about re-using split-1 models on ds2._
- _**The `eager_beta` batch (8), REFUTED or DEGENERATE, per the campaign's own verdict** (`2026-07-30-eager-margin-calibrated-refutation.md`): `run-20260730-175402-cd006-seq-dualgru` (β0 control), `-0d310` (β92/m0.8304), `run-20260730-184556-b08c1` (β35/m0.70) — **three PROVABLY INERT arms that are bit-identical reproductions of the now-pinned `18b4a`**: identical recall (0.11880 = 170/1,431) and identical facets to 5 dp (ild 0.45061, conc 0.37059, music 0.45535, mood 0.47985), i.e. keeping them would seat three more copies of one weight set (standing "keep ONE representative" precedent: `1c1a3`/`bdcbc`/`97c1e`/`82a8f`/`d593e`/`b32ea`); `-47d7b` (β57/m0.5843, H 0.023771) and `run-20260730-175402-7c304` (β19/m0.5843, 0.023751) — same recall, H within 1.4e-5 of that control, i.e. redundant near-duplicates; `run-20260730-181405-84da4` (β9/m0.4, 0.021602 at recall 0.11461) — a REFUTED arm that scores BELOW its own control on the ranking metric, so it is dominated on both axes; `-53088` (β3/m0, recall 0.05381 = 77/1,431) and `-0bc1a` (β12/m0, recall **0.01398 = 20/1,431**) — **DEGENERATE, far below the 0.108 floor** (`0bc1a` is the campaign's "best-facet, worst-H" arm, exactly the anti-degenerate case the floor exists for). Excluding the tail matters as much as the head here: the standing lesson is that freeing a seat backfills from the SAME field._
- _**Redundant / damaged cap-campaign controls (2):** `run-20260730-235524-0ec1d-seq-blend` (C0r, cap 0 + gate, H 0.018699174464215984, recall 0.23201 = 332/1,431) — **bit-identical to the seated `507bb`** (every stored metric to the last digit; that identity is the campaign's own preflight gate), so keep ONE representative and keep the one the crown board cites. `run-20260730-234635-505ae-seq-blend` — **DAMAGED PROVENANCE: it succeeded with `markov_gate=true` in its hyperparams but returned UNGATED metrics** (H 0.008603039046494065 / recall 0.21174 / conc 0.69165, identical to the ungated champion row `1c361`) because the worker ran a stale checkout. It is both a mislabelled row and a duplicate of a seated one; excluding it also stops a misleading `best-seq-blend-…-505ae` alias from ever being minted. Its orphaned sibling `run-20260730-234635-7a297` (stuck at `running`, `/stop` 409s, `DELETE` refuses) needs **no delta** — a null-metric run is never selectable._

_**(4) THE "MMR WAS EVAL-ONLY" QUESTION — ADJUDICATED, AND IT COSTS NOBODY A SEAT.** Audited: **7 of the 11 previously-seated members ranked on `mmr_lambda < 1.0`** (`bf395` λ0.7, `6d1b7` λ0.7+gate, `20d9b` λ0.7/pool50, `147a6` λ0.8, `e1539` λ0.9, `36edb` λ0.7, `6cb38` λ0.9), and until the **2026-07-30b** fix that re-rank shaped the leaderboard while `predict_ranking`/`rank_topk` silently dropped it — so those rankings were, at the time, not reproducible in production. **No exclusion is warranted, because the defect is CLOSED, not outstanding:** `mmr_lambda`/`mmr_pool` are now threaded through `rank_topk`/`predict_ranking` at all five declaring call sites, so eval and serving agree and each stored H is once again an honest measurement of a policy the server will actually apply. Excluding these rows would punish them for a fixed orchestrator bug and would demote the metric leader — which pins cannot do and exclusion should not do while the primary-metric decision is pending with the user. **Two caveats recorded instead of acted on:** (a) an MMR-dependent seat is reproducible in production ONLY while the `spotify_tracks_content_metric` index is reachable — `mmr_rerank_order` **silently no-ops** when `music_vecs is None` and now fetches ~19,402 vectors per predict, so at equal H the `artist_cap` arms (no vectors, ports to a thin client, eval/serving identical BY CONSTRUCTION) are the more robust operating points; (b) all seven passed the registered MMR-validity gate at measurement time (`ild` separated from their own λ=1.0 control). **Metric-generation audit re-run today:** every one of the 15 seats re-derives its stored H from its own stored facets under the live 4-factor definition `max(mood_coh,0)·ild·(1−max(artist_adj,artist_conc))·clamp((music@10−0.34)/0.66,0,1)` — 12 to machine precision, and `3e5d4`/`18b4a`/`28dbb` to 3.6e-08 / 1.7e-07 / 4.2e-09 (the rescore-tool reconstruction residual, ≈1/2000 of the sharpest measured hw(ΔH)). **The group is generation-uniform; its rank ORDER carries information.**_

_**(5) SUSPICIOUS-METRIC CHECK ON THE NEW RANK 1 — FLAGGED, CHECKED, CLEARED.** `93d38` at H **0.048063** is +40.7% over the previous leader (0.034171), far outside any noise band, which is exactly the profile that usually means leakage. It is nonetheless GENUINE: (a) `n_test` 1,431 like every other seat, and **recall goes DOWN, not up** (0.17540 = 251/1,431 vs the gate's 332) — the opposite of a leakage signature; (b) the mechanism is deterministic and understood — a hard per-artist cap moves `artist_conc` 0.59821 → **0.11344**, and H multiplies by (1−eager); (c) it reproduces an off-run measurement computed on bit-identical cached scores **to every digit**, and its H re-derives from its own facets exactly; (d) the effect is hardened on a disjoint retrained split (Δconc −0.52699 CI<0, Δmusic@10 +0.02644 CI>0); (e) `artist_cap` is applied in BOTH `eval_from_scores` and `rank_topk`/`predict_ranking`, so eval and serving cannot diverge. Nothing else in the candidate field is near-perfect; no tiny-`n_test` artifact anywhere._

_**(6) RESULTING GROUP — 15 entries = 12 auto + 3 pinned, all on canonical `seq-20260715-131139`, all n_test 1,431, all on the live metric generation; membership and order verified IDENTICAL across a forced recompute.** 1 `best-seq-blend-20260730-235524-93d38` (H 0.048063, recall 0.17540 — gate+cap3, **new rank 1, auto-promoted by this pass**); 2 `best-seq-blend-20260730-234635-a6252` (0.040056, 0.19846 — gate+cap5, **auto-promoted**); 3 `best-seq-nexttrack-20260726-184115-bf395` (0.034171, 0.11461); 4 `best-seq-blend-20260728-012030-6d1b7` (0.030980, 0.21314); 5 `…-20260726-184116-20d9b` (0.030355, 0.12020); 6 `…-184115-147a6` (0.029458, 0.11950); 7 `…-184115-e1539` (0.026761, 0.12089); 8 `…-20260726-183333-9a0b2` (0.024538, 0.12299); **9 `best-seq-dualgru-20260725-143502-18b4a` (0.023785, 0.11880 — PINNED, walk Δvibe +0.030 CI>0)**; **10 `best-seq-dualgru-20260725-143502-3e5d4` (0.021520, 0.11740 — PINNED, THE DEPLOYED SHOWCASE MODEL)**; 11 `best-seq-blend-20260728-005349-507bb` (0.018699, **0.23201 = 332/1,431, best recall on record**); 12 `…-20260726-184115-36edb` (0.015289, 0.19357); 13 `…-184115-6cb38` (0.010435, 0.21034); 14 `…-20260726-183333-1c361` (0.008603, 0.21174 — the champion row); **15 `best-seq-blend-20260722-165345-28dbb` (0.008259, 0.17121 — PINNED, R+M 2-leg)**. Pins now 6 (3 inert legacy + 3 new); exclusion keys 81 → 95. **READ THE GROUP ACCORDINGLY, and this is now load-bearing: rank means highest `holisticness@10` on a ONE-SHOT top-10; it does NOT mean recommended, and it does NOT mean deployed. What is deployed is rank 10.**_

_**(7) FLAGGED, LEFT ALONE — and (a) and (b) are the user's decisions, not this agent's.** (a) **THE GROUP IS A POLICY MONOCULTURE, and curation cannot fix it: 11 of the 15 seats are eval-time POLICIES over just TWO weight sets** (6 × `blend-gru-markov-content-proj` at λ1.0/λ0.7/λ0.9/gate/gate+λ0.7/gate+cap3/gate+cap5, 5 × `gru-infonce-h256` at λ0.7–1.0), against 4 seats holding genuinely distinct weights (2 dual-tower pins + the R+M 2-leg + the bare champion row). This is structural: `holisticness@10` rewards de-eagering RE-RANKS, which are free to stack on one weight set, so a genuinely new ARCHITECTURE enters at the bottom of the board — which is precisely what happened to the deployed dual-tower and why it needed a pin. The instruments that would fix it are a PRIMARY-METRIC decision (pending with the user; `domain.toml` untouched here as instructed) and PROMOTION, neither of which is curation. (b) **The walk surface is measured for exactly 3 arms** (`3e5d4`, `18b4a`, the shipped GRU) — the two artist-cap arms now holding ranks 1–2 have NEVER been walk-evaluated, so their seats rest entirely on the one-shot metric whose adequacy is under review. A walk head-to-head of `93d38` vs `3e5d4` is the single highest-value missing measurement for this group. (c) **`run-20260726-161801-0df6f-seq-dualgru`** (`fusion_layers=1`, the record's "cheapest real de-eagering") stays excluded on the 0.108 recall floor (0.10273) — but it is the strongest candidate for un-exclusion the moment a walk eval covers it, since the floor is itself a one-shot criterion and the family is now reopened. (d) **`run-20260722-165345-047cc-seq-markov`** stays excluded, still 0.0011 below the floor; un-exclude on sight if `seq-markov` is re-run at ≥0.108 (it remains the only unrepresented predictor family). (e) **The 3 legacy pins stay and stay INERT** (`blend-gru-markov-content-proj`, `blend-gru-markov-content`, `gru-lstm-pca128`) — their runs pre-date the holisticness suite; they are correct standing instructions that bite on re-promotion. (f) **`test-r`** (stray promoted `seq-nexttrack` on `af11a`) — still recommended for USER deletion; still doubly blocked. (g) **The `spawn_recompute` remote-batch trigger gap is now at SIX confirmations** and is the reason every remote campaign needs a manual recompute or arrives invisible; it is a server defect, not a curation problem. (h) **`POST /api/best-models/predict` remains a NO-OP for this domain**, so the diversity arguments above are about the group's honesty AS THE SHORTLIST OF RECORD (UI + agents), not about consensus-prediction failure modes._

_**2026-08-01 — STAGE-STACK DE-FLOOD: the 5 REFUTED arms + the redundant capacity twin are EXCLUDED, only the in-batch CONTROL keeps its seat (best-model-selector; 14 exclude keys covering 7 keys-by-run-id + model-name aliases, ZERO pins/unpins/unexcludes, nothing promoted, nothing deleted, no restart). Resulting group = 15 entries (12 auto + 3 rankable pins), verified IDENTICAL across a forced `POST /api/best-models/recompute`. THIS IS CLEANUP, NOT RE-CROWNING: the champion did NOT move and no row of the crown board was touched.**_

_**(1) WHAT HAPPENED.** The stage-stack campaign (`2026-08-01-stage-stack-topology.md`) promoted all 7 arms **FOR MEASUREMENT ONLY** — every `notes` reads verbatim "promoted to enable walk evaluation; not a crown claim" — and the deterministic recompute at **2026-08-01T02:07:10Z** seated four of them (`stack-c0-gru256` r9, `stack-c1-gru537` r11, `stack-c-aggg256` r13, `stack-e-gagg259` r14) and **auto-created `best-seq-stack-20260801-003536-d612e` from arm D, the WORST arm in the batch** (recall 0.07687). The recompute ranks on `holisticness@10` and cannot see a verdict table, so five arms whose retrieval is broken (recall **0.0706–0.0860** vs the in-batch control's **0.12439 = 178/1,431**, all five below the standing **0.108 floor**, losses of **2.5–3.7 measured half-widths**) seated on H values of 0.017–0.024. **This is the textbook suspicious-metric case this curation exists to catch: H rewards de-eagering, and a model that retrieves almost nothing is trivially un-eager.** Exclusions below are made on the RECORDED VERDICT (V1 paired bootstrap 2000/rng1337 n=1431 vs the in-batch C0, AND V2 the generated walk at the matched a0.4/s0.3 cell), not on the H value._

_**(2) THE 5 REFUTED ARMS — EXCLUDED, each REFUTED on BOTH surfaces** (Δrecall@10 CI<0 *and* ΔH CI<0 on V1; Δvibe CI<0 with Δstride_err not CI<0 on V2). Keyed by run_id **and** by model-name alias, per the standing pairing precedent, so no alias can re-seat them:_
- _`run-20260801-003646-6e3bf-seq-stack` / `stack-a-agag299` (`ann+gru+ann+gru` h299) — worst on both surfaces: Δrecall −0.05381 [−0.06848, −0.03913], Δvibe −0.0375, and additionally loses genre variety (Δgenres −1.200 CI<0). Recall 0.07058 = 101/1,431._
- _`run-20260801-003646-07994-seq-stack` / `stack-b-agga299` (`ann+gru+gru+ann` h299) — Δrecall −0.03843 [−0.05381, −0.02306], ΔH −0.00682 CI<0, Δvibe −0.0258 CI<0. Recall 0.08595. (Its recall is an epoch-capped LOWER BOUND — see follow-up 2 of the report — which is a reason to RE-RUN it, not to seat it.)_
- _`run-20260801-003646-dbe68-seq-stack` / `stack-c-aggg256` (`ann+gru+gru+gru` h256) — Δrecall −0.04472 CI<0, Δvibe −0.0231 CI<0; also REFUTED at the DEPLOYED cell a0.8/s0.5 (Δvibe −0.0158 [−0.0267, −0.0059]) and NOT rescued by the stride probe (peaks at the cell it was measured in). Was seated at rank 13 on H 0.021442 at recall 0.07966._
- _`run-20260801-003536-d612e-seq-stack` / `stack-d-gggg229` / **`best-seq-stack-20260801-003536-d612e`** (`gru+gru+gru+gru` h229) — fires the V2 REFUTED clause **TWICE** (Δvibe −0.0140 CI<0 plus Δground −0.0250 CI<0) on top of Δrecall −0.04752 CI<0. **The server's own auto-created alias is excluded here too** — this is the arm the recompute picked to mint a `best-seq-stack-*` name from, at recall 0.07687._
- _`run-20260801-003646-8cc2d-seq-stack` / `stack-e-gagg259` (`gru+ann+gru+gru` h259) — Δrecall −0.04193 CI<0, Δvibe −0.0316 CI<0; the user's mechanism hypothesis for this arm is refuted in the report. Was seated at rank 14 on H 0.021053 at recall 0.08246._

_**(3) C1 — EXCLUDED as a REDUNDANT CAPACITY TWIN, not as a refutation.** `run-20260801-003536-bcedf-seq-stack` / `stack-c1-gru537` is **not** refuted: V1 is a **TIE** (Δrecall −0.00280 [−0.01188, +0.00559], ΔH straddles) and V2 is **indistinguishable** (Δvibe −0.0100 straddles). It is excluded because it is the *same* `stages=gru` topology as C0 at **3.24× the parameters (1,280,937 vs 394,944) for no measurable gain**, so seating both spends two of twelve slots on one architecture that is itself a functional twin of the `seq-nexttrack` h256 GRU already holding five seats. It is also a measurement-only promotion. Freeing that slot admitted **`best-seq-blend-20260728-005349-507bb` (H 0.018699, recall 0.23201 = 332/1,431 — the best recall on record)** at rank 12: a genuinely differently-behaved member in exchange for a duplicate. **Un-exclude on sight if C1 ever earns a seat on merit** (e.g. a capacity WIN under a different objective) — the exclusion is about redundancy, and redundancy is the kind of reason that expires._

_**(4) C0 — KEPT, and the trade-off is stated.** `stack-c0-gru256` (`run-20260801-003536-eaf81-seq-stack`, `stages=gru` h256, H 0.024378, recall **0.12439 = 178/1,431**) holds auto rank **9** with NO delta applied. It is legitimate on merit: it is the batch's **in-batch control**, its recall is mid-board and comfortably above the 0.108 floor, its H re-derives from its own facets under the live 4-factor definition, and it is the **only seat in the group whose walk behaviour was validated as interchangeable with a shipped artifact** (report G3: Δvibe −0.0033 and Δstride_err +0.0137 both straddle 0 vs the shipped `gru.onnx` at a0.4/s0.3). **The honest cost: it is architecturally REDUNDANT** with the five seated `best-seq-nexttrack-*` h256-GRU rows — it is a reproduction of that anchor (178 hits vs the on-record 176), not a new family, so `seq-stack` as a *predictor name* in the group overstates the diversity actually added. It is kept anyway because the alternative was strictly worse for the group: excluding it hands the slot to a **sixth** `seq-blend` row (`best-seq-blend-20260726-184115-36edb`, H 0.015289) and deepens the policy monoculture recorded on 2026-07-31, while removing the smallest model in the group (394,944 params) and the batch's own reference point. **Either call was defensible; this is the one taken and the reason it was taken.**_

_**(5) `dual-premlp256-f0` — EXCLUDED BY MODEL NAME, closing a naming hole.** Its run `run-20260726-161801-a6423-seq-dualgru` was already excluded (which is why its H **0.044541** — a would-be rank 2 — was not seated), but the 2026-07-31 promotion minted the model NAME `dual-premlp256-f0` on top of that run, leaving an un-keyed handle on a blocked run. Two independent reasons it must not rank: (a) it was **walk-refuted today** (Δvibe **−0.0255 CI<0** at a matched cell); (b) its stored metrics **LACK `artist_conc_at_k`** (retired metric generation), so its H is NOT comparable to the live 4-factor board — exactly the cross-generation inflation the 2026-07-31 note refused to seat for the six 2026-07-25 degenerate arms. Verified after the PUT: **no unexcluded run in the whole candidate field is missing `artist_conc_at_k`**, and every implausible H (the 0.133–0.197 dual-tower cluster) remains blocked._

_**(6) RESULTING GROUP — 15 entries = 12 auto + 3 rankable pins, all on canonical `seq-20260715-131139`, all n_test 1,431, order verified IDENTICAL across a forced recompute (`updated_at` 2026-08-01T02:36:57Z).** 1 `best-seq-blend-20260730-235524-93d38` (H 0.048063 — **champion, UNMOVED**); 2 `best-seq-blend-20260730-234635-a6252` (0.040056); 3 `best-seq-nexttrack-20260726-184115-bf395` (0.034171); 4 `best-seq-blend-20260728-012030-6d1b7` (0.030980); 5 `…-184116-20d9b` (0.030355); 6 `…-184115-147a6` (0.029458); 7 `…-184115-e1539` (0.026761); 8 `…-183333-9a0b2` (0.024538); **9 `stack-c0-gru256` (0.024378, recall 0.12439 — the ONLY surviving stage-stack seat)**; 10 `best-seq-dualgru-20260725-143502-18b4a` (0.023785, PINNED); 11 `best-seq-dualgru-20260725-143502-3e5d4` (0.021520, PINNED — THE DEPLOYED SHOWCASE MODEL); 12 `best-seq-blend-20260728-005349-507bb` (0.018699, recall 0.23201 — **new seat, took C1's slot**); 13 `…-184115-36edb` (0.015289); 14 `…-184115-6cb38` (0.010435); 15 `best-seq-blend-20260722-165345-28dbb` (0.008259, PINNED). Family mix: 7 `seq-blend`, 5 `seq-nexttrack`, 2 `seq-dualgru` (both pins), 1 `seq-stack`. Exclusion keys 95 → 109; pins UNCHANGED at 6; promoted-model count UNCHANGED at 94 (no auto-promotion was triggered — every seated candidate was already promoted). **The champion row `…-183333-1c361` dropped off the auto tail as a side effect of `507bb` entering; the OWED re-promotion of `blend-gru-markov-content-proj` onto a holisticness-carrying run remains the right fix for that naming, not a pin.**_

_**(7) WHAT THE NEXT RECOMPUTE WILL RE-ADMIT — the re-curation watchlist.** (a) **Nothing from this batch**: all 7 stage-stack runs and all 8 of their model aliases are now keyed, and the pairing (run_id + name) means even a freshly-minted `best-seq-stack-*` alias cannot seat them. (b) **The seats freed by this pass are held by the WEAKEST auto candidates in the field** — ranks 13–14 sit at H 0.015289 / 0.010435, so **ANY new run above H ≈ 0.0105 seats itself immediately**, and any new run above ≈ 0.0187 evicts `507bb`'s recall asset. The standing lesson holds: freeing a seat backfills from the same field. (c) **The `spawn_recompute` remote-batch trigger gap is at SEVEN confirmations** — this batch was remote (`snappler:1914390`, `queue:true`) and the group would have stayed frozen without a mutation; **every remote campaign still needs a manual `POST /api/best-models/recompute` before the group can be judged.** (d) **The report's follow-up 6 proposes DELETING the six refuted measurement promotions** — NOT done and NOT this agent's call; deletion is the user's, and exclusion was used precisely so the walk instruments can still LOAD these models. If they are ever deleted, the exclusion keys should STAY (a deleted model's run is otherwise re-promoted by the next recompute). (e) **A single `epochs=120` re-run of arm B** would arrive as a NEW run id and is NOT pre-blocked — judge it on its own recall, not on B's exclusion._

_**(8) FLAGGED, LEFT ALONE.** (a) **The policy monoculture recorded on 2026-07-31 is UNCHANGED and slightly deepened**: 12 of 15 seats are still eval-time policies over two weight sets (7 `seq-blend`, 5 `seq-nexttrack`), and this pass added one more `seq-blend` row. Curation cannot fix it — the instruments are the pending PRIMARY-METRIC decision and PROMOTION. (b) **The 3 legacy pins stay INERT** (`blend-gru-markov-content-proj`, `blend-gru-markov-content`, `gru-lstm-pca128`): their runs pre-date the holisticness suite so the server cannot rank them and they do not appear as entries — correct standing instructions that bite on re-promotion, unpinning would destroy them for zero gain. (c) **`test-r`** (stray promoted `seq-nexttrack` on `af11a`) — still recommended for USER deletion, still doubly blocked. (d) **`run-20260722-165345-047cc-seq-markov`** stays excluded 0.0011 under the floor; `seq-markov` remains the only unrepresented predictor family and should be un-excluded on sight if re-run at ≥0.108. (e) **`run-20260726-161801-0df6f-seq-dualgru`** stays excluded on the floor (0.10273), still the strongest un-exclusion candidate once a walk eval covers it. (f) **The orphaned run row `run-20260730-234635-7a297-seq-blend`** (stuck `running`) needs no delta — a null-metric run is never selectable. (g) **The crown board is UNTOUCHED**: recall@10 holder `507bb` at 0.23201 and the H crown `93d38` at 0.048063 both stand exactly as the campaign left them._

_**2026-08-02b — TOWER-BANK DE-FLOOD: the ENTIRE 2026-08-02 measurement batch is EXCLUDED and the group is restored EXACTLY to its 2026-08-01 post-de-flood shape (best-model-selector; 25 exclude keys = 12 keys-by-run-id + 13 model-name aliases, ZERO pins/unpins/unexcludes, NOTHING PROMOTED, NOTHING DELETED, no restart, orphan row NOT touched). Resulting group = 15 entries (12 auto + 3 rankable pins), verified IDENTICAL across a forced `POST /api/best-models/recompute` (`updated_at` 2026-08-02T22:16:41Z, then 22:17:53Z after the alias PUT). CLEANUP, NOT RE-CROWNING: the champion did NOT move, no row of the crown board was touched, and promoted-model count did NOT change by any action of this pass.**_

_**(1) WHAT HAPPENED.** The tower-bank campaign (`2026-08-02-tower-bank-parallel-towers.md`) promoted 8 arms **FOR MEASUREMENT ONLY** (every `notes` verbatim "promoted to enable walk evaluation; not a crown claim") and the deterministic recompute at **2026-08-02T19:54:03Z** seated three of them — `best-seq-nexttrack-20260802-182050-91b54` (C0) r8, `best-seq-bank-20260802-190415-1d662` (B4) r10, `best-seq-bank-20260802-183404-62604` (D2) r12 — taking the group to 15 entries and **evicting `507bb`, `36edb` and `6cb38`** from the auto tail. The report itself recommends the batch be cleared now that the walk numbers are banked. **The usual suspicious-metric heuristic does NOT apply here and was not used**: no arm is degenerate (recall 0.11321–0.12998, every arm clears the 0.108 floor — a contrast with the stage-stack batch), and these arms **INVERTED** the usual signature by *raising* `artist_conc` (0.35472–0.38247 vs C0's 0.35591), so none of them is de-eagering its way up H. The exclusions below rest on **(i) the recorded S1/S2 verdicts, (ii) proven functional identity to families already seated, and (iii) the noise band** — not on H._

_**(2) THE H ORDER ACROSS RANKS 8–12 CARRIED NO INFORMATION, WHICH IS WHY THE VERDICTS DECIDE.** All ten batch arms sit at H 0.0224536–0.0245383 — a total spread of **0.0020847**, INSIDE the family noise band hw(ΔH) **0.00116–0.00261** and comparable to the sharpest shared-weights hw(ΔH) 0.000382. Every arm is therefore "equal, likely" to the already-seated h256 anchor `9a0b2` (0.0245383) and to each other; H cannot rank them and was not asked to._

_**(3) FUNCTIONAL IDENTITY — the decisive point, and it is proven at bit level, not argued.** `seq-bank` is a NEW PREDICTOR NAME but NOT a new failure mode: the campaign's own gates show `views="latent"` is bit-identical to `SeqNextLatent` (P2 leg a, max|Δ| 0.00e+00 at T=1,2,5,17,40), `views="latent+latent"` is bit-identical to `DualTowerNextLatent` f0 (P2 leg b, plus P2c: 10/10 state-dict tensors at 0.00e+00 and per-epoch train+val loss to the last digit), and `views="latent+cummean"` reproduces the DEPLOYED engine's **1,431/1,431 ranked lists byte for byte** (`2026-08-02b` G-E0). So a `seq-bank` seat would buy the group a family LABEL over `seq-nexttrack` (5 seats) / `seq-dualgru` (2 pinned seats) weights it already holds — the deepest form of the "nominal family label overstates the diversity added" trap recorded for `stack-c0-gru256` on 2026-08-01, and here it is demonstrable rather than arguable. **No bank arm was pinned for family diversity, and that is the reason.**_

_**(4) THE 12 EXCLUDED RUNS AND THE REASON FOR EACH** (all `seq-20260715-131139`, all n_test 1,431, all S1-TIE vs C0, no arm S1-REFUTED). **`run-20260802-182050-91b54-seq-nexttrack`** (C0, H 0.0245383, 176 hits) + name `best-seq-nexttrack-20260802-182050-91b54` — a **bit-identical reproduction** of the ALREADY-SEATED `run-20260726-183333-9a0b2-seq-nexttrack` (G2a: recall, music@10, H, mrr all to the last digit), i.e. the same model twice in a 12-slot group. **`run-20260802-183404-91e31-seq-nexttrack`** (C1 h512, 0.0237953, 186) + `bank-c1-gru512` — S1-TIE / S2-NULL, the **7th** param-matched width tie; the width axis stays closed and its H is inside the noise band of five seated h256 rows. **`run-20260802-190415-1d662-seq-bank`** (B4, 0.0244609, 186) + `bank-b4-llll211` + `best-seq-bank-20260802-190415-1d662` — S1-TIE / S2-NULL and **Δ vs C1 = exactly 0.000000 [−0.00839, +0.00839]**: the batch's best arm is fully explained by capacity, not topology. **`run-20260802-183404-62604-seq-bank`** (D2, 0.0242558, 177) + `bank-d2-ll256` + `best-seq-bank-20260802-183404-62604` — S1-TIE / S2-NULL, and its `latent+latent` view pair is the bit-identical twin of the pinned `18b4a` family. **`run-20260802-190415-90866-seq-bank`** (B3, 0.0235773, 172) + `bank-b3-lll256` — LOSES to C1 **CI<0 (−0.00978)** and S2 Δgenres **−1.0250 CI<0**. **`run-20260802-183404-67185-seq-bank`** (B2, 0.0233569, 170) + `bank-b2-ll334` — LOSES to C1 **CI<0 (−0.01118)**; earned an S2-TRADE at the primary cell but flips to Δvibe **−0.0064 CI<0 at the DEPLOYED cell a0.8/s0.5**, i.e. refuted where the showcase actually runs. **`run-20260802-192627-17502-seq-bank`** (V3, 0.0232003, 170) + `bank-v3-lcc256` — same shape: S2-TRADE at a0.4/s0.3, Δvibe **−0.0112 CI<0** at a0.8/s0.5. **`run-20260802-192627-836b1-seq-bank`** (V2, 0.0224536, 166) + `bank-v2-lcd256` — **S2-REFUTED** at the primary cell (Δvibe −0.0144 CI<0) and loses to C1 CI<0. **`run-20260802-190415-b5928-seq-bank`** (V1, 0.0229874, 162) + `bank-v1-llc256` — **S2-REFUTED** (Δvibe −0.0174 CI<0, Δgenres −1.7250 CI<0), loses to C1 CI<0, lowest recall in the batch. **`run-20260802-182050-2e6f2-seq-bank`** (R2 dropout-0.0 control, 0.0237849, 170, NEVER PROMOTED) — loses to C1 CI<0 and is bit-identical to the pinned `18b4a` (G2b); excluded **pre-emptively** because the other nine exclusions would have lifted it into the auto top-12 and the recompute would have auto-promoted a control nobody promoted. Plus the two `2026-08-02b` runs: **`run-20260802-214618-625eb-seq-bank`** (E0, 0.0215204) + `dropout-e0-latent-cummean-d00` — reproduces the PINNED deployed engine `3e5d4` on all 1,431 ranked lists byte for byte, so seating it would double-count a pinned member; **`run-20260802-214618-a38be-seq-bank`** (E1, 0.0219557, 172) + `dropout-e1-latent-cummean-d01` — verdict **NOT REPLICATED** (Δrecall +0.002795 straddles), a measurement-only twin of the deployed view pair whose H sits 0.0004 from the pinned original, i.e. inside the noise band. Exclusion keys **109 → 134**._

_**(5) A RACE WAS OBSERVED AND HANDLED, AND IT IS THE REASON FOR TWO PUTs.** `dropout-e0-latent-cummean-d00` and `dropout-e1-latent-cummean-d01` were created by a CONCURRENT agent at **22:16:25Z**, three seconds before this pass's first PUT returned (22:16:28Z) — both carry the measurement-only note. The run-id keys already blocked them (the 22:16:41Z forced recompute seated neither and created nothing), and a second PUT added the two model-NAME aliases at 22:17:53Z to close the same naming hole `dual-premlp256-f0` opened on 2026-08-01. **Both `best-seq-bank-*` auto names and both `bank-*` measurement names are keyed for the two double-named runs (`62604`, `1d662`), so no un-keyed handle survives on any excluded run.**_

_**(6) RESULTING GROUP — IDENTICAL to the 2026-08-01 post-de-flood group, seat for seat.** 1 `best-seq-blend-20260730-235524-93d38` (H 0.048063 — **champion, UNMOVED**); 2 `…-234635-a6252` (0.040056); 3 `best-seq-nexttrack-20260726-184115-bf395` (0.034171); 4 `best-seq-blend-20260728-012030-6d1b7` (0.030980); 5 `…-184116-20d9b` (0.030355); 6 `…-184115-147a6` (0.029458); 7 `…-184115-e1539` (0.026761); 8 `…-183333-9a0b2` (0.024538); 9 `stack-c0-gru256` (0.024378); 10 `best-seq-dualgru-20260725-143502-18b4a` (0.023785, PINNED); 11 `best-seq-dualgru-20260725-143502-3e5d4` (0.021520, PINNED — THE DEPLOYED SHOWCASE MODEL); 12 `best-seq-blend-20260728-005349-507bb` (0.018699, recall **0.23201** — the recall@10 crown, RE-ADMITTED after the flood evicted it); 13 `…-184115-36edb` (0.015289, RE-ADMITTED); 14 `…-184115-6cb38` (0.010435, RE-ADMITTED); 15 `best-seq-blend-20260722-165345-28dbb` (0.008259, PINNED). Family mix UNCHANGED at 7 `seq-blend` / 5 `seq-nexttrack` / 2 `seq-dualgru` / 1 `seq-stack`. Pins UNCHANGED at 6. **All three re-admitted seats were already promoted, so ZERO auto-promotions were triggered by this pass** (the +2 model rows between the pre-pass and post-pass reads are the concurrent agent's, not this pass's)._

_**(7) NOTHING WAS DELETED, AND THE RECOMMENDATION TO DELETE IS DEFERRED TO THE USER WITH A NEW TECHNICAL REASON.** The campaign report recommends deleting the 8 measurement promotions; deletion is the user's call under this agent's standing rules, and there is now a concrete hazard: **a THIRD `seq-bank` campaign was IN FLIGHT during this pass** (`run-20260802-221545-4ee07-seq-bank` running, `run-20260802-221545-b63d0-seq-bank` queued — the `latent+cummean` dropout pair re-run at **seed 7**, i.e. the seed study the 2026-08-02b follow-up asks for). `walk_headtohead.py::promoted_score` loads PROMOTED MODELS BY NAME, so deleting bank handles mid-campaign could break an in-flight comparison, while **exclusion cannot** — it bars auto-selection only and leaves every artifact loadable. With the 12 run ids keyed, deletion is now purely cosmetic (it removes model rows and snapshots; it does NOT remove group members, and deleting without the exclusion would simply get the run re-promoted by the next recompute). Suggested when the in-flight batch has reported: `DELETE /api/models/{name}` for `bank-c1-gru512`, `bank-d2-ll256`, `bank-b2-ll334`, `bank-b3-lll256`, `bank-b4-llll211`, `bank-v1-llc256`, `bank-v2-lcd256`, `bank-v3-lcc256`, `best-seq-bank-20260802-190415-1d662`, `best-seq-bank-20260802-183404-62604`, `best-seq-nexttrack-20260802-182050-91b54`._

_**(8) FLAGGED, LEFT ALONE — and one thing DECIDED AGAINST.** (a) **THE GROUP WILL RE-FLOOD WITHIN THE HOUR AND A SECOND PASS IS OWED**: the two in-flight seed-7 `seq-bank` runs will land at H ≈ 0.0215–0.0220 if they track their seed-1337 twins, which is ABOVE the current auto tail (0.018699 / 0.015289 / 0.010435), so the recompute will seat and auto-promote them. They were **deliberately NOT pre-excluded** — pre-judging an unmeasured arm has no auditable reason, and this batch is the one that could produce the record's first SEED-CONFIRMED improvement to the deployed engine, which would be a legitimate seat and a pin candidate (unlike every arm excluded above, all of which are explicitly "not a crown claim"). **REHABILITATION TRIGGER, recorded so the exclusions above are not read as permanent: if the seed study resolves Δrecall CI>0 for dropout 0.1 on `latent+cummean`, `dropout-e1-latent-cummean-d01` / `run-20260802-214618-a38be-seq-bank` should be UNEXCLUDED, since its exclusion rests on "NOT REPLICATED", not on a refutation.** (b) **PINNING `507bb` WAS CONSIDERED AND DECLINED.** It is the record's one-shot recall@10 crown (0.23201, ~2× the GRU rows) and H cannot see that, which is the standard pin argument — but it re-enters the auto top-12 on merit at rank 12 with no delta, and its LOW H is exactly the artist-eager profile the crown metric exists to penalize, so fixing it permanently into a group used for consensus mood-holding predictions would inject the eagerness H suppresses. `6d1b7` already covers the high-recall end at recall 0.21314 with H 0.030980. **Either call was defensible; this is the one taken and the reason it was taken.** (c) The **policy monoculture is UNCHANGED** (12 of 15 seats are eval-time policies over two weight sets); curation still cannot fix it and this pass did not deepen it. (d) `run-20260726-161801-0df6f-seq-dualgru` stays excluded on the floor (0.10273), still the strongest un-exclusion candidate once a walk eval covers it. (e) `run-20260722-165345-047cc-seq-markov` stays excluded 0.0011 under the floor; `seq-markov` remains the only unrepresented predictor family and `seq-bank`, though newly registered, is NOT a diversity candidate (see (3)). (f) The six 2026-07-25 degenerate dualgru arms, `dual-premlp256-f0` and the 7 stage-stack keys all stay excluded — no reason went stale, and the campaign's f0 under-regularization finding does NOT rehabilitate any of them (no exclusion on record was made *because* of an f0-vs-f1 gap). (g) **`test-r`** still recommended for USER deletion, still doubly blocked. (h) **The orphan row `run-20260730-234635-7a297-seq-blend`** (stuck `running` since 2026-07-30) was NOT touched and needs no delta — a null-metric run is never selectable. (i) **The crown board is UNTOUCHED**: recall@10 holder `507bb` 0.23201 and H crown `93d38` 0.048063 both stand._

## Follow-ups
- ✅ **DISCHARGED (2026-08-02c) — the 2026-08-02b follow-ups "settle the f0 dropout question
  with seeds" and "take the walk read" WERE BOTH RUN. The dropout axis is CLOSED; do not
  spend more runs on it.** A positive recall effect is excluded at a resolvable floor finer
  than the effect; the remaining sign ambiguity would need ~10 seeds to settle and the payoff
  is a number near zero. `2026-08-02c-dropout-seed-grid-and-the-walk-read.md`.
- **NEW (2026-08-02c, OPTIONAL, 2 runs) — re-seed the `latent+latent` D2−R2 contrast IF anyone
  intends to quote its +0.0049 CI>0 result again.** It is the one live claim the new
  seed-variance number directly undermines. Optional only because the facts file now flags it
  as unconfirmed at the point of use.
- **NEW (2026-08-02c, 3 runs) — measure seed variance ONCE on `seq-blend`, where the champion
  lives.** The champion's margins (+0.013…+0.025) are comfortably above the `seq-bank` seed
  range, so this is not urgent — but the gated variant's +0.0014…+0.025 spread deserves a
  same-family seed figure rather than one imported from `seq-bank`.
- **NEW (2026-08-02c) — the `Δdrift −0.0226 CI<0` at a0.4/s0.3 is the campaign's one
  unexplained result.** If the dual-tower family is revisited on the walk, look there:
  dropout on the concatenated hidden state costs mood-holding late in the walk specifically.
- **NEW (2026-08-02b) — SETTLE THE f0 DROPOUT QUESTION WITH SEEDS (3–6 runs, NEEDS USER
  APPROVAL). This SUPERSEDES the 2026-08-02 follow-up 'retrain the deployed engine at
  dropout 0.1', which HAS NOW BEEN RUN and came back UNRESOLVED, not settled.** The single
  split cannot see a ~0.005 effect (measured hw 0.005250, 16 discordant rows), so repeat
  `seq-bank` `views=latent+cummean` h256 at `dropout` ∈ {0.0, 0.1} × `seed` ∈ {1337, 7, 42}
  and judge on the 3-seed paired mean. Do NOT re-run it as another single split — that
  buys nothing. `2026-08-02b-dropout-on-the-deployed-view-pair.md`.
- **NEW (2026-08-02b) — cheaper alternative to seeds, but a DESIGN question:** the contrast
  above is underpowered partly because both arms are weak in absolute terms (~12% recall),
  leaving little discordant mass. Measuring the same dropout delta on a STRONGER base (the
  champion blend's GRU leg) would resolve the same mechanism with more discordant rows at
  lower run cost — but it changes family, so it needs the experiment-designer, not a runner.
- **NEW (2026-08-02b) — DO NOT redeploy the showcase engine on current evidence.** E1
  (`run-20260802-214618-a38be-seq-bank`) is the better point estimate on every crown facet
  at equal cost, but its recall gain is not significant AND **no walk read was taken**; the
  standing one-cell rule plus the two TRADE arms that flipped to Δvibe CI<0 at exactly
  a0.8/s0.5 make a walk measurement mandatory before any redeployment discussion. User's
  call. `2026-08-02b-dropout-on-the-deployed-view-pair.md`.
- **NEW (2026-07-27c) — REGISTER + PROMOTE the gated champion (NEEDS USER APPROVAL; the
  campaign deliberately did neither).** A1 `run-20260728-005349-507bb-seq-blend` is the
  new best recall on record (0.23201 = 332/1431, paired-Δ CI>0 on TWO splits, MRR up on
  both) — suggested definition name `blend-gru-markov-content-proj-gated`. A3
  `run-20260728-012030-6d1b7-seq-blend` is the crown candidate
  (`...-gated-mmr-l07`, H 0.030980 at a recall TIE with the champion). Both are
  RECOMMENDED. A 3-seed confirm of A1 is the cheap pre-promotion check and IS meaningful
  here — `seq-blend` exposes `seed` and the InfoNCE projection is genuinely seed-varied
  (seeds 1337/7/99 are what crowned the current champion). 3 runs, ~30 min.
- **NEW (2026-07-27c) — the SOFT GATE, i.e. the boundary-free version of the same
  hypothesis (top scientific follow-up).** Scale the Markov leg by `trust = n_u/(n_u+8)`
  BEFORE `_zcand` instead of hard-abstaining at `n_u == 0`. This removes the one residual
  selection concern (the `n_u=0` bucket boundary was chosen with test knowledge), is
  strictly more general, and if it matches or beats the hard gate it retires the caveat
  entirely. 2 runs (canonical + 2nd split); both controls are already banked.
- **NEW (2026-07-27c) — AUDIT the content and GRU legs for the same confidence defect.**
  `_zcand` erases per-leg confidence generally (see §Pitfalls). Does the content leg have
  a query-dependent evidence measure — cold-item neighbourhood density, distance to the
  k-th neighbour — that is being erased the same way? Same mechanism, possibly the same
  size of prize, and it is the only lever class on this corpus that has just paid.
- **NEW (2026-07-27c) — RE-SCREEN the whole λ curve on the GATED champion.** The gate
  changed MMR's tier on the blend from PRICED to FREE, so the 2026-07-26 λ frontier's
  champion-side conclusions are stale. Screen OFFLINE from the A1 checkpoint per the
  registered screen-then-bank rule; spend native runs only on the 2–3 bankable points.
  λ0.7 is the only point measured so far.
- **NEW (2026-07-27c) — FRESHEN the `snappler` worker checkout (ORCHESTRATOR action).**
  It has no `markov_gate`; any `queue:true` `seq-blend` run there silently returns an
  ungated null. md5-verify `predictors/` after rsync, per the divergence pitfall.
- **NEW (2026-07-27c) — best-model-selector.** The deterministic recompute auto-promoted
  four of this batch (A3 rank 2, A1 rank 7, C3 rank 9, C1 rank 12); the group now holds
  five `seq-blend` entries across two generations, including in-batch CONTROLS (C1 is a
  duplicate of the champion; C3 is an ungated control kept only for the paired stat).
  Family diversity and control-exclusion are the judgment calls.
- **NEW (2026-07-27c) — report-curator (sync mode).** Fold
  `2026-07-27c-nexttrack-markov-gate-confirm.md` into `docs/experiments.tex` (+ the
  Spanish and next-track-family docs) and rebuild. Natural home = the counterpoint to the
  five architecture nulls and the four learned-combiner nulls: the win came from taking a
  vote AWAY, not from adding machinery — and the bucket decomposition is the figure.
  NOT auto-updated.
- **NEW (2026-07-26) — best-model-selector (DO FIRST, group is FLOODED):** the mandatory
  manual `POST /api/best-models/recompute` after the λ-frontier batch filled **all 12 auto
  slots with 2026-07-26 runs, 10 of them that campaign's**, and the recall champion is no
  longer in the auto group at all. Exclude the **TIER-3** arms by **BOTH keys** (the
  `88a4c` precedent — excluding only one key lets a run back in):
  `run-20260726-184115-1dc77-seq-blend` / `best-seq-blend-20260726-184115-1dc77` (A4 λ0.3,
  H 0.031626 at recall@10 0.10971, Δrecall −0.102 — currently **rank 2**) and
  `run-20260726-184115-7a2f7-seq-blend` / `best-seq-blend-20260726-184115-7a2f7` (A3 λ0.5,
  Δrecall −0.047). Also exclude the two **G2-defective** discarded controls whose stored
  metrics lack `artist_conc_at_k` and are therefore not code-comparable:
  `run-20260726-175258-6f5fb-seq-nexttrack` / `best-seq-nexttrack-20260726-175258-6f5fb`
  and `run-20260726-175258-c4968-seq-blend` / `best-seq-blend-20260726-175258-c4968`.
  **Judgment call left to the selector:** B3 (`run-20260726-184115-bf395-seq-nexttrack`,
  currently rank 1, H 0.034171 at recall 0.11461) is TIER-2 PRICED — a legitimate
  operating point above the 0.108 floor, NOT a degenerate arm. Note this is the FIRST
  curation where the whole rankable field is on ONE live metric generation, so the
  2026-07-26 'ranks on two dead metric generations' defect is now CLOSED.
- **NEW (2026-07-26) — R4 ROBUSTNESS CONFIRM for the λ frontier (needs its own approval;
  MANDATORY before any promotion):** for the best TIER-1 arm **B4**
  (`gru-infonce-h256-mmr-l07-pool50`), `seq-nexttrack` exposes no `seed` and reproduces
  bit-identically, so the pre-agreed confirm is a **second-split re-run on
  `seq-20260718-211238`**: the winning arm **+ its λ=1.0 control**, both on that split,
  same host. **2 runs, ~6 min.** Verdict = paired ΔH CI>0 on that split with recall ≥ 0.108
  there too (absolute recall is lower on the 0.64-cold split BY CONSTRUCTION — judge the
  paired Δ, per the crown-hardening precedent). Worth adding B2 + its control (2 more runs)
  since B2 is the simpler operating point and sits within ~1 half-width of B4.
- **NEW (2026-07-26) — report-curator (sync mode) for the λ-frontier campaign:** fold
  `2026-07-26-nexttrack-mmr-lambda-frontier.md` into `docs/experiments.tex` (+
  `docs/experiments.es.tex`) and rebuild the PDFs. Natural home = the FIRST POSITIVE
  campaign after five architecture nulls, and the pivot from architecture to **eval-time
  policy**: the same corpus that refuses to reward more machinery around the GRU rewards
  re-ordering its output. The `artist_conc` H1 verdict and the binding-facet switch are the
  two figures worth drawing. NOT auto-updated.
- **NEW (2026-07-26) — push the λ frontier on the GRU between 0.75 and 0.85, with a pool
  probe at the winner (~4 runs).** H1 CONFIRMED means real mechanism remains, and the
  binding-facet switch says the productive band is ABOVE λ0.7 (below it, `artist_adj`
  binds and falls ~2.5× slower, so each recall point buys less crown). Screen offline
  first per the refined eval-only pitfall.
- **NEW (2026-07-26) — test λ on the blend's Markov leg.** H1 CONFIRMED makes this the
  indicated sequel on the champion side (2026-07-22 localizes blend eagerness to the
  bigram leg, and MMR de-concentrates artists via sonic proximity). Design around the
  obstacle this campaign found: the champion's `mood_coh` COLLAPSES under diversification
  (0.3982→0.2378 at λ0.3, −40%) whereas the GRU's barely moves (−4%).
- **NEW (2026-07-26) — DROPPED as the next move: the explicit artist-diversity rank
  penalty.** This was the pre-registered **H2** sequel (port `seq-mood`'s `artist_penalty`
  to `seq-blend`/`seq-nexttrack`); H2 is REFUTED 8/8, so it is no longer indicated. Keep on
  file as a COMPLEMENT to sonic MMR rather than a replacement, since `artist_adj` becomes
  the binding facet at low λ and MMR moves it ~2.5× slower than `artist_conc`.
- **NEW (2026-07-26) — re-derive whether `MUSIC_FLOOR = 0.34` is still right.** The floor
  was calibrated to cancel recall-destroying H gains, on the observation that music@10
  falls in lockstep with recall. MMR is the FIRST lever that RAISES music@10 while
  diversifying (0.4538→0.4925 across the champion arms, 0.4558→0.4747 on the GRU), so the
  floor now shapes the λ optimum in a regime it was not calibrated for. Worth a look before
  it silently picks the operating point.
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
- ~~**A CONSTRAINED holisticness objective** … the MMR λ≈0.9 eval-time re-rank
  (`2026-07-22`) remains the only near-free holisticness lever on record.~~
  **RESOLVED POSITIVE 2026-07-26 (`2026-07-26-nexttrack-mmr-lambda-frontier.md`).** MMR is
  CONFIRMED as a genuine crown lever by a native sweep: all 8 λ<1 arms lift H with paired
  ΔH CI>0, and λ≈0.9 is indeed near-free on BOTH models (recall ties, 301/303 and 173/176).
  But the flag under-sold the result in one direction and over-sold it in another: the
  valuable operating points are **λ0.7–0.8 on the single GRU** (three TIER-1 arms above the
  all-time bar, best B4 λ0.7/pool50 H 0.030355 at a recall tie), NOT λ0.9 on the champion
  (H 0.010435, a real but tiny gain). The constrained-objective framing is VINDICATED — the
  ±0.015 recall floor + `Δrecall ≥ −0.03` clause is what separated the 3 TIER-1 arms from
  the 2 TIER-3 rejects (A4 reaches H 0.031626 at recall 0.10971, Δrecall −0.10). NOT yet
  promoted: awaits the R4 second-split confirm on `seq-20260718-211238` + user approval.
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
