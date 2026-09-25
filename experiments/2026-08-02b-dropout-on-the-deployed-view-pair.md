# Dropout on the deployed view pair — does the D2−R2 win replicate on `latent/cummean`?

> **CONTINUED AND SUPERSEDED IN PART by
> `2026-08-02c-dropout-seed-grid-and-the-walk-read.md` (same day).** This report's verdict
> was "NOT REPLICATED, but underpowered — the deployed engine's regularization status stays
> OPEN". The user authorized both of its follow-ups; the seed grid closed the question in
> the direction that mattered (**the positive effect is EXCLUDED at a resolvable floor of
> 0.003960, finer than the +0.004892 effect being hunted; signs disagree across seeds
> +4/−4/−13**) and produced the family's first seed-variance measurement, which
> **re-flags the D2−R2 win itself as unconfirmed**. The walk read was also taken. Everything
> below stands as measured — only the "OPEN / needs seeds" framing is superseded.

**2 runs, both remote on `snappler:879965` via `queue:true`, 13.3 min total compute,
13.4 min wall (21:46:18Z → 21:59:40Z), 0 failed / 0 interrupted, 0 contingency runs
(none were authorized).**

NOTHING DEPLOYED. NOTHING PROMOTED. NO DEFINITION SAVED. NO CHAMPION REGISTRATION. Crown
metric and `domain.toml` UNCHANGED. `models.toml` not hand-edited. The orphaned run row
`run-20260730-234635-7a297-seq-blend` was NOT touched. No server restart (the registry
edit and its gated restart happened *before* this campaign, in the setup step). The walk
phase (S2) was **not entered** — its trigger is "only if S1 REPLICATED", and S1 did not.

## Goal

The 2026-08-02 tower-bank campaign's one real finding was a by-product: at byte-identical
parameters on the `latent+latent` view pair, **dropout 0.1 beat dropout 0.0 by +0.004892
recall@10 [+0.000699, +0.009783] CI>0** (`seq-bank` D2 177 hits vs R2 170). Because
`DualTowerNextLatent` builds **zero `nn.Dropout` modules** at `fusion_layers=0` /
`layers_*=1` / `pre_hidden_*=0`, every `fusion_layers=0` arm on record trained
unregularized — *including the deployed showcase engine* (`dualgru.onnx`, `latent/cummean`
f0). This campaign asks the single question that follows: **is that dropout win a property
of the f0 path, or of the `latent+latent` view pair specifically?** If the former, the
deployed engine is leaving ~+0.005 recall on the table for free, with no architecture bet.

**Why this had to run on `seq-bank`, not `seq-dualgru`.** The naive version of this
follow-up is a provable no-op: `dropout` is *inert* on the `seq-dualgru` f0 path, so
re-POSTing the deployed config at a different `dropout` returns a bit-identical model.
The regularized twin must therefore be built on `seq-bank`, which applies dropout to the
concatenated hidden state. The substitution was discharged by a functional-equivalence
gate **before** any registry edit: `seq_bank` `views="latent+cummean"` h256 `dropout=0.0`
is bit-identical to `seq_dualgru` `view_a=latent`/`view_b=cummean` at `fusion_layers=0` —
max|Δ| **0.00e+00** on `forward` *and* `predict_next` at T = 1, 2, 5, 17, 40, max|Δ|
**0.00e+00** on the **padded/packed** path too (lengths [40, 17, 3] with zeroed tails,
checked explicitly because `cummean` accumulates over the padded tail), params **789,696**
both sides. `registry.toml` gained `"latent+cummean"` as an additive 8th option on the
`seq-bank` `views` enum (it was previously undeclared, so the POST would have been
rejected).

**Dataset** `seq-20260715-131139` (canonical PCA-192 sequence split, 7,154 sessions,
19,402 items, dim 192, **test sessions 1,431**, chronological cut 2024-08-24, cold-item
rate 0.5507). `seq-20260718-211238` was NOT used: all 1,431 of its test sessions sit
inside split-1's train set. **No dataset was built.**

**Reference (the deployed engine's own training run).**
`run-20260725-143502-3e5d4-seq-dualgru`, recall@10 `0.11740041928721176` = **168/1431**,
music@10 `0.445822509348973`, H@10 `0.021520424520401444`, mrr `0.05087471236637486`,
params 789,696, `claimed_by snappler:4101029` — i.e. produced **on the worker**, which is
why `queue:true` (not the hub) is the parity-correct place to reproduce it.

**Held constant on both runs — only `dropout` varies:** `views=latent+cummean`,
`hidden=256`, `loss=infonce`, `tau=0.07`, `epochs=40`, `patience=6`, `lr=0.001`,
`batch_size=128`, `val_fraction=0.15`, `seed=1337`, `k=10`, `mmr_lambda=1.0` (off),
`mmr_pool=200`, `artist_cap=0` (off). All 15 keys are declared on `seq-bank`; exactly
these 15 were sent, nothing else (the `unknown hyperparameter` pitfall killed 3 POSTs on
2026-08-01). The reference run's `val_fraction` was not declared on `seq-dualgru` but
defaults to `0.15` at `seq_dualgru.py:140`, so the held value matches it exactly.

## Outcome in one line

**The D2−R2 dropout win does NOT replicate on the deployed `latent/cummean` view pair:
Δrecall@10 = +0.002795 [−0.002795, +0.007704] STRADDLES (172 vs 168 hits, half-width
0.005250) — but the measurement is UNDERPOWERED rather than negative, because the
D2−R2 point estimate (+0.004892) sits INSIDE this campaign's CI and this campaign's point
estimate sits INSIDE D2−R2's CI, so "the effect is absent here" and "the effect is present
here and unresolved" are not distinguishable by these two runs; the honest read is that the
deployed engine's regularization status stays OPEN, and the campaign's hardest number is
the control — E0 reproduced the deployed engine's predictions on all 1,431 rows byte for
byte.**

## Gates

| gate | result |
|---|---|
| **P1 — worker md5 parity on the whole import closure** | **PASS.** `registry.toml` was known stale (worker `a556ab37…` vs hub `15fd3837…`) and was re-copied before the POSTs; after the copy all five files match hub↔worker exactly — `seq_bank.py` `7c58d53a2d3c3750502050afaabde238`, `seq_common.py` `09e7aef666455d49e33f187d73615bda`, `seq_nexttrack.py` `1f995226a63c5127016d5bccd749d512`, `seq_dualgru.py` `d263c4253905a1b78e963dcdabe74592`, `registry.toml` `15fd3837de3b5e983735b2dc4cc52346`. The `lensing-server` **binary** was checked as the distinct parity item it is: `3b7c3172a85de4c8f502b3425d8b3bbb` on both sides. No hub fallback was needed or used — both runs ran on `snappler:879965`. |
| **G-E0 — bit-identity, hard** | **PASS, and in its strongest form.** E0 returned `0.11740041928721176` = **168/1431** exactly, and per-row the two runs agree on **1,431/1,431 identical `top_k_ids`** — not merely the same hit count, the same ranked list on every test session. `music@10` `0.445821943` and `mrr` `0.05087471236637486` also bit-identical. (`holisticness@10` differs in the 8th decimal, `0.02152038836569451` vs the stored `0.021520424520401444` — a rescore-path float artifact, and the gate is explicitly judged on the hit count.) This simultaneously re-confirms the functional-equivalence gate end-to-end through the real training path, and confirms worker-host stability across 8 days. |
| **G1 — metric completeness** | **GREEN on both.** Each run carries all five crown facets including `artist_conc_at_k`, and **`music@10` is populated** on both (E0 `0.445822509348973`, E1 `0.446890249497208`) — no silent Qdrant absence, so no mid-run hub suspension corrupted either arm. |
| **G3 — walk instrument** | **NOT REACHED** — the walk phase is gated on S1 REPLICATED, which did not occur. |

## Results — S1, one-shot top-10

Paired bootstrap, 2,000 resamples, rng 1337, n_test 1,431, computed from the stored
per-row `predictions.json` artifacts (fetched read-only from the worker; nothing written
under `data/`). Both arms: **789,696 params**, early stop at **epoch 24**.

| arm | run id | `dropout` | params | hits | recall@10 | Δrecall vs **E0** | H@10 | music@10 | a_conc | mrr | best val | S1 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E1** | `run-20260802-214618-a38be-seq-bank` | **0.1** | 789,696 | **172** | **0.120195667** | **+0.002795 [−0.002795, +0.007704] straddles** | **0.02195565** | **0.446890** | 0.36535 | **0.05185008** | 6.5767 | **NOT REPLICATED** |
| E0 (control) | `run-20260802-214618-625eb-seq-bank` | 0.0 | 789,696 | 168 | 0.117400419 | — (bit-identical to the deployed reference) | 0.02152039 | 0.445822 | **0.36638** | 0.05087471 | **6.5763** | anchor (G-E0) |
| — reference | `run-20260725-143502-3e5d4-seq-dualgru` | 0.1 (**inert**) | 789,696 | 168 | 0.117400419 | — | 0.02152042 | 0.445822 | 0.36638 | 0.05087471 | — | prior record |

No run failed; there is no failed-run row to report.

**Both mandatory floors pass.** Absolute recall floor ≥ 0.108 (155/1431): E1 is
0.120196 = 172, clear. Δ`music@10` must not be CI<0: **+0.001068 [−0.002067, +0.004160]**
straddles. So E1 is not refuted by any floor — it simply fails to clear the win leg.

**Discordance structure.** Only **16 of 1,431** rows flip at all (10 gained by E1, 6 lost).
The whole contrast rests on a net of 4 sessions, which is precisely why the interval is
wide relative to the effect at stake.

**Context vs the h256 GRU anchor** (`run-20260726-183333-9a0b2-seq-nexttrack`, 176/1431 =
0.122990915): E1 −0.002795 [−0.013976, +0.007687] straddles; E0 −0.005590 [−0.017470,
+0.004892] straddles. Neither the regularized nor the unregularized deployed pair is
distinguishable from the single-GRU anchor — unchanged from the record.

## Findings

**1. The verdict is NOT REPLICATED, and the decision rule was applied mechanically.**
The rule required Δrecall@10 CI>0 **and** absolute recall ≥ 0.108 **and** Δ`music@10` not
CI<0. Legs two and three pass; leg one straddles. Under the rule that is NOT REPLICATED,
so no promotion, no walk phase, no save, nothing deployed. A +4-hit lead with a
half-width of 5.25 hits' worth of recall is a near-miss, and a near-miss is a near-miss.

**2. But "not replicated" here means "not resolvable", and the distinction matters more
than the verdict.** The measured half-width is **0.005250** — *larger than the entire
effect D2−R2 reported* (+0.004892). The two campaigns' intervals are mutually compatible
in both directions: D2−R2's point estimate is inside `[−0.002795, +0.007704]`, and this
campaign's point estimate is inside `[+0.000699, +0.009783]`. The sign also agrees
(+0.0028, same direction, ~57% of the D2−R2 magnitude). So these two runs **cannot
distinguish** "dropout does nothing on this view pair" from "dropout does here what it did
on `latent+latent`, and n=1431 with 16 discordant rows can't see it". Reporting this as
"the dropout win is localized to `latent+latent`" would overstate the evidence; the
correct statement is that **the deployed engine's regularization status remains OPEN**, and
that settling it needs either seeds or a larger test set, not another single split.

**3. The f0-inertness defect itself is untouched and still confirmed.** Nothing here
weakens the mechanical finding that `DualTowerNextLatent` constructs zero `nn.Dropout`
modules at f0 — that is a code fact, re-confirmed by the strongest possible empirical
signature: the deployed engine (nominally `dropout=0.1`) and a `seq-bank` twin at
**explicit `dropout=0.0`** produce **the same 1,431 ranked lists, byte for byte**. That is
now a *measured* demonstration that the deployed engine's declared dropout is a no-op,
not merely an inference from reading the constructor. What this campaign fails to
establish is only the *consequence* — whether fixing it pays.

**4. Fifth confirmation that validation loss does not rank retrieval, and this time it is
adversarial.** E0 has the **better** best-val InfoNCE (6.5763 vs 6.5767) and the **worse**
hit count (168 vs 172). 0.0004 of val loss, 4 hits, opposite direction, at byte-identical
parameters and the same early-stop epoch (24 both). Consistent with the tower-bank
campaign's Spearman(val, hits) = −0.2954 (p = 0.407); keep the epoch/val table
diagnostic-only.

**5. Wall-clock again carried no signal.** E0 6.1 min, E1 7.2 min, identical epoch counts
(24 both). Consistent with the registered pitfall (D2 15.7 min vs R2 4.7 min at identical
params, e14 both) — shared-CPU noise, and this run pair happens to be 2–3× faster than
that pair on the same worker for the same work.

**6. The trade-off picture, such as it is, is benign.** E1 moves every facet the same
direction or negligibly: H@10 +0.00044, music@10 +0.0011, mrr +0.00098, mood_coh +0.0053,
`artist_conc` −0.0010, ild −0.0014, artist_recall −0.0007, genre_recall −0.0014. There is
no visible regularization cost on the crown facets — no reason to expect that a
better-powered version of this test would find a *penalty*, only that it might find no
gain.

## Best on record after this work

| surface | best | value | run / model |
|---|---|---|---|
| one-shot recall@10, `seq-20260715-131139` | `seq-blend` markov-gated champion | **0.23201** | unchanged by this campaign |
| best single-GRU width point (not significant vs h256) | h512 | 186/1431 = 0.129979 | `run-20260802-183404-91e31-seq-nexttrack` (unchanged) |
| best architecture on the walk | dual `latent/cummean` f0 @ a0.8/s0.5 (**DEPLOYED**) | vibe 0.610 / stride_err 0.144 | `run-20260725-143502-3e5d4-seq-dualgru` (unchanged — no walk measurement was taken here) |
| best `latent/cummean` f0 one-shot point | **`seq-bank` `latent+cummean` h256 dropout 0.1** | 172/1431 = 0.120196 | `run-20260802-214618-a38be-seq-bank` (**a point estimate, not a significant win**) |
| dropout at `fusion_layers=0` | **1 of 2 view pairs shows CI>0; the deployed pair is UNRESOLVED, not negative** | — | this report + `2026-08-02-tower-bank-parallel-towers.md` |

**No champion changed.** The best-models group needs no intervention from this campaign
on merit — but the server's deterministic recompute ran on both completions regardless.

## Follow-ups

1. **Settle it with seeds, not with another single split (3–6 runs, needs user
   authorization).** This is the same experiment at `seed` ∈ {1337, 7, 42} on both dropout
   values, so the contrast is a 3-seed paired mean rather than one 16-row discordance. The
   record's own noise rule says a single-split lead of this size is "at least equal, likely
   better — needs the 3-seed check", and that is exactly the position E1 is in. This is the
   *only* clean way to close the question the campaign opened.
2. **Do not redeploy on this evidence.** E1 is the better point estimate on every crown
   facet and is bit-comparable to the deployed engine in cost, but (a) the recall gain is
   not significant, and (b) **no walk measurement was taken** — and the standing rule is
   that two TRADE arms in the last campaign flipped to Δvibe CI<0 at exactly the deployed
   cell a0.8/s0.5. Redeployment is the user's call and would need the walk read first.
3. **If seeds confirm, the fix generalizes beyond the deployed arm.** Every f0 arm on
   record is affected, so a confirmed effect is a retro-correction to the whole f0 family,
   not a one-model tune — and the three recorded "f0 beats f1" gaps (+0.016 / +0.020 /
   +0.0266) stay flagged as **not like-regularized** either way.
4. **A cheaper power fix exists and is worth considering before spending seeds:** this
   contrast has only 16 discordant rows because both arms are weak in absolute terms
   (~12% recall). Measuring the same dropout delta on a *stronger* base (the champion
   blend's GRU leg) would put more discordant mass in the test and resolve the same
   mechanism at lower run cost — but it changes the family, so it is a design question,
   not a runner's.

## Provenance

`registry.toml` carries the additive `"latent+cummean"` option on the `seq-bank` `views`
enum (made in this campaign's setup step, before any run, with the server restarted at the
zero-live-runs gate at that time). `predictors/seq_bank.py` is **unchanged** since the
tower-bank campaign (`7c58d53a2d3c3750502050afaabde238`). Both runs' artifacts (4 each)
are in Postgres via the worker upload path. This report and the
`experiments/PROJECT-FACTS.md` reconciliation are left uncommitted for the user.
