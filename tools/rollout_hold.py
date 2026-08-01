#!/usr/bin/env python3
"""`hold@N` — a GENERATION metric for autoregressive session rollouts.

WHY THIS EXISTS

Every metric this project ranks on is a ONE-STEP metric. `recall@10` asks whether
the true next track of a held-out session landed in the top 10. `holisticness@10`
(and its mood_coh / ild / artist_adj / music_rel factors) scores the internal
composition of ONE ranked list produced from ONE held-out prefix. Neither ever
feeds a model its own output.

But the product is a session GENERATOR: it takes a seed and rolls forward 20+
steps, each pick becoming the context for the next. The dominant failure mode in
that regime is not a bad step, it is a bad TRAJECTORY — a rollout whose energy
decays monotonically until a party seed has become a 3am comedown. Every
individual transition can be plausible while the arc is wrong, so a one-step
metric is structurally blind to it: the failure only exists once the model eats
its own predictions.

Measured instance (2026-07-27, 8 paired held-out seeds, 30-track anchored
prefixes, 20 steps): `seq-dualgru` faded 0.033 in energy across the rollout's
halves against `seq-nexttrack`'s 0.135, paired Δ −0.102 [−0.194, −0.019], and
that verdict survived all 8 leave-one-seed-out refits. On the SAME pair,
holisticness@10 differed by 0.003 and every level metric tied. The listener heard
it immediately. That gap is what this metric is for.

DEFINITION

On an N-step rollout from a seed prefix, for each legible axis a (energy,
valence, acousticness, tempo):

    fade_a  = |mean(a over 2nd half) - mean(a over 1st half)| / band_a
    level_a = |mean(a over rollout)  - seed_a|                / band_a

`band_a` is the p10..p90 width of the SAME statistic measured on the listener's
own real sessions, so axes with incompatible units (a 0-1 valence and a BPM)
combine, and so "a big fade" means big relative to how this person actually
listens rather than relative to an arbitrary constant.

    stasis  = max(0, jump_p10 - mean step-to-step sonic jump) / jump_p10

    hold@N  = 1 - clamp(mean_a(fade_a) + stasis, 0, 1)

WHY `stasis` IS NOT OPTIONAL — the anti-gaming term

The lesson of the 2026-07-25 dual-tower campaign is that a one-sided coherence
reward is degenerate. Its `cummean/cummean` arm posted holisticness@10 0.19715,
the highest on record, by predicting the session centroid forever — mood-coherent,
non-eager, and nearly useless (recall@10 0.066 = 95 of 1,431 against a baseline's
176). `seq-mood` does the same thing (H 0.2332 at recall 0.024). Both were caught
only by an external recall floor bolted onto the decision rule, because
holisticness rewards mood_coh monotonically and has no notion of moving too
little.

A pure fade metric has exactly the same hole: 20 near-identical tracks fade by
zero and would score a perfect 1.0. `stasis` closes it from inside the metric
rather than from a side condition — a rollout that moves LESS than the listener's
own 10th-percentile step is penalised on the same scale as one that lurches. The
target is a band, not an extremum, which is the property `holisticness` lacks.

VALIDITY — read before promoting this to anything

`hold@N` is a PROXY for "I would enjoy this playlist", and its only ground truth
is listener preference. As of 2026-07-27 that ground truth is n=1 blind vote
(playlist `cenamos`, dualgru preferred), and the playlist-lab blind-vote tally is
the instrument for collecting more. Until the votes can actually discriminate
these weights, this metric belongs in the DISPLAY-ONLY tier that `music@10`
started in. Promoting an unvalidated composite to primary is how the project
ended up needing a recall floor to defend `holisticness` in the first place;
replacing one unvalidated composite with another would repeat that, not fix it.

Equal axis weights are a placeholder, chosen because there is no data to fit
better ones. Of the three fade axes measured so far only ENERGY is robust to
leave-one-out; acousticness and valence point the same way and collapse when any
single seed is dropped, and all three are correlated (they move together in
music), so they are closer to one finding seen three ways than to three
confirmations.

USAGE

  # 1. calibrate the bands against the listener's real sessions (once per dataset)
  tools/rollout_hold.py calibrate --model data/models/<promoted> \
      --out data/rollout-bands.json

  # 2. score rollout payloads (the playlist lab's POST .../extend response)
  tools/rollout_hold.py score --bands data/rollout-bands.json a.json b.json

  # 3. paired A/B over a sweep file {seed: {tag: payload}}
  tools/rollout_hold.py paired --bands data/rollout-bands.json --sweep sweep.json

Run `calibrate` under the predictor venv (needs the artifact + Qdrant for the
mood table); `score` / `paired` are pure JSON and run anywhere.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

AXES = ("energy", "valence", "acousticness", "tempo")
# The rollout window a real session is compared over. Sessions shorter than this
# are still used (whole session); the halves are what matter, not the length.
MIN_SESSION = 12


def halves(values: list[float]) -> tuple[float, float] | None:
    """Mean of the first and second half of a series, ignoring gaps."""
    v = [x for x in values if x is not None and x == x]
    if len(v) < 4:
        return None
    h = len(v) // 2
    return st.mean(v[:h]), st.mean(v[h:])


# --------------------------------------------------------------------------- #
# calibrate — the listener's own fade distribution                            #
# --------------------------------------------------------------------------- #
def calibrate(model_dir: Path, window: int, out: Path) -> dict:
    """Measure |2nd-half − 1st-half| per axis over real sessions.

    This is the band a rollout's fade is judged against. It must come from real
    listening rather than from theory: real sessions are NOT flat (people do wind
    down), so a metric that rewarded zero fade would rank an unnaturally rigid
    generator above one that behaves like its listener."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "predictors"))
    import numpy as np
    from seq_common import load_artifact          # noqa: E402
    import seq_extend as se                       # noqa: E402

    art = load_artifact(model_dir)
    mood = se.load_mood_table(art)
    fades: dict[str, list[float]] = {a: [] for a in AXES}
    used = 0
    for s in art.train_sessions.tolist():
        seq = art.session(int(s)).astype(int)
        if seq.size < MIN_SESSION:
            continue
        seq = seq[:window]
        used += 1
        for a in AXES:
            hv = halves([float(x) for x in mood[a][seq]])
            if hv:
                fades[a].append(abs(hv[1] - hv[0]))
    bands = {}
    for a in AXES:
        if len(fades[a]) < 50:
            continue
        p10, p50, p90 = (float(x) for x in np.percentile(fades[a], [10, 50, 90]))
        bands[a] = {"p10": round(p10, 5), "p50": round(p50, 5), "p90": round(p90, 5),
                    "width": round(max(p90 - p10, 1e-9), 5), "n": len(fades[a])}
    doc = {"kind": "rollout-bands", "window": window, "sessions": used,
           "dataset": art.manifest.get("dataset_id") or str(model_dir.name),
           "fade": bands}
    out.write_text(json.dumps(doc, indent=1))
    print(f"calibrated on {used} real sessions (window {window}) -> {out}")
    for a, b in bands.items():
        print(f"  {a:<14} fade p10 {b['p10']:.4f}  p50 {b['p50']:.4f}  p90 {b['p90']:.4f}")
    return doc


# --------------------------------------------------------------------------- #
# score — one rollout                                                         #
# --------------------------------------------------------------------------- #
def score(payload: dict, bands: dict) -> dict:
    """`hold@N` and its parts for one lab rollout payload."""
    stops = payload.get("stops") or []
    report = payload.get("mood_report") or {}
    ax = {a["key"]: a for a in report.get("axes", [])}
    fade_bands = bands.get("fade", {})

    fades, levels, detail = [], [], {}
    for a in AXES:
        band = fade_bands.get(a)
        if not band or a not in ax:
            continue
        series = [s.get("mood", {}).get(a) for s in stops]
        hv = halves(series)
        if hv is None:
            continue
        raw_fade = abs(hv[1] - hv[0])
        f = raw_fade / band["width"]
        fades.append(f)
        detail[a] = {"fade_raw": round(raw_fade, 4), "fade_norm": round(f, 4),
                     "first_half": round(hv[0], 4), "second_half": round(hv[1], 4)}
        # Level deviation rides along as a diagnostic; it is NOT in hold@N,
        # because sitting at a different level than the seed is a legitimate
        # stylistic choice while decaying away from it is the failure mode.
        seed, val = ax[a].get("seed"), ax[a].get("value")
        lb = ax[a].get("band")
        if seed is not None and val is not None and lb:
            w = max(lb[2] - lb[0], 1e-9)
            levels.append(abs(val - seed) / w)
            detail[a]["level_norm"] = round(abs(val - seed) / w, 4)

    # stasis: moving LESS than the listener's own floor is its own failure.
    #
    # The shortfall is measured as a FRACTION OF THE FLOOR, not of the band width.
    # Normalising by the width was the first implementation and it was too weak to
    # do the job the term exists for: this listener's jump band is 0.21..0.98, so a
    # rollout of 20 near-identical tracks (jump 0.01) came out at only 0.26 penalty
    # -> hold 0.740, which OUTRANKED a real working generator's 0.477. A metric
    # whose anti-gaming guard can be beaten by the degeneracy it targets is worse
    # than no guard, because it launders the degeneracy as a good score.
    # Relative-to-floor: that same rollout is 95% short of the floor -> 0.95.
    stasis = 0.0
    sj = ax.get("sonic_jump")
    if sj and sj.get("value") is not None and sj.get("band"):
        p10 = sj["band"][0]
        if p10 > 0:
            stasis = max(0.0, (p10 - sj["value"]) / p10)

    fade = st.mean(fades) if fades else None
    hold = None if fade is None else max(0.0, 1.0 - min(fade + stasis, 1.0))
    return {"hold": None if hold is None else round(hold, 4),
            "fade": None if fade is None else round(fade, 4),
            "stasis": round(stasis, 4),
            "level_dev": round(st.mean(levels), 4) if levels else None,
            "n_stops": len(stops), "axes": detail}


def _fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def paired(sweep: dict, bands: dict) -> None:
    """Paired A/B across seeds, house statistic: 2,000-resample bootstrap over
    the SEED (the unit of pairing), rng 1337, 95% percentile CI."""
    import numpy as np

    tags = [t for t in next(iter(sweep.values())) if t != "seed"]
    rows = {t: {k: score(p[t], bands) for k, p in sweep.items()} for t in tags}
    # A rollout whose stops carry no acoustics at all scores None. Dropping the
    # whole PAIR (not just the one arm) keeps the comparison paired, and the count
    # is printed because a silently smaller n is how a sweep overstates itself.
    usable = [k for k in sweep if all(rows[t][k]["hold"] is not None for t in tags)]
    dropped = len(sweep) - len(usable)
    if dropped:
        print(f"note: {dropped} of {len(sweep)} pairs dropped — a rollout in each "
              f"had no acoustic coverage to score\n")
    print(f"{'seed':<9}{'why':<17}" + "".join(f"{t + ' hold':>17}" for t in tags))
    for k, p in ((k, sweep[k]) for k in usable):
        why = (p.get("seed") or {}).get("why", "")
        cells = "".join(
            f"{_fmt(rows[t][k]['hold']) + ' (f' + _fmt(rows[t][k]['fade'], 2) + ')':>17}"
            for t in tags)
        print(f"{k:<9}{why:<17}{cells}")
    print()
    for t in tags:
        hs = [rows[t][k]["hold"] for k in usable]
        print(f"  mean hold@N  {t:<12} {st.mean(hs):.4f}  (n={len(hs)})")
    if len(tags) == 2 and len(usable) >= 3:
        a = np.array([rows[tags[0]][k]["hold"] for k in usable])
        b = np.array([rows[tags[1]][k]["hold"] for k in usable])
        d = a - b
        rng = np.random.default_rng(1337)
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        verdict = (f"{tags[0]} holds better" if lo > 0
                   else f"{tags[1]} holds better" if hi < 0 else "TIE (CI straddles 0)")
        print(f"\n  paired Δ hold@N ({tags[0]} − {tags[1]}) = {d.mean():+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]  ->  {verdict}")
        # Leave-one-out: at n<=10 seeds a single seed can carry a verdict, and a
        # verdict that does not survive its own jackknife is not a finding.
        survived = 0
        for i in range(len(d)):
            dd = np.delete(d, i)
            r2 = np.random.default_rng(1337)
            bt = np.array([dd[r2.integers(0, len(dd), len(dd))].mean() for _ in range(2000)])
            l2, h2 = np.percentile(bt, [2.5, 97.5])
            survived += (l2 > 0) if lo > 0 else (h2 < 0) if hi < 0 else 0
        print(f"  leave-one-seed-out: verdict survives {survived}/{len(d)} refits")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="measure real-session fade bands")
    c.add_argument("--model", type=Path, required=True, help="promoted model dir")
    c.add_argument("--window", type=int, default=20, help="rollout length to match")
    c.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("score", help="score rollout payload(s)")
    s.add_argument("--bands", type=Path, required=True)
    s.add_argument("payloads", type=Path, nargs="+")

    p = sub.add_parser("paired", help="paired A/B over a sweep file")
    p.add_argument("--bands", type=Path, required=True)
    p.add_argument("--sweep", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "calibrate":
        calibrate(args.model, args.window, args.out)
        return
    bands = json.loads(args.bands.read_text())
    if args.cmd == "score":
        for f in args.payloads:
            r = score(json.loads(f.read_text()), bands)
            print(f"{f.name}: hold@{r['n_stops']} {_fmt(r['hold'], 4)}  "
                  f"fade {_fmt(r['fade'])}  stasis {_fmt(r['stasis'])}  "
                  f"level_dev {_fmt(r['level_dev'])}")
            for a, d in r["axes"].items():
                print(f"    {a:<14} {d['first_half']:.3f} -> {d['second_half']:.3f}"
                      f"  fade_norm {d['fade_norm']:.3f}")
    else:
        paired(json.loads(args.sweep.read_text()), bands)


if __name__ == "__main__":
    main()
