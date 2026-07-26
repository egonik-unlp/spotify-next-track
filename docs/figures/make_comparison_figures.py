#!/usr/bin/env python3
"""Generate the vector-PDF result plots for docs/model-comparison.tex.

Scope: the four figures owned by this script are the model-comparison
document's own result plots.  Every other nexttrack_*.pdf in this
directory is owned by make_figures.py and must NOT be regenerated here.

All data is HARD-CODED and transcribed from verified sources so the
document is reproducible from this file alone:
  * live API run records (GET /api/runs, /api/runs/<id>/predictions),
  * the campaign reports in experiments/,
  * experiments/PROJECT-FACTS.md.
Each fig_<name>() cites its provenance in the comment block above it.
Run from anywhere: outputs land next to this script (docs/figures/).

    predictors/.venv/bin/python docs/figures/make_comparison_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent

# Same style block as make_figures.py — these figures must read as family.
plt.rcParams.update(
    {
        "figure.figsize": (6.0, 3.5),
        "figure.constrained_layout.use": True,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)

ACCENT = "#2563eb"  # blue   — primary series, baselines, controls
GOOD = "#16a34a"  # green  — winners, free arms, desired direction
BAD = "#dc2626"  # red    — losers, floors, harmful arms
MUTED = "#9ca3af"  # gray   — references, ties, tie bands

# Per-family colours, reused verbatim from make_figures.py's
# fig_nexttrack_music_scatter so the two documents index onto each other.
FAM = {
    "blend": ACCENT,  # seq-blend
    "nexttrack": GOOD,  # seq-nexttrack
    "dualgru": MUTED,  # seq-dualgru
    "stacker": "#0891b2",  # seq-stacker (cyan)
    "ann": "#ea580c",  # seq-ann (orange)
    "markov": "#9333ea",  # seq-markov (purple) — a lookup, not a model
}
INK = "#374151"
INK2 = "#6b7280"

afmt = FuncFormatter(lambda v, _: f"{v:.3f}")

# Canonical split seq-20260715-131139 constants (verified against the
# dataset artifact data/datasets/seq-20260715-131139/).
N_TEST = 1431
N_WARM = 643
N_COLD = 788
COLD_RATE = 0.550664
SEEN_ONLY_CEILING = 0.449336  # 643/1431

MARKOV_BAR = 0.106918  # run-20260722-165345-047cc-seq-markov, 153/1431
GRU_BAR = 0.122991  # run-20260726-183333-9a0b2-seq-nexttrack, 176/1431
CHAMP = 0.211740  # run-20260726-183333-1c361-seq-blend, 303/1431
RECALL_FLOOR = GRU_BAR - 0.015  # 0.10799 — the pre-registered anti-degenerate floor
H_BAR = 0.024872  # run-20260725-143502-be11b, highest gen-3 H among healthy models
# The gen-3 H band spanned by the TEN separately-trained architecture arms that
# carry a live value (PROJECT-FACTS.md, re-derived from stored per-row facets):
# 0.02487 h425 ... 0.02127 T1-f1.  Spread 0.0036 against hw(Delta-H) ~0.0018.
H_BAND_LO = 0.021272
H_BAND_HI = 0.024872


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {name}")


# ---------------------------------------------------------------------------
# Fig 1: the model landscape.
#
# EVERY IN-SCOPE TRAINED MODEL on the canonical split, on recall@10.
# In scope = parametric, actually trained: seq-blend, seq-nexttrack,
# seq-dualgru, seq-stacker, seq-ann.  OUT of scope and therefore drawn as
# REFERENCE LINES only: the first-order Markov lookup (0.106918) and the
# popularity/recency floors (popularity 0.00070, off-scale).
#
# Eval-time MMR re-rank arms are EXCLUDED: they share bit-identical trained
# weights with their parent, so they are a policy on a model, not a model.
# They get their own figure (mc_lambda_frontier.pdf).
#
# Only ONE marginal 95% CI is on record for these rows (the champion's,
# [0.191, 0.233], from 2026-07-18-nexttrack-literature-fit-campaign.md); the
# record's decisive statistic is the PAIRED per-session delta, so the
# family-level paired half-width is drawn as an explicit resolution bar
# instead of fabricating marginal intervals.
#
# Sources: GET /api/runs; 2026-07-15c-nexttrack-phase2-model-sweep.md,
# 2026-07-18-nexttrack-literature-fit-campaign.md,
# 2026-07-18b-nexttrack-projection-registration-and-scan.md,
# 2026-07-25-nexttrack-dual-tower-fusion-scan.md,
# 2026-07-26-nexttrack-pre-encoder-scan.md, PROJECT-FACTS.md.
# ---------------------------------------------------------------------------

# (label, recall@10, hits, family, marginal CI or None, role)
LANDSCAPE = [
    ("proj-blend $\\tau$0.10 (scan cell)", 0.222921, 319, "blend", None, "scan"),
    ("$R'{+}M{+}C'$ $\\tau$0.07 — CHAMPION", 0.211740, 303, "blend", (0.191, 0.233), "champ"),
    ("proj-blend rank-64", 0.205451, 294, "blend", None, ""),
    ("proj-blend $\\tau$0.05", 0.192872, 276, "blend", None, ""),
    ("$R{+}M{+}C$ (prior crown)", 0.185884, 266, "blend", None, ""),
    ("$R{+}M$ (2-leg)", 0.171209, 245, "blend", None, ""),
    ("proj-blend BPR, full rank", 0.157932, 226, "blend", None, ""),
    ("proj-blend BPR, rank-64", 0.152341, 218, "blend", None, ""),
    ("GRU h454 (width ctrl)", 0.125087, 179, "nexttrack", None, ""),
    ("GRU h256 $+$ anti-eager $\\beta$0.05", 0.123690, 177, "nexttrack", None, ""),
    ("dual A-bare/B-preMLP256, $f0$", 0.123690, 177, "dualgru", None, ""),
    ("GRU h256 infonce — workhorse bar", 0.122991, 176, "nexttrack", None, "bar"),
    ("GRU h425 (width ctrl)", 0.120894, 173, "nexttrack", None, ""),
    ("LSTM h256", 0.118798, 170, "nexttrack", None, ""),
    ("dual latent/latent, $f0$", 0.118798, 170, "dualgru", None, ""),
    ("dual latent/cummean, $f0$", 0.117400, 168, "dualgru", None, ""),
    ("GRU h297 (width ctrl)", 0.113208, 162, "nexttrack", None, ""),
    ("MLP256$\\to$GRU256 pre-encoder", 0.109015, 156, "nexttrack", None, ""),
    ("dual latent/latent, $f1$", 0.102725, 147, "dualgru", None, ""),
    ("dual both-preMLP256, $f1$", 0.097834, 140, "dualgru", None, ""),
    ("dual A-bare/B-preMLP256, $f1$", 0.097135, 139, "dualgru", None, ""),
    ("dual latent/cummean, $f1$", 0.097135, 139, "dualgru", None, ""),
    ("dual latent/delta, $f1$", 0.091544, 131, "dualgru", None, ""),
    ("ANN pooled-MLP (PCA-128, off-split)", 0.090147, 129, "ann", None, "offsplit"),
    ("dual delta/cummean, $f1$", 0.086653, 124, "dualgru", None, ""),
    ("XGB stacker $M{+}G{+}A$", 0.085255, 122, "stacker", None, ""),
    ("dual cummean/cummean, $f1$", 0.066387, 95, "dualgru", None, ""),
    ("dual delta/delta, $f1$", 0.058001, 83, "dualgru", None, ""),
]


def fig_model_landscape():
    rows = sorted(LANDSCAPE, key=lambda r: -r[1])
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(6.8, 6.1))

    # Out-of-scope reference lines.
    ax.axvline(
        MARKOV_BAR,
        color=MUTED,
        ls="--",
        lw=1.3,
        zorder=1,
        label=f"$M$ Markov lookup bar ({MARKOV_BAR:.3f}) — NOT a compared model",
    )
    ax.axvline(
        GRU_BAR,
        color=INK,
        ls=":",
        lw=1.0,
        alpha=0.65,
        zorder=1,
        label=f"single-model bar: GRU h256 infonce ({GRU_BAR:.3f})",
    )

    for yi, (label, v, hits, fam, ci, role) in zip(y, rows):
        c = FAM[fam]
        if ci is not None:
            ax.plot([ci[0], ci[1]], [yi, yi], color=c, lw=1.6, solid_capstyle="butt", zorder=3)
            for b in ci:
                ax.plot([b, b], [yi - 0.22, yi + 0.22], color=c, lw=1.2, zorder=3)
        if role == "champ":
            ax.plot(
                v, yi, marker="*", ms=13, color=c, mec="black", mew=0.6, ls="", zorder=6
            )
        elif role == "offsplit":
            ax.plot(v, yi, marker="o", ms=6, mfc="white", mec=c, mew=1.4, ls="", zorder=5)
        elif role == "scan":
            ax.plot(v, yi, marker="o", ms=6, mfc="white", mec=c, mew=1.4, ls="", zorder=5)
        elif role == "bar":
            ax.plot(
                v, yi, marker="s", ms=6.4, color=c, mec="black", mew=0.5, ls="", zorder=6
            )
        else:
            ax.plot(
                v, yi, marker="o", ms=5.4, color=c, mec="white", mew=0.4, ls="", zorder=5
            )
        # Value labels are right-aligned in a fixed column so they never
        # collide with the two reference lines.
        ax.text(
            0.2725,
            yi,
            f"{v:.4f} ({hits})",
            va="center",
            ha="right",
            fontsize=6.1,
            color=INK2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    for lbl, r in zip(ax.get_yticklabels(), rows):
        if r[5] in ("champ", "bar"):
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    ax.set_xlabel("Recall@10 (value and raw hits of 1,431 test sessions)")
    ax.xaxis.set_major_formatter(afmt)
    ax.set_xlim(0.045, 0.276)
    ax.set_ylim(-2.6, len(rows) + 3.5)
    ax.grid(axis="y", visible=False)

    # Explicit resolution bar: the record's decisive statistic is the paired
    # per-session delta, whose family half-width at n=1,431 is ~0.0124.
    ybar = -1.9
    ax.errorbar(
        0.105,
        ybar,
        xerr=0.0124,
        color=BAD,
        lw=1.5,
        capsize=3,
        marker="|",
        ms=0,
        zorder=6,
    )
    ax.text(
        0.1262,
        ybar,
        "paired-$\\Delta$ resolution at $n{=}1{,}431$: $\\pm0.0124$ half-width\n"
        "(measured mean over the 2026-07-25/26 batches)",
        va="center",
        ha="left",
        fontsize=6.4,
        color=INK,
    )

    handles, labels = ax.get_legend_handles_labels()
    for fam, name in [
        ("blend", "seq-blend (fixed z-blend; $R'$/$C'$ legs learned)"),
        ("nexttrack", "seq-nexttrack (GRU / LSTM)"),
        ("dualgru", "seq-dualgru (two towers $+$ fusion)"),
        ("stacker", "seq-stacker (XGB meta-learner)"),
        ("ann", "seq-ann (pooled MLP)"),
    ]:
        handles.append(plt.Line2D([], [], color=FAM[fam], marker="o", ls="", label=name))
    handles.append(
        plt.Line2D(
            [],
            [],
            color=MUTED,
            marker="o",
            mfc="white",
            mew=1.4,
            ls="",
            label="hollow: scan cell held sub-noise, or off-split",
        )
    )
    handles.append(
        plt.Line2D([], [], color=FAM["blend"], marker="*", ms=11, mec="black",
                   mew=0.6, ls="", label="star: the promoted champion")
    )
    handles.append(
        plt.Line2D([], [], color=FAM["nexttrack"], marker="s", ms=6, mec="black",
                   mew=0.5, ls="",
                   label="square: the single-model bar (also its own dotted line)")
    )
    ax.legend(handles=handles, loc="upper left", fontsize=6.2, framealpha=0.95)

    ax.set_title(
        "Twenty-seven canonical-split arms plus the off-split ANN twin;\n"
        "only NEW SIGNAL moves the axis — a content leg, then a learned metric"
    )
    save(fig, "mc_model_landscape.pdf")


# ---------------------------------------------------------------------------
# Fig 2: the two boards are near-inverted.
#
# recall@10 (the record's comparability currency) against LIVE-GENERATION
# holisticness@10 (the server's declared primary metric,
# GET /api/best-models -> primary_metric).  Only runs whose stored H is
# generation 3 are plotted:
#   H = clamp(mood_coh,0) x ild x (1 - max(artist_adj, artist_conc))
#       x clamp((music@10 - 0.34)/0.66, 0, 1)
# Runs whose stored H is gen 1 or gen 2 (nine on this split, all remote-only
# runs that never emitted artist_conc_at_k) are EXCLUDED rather than
# rescaled — that includes the degenerate cummean/cummean arm, which is why
# the degenerate corner is annotated as a REGION, with its gen-1 value
# labelled as gen-1.
#
# Sources: GET /api/runs, GET /api/best-models;
# 2026-07-26-nexttrack-mmr-lambda-frontier.md,
# 2026-07-25-nexttrack-dual-tower-fusion-scan.md, PROJECT-FACTS.md.
# ---------------------------------------------------------------------------

# (recall, H_gen3, family, is_mmr_rerank, label or None)
BOARDS = [
    (0.211740, 0.008603, "blend", False, "$R'{+}M{+}C'$ champion"),
    (0.210342, 0.010435, "blend", True, None),
    (0.193571, 0.015289, "blend", True, None),
    (0.171209, 0.008259, "blend", False, "$R{+}M$"),
    (0.164221, 0.023576, "blend", True, None),
    (0.109713, 0.031626, "blend", True, "$R'{+}M{+}C'$ at $\\lambda$0.3\nTIER-3 REJECTED, $-0.102$ recall"),
    (0.125087, 0.023988, "nexttrack", False, None),
    (0.123690, 0.024755, "nexttrack", False, None),
    (0.122991, 0.024538, "nexttrack", False, "GRU h256 bar"),
    (0.120894, 0.024872, "nexttrack", False, None),
    (0.120894, 0.026761, "nexttrack", True, None),
    (0.120196, 0.030355, "nexttrack", True, "GRU $+$ MMR $\\lambda$0.7/pool 50"),
    (0.119497, 0.029458, "nexttrack", True, None),
    (0.118798, 0.022941, "nexttrack", False, None),
    (0.114605, 0.034171, "nexttrack", True, "GRU $+$ MMR $\\lambda$0.7/pool 200\n(crown rank 1, 164/1431)"),
    (0.109015, 0.024102, "nexttrack", False, None),
    (0.118798, 0.023785, "dualgru", False, None),
    (0.117400, 0.021520, "dualgru", False, None),
    (0.102725, 0.022841, "dualgru", False, None),
    (0.099930, 0.021877, "dualgru", False, None),
    (0.099930, 0.022050, "dualgru", False, None),
    (0.097135, 0.021272, "dualgru", False, None),
]

LABEL_OFFSET = {
    "$R'{+}M{+}C'$ champion": (-0.0045, -0.0009, "right"),
    "$R{+}M$": (0.0035, 0.0004, "left"),
    "$R'{+}M{+}C'$ at $\\lambda$0.3\nTIER-3 REJECTED, $-0.102$ recall": (-0.0026, 0.0000, "right"),
    "GRU h256 bar": (0.0000, 0.0000, "left"),  # drawn with an arrow, see below
    "GRU $+$ MMR $\\lambda$0.7/pool 50": (0.0040, -0.0008, "left"),
    "GRU $+$ MMR $\\lambda$0.7/pool 200\n(crown rank 1, 164/1431)": (0.0042, 0.0009, "left"),
}


def fig_two_boards():
    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    # The anti-degenerate recall floor: below it, an H gain is bought by
    # predicting worse.  Pre-registered as baseline minus 0.015.
    ax.axvspan(0.085, RECALL_FLOOR, color=BAD, alpha=0.07, zorder=0)
    ax.axvline(RECALL_FLOOR, color=BAD, ls="--", lw=1.1, zorder=1)
    ax.text(
        0.0875,
        0.0138,
        "DEGENERATE REGION — below the\n"
        "pre-registered $\\pm0.015$ recall floor\n"
        "(0.108). Crown gains here are bought\n"
        "by predicting worse: cummean/cummean\n"
        "reached $H_{\\mathrm{gen1}}$ 0.197 on 95 of 1,431\n"
        "hits (gen-1 value; it stores no gen-3 $H$,\n"
        "so it cannot be plotted here).",
        fontsize=6.2,
        color=BAD,
        va="top",
        ha="left",
        bbox=dict(fc="#f9fafb", ec=BAD, lw=0.5, alpha=0.92, boxstyle="round,pad=0.3"),
        zorder=7,
    )

    # The HEALTHY BAND: the 10 separately-trained architecture arms that carry a
    # live H span 0.02127-0.02487 (PROJECT-FACTS, re-derived).  Its width is only
    # ~2x the measured hw(Delta-H), so the ordering INSIDE it is noise -- while
    # the champion (far below) and the MMR arms (far above) are outside it.
    ax.axhspan(H_BAND_LO, H_BAND_HI, color=GOOD, alpha=0.10, zorder=0)
    ax.axhline(H_BAR, color=MUTED, ls=":", lw=1.1, zorder=1)
    ax.text(
        0.0878,
        H_BAND_HI + 0.0005,
        f"HEALTHY BAND {H_BAND_LO:.5f}$-${H_BAND_HI:.5f} "
        "— the 10 separately-trained architecture arms",
        fontsize=6.2,
        color=GOOD,
        ha="left",
        va="bottom",
    )

    for fam in ("dualgru", "nexttrack", "blend"):
        pts = [p for p in BOARDS if p[2] == fam]
        for r, h, _f, mmr, label in pts:
            c = FAM[fam]
            if mmr:
                ax.plot(r, h, marker="o", ms=7, mfc="white", mec=c, mew=1.5, ls="", zorder=5)
            else:
                ax.plot(
                    r, h, marker="o", ms=6.6, color=c, mec="white", mew=0.5, ls="", zorder=4
                )

    # The M lookup: plotted for orientation only, never a compared model.
    ax.plot(
        MARKOV_BAR,
        0.004409,
        marker="D",
        ms=6,
        color=FAM["markov"],
        mec="white",
        mew=0.5,
        ls="",
        zorder=5,
    )
    ax.text(
        MARKOV_BAR + 0.0035,
        0.004409 - 0.0004,
        "$M$ lookup (bar, not a compared model)",
        fontsize=6.3,
        color=FAM["markov"],
        va="center",
    )

    for r, h, fam, mmr, label in BOARDS:
        if label is None or label == "GRU h256 bar":
            continue
        dx, dy, ha = LABEL_OFFSET[label]
        ax.text(
            r + dx,
            h + dy,
            label,
            fontsize=6.4,
            color=FAM[fam],
            ha=ha,
            va="center",
            fontweight="bold" if label == "$R'{+}M{+}C'$ champion" else "normal",
        )

    # The GRU control sits inside a dense cluster; label it from clear space.
    ax.annotate(
        "GRU h256 bar\n($\\lambda{=}1.0$ control)",
        xy=(GRU_BAR, 0.024538),
        xytext=(0.1292, 0.0190),
        fontsize=6.4,
        color=FAM["nexttrack"],
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color=FAM["nexttrack"], lw=0.8),
    )

    # The inversion, stated with the two verified endpoints.
    ax.annotate(
        "",
        xy=(0.2103, 0.0100),
        xytext=(0.1163, 0.0333),
        arrowprops=dict(
            arrowstyle="<->", color=INK, lw=1.0, ls="--", shrinkA=6, shrinkB=6,
            connectionstyle="arc3,rad=-0.22",
        ),
    )
    ax.text(
        0.1962,
        0.0222,
        "THE INVERSION (dashed guide)\nrecall leader is crown rank 8 of 9;\ncrown leader is an eval-time\nre-rank of the plain GRU",
        fontsize=6.4,
        color=INK,
        ha="center",
        va="center",
        bbox=dict(fc="#f9fafb", ec="#d1d5db", lw=0.6, boxstyle="round,pad=0.3"),
    )

    # Resolution: the healthy-model H spread is only ~2x the measured
    # half-width, so the crown ordering carries no information.
    ax.errorbar(
        0.1345,
        0.0132,
        yerr=0.0018,
        color=BAD,
        lw=1.4,
        capsize=3,
        zorder=6,
    )
    ax.text(
        0.1375,
        0.0132,
        "measured hw($\\Delta H$) $\\approx$ 0.0018 against a\n"
        "healthy-model $H$ spread of only $\\approx$0.0036:\n"
        "the crown ORDERING carries no information",
        fontsize=6.2,
        color=BAD,
        ha="left",
        va="center",
    )

    ax.set_xlabel("Recall@10 — the record's comparability currency")
    ax.set_ylabel("holisticness@10 (gen 3, live)\n— the server's declared primary metric")
    ax.xaxis.set_major_formatter(afmt)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.set_xlim(0.085, 0.2295)
    ax.set_ylim(0.0025, 0.0375)

    handles, labels = ax.get_legend_handles_labels()
    for fam, name in [
        ("blend", "seq-blend"),
        ("nexttrack", "seq-nexttrack"),
        ("dualgru", "seq-dualgru"),
    ]:
        handles.append(plt.Line2D([], [], color=FAM[fam], marker="o", ls="", label=name))
    handles.append(
        plt.Line2D(
            [],
            [],
            color=MUTED,
            marker="o",
            mfc="white",
            mew=1.5,
            ls="",
            label="hollow: eval-time MMR re-rank of the filled parent",
        )
    )
    ax.legend(handles=handles, loc="upper right", fontsize=6.2, framealpha=0.95)

    ax.set_title(
        "The two boards are near-inverted: the recall champion sits BELOW the whole\n"
        "healthy band on the live crown, and no crown gain comes from better training"
    )
    save(fig, "mc_two_boards.pdf")


# ---------------------------------------------------------------------------
# Fig 3: the MMR lambda frontier, one traced curve per anchor.
#
# lambda is EVAL-ONLY: the trained weights are bit-identical across the arms
# of one anchor (the lambda=1.0 control reproduces its parent exactly), which
# is why the pairing is tight and the instrument is 3-33x sharper than the
# cross-model family figure.  MMR itself is greedy maximal-marginal-relevance
# re-ranking over the top mmr_pool candidates in the 517-d musical-distance
# space: pick argmax_i [ lambda*rel(i) - (1-lambda)*max_{j in S} cos(M_i,M_j) ].
#
# TIER-1 "free" = paired Delta-recall CI straddles zero AND Delta-H CI > 0.
# TIER-2 "priced" / TIER-3 "rejected" = Delta-recall CI < 0.
# NOTHING in this campaign was promoted; six definitions were registered.
#
# Source: 2026-07-26-nexttrack-mmr-lambda-frontier.md (all ten arms,
# run-20260726-1841xx), plus the lambda=1.0 controls run-20260726-183333-1c361
# and run-20260726-183333-9a0b2.
# ---------------------------------------------------------------------------

# (lambda, recall, H_gen3, tier)   tier: free | priced | rejected | control
CHAMP_ARMS = [
    (1.0, 0.211740, 0.008603, "control"),
    (0.9, 0.210342, 0.010435, "free"),
    (0.7, 0.193571, 0.015289, "priced"),
    (0.5, 0.164221, 0.023576, "rejected"),
    (0.3, 0.109713, 0.031626, "rejected"),
]
GRU_ARMS = [
    (1.0, 0.122991, 0.024538, "control"),
    (0.9, 0.120894, 0.026761, "free"),
    (0.8, 0.119497, 0.029458, "free"),
    (0.7, 0.114605, 0.034171, "priced"),
]
GRU_POOL50 = (0.7, 0.120196, 0.030355, "free")  # mmr_pool 50, off the pool-200 trace

TIER_EDGE = {"control": INK, "free": GOOD, "priced": "#9ca3af", "rejected": BAD}

# Per-point lambda-label placement (dx, dy, ha, va) — the GRU cluster is dense.
LAM_OFFSET = {
    ("nexttrack", 1.0): (0.0020, -0.0007, "left", "center"),
    ("nexttrack", 0.9): (-0.0019, 0.0000, "right", "center"),
    ("nexttrack", 0.8): (-0.0021, -0.0006, "right", "center"),
    ("nexttrack", 0.7): (0.0018, 0.0009, "left", "center"),
    ("blend", 1.0): (0.0022, -0.0009, "left", "center"),
    ("blend", 0.9): (0.0000, 0.0013, "center", "bottom"),
    ("blend", 0.7): (0.0000, 0.0013, "center", "bottom"),
    ("blend", 0.5): (0.0026, 0.0005, "left", "center"),
    ("blend", 0.3): (0.0000, 0.0021, "center", "bottom"),
}


def fig_lambda_frontier():
    fig, ax = plt.subplots(figsize=(6.8, 4.1))

    # The GRU's anti-degenerate recall floor.
    ax.axvspan(0.100, RECALL_FLOOR, color=BAD, alpha=0.08, zorder=0)
    ax.axvline(RECALL_FLOOR, color=BAD, ls="--", lw=1.1, zorder=1)
    ax.text(
        RECALL_FLOOR - 0.0021,
        0.0212,
        "GRU's $\\pm0.015$ recall floor (0.108)",
        fontsize=6.2,
        color=BAD,
        ha="center",
        va="center",
        rotation=90,
    )
    # The champion's own floor sits at its own control minus 0.015.
    ax.axvline(CHAMP - 0.015, color=BAD, ls=":", lw=1.0, zorder=1)
    ax.text(
        CHAMP - 0.015 + 0.0021,
        0.0212,
        "champion's own floor (0.197)",
        fontsize=6.2,
        color=BAD,
        ha="center",
        va="center",
        rotation=90,
    )

    ax.axhline(H_BAR, color=MUTED, ls=":", lw=1.1, zorder=1)
    ax.text(
        0.229,
        H_BAR + 0.0005,
        f"all-time healthy $H$ bar {H_BAR:.4f}",
        fontsize=6.3,
        color=INK2,
        ha="right",
        va="bottom",
    )

    for arms, fam, name, lw, z in [
        (CHAMP_ARMS, "blend", "$R'{+}M{+}C'$ champion (arm A) — shallow", 1.3, 2),
        (GRU_ARMS, "nexttrack", "single GRU h256 (arm B) — steep", 2.6, 3),
    ]:
        c = FAM[fam]
        xs = [a[1] for a in arms]
        ys = [a[2] for a in arms]
        ax.plot(xs, ys, color=c, lw=lw, ls="-", zorder=z, label=name)
        for lam, r, h, tier in arms:
            # Tier is a ring OUTSIDE the marker; family is the fill. Encoding
            # them as edge-vs-face on one marker made the edge dominate.
            ax.plot(
                r, h, marker="o", ms=12.5, mfc="none", mec=TIER_EDGE[tier], mew=1.5,
                ls="", zorder=4,
            )
            ax.plot(
                r, h, marker="o", ms=6.2, mfc=c, mec="white", mew=0.5, ls="", zorder=5
            )
            dx, dy, ha, va = LAM_OFFSET[(fam, lam)]
            ax.text(
                r + dx,
                h + dy,
                f"$\\lambda${lam:g}/200" if lam < 1.0 else "$\\lambda$1.0 (MMR off)",
                fontsize=6.3,
                color=c,
                ha=ha,
                va=va,
            )

    # The pool-50 arm is NOT part of the fixed-pool-200 sweep, so it is drawn
    # deliberately OFF the traced curve — plotting it on the line would imply a
    # monotone sweep it does not belong to.
    lam, r, h, tier = GRU_POOL50
    ax.plot(r, h, marker="o", ms=12.5, mfc="none", mec=GOOD, mew=1.5, ls="", zorder=4)
    ax.plot(
        r, h, marker="*", ms=11, mfc=FAM["nexttrack"], mec="white", mew=0.5, ls="", zorder=6
    )
    ax.text(
        r + 0.0024, h - 0.0011, "$\\lambda$0.7/50", fontsize=6.3,
        color=FAM["nexttrack"], ha="left", va="center",
    )
    ax.annotate(
        "$\\lambda$0.7, pool 50 — BEST FREE ARM (star, drawn OFF\n"
        "the pool-200 trace because it changes the pool).\n"
        "$H$ 0.0304, $+22.1\\%$ over the bar, at a recall\n"
        "tie ($\\Delta$ $-0.0029$, straddles zero)",
        xy=(r, h),
        xytext=(0.1352, 0.0322),
        fontsize=6.3,
        color=FAM["nexttrack"],
        va="center",
        ha="left",
        arrowprops=dict(arrowstyle="->", color=FAM["nexttrack"], lw=0.8),
    )

    # The verdict annotation: at the same lambda the GRU is 2.24x the champion.
    ax.annotate(
        "",
        xy=(0.1155, 0.0336),
        xytext=(0.1932, 0.0148),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0, ls="--", shrinkA=4, shrinkB=4),
    )
    ax.text(
        0.1585,
        0.0107,
        "dashed guide: at the SAME $\\lambda$0.7 the GRU reaches $H$ 0.0342\n"
        "and the champion only 0.0153 — a $2.24{\\times}$ gap. The champion pays\n"
        "twice: diversifying away from the seed artist its Markov leg lives on\n"
        "also walks it out of the prefix mood neighbourhood. At this MATCHED\n"
        "$\\lambda$ its mood_coh falls $0.398\\to0.364$ ($-8.5\\%$) against the GRU's\n"
        "$-4.2\\%$; only at $\\lambda$0.3, where no GRU arm was run, does it\n"
        "collapse to 0.238 ($-40\\%$).",
        fontsize=6.3,
        color=INK,
        ha="center",
        va="center",
        bbox=dict(fc="#f9fafb", ec="#d1d5db", lw=0.6, boxstyle="round,pad=0.32"),
    )

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("holisticness@10 (gen 3, live)")
    ax.xaxis.set_major_formatter(afmt)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.set_xlim(0.100, 0.2320)
    ax.set_ylim(0.005, 0.0375)

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color="white", marker="o", ms=9, mfc="none", mec=INK, mew=1.5,
                   ls="", label="ring $=$ $\\lambda{=}1.0$ control (MMR off)"),
        plt.Line2D([], [], color="white", marker="o", ms=9, mfc="none", mec=GOOD, mew=1.5,
                   ls="", label="TIER 1 FREE: $\\Delta H>0$, $\\Delta$recall straddles zero"),
        plt.Line2D([], [], color="white", marker="o", ms=9, mfc="none", mec=MUTED, mew=1.5,
                   ls="", label="TIER 2 priced: $\\Delta$recall CI$<$0"),
        plt.Line2D([], [], color="white", marker="o", ms=9, mfc="none", mec=BAD, mew=1.5,
                   ls="", label="TIER 3 rejected (frontier mapping only)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.1, framealpha=0.95)

    ax.set_title(
        "MMR is a free crown lever on the plain GRU, an expensive one on the champion:\n"
        "the GRU's trace climbs almost vertically, the champion's slides sideways"
    )
    save(fig, "mc_lambda_frontier.pdf")


# ---------------------------------------------------------------------------
# Fig 4: why fusion works — cold/warm coverage is almost complementary.
#
# A test target is COLD if its item index never appears in any of the 5,723
# TRAIN sessions.  788 of 1,431 targets (55.07%) are cold, so any model that
# can only retrieve items it observed in training — a co-occurrence table, a
# free learned item-embedding table, a Markov bigram — is capped at
# recall@10 643/1431 = 0.4493.
#
# Warm/cold splits for M, R and the champion were recomputed from
# GET /api/runs/<id>/predictions against the dataset artifact and reproduce
# the reported figures.  The R'/C' leg splits exist ONLY as off-server driver
# figures in 2026-07-18-nexttrack-literature-fit-campaign.md (no server run,
# no raw hit counts) and are hatched to mark that provenance.
#
# Sources: dataset artifact data/datasets/seq-20260715-131139/;
# GET /api/runs/<id>/predictions for run-20260722-165345-047cc (M),
# run-20260726-183333-9a0b2 (R), run-20260718-123738-38725 (R+M+C),
# run-20260722-165345-28dbb (R+M), run-20260726-183333-1c361 (champion);
# 2026-07-18-nexttrack-literature-fit-campaign.md for the R'/C' legs.
# ---------------------------------------------------------------------------

# (label, warm, cold, overall or None, report_only)
COLDWARM = [
    ("$M$\nMarkov\nlookup", 0.2255, 0.0102, 0.106918, False),
    ("$R$\nGRU h256", 0.1198, 0.1256, 0.122991, False),
    ("$C'$\nproj.\ncontent-kNN", 0.151, 0.152, None, True),
    ("$R'$\nproj. GRU", 0.159, 0.184, None, True),
    ("$R{+}M$", 0.2255, 0.1269, 0.171209, False),
    ("$R{+}M{+}C$", 0.2255, 0.1536, 0.185884, False),
    ("$R'{+}M{+}C'$\nchampion", 0.2395, 0.1891, 0.211740, False),
]

WARM_C = MUTED
COLD_C = ACCENT


def fig_cold_warm():
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(7.4, 3.8), gridspec_kw={"width_ratios": [2.95, 1.05]}
    )

    x = np.arange(len(COLDWARM))
    w = 0.38
    for i, (label, warm, cold, overall, report_only) in enumerate(COLDWARM):
        hatch = "///" if report_only else None
        axl.bar(
            i - w / 2,
            warm,
            w,
            color=WARM_C,
            edgecolor="white",
            lw=0.5,
            hatch=hatch,
            zorder=3,
        )
        axl.bar(
            i + w / 2,
            cold,
            w,
            color=COLD_C,
            edgecolor="white",
            lw=0.5,
            hatch=hatch,
            zorder=3,
        )
        # Stagger the pair's value labels when the two bars are near-equal.
        dy_cold = 0.005 + (0.016 if abs(warm - cold) < 0.014 else 0.0)
        axl.text(i - w / 2, warm + 0.005, f"{warm:.3f}", ha="center", fontsize=6.0, color=INK)
        axl.text(i + w / 2, cold + dy_cold, f"{cold:.3f}", ha="center", fontsize=6.0, color=INK)

    axl.set_xticks(x)
    axl.set_xticklabels([c[0] for c in COLDWARM], fontsize=6.6)
    axl.set_ylabel("Recall@10 within the half")
    axl.set_ylim(0, 0.375)
    axl.set_xlim(-0.62, len(COLDWARM) - 0.38)
    axl.yaxis.set_major_formatter(afmt)
    axl.grid(axis="x", visible=False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=WARM_C, label=f"WARM ({N_WARM} of {N_TEST})"),
        plt.Rectangle((0, 0), 1, 1, color=COLD_C, label=f"COLD ({N_COLD} of {N_TEST})"),
        plt.Rectangle(
            (0, 0), 1, 1, fc="white", ec=MUTED, hatch="///",
            label="report-provenance leg\n(no server run, no raw counts)",
        ),
    ]
    axl.legend(handles=handles, loc="upper right", fontsize=6.0, framealpha=0.95)
    axl.text(
        -0.55,
        0.370,
        "$M$ is structurally WARM-ONLY — 8 cold hits of 788,\n"
        "a $22{\\times}$ gap. $R$, $R'$ and $C'$ retrieve in a content\n"
        "space, so they need never have SEEN the target:\n"
        "cold $\\approx$ warm, and $R$ is slightly BETTER on cold.\n"
        "The fusion therefore lifts BOTH halves at once —\n"
        "the champion lands 154/643 warm and 149/788 cold.",
        fontsize=6.3,
        color=INK,
        ha="left",
        va="top",
        bbox=dict(fc="#f9fafb", ec="#d1d5db", lw=0.6, boxstyle="round,pad=0.3"),
    )
    axl.set_title(
        "The champion's legs are correct on almost disjoint halves\n"
        "of the test set — that is what the fusion buys"
    )

    # Right panel: the split's composition IS the seen-items-only ceiling.
    axr.bar(0, SEEN_ONLY_CEILING, 0.62, color=WARM_C, edgecolor="white", lw=0.6, zorder=3)
    axr.bar(
        0,
        COLD_RATE,
        0.62,
        bottom=SEEN_ONLY_CEILING,
        color=COLD_C,
        edgecolor="white",
        lw=0.6,
        zorder=3,
    )
    axr.axhline(SEEN_ONLY_CEILING, color=BAD, ls="--", lw=1.2, zorder=4)
    axr.text(
        0.38,
        SEEN_ONLY_CEILING + 0.020,
        "a seen-items-only\nmodel is capped at\n"
        f"recall@10 $\\leq$ {SEEN_ONLY_CEILING:.4f}",
        fontsize=6.2,
        color=BAD,
        ha="left",
        va="bottom",
    )
    axr.text(
        0,
        SEEN_ONLY_CEILING / 2,
        f"WARM\n{N_WARM}/{N_TEST}\n{SEEN_ONLY_CEILING:.3f}",
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
    )
    axr.text(
        0,
        SEEN_ONLY_CEILING + COLD_RATE / 2,
        f"COLD\n{N_COLD}/{N_TEST}\n{COLD_RATE * 100:.1f}%",
        ha="center",
        va="center",
        fontsize=6.2,
        color="white",
        fontweight="bold",
    )
    axr.plot(
        0.66, CHAMP, marker="*", ms=13, color=FAM["blend"], mec="black", mew=0.5, zorder=5
    )
    axr.text(
        0.80,
        CHAMP,
        f"champion Recall@10\n{CHAMP:.4f} — a DIFFERENT\nquantity, same 0-1 scale",
        fontsize=6.0,
        color=FAM["blend"],
        ha="left",
        va="center",
    )
    axr.set_xlim(-0.45, 1.55)
    axr.set_ylim(0, 1.0)
    axr.set_xticks([])
    axr.set_ylabel("fraction of the 1,431 test targets (bars)\n"
                   "/ Recall@10 (star)")
    axr.grid(axis="x", visible=False)
    axr.set_title("Cold is the\nMAJORITY half")

    save(fig, "mc_cold_warm.pdf")


if __name__ == "__main__":
    fig_model_landscape()
    fig_two_boards()
    fig_lambda_frontier()
    fig_cold_warm()
