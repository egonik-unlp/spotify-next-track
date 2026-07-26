#!/usr/bin/env python3
"""Generate the CONCEPT-ILLUSTRATION vector PDFs for the next-track
model-comparison document.

These are the *mechanism* diagrams: they make a mechanism intuitive to a
mathematically literate reader who is not a recommender-systems expert.
They deliberately do NOT duplicate the measurement figures already produced
by make_figures.py --- in particular nexttrack_rectifier.pdf already carries
the rectifier DECODABILITY measurement, so mc_relu_sign_loss.pdf carries the
complementary GEOMETRIC intuition and nothing else.

Every quantity printed on a panel is either (a) transcribed from an
experiments/*.md report or a source docstring, or (b) measured directly here
from a dataset artifact / a live model run --- each figure's comment block
names which. Schematic panels that stand in for a high-dimensional geometry
say so on the panel itself.

One fig_<name>() per figure; outputs land next to this script. Style block is
copied from make_figures.py so the two families read as one.

Run under the shared predictor venv (matplotlib / numpy):
  predictors/.venv/bin/python docs/figures/make_concept_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent

# Consistent style --- identical to make_figures.py.
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

ACCENT = "#2563eb"  # blue
GOOD = "#16a34a"  # green
BAD = "#dc2626"  # red
MUTED = "#9ca3af"  # gray

INK = "#374151"  # annotation text
FAINT = "#6b7280"  # secondary annotation text
PANEL = "#f9fafb"  # callout box face
PANEL_EC = "#d1d5db"  # callout box edge

afmt = FuncFormatter(lambda v, _: f"{v:.3f}")


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {name}")


def callout(ax, x, y, text, fontsize=6.6, color=INK, ha="left", va="top"):
    """The house annotation box (fc #f9fafb / ec #d1d5db)."""
    ax.text(
        x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=PANEL_EC, lw=0.7),
    )


def bare(ax, equal=True):
    """Schematic panel: no axes, no grid, optionally equal aspect."""
    if equal:
        ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ----------------------------------------------------------------------------
# Concept 1: WHY A ReLU DESTROYS A SIGNED PCA REPRESENTATION -- the geometry.
#
# The expressivity algebra is what makes this the ONLY variable: a GRU's gates
# already apply a linear input transform W_i x, so prepending a LINEAR map V
# gives W_i(Vx) = (W_i V)x, and at pre_hidden >= latent_dim a generically
# full-rank V leaves the reachable set of matrices unchanged -- a linear
# pre-encoder is PROVABLY expressivity-neutral. The ReLU is therefore the only
# expressivity change in the refuted pre-encoder arm, and it is applied to PCA
# coordinates, which are zero-centred and near-symmetric BY CONSTRUCTION.
#
# NUMBERS ON THIS PANEL AND WHERE THEY COME FROM:
#  * the plotted 28-coordinate vector is item 16921's first 28 PCA-192
#    coordinates, read here from
#    data/datasets/seq-20260715-131139/item_latents.f32 (measured, this script).
#  * frac_negative over the WHOLE item space = 0.5046 (19,402 items x 192 dims),
#    measured here from the same artifact -- the "near-symmetric by
#    construction" claim, stated as a property of the representation.
#  * frac_exact_zero on the rectified tap = 0.5327, matching the un-rectified
#    tap's frac_negative_coords = 0.5327 to four digits, and 256 x (1 - 0.5327)
#    = 119.6 live units for 192 signed directions. Those are the MEASURED tap
#    figures from tools/rectifier_control.py, recorded in PROJECT-FACTS.md and
#    2026-07-26-nexttrack-pre-encoder-scan.md Phase B4. (The tap fraction 0.5327
#    is measured after the learned affine map V, which is why it differs
#    slightly from the raw item space's 0.5046 -- both are printed.)
#  * the concat[relu(x), relu(-x)] identity x = relu(x) - relu(-x) is the exact
#    lossless construction; a plain ReLU at 2x width is NOT lossless, which is
#    the nuance the right panel is drawn to show.
# (2026-07-26-nexttrack-pre-encoder-scan.md; PROJECT-FACTS.md rectifier block.)
# ----------------------------------------------------------------------------
def fig_mc_relu_sign_loss():
    # Item 16921, first 28 PCA-192 coordinates (read from the dataset artifact).
    v = np.array([
        0.131, -0.286, -0.445, -0.032, -0.200, -0.012, -0.016, 0.492,
        -0.025, -0.042, 0.000, -0.120, -0.145, -0.047, -0.221, -0.266,
        0.093, 0.141, 0.018, -0.107, 0.165, 0.047, 0.040, 0.034,
        -0.069, 0.051, -0.024, 0.052,
    ])
    frac_neg_space = 0.5046   # measured over all 19,402 x 192 coordinates
    frac_zero_tap = 0.5327    # measured frac_exact_zero on the rectified tap
    D = 192                   # signed input directions to be preserved

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.3, 3.5), gridspec_kw={"width_ratios": [1.38, 1.0]}
    )

    # -- LEFT: one signed coordinate vector, before and after the rectifier ----
    x = np.arange(v.size)
    neg = v < 0
    axL.axhline(0, color=INK, lw=1.0, zorder=3)
    axL.vlines(x[~neg], 0, v[~neg], color=ACCENT, lw=2.4, zorder=2)
    axL.vlines(x[neg], 0, v[neg], color=BAD, lw=2.4, zorder=2)
    axL.plot(x[~neg], v[~neg], "o", color=ACCENT, ms=4.2, zorder=4)
    axL.plot(x[neg], v[neg], "o", color=BAD, ms=4.2, zorder=4)
    # the rectified image: positives survive, negatives land exactly on zero
    axL.plot(x[neg], np.zeros(neg.sum()), "x", color=BAD, ms=6.0, mew=1.6,
             zorder=5)
    axL.plot(x[~neg], v[~neg], "o", mfc="none", mec=GOOD, ms=8.0, mew=1.3,
             zorder=5)

    axL.set_xlim(-1, v.size)
    axL.set_ylim(-0.64, 0.68)
    axL.set_xlabel("PCA coordinate index (first 28 of 192)")
    axL.set_ylabel("coordinate value")
    axL.grid(axis="x", visible=False)
    axL.set_title("A signed, zero-centred coordinate vector\n"
                  "and its rectified image", fontsize=8.6)
    axL.text(-0.4, 0.645, "green ring $=$ survives", fontsize=6.4,
             color=GOOD, va="top")
    axL.text(13.5, -0.505,
             "every negative coordinate is mapped to EXACTLY 0 (red $\\times$);\n"
             "ReLU is not injective, so the sign is gone for good",
             fontsize=6.5, color=BAD, ha="center", va="top")
    callout(
        axL, 0.985, 0.955,
        "PCA coordinates are zero-centred by\n"
        f"construction: {frac_neg_space:.4f} of all\n"
        "$19{,}402\\times192$ coordinates are negative\n"
        f"(rectified tap, measured: {frac_zero_tap:.4f})",
        fontsize=6.2, ha="right", va="top",
    )

    # -- RIGHT: the width budget ---------------------------------------------
    rows = [
        ("width 256 ($1.33\\times$)\nas built", 256 * (1 - frac_zero_tap), BAD, False),
        ("width 384 ($2.0\\times$)\nplain ReLU", 384 * (1 - frac_zero_tap), MUTED, False),
        ("width 384 as\nconcat[relu($x$), relu($-x$)]", 192.0, GOOD, True),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, live, col, exact) in zip(y, rows):
        axR.barh(yi, live, 0.50, color=col, edgecolor="white", linewidth=0.8)
        axR.text(live + 6, yi, f"{live:.1f}" + (" exact" if exact else ""),
                 va="center", fontsize=7.0, color=col)
    axR.axvline(D, color=INK, ls="--", lw=1.3)
    axR.text(D, 3.02, f"{D} signed directions must survive", fontsize=6.5,
             color=INK, ha="center", va="top")

    axR.set_yticks(y)
    axR.set_yticklabels([r[0] for r in rows], fontsize=6.9)
    axR.set_xlim(0, 262)
    axR.set_ylim(-1.75, 3.12)
    axR.set_xlabel("live (non-zero) units after the rectifier")
    axR.grid(axis="y", visible=False)
    axR.set_title("Half the units are dead: the layer must\n"
                  "be over-complete just to break even", fontsize=8.6)
    callout(
        axR, 0.015, 0.245,
        "$x=\\mathrm{relu}(x)-\\mathrm{relu}(-x)$: the sign survives\n"
        "only if BOTH halves are kept. A plain ReLU at\n"
        "$2\\times$ width still averages $<192$ live units ---\n"
        "width alone is not the fix, the split is.",
        fontsize=6.2,
    )

    fig.suptitle(
        "A rectifier on a signed PCA latent is lossy compression dressed as capacity:\n"
        f"{256 * (1 - frac_zero_tap):.0f} live units are left to carry {D} signed directions",
        fontsize=9.6,
    )
    save(fig, "mc_relu_sign_loss.pdf")


# ----------------------------------------------------------------------------
# Concept 2: WHY TWO TOWERS OVER DIFFERENT CAUSAL VIEWS CANNOT ADD INFORMATION.
#
# The three tower views are all deterministic, strictly causal functions of the
# SAME prefix (predictors/seq_dualgru.py:179-196):
#     latent  = x_t
#     delta   = x_t - x_{t-1}
#     cummean = (1/t) * sum_{i<=t} x_i     (1-based t; the code divides the
#                                          running cumsum by 1,2,3,...)
# so sigma(view at t) is a SUBSET of sigma(x_0..x_t): tower B's input is
# measurable with respect to what tower A already receives. A GRU over raw
# latents additionally computes both derived views internally -- its update gate
# is a leaky accumulator (a running mean is inside its hypothesis space) and the
# candidate/gate interaction covers first differences. So the only thing on
# offer is ENSEMBLE DIVERSITY, and joint end-to-end training through one shared
# loss is precisely the scheme that destroys it.
#
# MEASURED NUMBERS ON THIS PANEL (2026-07-25-nexttrack-dual-tower-fusion-scan.md):
#  * best dual arm D-LL-f0 0.11880 (170/1431) vs the C0 single-GRU baseline
#    0.12299 (176/1431): paired Delta -0.00419 [-0.01258, +0.00419] STRADDLES.
#  * latent/cummean vs its latent/latent parent -0.00559 [-0.01747, +0.00699]
#    STRADDLES; latent/delta vs the same parent -0.01118 [-0.02306, +0.00070]
#    STRADDLES -- an explicit derived view adds nothing beside a raw-latent tower.
#  * the ONE confirmed complementarity: delta/cummean beats BOTH its parents
#    CI>0 (+0.02865 vs delta/delta, +0.02027 vs cummean/cummean) at a hopeless
#    absolute 0.08665 -- the mechanism is real, it just lands on the two views
#    that are individually hopeless.
#  * the SAE read on the dual arm found real new linear geometry between the
#    towers (CKA 0.458) but +0 new next-item concepts.
# ----------------------------------------------------------------------------
def fig_mc_view_redundancy():
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.4, 4.15), gridspec_kw={"width_ratios": [1.06, 1.0]}
    )
    for ax in (axL, axR):
        bare(ax, equal=False)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    # -- LEFT: nested information sets ---------------------------------------
    axL.add_patch(FancyBboxPatch(
        (0.30, 2.55), 9.40, 6.45, boxstyle="round,pad=0.10",
        fc="#f3f4f6", ec=MUTED, lw=1.1, zorder=1))
    axL.text(5.0, 9.62,
             "$\\sigma(x_0,\\dots,x_t)$ --- everything the prefix determines",
             ha="center", va="top", fontsize=7.2, color=INK, style="italic")

    axL.add_patch(Ellipse((5.0, 6.00), 8.5, 5.0, fc="#dbeafe", ec=ACCENT,
                          lw=1.6, zorder=2))
    axL.text(5.0, 8.15, "tower A: raw latent $x_t$", ha="center", va="top",
             fontsize=7.8, color=ACCENT, fontweight="bold", zorder=6)
    axL.text(5.0, 7.62,
             "(the recurrence's own gates already accumulate\n"
             "a running mean and a first difference)",
             ha="center", va="top", fontsize=6.3, color=ACCENT, style="italic",
             zorder=6)

    # Inner ellipses are kept strictly inside tower A's boundary -- the whole
    # claim of the panel is a SUBSET relation, so the drawing must honour it.
    axL.add_patch(Ellipse((3.25, 5.35), 3.2, 2.3, fc="#e5e7eb", ec=MUTED,
                          lw=1.3, zorder=3))
    axL.text(3.25, 5.78, "tower B: delta", ha="center", va="center",
             fontsize=7.0, color=INK, zorder=6)
    axL.text(3.25, 4.98, "$x_t-x_{t-1}$", ha="center", va="center",
             fontsize=7.0, color=FAINT, zorder=6)

    axL.add_patch(Ellipse((6.75, 5.35), 3.2, 2.3, fc="#e5e7eb", ec=MUTED,
                          lw=1.3, zorder=3))
    axL.text(6.75, 5.85, "tower B: cummean", ha="center", va="center",
             fontsize=7.0, color=INK, zorder=6)
    axL.text(6.75, 4.95, "$\\frac{1}{t}\\sum_{i\\leq t} x_i$", ha="center",
             va="center", fontsize=7.2, color=FAINT, zorder=6)

    axL.text(5.0, 3.32,
             "both are deterministic functions of tower A's own input\n"
             "$\\Rightarrow$ a re-compression, never new information",
             ha="center", va="top", fontsize=6.8, color=BAD, fontweight="bold",
             zorder=6)

    callout(
        axL, 0.02, 0.225,
        "best dual arm 0.11880 (170/1431) vs baseline\n"
        "0.12299 (176/1431): $\\Delta$ $-0.00419$ [$-0.013$, $+0.004$]\n"
        "TIE. Adding an explicit derived view beside the\n"
        "raw one is also a tie --- both straddle zero.",
        fontsize=6.2,
    )
    axL.set_title("Same-sequence views: tower B $\\subseteq$ tower A",
                  fontsize=8.8)

    # -- RIGHT: genuine complementarity --------------------------------------
    axR.add_patch(Ellipse((3.85, 6.05), 5.7, 4.7, fc="#dbeafe", ec=ACCENT,
                          lw=1.6, alpha=0.85, zorder=2))
    axR.add_patch(Ellipse((6.15, 6.05), 5.7, 4.7, fc="#dcfce7", ec=GOOD,
                          lw=1.6, alpha=0.72, zorder=2))
    axR.text(1.85, 6.05, "private\nto A", ha="center", va="center",
             fontsize=7.0, color=ACCENT, fontweight="bold", zorder=6)
    axR.text(8.15, 6.05, "private\nto B", ha="center", va="center",
             fontsize=7.0, color=GOOD, fontweight="bold", zorder=6)
    axR.text(5.0, 6.05, "shared", ha="center", va="center", fontsize=6.8,
             color=INK, zorder=6)
    axR.text(2.85, 9.62, "stream A", ha="center", va="top", fontsize=7.6,
             color=ACCENT, fontweight="bold")
    axR.text(7.15, 9.62, "stream B", ha="center", va="top", fontsize=7.6,
             color=GOOD, fontweight="bold")
    axR.text(5.0, 8.95, "neither is a function of the other", ha="center",
             va="top", fontsize=6.5, color=INK, style="italic")
    axR.text(5.0, 3.32,
             "each private crescent is signal the other\n"
             "stream cannot supply $\\Rightarrow$ fusion can pay",
             ha="center", va="top", fontsize=6.8, color=GOOD,
             fontweight="bold")

    callout(
        axR, 0.02, 0.225,
        "measured once, and only once: delta$+$cummean beats\n"
        "BOTH parents CI$>$0 ($+0.02865$ and $+0.02027$) --- at a\n"
        "hopeless 0.08665. Real mechanism, two individually\n"
        "hopeless views. Beside a raw-latent tower instead:\n"
        "CKA 0.458 of new geometry, $+0$ new next-item concepts.",
        fontsize=6.2,
    )
    axR.set_title("Genuine multi-modal fusion (the contrast)", fontsize=8.8)

    fig.suptitle(
        "The dual-tower null in one picture: a second encoder over a derived view of the\n"
        "same sequence can only re-compress, and joint training removes even the diversity",
        fontsize=9.6,
    )
    save(fig, "mc_view_redundancy.pdf")


# ----------------------------------------------------------------------------
# Concept 3: THE EIGEN-SHAPE OF A MODEL'S NEXT-TRACK BELIEF.
#
# predictors/seq_extend.py::candidate_eigen takes the top-M candidates' latents,
# weights each by its share of score mass, centres them and takes the SVD. The
# squared singular values, as variance SHARES, say how the plausible
# continuations are distributed in the content space; the participation ratio
# pr = 1/sum(s_i^2) is the EFFECTIVE NUMBER of directions the belief spreads
# over (1 = a single axis). A dominant leading share means the model has a
# direction of travel and is choosing HOW FAR, not WHAT comes next -- which
# entropy over items cannot see, because a tight cluster of 50 near-identical
# tracks has high item entropy but pr ~ 1.
#
# MEASURED HERE, 2026-07-26, BY ME -- not transcribed from a report:
#   predictors/.venv/bin/python predictors/seq_extend.py extend \
#     --model data/models/blend-gru-markov-content-proj --predictor seq-blend
#   predictors/.venv/bin/python predictors/seq_extend.py extend \
#     --model data/models/best-seq-nexttrack-20260726-183333-9a0b2 \
#     --predictor seq-nexttrack
#   IDENTICAL seed prefix (vocab indices [10246, 15570, 11233, 16024, 5354]),
#   10 steps, default policy, pool M=50.
# Champion blend:  leading share 0.5907 -> 0.7036 -> 0.8968 -> 0.9720 over four
#   successive steps, pr 2.35 -> 1.79 -> 1.23 -> 1.06 (mean pr 3.14).
# Single GRU on the same seed: pr 2.07-8.64, mean 5.87, leading share mean 0.349.
# The collapsed step reproduces the album-lock signature recorded in
# predictors/seq_extend.py:553-555 ("pr 1.25, leading share 0.89,
# single-genre pool") to two decimals -- an independent corroboration.
# The right-hand cloud panel is a 2-D SCHEMATIC whose aspect ratio is set to
# sqrt(lambda_1/lambda_2) from the measured spectra (6.65 and 1.17); it says so
# on the panel.
# ----------------------------------------------------------------------------
def fig_mc_candidate_eigen():
    # measured trajectories, same seed, 10 steps
    pr_blend = [2.35, 1.79, 1.23, 1.06, 1.31, 2.23, 5.76, 5.89, 7.69, 2.08]
    pr_gru = [6.08, 6.24, 2.07, 4.52, 7.11, 6.13, 5.50, 5.54, 6.87, 8.64]
    steps = np.arange(1, 11)

    spec_gru = [0.2558, 0.1874, 0.1264, 0.1015]     # GRU, step 5, pr 7.11
    spec_b0 = [0.5907, 0.2494, 0.1200, 0.0137]      # blend, step 1, pr 2.35
    spec_b3 = [0.9720, 0.0220, 0.0046, 0.0004]      # blend, step 4, pr 1.06

    fig = plt.figure(figsize=(7.4, 4.85))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.60],
                          width_ratios=[1.08, 0.92])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # -- (a) the measured pr trajectory --------------------------------------
    axA.axvspan(0.85, 4.15, color=BAD, alpha=0.07, zorder=0)
    axA.axhspan(5, 7, color=GOOD, alpha=0.10, zorder=0)
    axA.text(11.45, 6.0, "spread\nbelief", fontsize=6.3, color=GOOD,
             ha="center", va="center")
    # HONEST: the GRU's step-3 point (pr 2.07) falls BELOW the shaded band, so
    # the claim is monotone-collapse vs NO TREND, not inside-band vs outside.
    axA.annotate("the GRU's own minimum\n(2.07) is below the band:\n"
                 "no trend, not a floor",
                 xy=(3, 2.07), xytext=(4.30, 2.60), fontsize=6.0, color=ACCENT,
                 ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    axA.axhline(1.25, color=BAD, ls=":", lw=1.3, zorder=1)
    axA.text(0.75, 0.72, "recorded album-lock signature (pr 1.25)",
             fontsize=6.1, color=BAD, va="center")
    axA.plot(steps, pr_gru, "-o", color=ACCENT, lw=1.7, ms=4.0,
             label="single GRU (mean 5.87, range 2.07-8.64)")
    axA.plot(steps, pr_blend, "-o", color=BAD, lw=1.9, ms=4.0,
             label="champion blend (mean 3.14)")
    axA.text(2.5, 11.35, "collapse over four\nsuccessive steps", fontsize=6.3,
             color=BAD, ha="center", va="top", fontweight="bold")
    axA.set_xlim(0.5, 12.0)
    axA.set_ylim(0, 11.8)
    axA.set_xticks([1, 3, 5, 7, 9])
    axA.set_xlabel("generation step (same seed)")
    axA.set_ylabel("participation ratio $1/\\sum s_i^2$")
    axA.legend(loc="upper right", fontsize=6.2)
    axA.grid(axis="x", visible=False)
    axA.set_title("effective number of directions, per step", fontsize=8.4)

    # -- (b) the spectra -----------------------------------------------------
    x = np.arange(4)
    w = 0.26
    axB.bar(x - w, spec_gru, w, color=ACCENT, edgecolor="white", linewidth=0.7,
            label="GRU, step 5 (pr 7.11)")
    axB.bar(x, spec_b0, w, color=MUTED, edgecolor="white", linewidth=0.7,
            label="blend step 1, pr 2.35")
    axB.bar(x + w, spec_b3, w, color=BAD, edgecolor="white", linewidth=0.7,
            label="blend step 4, pr 1.06")
    axB.text(0 + w, spec_b3[0] + 0.025, "0.972", ha="center", fontsize=6.6,
             color=BAD, fontweight="bold")
    axB.text(0 - w, spec_gru[0] + 0.025, "0.256", ha="center", fontsize=6.6,
             color=ACCENT)
    axB.set_xticks(x)
    axB.set_xticklabels(["$\\lambda_1$", "$\\lambda_2$", "$\\lambda_3$",
                         "$\\lambda_4$"])
    axB.set_ylim(0, 1.10)
    axB.set_xlabel("candidate-covariance eigenvalue")
    axB.set_ylabel("variance share (sums to 1)")
    axB.legend(loc="upper right", fontsize=6.2)
    axB.grid(axis="x", visible=False)
    axB.set_title("several comparable directions, or one", fontsize=8.4)

    # -- (c) what that means geometrically -----------------------------------
    # Equal aspect is load-bearing here: the drawn aspect of each cloud IS the
    # measured sqrt(lambda_1/lambda_2), so it must not be distorted by the axes.
    bare(axC, equal=True)
    axC.set_xlim(-3.6, 3.6)
    axC.set_ylim(-0.80, 0.80)
    rng = np.random.default_rng(1337)

    # isotropic cloud: aspect sqrt(l1/l2) from the MEASURED GRU spectrum
    a_gru = np.sqrt(spec_gru[0] / spec_gru[1])       # 1.17
    p = rng.normal(size=(52, 2)) * np.array([0.115 * a_gru, 0.115])
    th = np.deg2rad(28.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    p = p @ R.T + np.array([-1.85, 0.02])
    axC.plot(p[:, 0], p[:, 1], "o", color=ACCENT, ms=3.2, alpha=0.85,
             mec="white", mew=0.3, zorder=4)
    axC.add_patch(Circle((-1.85, 0.02), 0.335, fc="none", ec=ACCENT, lw=1.0,
                         ls="--", zorder=3))
    axC.text(-1.85, 0.60, "pr $\\approx$ 7", ha="center", va="center",
             fontsize=7.6, color=ACCENT, fontweight="bold")
    axC.text(-1.85, -0.50, "WHAT comes next is still open", ha="center",
             va="center", fontsize=6.6, color=ACCENT)

    # collapsed cloud: aspect sqrt(l1/l2) from the MEASURED blend spectrum
    a_b = np.sqrt(spec_b3[0] / spec_b3[1])           # 6.65
    q = rng.normal(size=(52, 2)) * np.array([0.052 * a_b, 0.052])
    th2 = np.deg2rad(-13.0)
    R2 = np.array([[np.cos(th2), -np.sin(th2)], [np.sin(th2), np.cos(th2)]])
    q = q @ R2.T + np.array([1.85, 0.02])
    axC.plot(q[:, 0], q[:, 1], "o", color=BAD, ms=3.2, alpha=0.85,
             mec="white", mew=0.3, zorder=4)
    d = np.array([np.cos(th2), np.sin(th2)]) * 0.82
    axC.annotate("", xy=(1.85 + d[0], 0.02 + d[1]),
                 xytext=(1.85 - d[0], 0.02 - d[1]),
                 arrowprops=dict(arrowstyle="<->", color=BAD, lw=1.5),
                 zorder=5)
    axC.text(1.85, 0.60, "pr $\\approx$ 1", ha="center", va="center",
             fontsize=7.6, color=BAD, fontweight="bold")
    axC.text(1.85, -0.50, "only HOW FAR along one line is left", ha="center",
             va="center", fontsize=6.6, color=BAD)
    axC.text(0.0, 0.02,
             "2-D schematic; each cloud's\n"
             "drawn aspect is the measured\n"
             f"$\\sqrt{{\\lambda_1/\\lambda_2}}$ ({a_gru:.2f} and {a_b:.2f})",
             ha="center", va="center", fontsize=5.9, color=FAINT,
             style="italic")
    axC.set_title("a collapsed belief is a direction, not a choice",
                  fontsize=8.4)

    fig.suptitle(
        "The eigen-shape of a next-track belief: the champion blend's candidate cloud collapses onto ONE axis\n"
        "(leading share 0.591$\\to$0.704$\\to$0.897$\\to$0.972 over four steps) where the single GRU shows no trend\n"
        "AUTHOR-RUN, SINGLE SEED, ONE PREFIX — not a campaign result and not in the experiment record",
        fontsize=9.2,
    )
    save(fig, "mc_candidate_eigen.pdf")


# ----------------------------------------------------------------------------
# Concept 4: HOW GREEDY MMR RE-RANKS, AND WHY IT DE-EAGERS ARTISTS FOR FREE.
#
# The rule, verbatim from predictors/seq_common.py::mmr_rerank_order: take the
# top `mmr_pool` candidates in base-score order, min-max normalize their base
# scores to rel in [0,1] over that pool, then build the top-k GREEDILY, at each
# step selecting
#       argmax_i  lambda * rel(i) - (1 - lambda) * max_{j in selected} cos(M_i, M_j)
# where M is the 517-d `balanced` musical-distance vector. lambda = 1 recovers
# the pure score order exactly (an identity gate, which is why lambda arms of
# one model share BIT-IDENTICAL trained weights).
#
# Nothing in that rule mentions artists. Artists de-concentrate anyway because
# same-artist tracks are sonic near-neighbours, so the max-similarity penalty
# preferentially evicts them -- the campaign's H1 ("responsive") hypothesis.
#
# LEFT PANEL is a 2-D SCHEMATIC: the real similarity is the cosine between
# 517-d vectors, which is not drawable, so proximity in the drawing stands in
# for it. The greedy selection rule itself is applied VERBATIM to the drawn
# points, and the artist-concentration numbers quoted for the two columns are
# the normalized Herfindahl (sum p^2 - 1/n)/(1 - 1/n) of the illustration's own
# selected sets -- illustration arithmetic, labelled as such.
#
# MEASURED NUMBERS (2026-07-26-nexttrack-mmr-lambda-frontier.md), single GRU,
# lambda 1.0 -> 0.7 at mmr_pool 200:
#     ild        0.4536 -> 0.5314   (the penalty term itself, rises by construction)
#     artist_conc 0.3559 -> 0.2707  (-0.0852)
#     artist_adj  0.3474 -> 0.3129  (-0.0345)
#     music@10   0.4558 -> 0.4747   (RISES -- what separates this from the
#                                    degenerate "de-eager by predicting worse")
#     mood_coh   0.4788 -> 0.4585   (-4% only)
# H1 verdict: |Delta artist_conc| >= |Delta artist_adj| in 8 of 8 arms, ratio
# 1.59x to 3.06x; H2 ("sticky") required < 0.5x and is off by 3-6x the wrong way.
# ----------------------------------------------------------------------------
def fig_mc_mmr_geometry():
    rng = np.random.default_rng(20260726)

    # --- the illustration's candidate pool --------------------------------
    # Artist A0 is a TIGHT SONIC CLUSTER that also owns the head of the base
    # ranking -- the "album-eager" block the re-rank has to break up. Every
    # other artist forms its own looser cluster lower down the ranking. The
    # per-artist relevance bands encode "eager": the model's own top scores are
    # dominated by one artist, with decent alternatives just below it.
    spec = [
        # artist, cx, cy, n, spread, (rel_lo, rel_hi)
        (0, 0.62, 0.66, 7, 0.060, (0.84, 1.00)),
        (1, 0.24, 0.78, 4, 0.075, (0.58, 0.80)),
        (2, 0.82, 0.25, 4, 0.075, (0.52, 0.76)),
        (3, 0.28, 0.24, 5, 0.085, (0.34, 0.62)),
        (4, 0.55, 0.10, 3, 0.062, (0.26, 0.50)),
        (5, 0.11, 0.47, 3, 0.060, (0.20, 0.44)),
        (6, 0.90, 0.62, 3, 0.060, (0.30, 0.56)),
    ]
    pts, art, rel = [], [], []
    for a, cx, cy, k, sc, (lo, hi) in spec:
        pts.append(np.array([cx, cy]) + rng.normal(scale=sc, size=(k, 2)))
        art += [a] * k
        rel += list(rng.uniform(lo, hi, k))
    P = np.clip(np.vstack(pts), 0.04, 0.96)
    art = np.array(art)
    rel = np.array(rel)
    n = len(art)

    # Similarity: a sonic neighbourhood of scale SG. In the real rule this is
    # cos(M_i, M_j) between 517-d `balanced` vectors; here proximity stands in
    # for it so the geometry is drawable (the panel says so).
    SG = 0.15
    Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    SIM = np.exp(-Dm ** 2 / (2 * SG ** 2))

    def greedy(lam, k):
        """seq_common.mmr_rerank_order, applied verbatim to the drawn points."""
        sel, rem = [], list(range(n))
        while rem and len(sel) < k:
            best, bv = rem[0], -1e18
            for c in rem:
                div = max((SIM[c, s] for s in sel), default=0.0)
                v = lam * rel[c] - (1.0 - lam) * div
                if v > bv:
                    bv, best = v, c
            sel.append(best)
            rem.remove(best)
        return sel

    K = 8
    sel_pure = greedy(1.0, K)          # lambda = 1 is exactly the score order
    sel_mmr = greedy(0.7, K)
    evicted = [i for i in sel_pure if i not in sel_mmr]

    def herf(idx):
        """Normalized Herfindahl over the list's artists: 0 = all different."""
        _, c = np.unique(art[idx], return_counts=True)
        pr = c / c.sum()
        m = len(idx)
        return (float((pr ** 2).sum()) - 1 / m) / (1 - 1 / m)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.4, 3.9), gridspec_kw={"width_ratios": [1.46, 1.0]}
    )

    # -- LEFT: the greedy geometry ------------------------------------------
    # Equal aspect: the penalty reach is drawn as a CIRCLE, so the axes must
    # not distort it. The right-hand strip of the data range is deliberately
    # empty and carries the annotations.
    bare(axL, equal=True)
    axL.set_xlim(0.0, 1.80)
    axL.set_ylim(-0.02, 1.16)

    # No artist uses "X" or a ring: those are reserved for the DROP marker, so a
    # reader can always decode a dropped candidate's artist from its own shape.
    marks = ["o", "s", "^", "D", "v", "P", "p"]
    for a in range(art.max() + 1):
        m = art == a
        axL.scatter(P[m, 0], P[m, 1], s=15 + 96 * rel[m], marker=marks[a],
                    facecolor=(BAD if a == 0 else "#cbd5e1"),
                    edgecolor=("white" if a == 0 else MUTED),
                    linewidths=0.6, alpha=0.92, zorder=3)

    # the penalty reach of the first three picks
    for r, i in enumerate(sel_mmr[:3], start=1):
        # An ISO-PENALTY CONTOUR, not a cutoff radius: the real penalty
        # max_j cos(M_i, M_j) is continuous, so a sufficiently relevant
        # candidate survives inside the contour (two do, below).
        axL.add_patch(Circle(P[i], SG, fc="none", ec=GOOD, lw=0.9, ls="--",
                             alpha=0.55, zorder=1))
        axL.scatter(*P[i], s=150, marker="o", facecolor="none", edgecolor=GOOD,
                    linewidths=1.8, zorder=5)
        # redraw the picked candidate's own marker so it reads inside the ring
        axL.scatter(*P[i], s=34, marker=marks[int(art[i])],
                    facecolor=(BAD if art[i] == 0 else "#94a3b8"),
                    edgecolor="white", linewidths=0.5, zorder=6)
        axL.annotate(str(r), P[i], textcoords="offset points", xytext=(9, 6),
                     fontsize=7.6, color=GOOD, fontweight="bold", zorder=8)

    # the same-artist candidates the re-rank DROPS from the top-8
    drop_rank = {i: k for k, i in
                 enumerate(sorted(evicted, key=lambda i: -rel[i]), start=1)}
    for i in evicted:
        # The dropped candidate keeps its OWN artist marker (so its artist stays
        # readable) and is struck through with a red ring.
        axL.scatter(*P[i], s=34, marker=marks[int(art[i])],
                    facecolor=(BAD if art[i] == 0 else "#94a3b8"),
                    edgecolor="white", linewidths=0.5, zorder=6)
        axL.scatter(*P[i], s=125, marker="o", facecolor="none",
                    edgecolor="white", linewidths=2.6, zorder=6)
        axL.scatter(*P[i], s=125, marker="o", facecolor="none", edgecolor=BAD,
                    linewidths=1.4, zorder=7)
        d = 0.0225
        axL.plot([P[i][0] - d, P[i][0] + d], [P[i][1] + d, P[i][1] - d],
                 color=BAD, lw=1.4, zorder=8)
        # Index each drop so the reader can count three of them even where the
        # A0 cluster is tight.
        axL.annotate(f"d{drop_rank[i]}", P[i], textcoords="offset points",
                     xytext=(-13, -9), fontsize=6.6, color=BAD,
                     fontweight="bold", zorder=9)

    axL.text(0.02, 1.135,
             "$\\arg\\max_i\\;\\lambda\\,\\mathrm{rel}(i)-(1-\\lambda)"
             "\\max_{j\\in\\mathrm{selected}}\\cos(M_i,M_j)$",
             fontsize=8.0, color=INK, va="top")
    axL.text(0.02, 1.045,
             "marker size $=$ base relevance $\\cdot$ marker SHAPE $=$ artist "
             "(A0 red, all others gray)",
             fontsize=6.1, color=FAINT, va="top")

    axL.text(1.02, 0.965,
             "green ring $+$ number $=$ pick order\n"
             "green dashed circle $=$ an ISO-PENALTY\n"
             "  contour of that pick. The penalty is\n"
             "  CONTINUOUS, not a cutoff: two\n"
             "  candidates survive inside one.\n"
             "red struck ring $=$ dropped from the\n"
             "  top-8 (its artist shape is kept)",
             fontsize=6.4, color=INK, va="top")
    axL.text(1.02, 0.475,
             f"all {len(evicted)} dropped candidates (d1--d3)\n"
             "are artist A0 (relevance "
             + " / ".join(f"{rel[i]:.2f}" for i in
                          sorted(evicted, key=lambda i: -rel[i]))
             + "),\neach inside an already-picked\ntrack's sonic neighbourhood.",
             fontsize=6.4, color=BAD, va="top")
    axL.text(1.02, -0.005,
             "2-D stand-in: the real similarity is\n"
             "the cosine between 517-d sonic vectors.\n"
             "The greedy rule itself is verbatim.",
             fontsize=5.8, color=FAINT, va="bottom", style="italic")
    axL.set_title("Greedy MMR at $\\lambda=0.7$: each pick pushes the next one\n"
                  "out of its own sonic neighbourhood", fontsize=8.6)

    # -- RIGHT: the resulting lists, and the free de-concentration ----------
    bare(axR, equal=False)
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)

    cols = [(2.55, sel_pure, "$\\lambda=1.0$", "pure relevance", MUTED),
            (6.55, sel_mmr, "$\\lambda=0.7$", "MMR re-rank", GOOD)]
    for cx, sel, lab, sub, col in cols:
        axR.text(cx + 0.75, 9.75, lab, ha="center", va="top", fontsize=7.8,
                 color=col, fontweight="bold")
        axR.text(cx + 0.75, 9.10, sub, ha="center", va="top", fontsize=6.4,
                 color=col)
        for r, i in enumerate(sel):
            yy = 8.32 - r * 0.70
            a = int(art[i])
            axR.add_patch(FancyBboxPatch(
                (cx, yy - 0.27), 1.50, 0.54, boxstyle="round,pad=0.02",
                fc=(BAD if a == 0 else "#e5e7eb"),
                ec=("white" if a == 0 else MUTED), lw=0.7, zorder=3))
            axR.text(cx + 0.75, yy, f"A{a}", ha="center", va="center",
                     fontsize=6.8, zorder=5,
                     color=("white" if a == 0 else INK),
                     fontweight=("bold" if a == 0 else "normal"))
        axR.text(cx + 0.75, 2.72,
                 f"{len(set(art[sel].tolist()))} artists in {K}\n"
                 f"Herfindahl {herf(sel):.3f}",
                 ha="center", va="top", fontsize=6.6, color=col,
                 fontweight="bold")

    axR.annotate("", xy=(6.38, 5.30), xytext=(4.22, 5.30),
                 arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.6))
    axR.text(5.30, 7.10, "no\nartist\nterm in\nthe rule", ha="center",
             va="top", fontsize=6.4, color=GOOD, fontweight="bold",
             linespacing=1.25)

    axR.text(5.0, 1.70,
             "MEASURED, single GRU, $\\lambda$ 1.0$\\to$0.7 (pool 200):\n"
             "ild 0.4536$\\to$0.5314    artist_conc 0.3559$\\to$0.2707\n"
             "artist_adj 0.3474$\\to$0.3129    music@10 0.4558$\\to$0.4747\n"
             "$|\\Delta$conc$|$ exceeds $|\\Delta$adj$|$ in 8 of 8 arms, "
             "1.59--3.06$\\times$",
             ha="center", va="top", fontsize=6.1, color=INK,
             bbox=dict(boxstyle="round,pad=0.34", fc=PANEL, ec=PANEL_EC,
                       lw=0.7))
    axR.set_title("Artist A0 is a sonic cluster, so it is\n"
                  "de-concentrated without being named", fontsize=8.6)

    fig.suptitle(
        "Why sonic diversification de-eagers artists for free: same-artist tracks are sonic\n"
        "near-neighbours, so the max-similarity penalty evicts them first (the H1 verdict)",
        fontsize=9.6,
    )
    save(fig, "mc_mmr_geometry.pdf")


if __name__ == "__main__":
    fig_mc_relu_sign_loss()
    fig_mc_view_redundancy()
    fig_mc_candidate_eigen()
    fig_mc_mmr_geometry()
