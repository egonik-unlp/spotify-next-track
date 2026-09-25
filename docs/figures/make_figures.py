#!/usr/bin/env python3
"""Generate the vector-PDF figures for docs/experiments.tex.

All data is transcribed from the experiment reports in experiments/.
Each figure cites its source report in a comment; one fig_<name>()
function per figure. Run from anywhere: outputs land next to this script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent

# Consistent style
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

# Domain style header — keep axis units/labels going through these instead
# of hardcoding the unit per figure (values mirror [metrics] in domain.toml).
PRIMARY_METRIC = "MAE"
VALUE_UNIT = "engagement"
VALUE_AXIS = f"{PRIMARY_METRIC} ({VALUE_UNIT})"
kfmt = FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")

# Rotation is the active target: a binary [0,1] taste-fit signal scored by
# AUC (classification) — its values are O(1) probabilities, so a plain 3-dp
# formatter reads them rather than the thousands-separator engagement kfmt.
afmt = FuncFormatter(lambda v, _: f"{v:.3f}")


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {name}")


# Figures are added by the report-curator agent, one fig_<name>() per
# figure, each citing its source report.


# Fig 1: the AE-latent standalone signal washes out as the kNN neighborhood
# widens, and never reaches the metadata-only tree
# (2026-06-09-ae-xgboost-blend.md, "The mae_ae(k) curve").
def fig_ae_latent_knn():
    k = [10, 25, 50, 100, 200]
    mae_ae = [0.4259, 0.4403, 0.4510, 0.4609, 0.4702]
    ridge = 0.4623  # ridge-on-latent arm, between k=100 and k=200
    meta_xgb = 0.3879  # metadata-only XGBoost (the branch the blend collapses to)
    ae_only_xgb = 0.4098  # xgboost on AE-latent only (64 cols), on record

    fig, ax = plt.subplots()
    ax.plot(k, mae_ae, "o-", color=ACCENT, label="AE-latent kNN ($mae_{ae}$)")
    ax.axhline(ridge, color=MUTED, ls=":", lw=1.4, label=f"ridge-on-latent ({ridge:.3f})")
    ax.axhline(
        ae_only_xgb, color=MUTED, ls="--", lw=1.2,
        label=f"AE-latent-only xgboost ({ae_only_xgb:.3f})",
    )
    ax.axhline(
        meta_xgb, color=GOOD, ls="-", lw=1.6,
        label=f"metadata-only XGBoost ({meta_xgb:.3f})",
    )
    ax.set_xscale("log")
    ax.set_xticks(k)
    ax.set_xticklabels([str(v) for v in k])
    ax.set_xlabel("kNN neighborhood size $k$")
    ax.set_ylabel("MAE (P(rotation))")
    ax.set_title("The song-AE latent is a dead modality: even its best read\nnever reaches the metadata tree")
    ax.legend(loc="center right", fontsize=7.5)
    save(fig, "ae_latent_knn.pdf")


# Fig 2: on a binary target, low SVR MAE is an epsilon-tube artifact — the
# honest R² says every kernel is worse than the mean
# (2026-06-09-svm-kernel-rotation.md, Results + Finding 3).
def fig_svm_mae_vs_r2():
    labels = ["xgboost\n(champion)", "svm\nrbf", "svm\nlinear", "svm\npoly", "svm\nsigmoid"]
    mae = [0.3879, 0.3847, 0.3901, 0.3814, 0.6240]
    r2 = [0.191, -0.130, -0.198, -0.137, -8.479]
    colors = [GOOD, BAD, BAD, BAD, BAD]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.3))
    x = np.arange(len(labels))

    ax1.bar(x, mae, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("MAE (P(rotation))")
    ax1.set_title("MAE: SVR looks competitive\n(the $\\epsilon$-tube artifact)")
    ax1.axhline(0.3879, color=MUTED, ls=":", lw=1.0)

    ax2.bar(x, r2, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("$R^2$")
    ax2.axhline(0.0, color="black", lw=1.0)
    ax2.set_title("$R^2$: every SVR kernel is\nworse than the mean")
    # sigmoid R^2 = -8.48 would flatten the scale; clip the view and annotate.
    ax2.set_ylim(-0.6, 0.3)
    ax2.annotate(
        "sigmoid $R^2=-8.48$\n(off scale)", xy=(4, -0.55), xytext=(2.4, -0.45),
        fontsize=7, color=BAD, ha="left",
        arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8),
    )
    save(fig, "svm_mae_vs_r2.pdf")


# Fig 3: the honest AUC bake-off — xgboost-classifier beats every dense
# distance/margin family by many noise bands
# (2026-06-09-rotation-classification.md, Results + Findings).
def fig_auc_bakeoff():
    labels = [
        "xgboost-classifier\n(champion)",
        "svc poly",
        "logistic",
        "svc rbf",
        "svc linear",
        "svc sigmoid",
    ]
    auc = [0.7429, 0.6855, 0.6762, 0.6722, 0.6363, 0.6255]
    colors = [GOOD] + [MUTED] * 5
    band = 0.0126  # 2-sigma AUC noise band, measured this campaign

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, auc, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC (P(rotation))")
    ax.set_xlim(0.5, 0.78)
    ax.xaxis.set_major_formatter(afmt)
    ax.axvline(0.5, color="black", lw=1.0, label="chance (0.5)")
    # band envelope around the champion
    ax.axvspan(0.7429 - band, 0.7429, color=GOOD, alpha=0.12)
    for yi, a in zip(y, auc):
        ax.text(a + 0.003, yi, f"{a:.3f}", va="center", fontsize=7.5)
    ax.set_title("Honest AUC: the tree beats every dense family by 4.6--9.3 noise bands")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "auc_bakeoff.pdf")


# Fig 4: the rotation AUC noise band — three split seeds of the champion
# config, with the 2-sigma envelope
# (2026-06-09-rotation-classification.md, decision rule step 2).
def fig_auc_noise_band():
    seeds = ["s42", "s17", "s101"]
    auc = [0.742934, 0.730659, 0.734469]
    mean = 0.736020
    sigma = 0.00628
    band = 0.01257  # 2 sigma

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(seeds))
    ax.axhspan(mean - band, mean + band, color=ACCENT, alpha=0.12, label="$\\pm 2\\sigma$ band (0.0126)")
    ax.axhline(mean, color=ACCENT, ls="--", lw=1.2, label=f"mean {mean:.4f}")
    ax.plot(x, auc, "o", color=GOOD, markersize=9)
    for xi, a in zip(x, auc):
        ax.text(xi, a + 0.0013, f"{a:.4f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("data-split seed")
    ax.set_ylabel("AUC (P(rotation))")
    ax.yaxis.set_major_formatter(afmt)
    ax.set_ylim(mean - 2.2 * band, mean + 2.2 * band)
    ax.set_title("First measured rotation AUC band: $\\sigma=0.0063$, $2\\sigma=0.0126$")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "auc_noise_band.pdf")


# Fig 5: the four-family bake-off on the NEW full-64-d corpus — the family
# ordering survives the corpus change; the two tree families lead, the dense
# net is last (2026-06-12-classifier-bakeoff-p64.md, Phase 1).
def fig_bakeoff_p64():
    labels = [
        "xgboost-classifier\n(winner)",
        "catboost-classifier\n(2nd tree, in-band)",
        "ngboost-classifier",
        "pyramid-mlp-classifier\n(dense net)",
    ]
    auc = [0.7204, 0.7111, 0.6917, 0.6781]
    colors = [GOOD, ACCENT, MUTED, BAD]
    band = 0.0126  # 2-sigma AUC band (old-corpus estimate, approximate here)
    winner = 0.7204

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, auc, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC (P(rotation))")
    ax.set_xlim(0.5, 0.76)
    ax.xaxis.set_major_formatter(afmt)
    ax.axvline(0.5, color="black", lw=1.0, label="chance (0.5)")
    # in-band strip below the winner: catboost falls inside it
    ax.axvspan(winner - band, winner, color=GOOD, alpha=0.12, label="$2\\sigma$ band below winner")
    for yi, a in zip(y, auc):
        ax.text(a + 0.003, yi, f"{a:.3f}", va="center", fontsize=7.5)
    ax.set_title("New full-64-d corpus: two tree families lead, the dense net is last")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "bakeoff_p64.pdf")


# Fig 6: xgboost depth × learning-rate sweep on the new corpus — depth is the
# dominant axis, high lr hurts; the best cell beats the default sub-band
# (2026-06-12-classifier-bakeoff-p64.md, Phase 2).
def fig_xgb_depth_lr_heatmap():
    depths = [4, 6, 8]
    lrs = [0.03, 0.05, 0.1]
    # AUC[depth_idx][lr_idx], transcribed from the Phase-2 sweep table.
    auc = np.array([
        [0.7074, 0.7094, 0.7119],  # depth 4
        [0.7205, 0.7204, 0.7153],  # depth 6
        [0.7261, 0.7196, 0.7173],  # depth 8
    ])
    best = (2, 0)  # depth 8, lr 0.03

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    im = ax.pcolormesh(
        np.arange(len(lrs) + 1), np.arange(len(depths) + 1), auc,
        cmap="viridis", shading="flat",
    )
    ax.set_xticks(np.arange(len(lrs)) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in lrs])
    ax.set_yticks(np.arange(len(depths)) + 0.5)
    ax.set_yticklabels([str(d) for d in depths])
    ax.set_xlabel("learning rate")
    ax.set_ylabel("max_depth")
    ax.grid(False)
    for i in range(len(depths)):
        for j in range(len(lrs)):
            is_best = (i, j) == best
            ax.text(
                j + 0.5, i + 0.5, f"{auc[i, j]:.4f}",
                ha="center", va="center", fontsize=8,
                color="white" if auc[i, j] < 0.718 else "black",
                fontweight="bold" if is_best else "normal",
            )
            if is_best:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor=BAD, lw=2.2))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("AUC (P(rotation))")
    cbar.formatter = afmt
    cbar.update_ticks()
    ax.set_title("Depth is the lever, fast lr hurts;\nbest (depth 8, lr 0.03) is only sub-band over default")
    save(fig, "xgb_depth_lr_heatmap.pdf")


# Fig 7: the dense-net diagnostic signature — genuine AUC ranking signal
# coexisting with blown log-loss (calibration ceiling), contrasted with the
# trees. Source: bake-off (xgb/catboost/ngboost/mlp logloss) +
# arch scan (best converged MLP, and the random-baseline log-loss reference)
# (2026-06-12-classifier-bakeoff-p64.md + 2026-06-12-pyramid-mlp-arch-scan.md).
def fig_mlp_calibration_signature():
    labels = [
        "xgboost",
        "catboost",
        "ngboost",
        "pyramid-mlp\n(arch scan best)",
        "pyramid-mlp\n(bake-off)",
    ]
    auc = [0.7204, 0.7111, 0.6917, 0.6792, 0.6781]
    logloss = [0.6075, 0.6132, 0.6293, 2.0584, 2.4085]
    colors = [GOOD, GOOD, MUTED, BAD, BAD]
    rand_logloss = 0.6900  # log-loss of a constant base-rate predictor (~0.445)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.4))
    x = np.arange(len(labels))

    ax1.bar(x, auc, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylim(0.5, 0.74)
    ax1.yaxis.set_major_formatter(afmt)
    ax1.axhline(0.5, color="black", lw=1.0)
    ax1.set_ylabel("AUC (P(rotation))")
    ax1.set_title("AUC: the net DOES rank\n(genuine signal, $\\approx 0.68$)")

    ax2.bar(x, logloss, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.axhline(rand_logloss, color=BAD, ls="--", lw=1.2, label="random baseline ($\\approx 0.69$)")
    ax2.set_ylabel("log-loss (lower better)")
    ax2.set_title("log-loss: but it is confidently WRONG\n(worse than the random baseline)")
    ax2.legend(loc="upper left", fontsize=7)
    save(fig, "mlp_calibration_signature.pdf")


# Fig 8: the pyramid-MLP architecture scan refutes the budget hypothesis —
# every geometry self-stops at 73–100 iters (well below the 800 cap and even
# the old 300 cap), and the whole family lands inside one noise band
# (2026-06-12-pyramid-mlp-arch-scan.md, Results).
def fig_mlp_arch_scan():
    arms = [
        "M1 [512,256]",
        "M0 [256,128,64]",
        "M4 [384,307,246]",
        "M2 [256,128,64,32]",
        "M3 [512,169,56]",
        "M5 [512,307,184,111]",
    ]
    auc = [0.6792, 0.6781, 0.6777, 0.6737, 0.6736, 0.6712]
    iters = [75, 73, 87, 100, 89, 84]
    band = 0.0126
    default = 0.6781  # M0 re-baseline (== the under-trained 300-iter default)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    y = np.arange(len(arms))[::-1]

    # left: AUC, whole family inside one band of the default
    ax1.barh(y, auc, color=[GOOD] + [MUTED] * 5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(arms, fontsize=7)
    ax1.set_xlim(0.66, 0.685)
    ax1.xaxis.set_major_formatter(afmt)
    ax1.axvspan(default - band, default + band, color=ACCENT, alpha=0.12, label="$\\pm 2\\sigma$ band")
    ax1.axvline(default, color=ACCENT, ls="--", lw=1.0)
    for yi, a in zip(y, auc):
        ax1.text(a - 0.0006, yi, f"{a:.4f}", va="center", ha="right", fontsize=6.5, color="white")
    ax1.set_xlabel("AUC (P(rotation))")
    ax1.set_title("Every geometry is inside one band\n(architecture is exhausted)")
    ax1.legend(loc="lower right", fontsize=6.5)

    # right: stopping iteration vs the caps — budget was never binding
    ax2.barh(y, iters, color=MUTED)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.axvline(300, color=ACCENT, ls=":", lw=1.4, label="old cap (300)")
    ax2.axvline(800, color=BAD, ls="--", lw=1.4, label="raised cap (800)")
    for yi, it in zip(y, iters):
        ax2.text(it + 6, yi, f"{it}", va="center", fontsize=7)
    ax2.set_xlim(0, 850)
    ax2.set_xlabel("iterations to self-stop")
    ax2.set_title("All self-stop at 73--100 iters:\nbudget was never the constraint")
    ax2.legend(loc="lower right", fontsize=6.5)
    save(fig, "mlp_arch_scan.pdf")


# Fig 9: the burn-deep-classifier topology bake-off resolves the open question.
# Left: the three topologies (plain mlp wins, all inside one band, embeddings
# edges wide_deep). Right: the calibration recovery — the Rust/burn net with RAW
# one-hots drops log-loss from the sklearn pyramid's ~2.0 to the trees' ~0.62
# regime (2026-06-12-burn-deep-topology.md, Phase 1 + Findings (i)).
def fig_burn_topology():
    # Left panel: Phase-1 topology AUC (winner mlp, then embeddings, wide_deep).
    topo = ["mlp\n(plain)", "embeddings", "wide_deep"]
    topo_auc = [0.7001, 0.6934, 0.6918]
    band = 0.0126  # 2-sigma AUC band (old-corpus estimate, approximate here)
    winner = 0.7001
    topo_colors = [GOOD, ACCENT, MUTED]

    # Right panel: the calibration recovery — sklearn pyramid vs the burn net,
    # the headline finding (log-loss 2.0 -> 0.62, into the tree regime).
    calib_labels = [
        "sklearn\npyramid-mlp",
        "burn-deep\n(mlp, raw 1-hot)",
        "xgboost\n(tree tier)",
    ]
    calib_logloss = [2.0584, 0.6211, 0.6075]
    calib_colors = [BAD, GOOD, MUTED]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4))

    # Left: topology comparison, all inside one band of the plain-mlp winner.
    x = np.arange(len(topo))
    ax1.bar(x, topo_auc, color=topo_colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(topo, fontsize=7.5)
    ax1.set_ylim(0.5, 0.72)
    ax1.yaxis.set_major_formatter(afmt)
    ax1.axhline(0.5, color="black", lw=1.0)
    ax1.axhspan(winner - band, winner, color=GOOD, alpha=0.12, label="$2\\sigma$ band below winner")
    for xi, a in zip(x, topo_auc):
        ax1.text(xi, a + 0.004, f"{a:.4f}", ha="center", fontsize=7.5)
    ax1.set_ylabel("AUC (P(rotation))")
    ax1.set_title("Plain mlp wins; entity embeddings\nstarved at low cardinality")
    ax1.legend(loc="lower center", fontsize=6.5)

    # Right: the calibration fix — the headline.
    xc = np.arange(len(calib_labels))
    ax2.bar(xc, calib_logloss, color=calib_colors)
    ax2.set_xticks(xc)
    ax2.set_xticklabels(calib_labels, fontsize=7)
    rand_logloss = 0.6900  # constant base-rate predictor (~0.445)
    ax2.axhline(rand_logloss, color=MUTED, ls="--", lw=1.0, label="random baseline ($\\approx 0.69$)")
    for xi, ll in zip(xc, calib_logloss):
        ax2.text(xi, ll + 0.04, f"{ll:.3f}", ha="center", fontsize=7.5)
    ax2.annotate(
        "", xy=(1, 0.72), xytext=(0, 2.0),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6),
    )
    ax2.text(0.5, 1.45, "raw one-hots\nfix calibration", ha="center", fontsize=7, color=ACCENT)
    ax2.set_ylabel("log-loss (lower better)")
    ax2.set_title("The headline: log-loss $2.0\\to 0.62$\n(into the tree regime)")
    ax2.legend(loc="upper right", fontsize=6.5)
    save(fig, "burn_topology.pdf")


# Fig 10: raising identity cardinality flips the topology ranking and lifts the
# ANN into the tree tier — the paired re-run on the HIGH-VOCAB matrix.
# Left: the embeddings-vs-mlp reversal across the two cardinalities (60/40 ->
# 300/120) — embeddings goes from below mlp to above it (a +0.0130 swing).
# Right: the five-family AUC ranking on the high-vocab matrix — the saved
# embeddings net reaches the bottom of the tree tier
# (2026-06-12-high-vocab-topology.md, Phase 0 + Phase 1 + Findings (i),(iii)).
def fig_burn_highvocab():
    # Left panel: the paired emb-vs-mlp comparison across the two matrices.
    # 60/40 matrix (the prior burn-deep run): mlp beats embeddings by +0.0067.
    # high-vocab matrix (this run): embeddings beats mlp by +0.0063 — reversal.
    grp_labels = ["60/40 matrix\n(artist 60, genre 40)", "high-vocab matrix\n(artist 300, genre 120)"]
    emb = [0.6934, 0.7192]
    mlp = [0.7001, 0.7129]

    # Right panel: the five-family AUC ranking on the high-vocab matrix.
    band = 0.0126  # 2-sigma AUC band (old-corpus estimate, approximate here)
    fam_labels = [
        "xgboost\n(in-dataset leader)",
        "catboost",
        "burn-deep emb\n(Phase-2 best)",
        "burn-deep emb\n(saved)",
        "burn-deep mlp",
        "burn-deep wide_deep",
    ]
    fam_auc = [0.7299, 0.7252, 0.7228, 0.7192, 0.7129, 0.7061]
    fam_colors = [GOOD, GOOD, ACCENT, ACCENT, MUTED, MUTED]
    tree_floor = 0.7252  # bottom of the in-dataset tree tier (catboost)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # Left: grouped bars, embeddings vs mlp, across the two cardinalities.
    x = np.arange(len(grp_labels))
    w = 0.36
    b_emb = ax1.bar(x - w / 2, emb, w, color=ACCENT, label="embeddings")
    b_mlp = ax1.bar(x + w / 2, mlp, w, color=MUTED, label="mlp")
    ax1.set_xticks(x)
    ax1.set_xticklabels(grp_labels, fontsize=6.8)
    ax1.set_ylim(0.66, 0.73)
    ax1.yaxis.set_major_formatter(afmt)
    ax1.set_ylabel("AUC (P(rotation))")
    for bars, vals in ((b_emb, emb), (b_mlp, mlp)):
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.0008, f"{v:.4f}",
                     ha="center", fontsize=6.3)
    # arrows annotating who wins in each regime
    ax1.annotate("mlp wins\n($-0.0067$)", xy=(0, 0.704), xytext=(0, 0.722),
                 ha="center", fontsize=6.5, color=BAD,
                 arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
    ax1.annotate("embeddings wins\n($+0.0063$)", xy=(1, 0.722), xytext=(1, 0.668),
                 ha="center", fontsize=6.5, color=GOOD,
                 arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8))
    ax1.set_title("Higher cardinality FLIPS the ranking\n(emb vs mlp, a $+0.0130$ swing)")
    ax1.legend(loc="lower left", fontsize=6.5)

    # Right: five-family ranking on the high-vocab matrix, tree-tier floor shaded.
    y = np.arange(len(fam_labels))[::-1]
    ax2.barh(y, fam_auc, color=fam_colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels(fam_labels, fontsize=6.3)
    ax2.set_xlim(0.69, 0.735)
    ax2.xaxis.set_major_formatter(afmt)
    # tree tier: from the band below catboost up to xgboost
    ax2.axvspan(tree_floor - band, fam_auc[0], color=GOOD, alpha=0.10,
                label="in-dataset tree tier")
    ax2.axvline(tree_floor, color=GOOD, ls="--", lw=1.0)
    for yi, a in zip(y, fam_auc):
        ax2.text(a + 0.0006, yi, f"{a:.4f}", va="center", fontsize=6.3)
    ax2.set_xlabel("AUC (P(rotation))")
    ax2.set_title("The saved embeddings net reaches\nthe bottom of the tree tier")
    ax2.legend(loc="lower right", fontsize=6.5)
    save(fig, "burn_highvocab.pdf")


# Fig 11: the GBDT-leaf-embedding -> head ceiling probe (Facebook GBDT->LR
# generalized to GBDT->MLP). Every head trained on the frozen champion's one-hot
# leaf indices caps BELOW the bare tree on AUC (left) AND blows calibration
# (right): the leaf one-hots are a strictly lossier view of a tree that already
# scores better directly (2026-06-15-gbdt-leaf-head-ceiling-probe.md, Results +
# Findings). On the 60/40 full-64-d matrix ds-20260609-204419-p64-s42, seed 42.
def fig_leaf_head_ceiling():
    labels = [
        "bare xgboost\n(C0 control)",
        "GBDT->linear\n(T1)",
        "GBDT->mlp[256,128]\n(T3)",
        "GBDT->mlp[128]\n(T2)",
    ]
    auc = [0.7204, 0.6978, 0.6921, 0.6897]
    logloss = [0.6075, 1.3119, 1.1813, 0.8162]
    colors = [GOOD, BAD, BAD, BAD]
    band = 0.0126  # 2-sigma AUC band
    champ = 0.7204
    refute = champ - band  # 0.7078 refutation threshold

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    x = np.arange(len(labels))

    # Left: AUC — every head below the refutation threshold (champion - band).
    ax1.bar(x, auc, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylim(0.66, 0.73)
    ax1.yaxis.set_major_formatter(afmt)
    ax1.axhline(champ, color=GOOD, ls="-", lw=1.2)
    ax1.axhspan(refute, champ, color=GOOD, alpha=0.12, label="$2\\sigma$ band below champion")
    ax1.axhline(refute, color=GOOD, ls="--", lw=1.0, label=f"refutation thr. ({refute:.4f})")
    for xi, a in zip(x, auc):
        ax1.text(xi, a + 0.0008, f"{a:.4f}", ha="center", fontsize=6.8)
    ax1.set_ylabel("AUC (P(rotation))")
    ax1.set_title("Every head caps below the bare tree\n(1.8--2.4 bands; REFUTED)")
    ax1.legend(loc="lower right", fontsize=6.3)

    # Right: log-loss — every head mis-calibrated vs the tree.
    ax2.bar(x, logloss, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.5)
    rand_logloss = 0.6900  # constant base-rate predictor (~0.445)
    ax2.axhline(rand_logloss, color=MUTED, ls="--", lw=1.0, label="random baseline ($\\approx 0.69$)")
    ax2.axhline(0.6075, color=GOOD, ls=":", lw=1.2, label="tree regime (0.608)")
    for xi, ll in zip(x, logloss):
        ax2.text(xi, ll + 0.03, f"{ll:.3f}", ha="center", fontsize=6.8)
    ax2.set_ylabel("log-loss (lower better)")
    ax2.set_title("And every head is mis-calibrated\n(log-loss 0.82--1.31 vs 0.61)")
    ax2.legend(loc="upper right", fontsize=6.3)
    save(fig, "leaf_head_ceiling.pdf")


# Fig 12: the identity-resolution vocab-cap scan on the bare GBDT — the cap->AUC
# curve for artist/genre one-hot cardinality, holding the champion xgboost
# config (and catboost as a secondary arm) fixed and moving only vocab_top_n.
# The first jump (60->300) carries the gain; past 300/120 the lever pays only
# sub-band returns that wobble inside the noise envelope, and the top-of-curve
# all-vocab point (6633 cols) could not be built (host memory ceiling) — marked
# as unmeasured, NOT a modeling result. Catboost tracks xgboost qualitatively.
# (2026-06-15-vocab-cap-scan-gbdt.md, "The cap -> AUC curve" + Findings).
def fig_vocab_cap_scan():
    caps = ["60/40", "300/120", "500/200", "1000/300", "all\n5594/962"]
    ncols = [177, 497, 777, 1377, 6633]
    x = np.arange(len(caps))
    # AUC by artist/genre cap; the all-vocab point is unmeasured (build failed).
    xgb = [0.72044, 0.72988, 0.72966, 0.73490, np.nan]
    cat = [0.71105, 0.72516, 0.72372, 0.72633, np.nan]
    band = 0.0126  # 2-sigma AUC band (rotation-classification estimate)
    anchor = 0.72988  # the 300/120 xgboost anchor the scan is read against
    best = (3, 0.73490)  # 1000/300 xgboost, new in-lineage single-split best

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    # band envelope above the 300/120 anchor: a >=1-band beat would clear its top
    ax.axhspan(anchor, anchor + band, color=GOOD, alpha=0.10,
               label="$+2\\sigma$ band over 300/120 anchor")
    ax.axhline(anchor, color=GOOD, ls="--", lw=1.0)

    xm = np.arange(4)  # measured points only (drop the failed all-vocab x)
    ax.plot(xm, xgb[:4], "o-", color=ACCENT, lw=1.8, label="xgboost (champion config)")
    ax.plot(xm, cat[:4], "s--", color=MUTED, lw=1.4, label="catboost (secondary)")
    for xi, a in zip(xm, xgb[:4]):
        ax.text(xi, a + 0.0010, f"{a:.4f}", ha="center", fontsize=7, color=ACCENT)
    # mark the new in-lineage best
    ax.annotate("new in-lineage best\n(+0.0050, sub-band)", xy=best,
                xytext=(2.1, 0.7368), ha="center", fontsize=6.8, color=GOOD,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8))
    # the unmeasured all-vocab point: a hatched marker on the axis, no value
    ax.axvline(4, color=BAD, ls=":", lw=1.2)
    ax.scatter([4], [anchor], marker="x", s=90, color=BAD, zorder=5)
    ax.annotate("all-vocab (6633 cols)\nNOT measured\n(build memory ceiling)",
                xy=(4, anchor), xytext=(3.05, 0.7250), ha="center",
                fontsize=6.5, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({n} cols)" for c, n in zip(caps, ncols)], fontsize=6.8)
    ax.set_xlabel("artist / genre one-hot cap (column count)")
    ax.set_ylabel("AUC (P(rotation))")
    ax.yaxis.set_major_formatter(afmt)
    ax.set_ylim(0.705, 0.740)
    ax.set_title("Identity resolution is a largely-spent lever: the first jump pays,\npast 300/120 the returns are sub-band")
    ax.legend(loc="lower right", fontsize=6.8)
    save(fig, "vocab_cap_scan.pdf")


# Fig 13: the burn-ae-classifier capacity scan — the AE-transplant net.
# LEFT: 12 fine-tuned configs span one 0.0054 strip entirely inside the 0.0126
# band, well below the burn-deep mlp and the xgboost tree, so capacity (latent
# size, head depth/width, activation) is a spent lever. RIGHT: the ONE real
# effect — freezing the MSE-pretrained encoder is safe under a (near-)linear
# head (cfg11) but collapses under a DEEP head (cfg10, -1.2 bands), because a
# deep head stacked on a frozen sub-optimal representation just underfits the
# label. On the 60/40 full-64-d matrix ds-20260609-204419-p64-s42, seed 42.
# (2026-06-15-burn-ae-classifier-capacity-scan.md, Results + Findings iii.)
def fig_burn_ae_capacity():
    # the 12 fine-tuned configs (freeze=false), sorted high->low
    ft_auc = [0.69719, 0.69712, 0.69657, 0.69645, 0.69561, 0.69541,
              0.69340, 0.69302, 0.69242, 0.69240, 0.69175]
    # (cfg11 0.69530 is frozen+shallow -> shown on the right panel, not here)
    baseline = 0.69225   # prior baseline run (enc[128,64]/clf[64], fine-tune)
    burn_mlp = 0.7038    # comparable dense ANN (burn-deep mlp) on this matrix
    xgb = 0.7204         # in-corpus xgboost champion on this matrix
    band = 0.0126        # 2-sigma AUC band

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # LEFT: the flat capacity strip vs the two reference tiers.
    xj = np.random.default_rng(0).uniform(-0.12, 0.12, len(ft_auc))
    ax1.scatter(xj, ft_auc, s=34, color=ACCENT, zorder=4,
                label="12 fine-tuned configs")
    lo, hi = min(ft_auc), max(ft_auc)
    ax1.axhspan(lo, hi, color=ACCENT, alpha=0.10,
                label=f"whole strip ({hi - lo:.4f} wide)")
    ax1.axhline(baseline, color=MUTED, ls=":", lw=1.2,
                label=f"baseline ({baseline:.4f})")
    ax1.axhline(burn_mlp, color=MUTED, ls="--", lw=1.4,
                label=f"burn-deep mlp ({burn_mlp:.4f})")
    ax1.axhline(xgb, color=GOOD, ls="-", lw=1.6,
                label=f"xgboost champion ({xgb:.4f})")
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_xticks([])
    ax1.set_ylim(0.688, 0.723)
    ax1.yaxis.set_major_formatter(afmt)
    ax1.set_ylabel("AUC (P(rotation))")
    ax1.set_title("Capacity is a spent lever:\n12 configs in one sub-band strip")
    ax1.legend(loc="lower right", fontsize=6.2)

    # RIGHT: freeze vs fine-tune, paired for a shallow head and a deep head.
    grp = ["shallow head\n(clf [128])", "deep head\n(clf [128,64])"]
    finetune = [0.69712, 0.69719]   # cfg5, cfg3
    frozen = [0.69530, 0.68210]     # cfg11, cfg10 (collapse)
    x = np.arange(len(grp))
    w = 0.36
    b_ft = ax2.bar(x - w / 2, finetune, w, color=ACCENT, label="fine-tune")
    b_fz = ax2.bar(x + w / 2, frozen, w, color=BAD, label="frozen encoder")
    ax2.set_xticks(x)
    ax2.set_xticklabels(grp, fontsize=7)
    ax2.set_ylim(0.675, 0.702)
    ax2.yaxis.set_major_formatter(afmt)
    ax2.set_ylabel("AUC (P(rotation))")
    for bars, vals in ((b_ft, finetune), (b_fz, frozen)):
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.0007, f"{v:.4f}",
                     ha="center", fontsize=6.3)
    ax2.annotate("collapse\n($-0.0151$, $-1.2$ bands)", xy=(1 + w / 2, 0.68210),
                 xytext=(1.05, 0.694), ha="center", fontsize=6.3, color=BAD,
                 arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
    ax2.set_title("Freezing is safe only under a\n(near-)linear head")
    ax2.legend(loc="lower left", fontsize=6.5)
    save(fig, "burn_ae_capacity.pdf")


# Fig 14: per-model SAE concept allocation — at EQUAL AUC two burn-deep-clf nets
# capture the same decodable taste-fit signal but LAY IT OUT differently. The
# pyramid [256,128,64] peaks its interpretable concepts in the middle 128-d
# layer (72) then compresses ~40% out at the 64-d neck (42), while the uniform
# rect [128,128,128] spreads them and RETAINS more to the end (17->66->54). A
# representational difference the AUC leaderboard cannot see.
# (2026-07-12-pyramid-vs-rect-sae.md, per-model SAE table + Findings.)
def fig_sae_pyramid_rect():
    layers = ["layer 1", "layer 2", "layer 3\n(output)"]
    pyr = [11, 72, 42]   # [256,128,64] — peaks mid, compresses at the neck
    rect = [17, 66, 54]  # [128,128,128] — broader, retains more to the end
    x = np.arange(len(layers))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    b_p = ax.bar(x - w / 2, pyr, w, color=ACCENT,
                 label="pyramid [256,128,64] (AUC 0.6986)")
    b_r = ax.bar(x + w / 2, rect, w, color=MUTED,
                 label="uniform [128,128,128] (AUC 0.6994)")
    for bars, vals in ((b_p, pyr), (b_r, rect)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0, str(v),
                    ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_ylabel("interpretable concepts (SAE)")
    ax.set_ylim(0, 85)
    ax.annotate("neck squeezes\n~40% out", xy=(2 - w / 2, 42),
                xytext=(1.35, 20), ha="center", fontsize=6.8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    ax.set_title("Same AUC, different allocation: the pyramid consolidates\nconcepts into its bottleneck; the uniform net keeps a broad set")
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "sae_pyramid_rect.pdf")


# Fig 15: the burn-deep-classifier capacity scan on the HIGH-VOCAB matrix —
# CEILING. Two one-axis geometry ladders (embeddings, mlp) at fixed lr; neither
# widening nor deepening beats its [128,64] control by >1 band, and deepening
# EMBEDDINGS to 3 layers actively HURTS (-0.0152, the only out-of-band move).
# The best ANN (0.72274) still sits ~0.2-0.6 band below the tree tier: capacity,
# not the embeddings mechanism, was the lever, and it is saturated.
# On ds-20260612-143014-p64-s42 (497 cols). (2026-07-12-high-vocab-capacity-scan.md.)
def fig_highvocab_capacity():
    geom = ["[128,64]\n(control)", "[256,128]\n(wider)", "[256,128,64]\n(deeper)"]
    emb = [0.72274, 0.71798, 0.70750]
    mlp = [0.71229, 0.71576, 0.71800]
    x = np.arange(len(geom))
    band = 0.0126
    xgb = 0.7299     # in-dataset tree leader
    cat = 0.7252     # 2nd tree family (bottom of the tree tier)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    # tree tier band: from a noise band below catboost up to xgboost
    ax.axhspan(cat - band, xgb, color=GOOD, alpha=0.10, label="in-dataset tree tier")
    ax.axhline(xgb, color=GOOD, ls="-", lw=1.2)
    ax.axhline(cat, color=GOOD, ls="--", lw=1.0)
    ax.plot(x, emb, "o-", color=ACCENT, lw=1.8, label="embeddings (ed8, lr 5e-4)")
    ax.plot(x, mlp, "s--", color=MUTED, lw=1.6, label="mlp (lr 1e-3)")
    for xi, v in zip(x, emb):
        ax.text(xi, v + 0.0011, f"{v:.4f}", ha="center", fontsize=6.8, color=ACCENT)
    for xi, v in zip(x, mlp):
        ax.text(xi, v - 0.0018, f"{v:.4f}", ha="center", fontsize=6.8, color="#4b5563")
    ax.annotate("3rd embedding layer HURTS\n($-0.0152$, $-1.2$ bands)", xy=(2, 0.70750),
                xytext=(1.15, 0.7045), ha="center", fontsize=6.5, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(geom, fontsize=7.5)
    ax.set_xlabel("hidden geometry")
    ax.set_ylabel("AUC (P(rotation))")
    ax.yaxis.set_major_formatter(afmt)
    ax.set_ylim(0.702, 0.734)
    ax.set_title("Capacity is saturated at high vocab: widening buys nothing,\ndeepening embeddings hurts, and the ANN tier stays below the trees")
    ax.legend(loc="lower right", fontsize=6.8)
    save(fig, "highvocab_capacity.pdf")


# Fig 16: taste drift — the honest forward-looking read. LEFT: a chronological
# (train-earlier / test-later) split drops AUC ~0.11-0.12 for EVERY model family
# vs the random split, so the drop is the data/taste, not a model quirk. RIGHT:
# walk-forward (train on first_played < Y, test on year Y) lands at 0.59-0.67
# every single year, never near the 0.735 random baseline, and more history
# does not close the gap. The leaderboard AUC is an upper bound on same-era
# ranking; forward skill is ~0.62. (2026-07-12-taste-drift-temporal-split.md.)
def fig_taste_drift():
    fams = ["xgboost\n(champion)", "catboost", "logistic"]
    rand = [0.73490, 0.72889, 0.72141]
    chrono = [0.61894, 0.61294, 0.61234]
    x = np.arange(len(fams))
    w = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.5))

    # LEFT: random vs chronological, per family.
    b_r = ax1.bar(x - w / 2, rand, w, color=GOOD, label="random split")
    b_c = ax1.bar(x + w / 2, chrono, w, color=BAD, label="chronological split")
    ax1.set_xticks(x)
    ax1.set_xticklabels(fams, fontsize=7)
    ax1.set_ylim(0.55, 0.75)
    ax1.yaxis.set_major_formatter(afmt)
    ax1.set_ylabel("AUC (P(rotation))")
    for bars, vals in ((b_r, rand), (b_c, chrono)):
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}",
                     ha="center", fontsize=6.3)
    for xi, (r, c) in enumerate(zip(rand, chrono)):
        ax1.annotate("", xy=(xi + w / 2, c + 0.004), xytext=(xi - w / 2, r - 0.004),
                     arrowprops=dict(arrowstyle="->", color="#4b5563", lw=0.8))
    ax1.text(1.0, 0.565, "drift $\\approx-0.11$ for every family",
             ha="center", fontsize=6.8, color=BAD)
    ax1.set_title("A chronological split costs ~0.11 AUC\nacross every model family")
    ax1.legend(loc="upper right", fontsize=6.5)

    # RIGHT: walk-forward AUC by test year vs the random and chrono references.
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    wf = [0.6001, 0.6272, 0.6548, 0.6577, 0.6697, 0.5895, 0.6318, 0.6213]
    rand_base = 0.73490
    chrono_ref = 0.61894
    ax2.plot(years, wf, "o-", color=ACCENT, lw=1.8, label="walk-forward (test year Y)")
    ax2.axhline(rand_base, color=GOOD, ls="-", lw=1.4,
                label=f"random baseline ({rand_base:.3f})")
    ax2.axhline(chrono_ref, color=BAD, ls="--", lw=1.2,
                label=f"chrono split ({chrono_ref:.3f})")
    ax2.fill_between([2018.5, 2026.5], chrono_ref, rand_base, color=BAD, alpha=0.06)
    ax2.set_xlim(2018.5, 2026.5)
    ax2.set_ylim(0.55, 0.75)
    ax2.yaxis.set_major_formatter(afmt)
    ax2.set_xlabel("test year")
    ax2.set_ylabel("AUC (P(rotation))")
    ax2.set_title("Predicting any single future year lands at\n0.59--0.67, never near the random baseline")
    ax2.legend(loc="upper right", fontsize=6.3)
    save(fig, "taste_drift.pdf")



# Fig 17: the next-track item-representation compression bake-off — Recall@10
# vs latent dim for PCA and the AE, each read by the SAME frozen GRU
# (gru-infonce-h256). Latent DIM, not the compressor, is the lever: both series
# climb steeply 32->64; PCA keeps climbing to 128 and OVERTAKES the AE, while
# the AE plateaus (128 dips below its own 64). At dim 32 the AE edges PCA — a
# low-capacity crossover that reverses once dim is generous, refuting the "AE
# preserves acoustic blocks" hypothesis. The AE-64 control (the champion's
# space) and the PCA-128 winner are marked; error bars are the bootstrap 95% CI
# over the 1431 test sessions.
# (2026-07-15-nexttrack-compression-representation-bakeoff.md, Results.)
def fig_nexttrack_compression():
    dims = [32, 64, 128]
    pca = [0.0412, 0.1041, 0.1146]
    pca_lo = [0.0314, 0.0881, 0.0985]
    pca_hi = [0.0517, 0.1209, 0.1321]
    ae = [0.0545, 0.0957, 0.0929]
    ae_lo = [0.0426, 0.0811, 0.0783]
    ae_hi = [0.0664, 0.1111, 0.1076]
    AE_COLOR = "#4b5563"  # dark gray, distinct from PCA's accent blue

    def yerr(v, lo, hi):
        return [[a - b for a, b in zip(v, lo)], [b - a for a, b in zip(v, hi)]]

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.errorbar(dims, pca, yerr=yerr(pca, pca_lo, pca_hi), fmt="o-", color=ACCENT,
                lw=1.9, capsize=3, label="PCA (linear)")
    ax.errorbar(dims, ae, yerr=yerr(ae, ae_lo, ae_hi), fmt="s--", color=AE_COLOR,
                lw=1.6, capsize=3, label="AE (nonlinear)")

    # the AE-64 control: the champion's item space, one arbitrary prior choice
    ax.scatter([64], [0.0957], s=150, facecolors="none", edgecolors=BAD,
               linewidths=1.8, zorder=6)
    ax.annotate("AE-64 control\n(champion's space)", xy=(64, 0.0957),
                xytext=(64, 0.055), ha="center", fontsize=6.8, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
    # the PCA-128 winner
    ax.scatter([128], [0.1146], marker="*", s=180, color=GOOD, zorder=7)
    ax.annotate("PCA-128 winner\n(paired-$\\Delta$ +0.0189, CI$>$0)", xy=(128, 0.1146),
                xytext=(96, 0.128), ha="center", fontsize=6.8, color=GOOD,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8))
    # the low-dim crossover: AE edges PCA at 32
    ax.annotate("at dim 32 the AE\nedges PCA (crossover)", xy=(32, 0.0480),
                xytext=(38, 0.088), ha="left", fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlim(28, 150)
    ax.set_ylim(0.02, 0.145)
    ax.set_xlabel("item-latent dimension")
    ax.set_ylabel("Recall@10 (next-distinct, frozen GRU)")
    ax.yaxis.set_major_formatter(afmt)
    ax.set_title("Latent dimension, not the compressor, is the lever:\nPCA overtakes the AE once dim is generous")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "nexttrack_compression.pdf")

# Fig 18: the next-track compression DIM curve, extended to PCA-192/256 and
# AE-192. The bake-off (Fig 17) stopped at 128 with PCA still climbing; the
# dim-extension leg locates the ceiling: PCA PEAKS at ~192 (R@10 0.123) and
# TURNS OVER by 256 (0.117), while the AE plateaus flat at ~0.09-0.10 from dim
# 64 upward. The PCA>=AE gap is widest at 192 (0.123 vs 0.098). PCA-192 is the
# point-optimum but its paired-Delta vs PCA-128 grazes 0, so PCA-128 stays the
# Phase-2 base. Error bars are bootstrap 95% CIs over the 1431 test sessions.
# (2026-07-15b-nexttrack-compression-dim-extension.md, Findings 1-2.)
def fig_nexttrack_dim_curve():
    pca_d = [32, 64, 128, 192, 256]
    pca = [0.0412, 0.1041, 0.1146, 0.1230, 0.1167]
    pca_lo = [0.0314, 0.0881, 0.0985, 0.1069, 0.1006]
    pca_hi = [0.0517, 0.1209, 0.1321, 0.1405, 0.1335]
    ae_d = [32, 64, 128, 192]
    ae = [0.0545, 0.0957, 0.0929, 0.0978]
    ae_lo = [0.0426, 0.0811, 0.0783, 0.0825]
    ae_hi = [0.0664, 0.1111, 0.1076, 0.1132]
    AE_COLOR = "#4b5563"  # dark gray, distinct from PCA's accent blue

    def yerr(v, lo, hi):
        return [[a - b for a, b in zip(v, lo)], [b - a for a, b in zip(v, hi)]]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.errorbar(pca_d, pca, yerr=yerr(pca, pca_lo, pca_hi), fmt="o-", color=ACCENT,
                lw=1.9, capsize=3, label="PCA (linear)")
    ax.errorbar(ae_d, ae, yerr=yerr(ae, ae_lo, ae_hi), fmt="s--", color=AE_COLOR,
                lw=1.6, capsize=3, label="AE (nonlinear)")

    # the PCA peak at 192
    ax.scatter([192], [0.1230], marker="*", s=200, color=GOOD, zorder=7)
    ax.annotate("PCA-192 peak\n(point-optimum; $\\Delta$-CI grazes 0)",
                xy=(192, 0.1230), xytext=(150, 0.137), ha="center", fontsize=6.8,
                color=GOOD, arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8))
    # the turnover at 256
    ax.annotate("turns over\nby 256", xy=(256, 0.1167), xytext=(256, 0.083),
                ha="center", fontsize=6.8, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
    # PCA-128 base
    ax.annotate("PCA-128\n(Phase-2 base)", xy=(128, 0.1146), xytext=(96, 0.128),
                ha="center", fontsize=6.5, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    # AE-64 control
    ax.scatter([64], [0.0957], s=140, facecolors="none", edgecolors="#4b5563",
               linewidths=1.6, zorder=6)
    ax.annotate("AE-64 control", xy=(64, 0.0957), xytext=(64, 0.066),
                ha="center", fontsize=6.5, color=AE_COLOR,
                arrowprops=dict(arrowstyle="->", color=AE_COLOR, lw=0.8))

    ax.set_xscale("log", base=2)
    ax.set_xticks(pca_d)
    ax.set_xticklabels([str(d) for d in pca_d])
    ax.set_xlim(28, 300)
    ax.set_ylim(0.02, 0.15)
    ax.set_xlabel("item-latent dimension")
    ax.set_ylabel("Recall@10 (next-distinct, frozen GRU)")
    ax.yaxis.set_major_formatter(afmt)
    ax.set_title("The dim ceiling is found: PCA peaks at ~192, then turns over;\nthe AE plateaus and never catches it")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "nexttrack_dim_curve.pdf")


# Fig 19: the Phase-2 model-family sweep on the rich PCA spaces. The FIXED
# z-blend is the combiner of record: R+M+C on PCA-192 is the new champion ROW
# (R@10 0.1859), the content leg is genuinely additive on the rich space, and
# the LEARNED XGB stacker badly UNDERPERFORMS the z-blend (0.08-0.09), losing
# even to its own single GRU base leg. LSTM edges GRU as the best single model.
# The first-order Markov bar (0.107) is drawn as the incumbent reference.
# (2026-07-15c-nexttrack-phase2-model-sweep.md, full sweep.)
def fig_nexttrack_phase2_families():
    labels = [
        "R+M+C z-blend, PCA-192\n(NEW CHAMPION ROW)",
        "R+M+C z-blend, PCA-128",
        "R+M z-blend, AE-64\n(prior champion)",
        "R+M z-blend, PCA-192",
        "LSTM single, PCA-128\n(best single)",
        "GRU single, PCA-192",
        "GRU single, PCA-128",
        "ANN pooled-MLP, PCA-128",
        "XGB stacker M+G, PCA-128",
        "XGB stacker M+G+A, PCA-192",
    ]
    r10 = [0.1859, 0.1824, 0.1726, 0.1712, 0.1272, 0.1230, 0.1146, 0.0901, 0.0867, 0.0853]
    # green = fixed-blend champions; accent = other fixed blends; muted = single
    # models; red = the learned XGB stacker (the underperformer).
    colors = [GOOD, GOOD, ACCENT, ACCENT, MUTED, MUTED, MUTED, MUTED, BAD, BAD]
    markov = 0.107

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, r10, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlabel("Recall@10 (next-distinct)")
    ax.set_xlim(0.0, 0.215)
    ax.xaxis.set_major_formatter(afmt)
    # the first-order Markov bar (app incumbent)
    ax.axvline(markov, color="black", ls="--", lw=1.1, label="first-order Markov bar (0.107)")
    # the champion line
    ax.axvline(0.1859, color=GOOD, ls=":", lw=1.2, label="champion row (0.186)")
    for yi, v in zip(y, r10):
        ax.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=7)
    ax.set_title("Phase 2: the fixed z-blend is the combiner; the learned XGB\nstacker underperforms even its own GRU base leg")
    ax.legend(loc="lower right", fontsize=7.2)
    save(fig, "nexttrack_phase2_families.pdf")


# Fig 20: the musical-distance metric (music@10) back-filled across the 53
# next-track runs — Recall@10 (x) vs music@10 (y), one point per run, colored by
# predictor family. The metric CORRELATES with exact recall (r=0.609, stated in
# the report) but crowns a DIFFERENT winner: the Recall@10 champion is a
# seq-blend run (0.223), while the musically-closest predictions come from
# seq-nexttrack (which regresses the next track's content latent and retrieves
# by cosine) — best music@10 0.482, highest family mean 0.448 — even though its
# exact recall is lower. The seq-popularity floor is correctly worst (0.185), so
# the metric is not trivially saturated. music@10 is DISPLAY-ONLY; recall@10
# stays primary. (2026-07-20-musical-distance-metric.md, sec. 4-5.)
def fig_nexttrack_music_scatter():
    # (recall@10, music@10) per run, grouped by predictor family. Transcribed
    # verbatim from the full 53-run backfill table (report sec. 5).
    fam = {
        "seq-blend": {
            "color": ACCENT, "marker": "o",
            "r": [0.223, 0.217, 0.216, 0.212, 0.212, 0.209, 0.208, 0.205, 0.205,
                  0.200, 0.195, 0.194, 0.193, 0.186, 0.173, 0.173, 0.171, 0.171,
                  0.167, 0.158, 0.152, 0.128, 0.117],
            "m": [0.461, 0.460, 0.455, 0.454, 0.454, 0.452, 0.454, 0.452, 0.455,
                  0.458, 0.448, 0.447, 0.447, 0.449, 0.441, 0.441, 0.443, 0.396,
                  0.444, 0.434, 0.429, 0.383, 0.377],
        },
        "seq-nexttrack": {
            "color": GOOD, "marker": "s",
            "r": [0.159, 0.157, 0.157, 0.153, 0.149, 0.148, 0.138, 0.127, 0.123,
                  0.123, 0.123, 0.123, 0.123, 0.123, 0.122, 0.117, 0.115, 0.104,
                  0.098, 0.096, 0.096, 0.093, 0.055, 0.041],
            "m": [0.465, 0.482, 0.462, 0.475, 0.472, 0.450, 0.458, 0.451, 0.456,
                  0.456, 0.456, 0.456, 0.456, 0.456, 0.450, 0.456, 0.457, 0.437,
                  0.436, 0.432, 0.432, 0.436, 0.368, 0.392],
        },
        "seq-markov": {"color": "#9333ea", "marker": "D", "r": [0.107], "m": [0.407]},
        "seq-ann": {"color": "#ea580c", "marker": "^", "r": [0.090], "m": [0.426]},
        "seq-stacker": {"color": "#0891b2", "marker": "v",
                        "r": [0.087, 0.085, 0.080], "m": [0.418, 0.408, 0.409]},
        "seq-popularity": {"color": BAD, "marker": "X", "r": [0.001], "m": [0.185]},
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, d in fam.items():
        ax.scatter(d["r"], d["m"], s=42, color=d["color"], marker=d["marker"],
                   edgecolors="white", linewidths=0.4, zorder=4, label=name)

    # least-squares guide line (visual aid only; the r=0.609 is transcribed).
    allr = [v for d in fam.values() for v in d["r"]]
    allm = [v for d in fam.values() for v in d["m"]]
    a, b = np.polyfit(allr, allm, 1)
    xs = np.array([min(allr), max(allr)])
    ax.plot(xs, a * xs + b, color=MUTED, ls="--", lw=1.2, zorder=2,
            label="least-squares fit")

    # the recall@10 champion (seq-blend a781c) — top on exact recall
    ax.annotate("Recall@10 champion\n(seq-blend, 0.223)", xy=(0.223, 0.461),
                xytext=(0.150, 0.492), ha="center", fontsize=6.8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    # the music@10 champion (seq-nexttrack d3d4a) — musically closest misses
    ax.annotate("music@10 champion\n(seq-nexttrack, 0.482)", xy=(0.157, 0.482),
                xytext=(0.052, 0.500), ha="center", fontsize=6.8, color=GOOD,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8))
    # the popularity floor — correctly worst
    ax.annotate("seq-popularity floor\n(0.185, correctly worst)", xy=(0.001, 0.185),
                xytext=(0.062, 0.232), ha="center", fontsize=6.8, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))

    ax.set_xlim(-0.01, 0.245)
    ax.set_ylim(0.16, 0.515)
    ax.set_xlabel("Recall@10 (next-distinct, exact hit)")
    ax.set_ylabel("music@10 (best cosine in the balanced space)")
    ax.xaxis.set_major_formatter(afmt)
    ax.yaxis.set_major_formatter(afmt)
    ax.set_title("music@10 correlates with Recall@10 ($r=0.609$) but crowns a\ndifferent winner: the recall leader is not the musically-closest")
    ax.legend(loc="lower right", fontsize=6.8, ncol=2)
    save(fig, "nexttrack_music_scatter.pdf")


# Fig 21: the learned-content-projection champion HARDENED across two disjoint
# leak-free splits. R'+M+C' beats the R+M+C crown on BOTH the canonical
# 0.55-cold split (paired-D +0.0259) and a disjoint, colder 0.64-cold
# earlier-holdout split (paired-D +0.0426): the lead WIDENS on the colder split,
# exactly as the cold-reaching content leg predicts. Absolute R@10 is lower on
# the colder split BY CONSTRUCTION; the verdict is the paired-D, not absolute
# reproduction. (2026-07-18-nexttrack-literature-fit-campaign.md, Tier A1;
# 2026-07-18b-nexttrack-projection-registration-and-scan.md, Second-split hardening.)
def fig_nexttrack_projection_hardening():
    groups = ["canonical split\n(cold 0.55)", "disjoint earlier split\n(cold 0.64)"]
    proj = [0.2117, 0.1705]
    rmc = [0.1859, 0.1279]
    rm = [0.1712, 0.1174]
    dtext = ["paired-$\\Delta$ +0.0259\n[+0.012, +0.041]",
             "paired-$\\Delta$ +0.0426\n[+0.028, +0.057]"]
    x = np.arange(len(groups))
    w = 0.26

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars_p = ax.bar(x - w, proj, w, color=GOOD, label="R$'$+M+C$'$ projection (champion)")
    bars_c = ax.bar(x, rmc, w, color=ACCENT, label="R+M+C crown (prior)")
    bars_m = ax.bar(x + w, rm, w, color=MUTED, label="R+M 2-leg")
    for bars in (bars_p, bars_c, bars_m):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.003,
                    f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=6.6)
    for xi, t in zip(x, dtext):
        ax.text(xi, 0.246, t, ha="center", va="top", fontsize=6.6, color=GOOD)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0.0, 0.30)
    ax.set_ylabel("recall@10 (P(next-track hit))")
    ax.yaxis.set_major_formatter(afmt)
    ax.set_title("The projection lead WIDENS on the colder split:\ntwo-split hardening of the next-track champion")
    ax.legend(loc="upper right", fontsize=7.0)
    save(fig, "nexttrack_projection_hardening.pdf")


# Fig 22: the item-vector variance budget, and why the champion projection is the
# fix. The equal-weighted 4-block PCA-192 orders dimensions by RAW variance, so
# 87% of the captured budget goes to the behaviorally-inert numeric+acoustic
# blocks (43%+44% of raw input variance = popularity/era/loudness), reconstructed
# to R2=1.000, while the signal-carrying text block (2.5% of raw variance,
# L2-normalized -> tiny per-dim scale) is under-preserved (R2=0.717). Per-column
# whitening FLIPS the captured budget to 73% text / 21% categorical (text
# R2=0.905). The learned InfoNCE projection learns the same reweighting from data.
# (2026-07-18c-nexttrack-representation-exploration.md, EVR preflight.)
def fig_nexttrack_evr_budget():
    blocks = ["text\n(artist/genre\nsignal)", "numeric+acoustic\n(popularity/era/\nloudness)", "categorical"]
    full = [2.5, 87.0, 10.5]   # equal-weight PCA-192 captured budget == raw variance share
    whit = [73.0, 6.0, 21.0]   # per-column standardize (whitening) captured budget
    x = np.arange(len(blocks))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    b1 = ax.bar(x - w / 2, full, w, color=MUTED, label="equal-weight PCA-192 (variance-ordered)")
    b2 = ax.bar(x + w / 2, whit, w, color=GOOD, label="per-column whitening")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2,
                    f"{r.get_height():.0f}%", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, fontsize=7.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of captured variance budget (%)")
    ax.set_title("The equal-weighted PCA-192 spends 87% of its budget on inert\nnumeric+acoustic; whitening FLIPS it to text (the signal block)")
    ax.annotate("text $R^2$: 0.717 $\\to$ 0.905", xy=(0, 73), xytext=(0.34, 52),
                ha="left", fontsize=7.0, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    ax.legend(loc="upper right", fontsize=7.2)
    save(fig, "nexttrack_evr_budget.pdf")


# Fig 23: the representation handicap is real for the RAW GRU but ABSORBED by the
# champion projection. Per-variant paired-D on exact hit@10 vs the same-readout
# PCA-192 baseline, under readout C (frozen gru-infonce-h256, raw representation)
# and readout A (champion projection blend-gru-markov-content-proj). Six of eight
# whitening / acoustic-drop / metric spaces LIFT the raw GRU (readout C, CI>0),
# but NONE lift the champion (readout A) -- the learned InfoNCE map already
# re-weights away the loud acoustic/numeric directions, so whitening the input
# first is redundant at the champion ceiling.
# (2026-07-18c-nexttrack-representation-exploration.md, Verdict 1.)
def fig_nexttrack_representation():
    # sorted by readout-C delta descending
    variants = ["V5 balanced", "V3 std-noaco", "V4 textcat", "V2 std",
                "V7 std-txtcatup", "V1 noaco", "V10 text-only", "V6 sonic-64"]
    dC = [0.0356, 0.0342, 0.0335, 0.0300, 0.0259, 0.0252, 0.0147, -0.0014]
    dA = [0.0042, 0.0049, -0.0168, -0.0035, -0.0063, -0.0028, -0.0119, -0.0182]
    y = np.arange(len(variants))[::-1]
    h = 0.38

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.barh(y + h / 2, dC, h, color=GOOD, label="readout C: frozen GRU (raw representation)")
    ax.barh(y - h / 2, dA, h, color=ACCENT, label="readout A: champion projection")
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(variants, fontsize=7.6)
    ax.set_xlabel("paired-$\\Delta$ exact hit@10 vs same-readout PCA-192 baseline")
    ax.xaxis.set_major_formatter(afmt)
    ax.set_xlim(-0.03, 0.052)
    ax.set_title("Whitening/acoustic-drop lifts the RAW GRU (readout C) but NOT the\nchampion projection (readout A): the projection already fixes it")
    ax.legend(loc="lower right", fontsize=7.2)
    save(fig, "nexttrack_representation.pdf")


# Fig 24: the MMR-lambda anti-eager trade-off on the next-track champion. The
# eval-time MMR re-rank (mmr_lambda, 1.0=off) is the STRONG holisticness lever:
# as lambda drops off->0.3 recall@10 falls (the only thing traded), while ild@10
# RISES (more diverse) and artist_adj@10 FALLS (less artist-eager). lambda=0.9 is
# essentially FREE (recall dD -0.0014, inside the +/-0.015 noise band = TIE) yet
# already lifts ild 0.406->0.436; lambda~0.7 buys a big ild gain (+0.114) and real
# de-eagering (-0.039) for ~2 recall points. music@10 RISES monotonically as you
# diversify (0.454->0.492) -- sonic closeness and list diversity are not in
# tension; exact recall is the only cost. All from the champion A1 checkpoint,
# offline (mmr_pool=200). (2026-07-22-session-holisticness-and-antieager-levers.md,
# Phase C.)
def fig_nexttrack_mmr_tradeoff():
    labels = ["off\n(1.0)", "0.9", "0.7", "0.5", "0.3"]
    x = np.arange(len(labels))
    recall = [0.2117, 0.2103, 0.1936, 0.1642, 0.1097]
    ild = [0.406, 0.436, 0.520, 0.653, 0.815]
    artist_adj = [0.610, 0.603, 0.571, 0.486, 0.293]
    music = [0.454, 0.462, 0.476, 0.487, 0.492]
    band = 0.015  # recall@10 practical half-width; a dD inside it is a TIE

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    # left axis: recall@10 (the cost), with the noise band around the off point
    ax.axhspan(recall[0] - band, recall[0] + band, color=BAD, alpha=0.08,
               label="recall $\\pm0.015$ tie band")
    ax.plot(x, recall, "o-", color=BAD, lw=2.0, zorder=5, label="recall@10 (cost)")
    for xi, v in zip(x, recall):
        ax.text(xi, v - 0.020, f"{v:.3f}", ha="center", fontsize=6.6, color=BAD)
    ax.set_ylim(0.05, 0.87)
    ax.set_ylabel("recall@10 (P(next-track hit))", color=BAD)
    ax.tick_params(axis="y", labelcolor=BAD)
    ax.yaxis.set_major_formatter(afmt)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("MMR $\\lambda$ (1.0 = off; lower = more diversified)")
    ax.grid(False)

    # right axis: holisticness levers (ild up = good, artist_adj down = good)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(x, ild, "s-", color=GOOD, lw=1.8, label="ild@10 $\\uparrow$ (diversity)")
    ax2.plot(x, artist_adj, "D--", color=ACCENT, lw=1.8,
             label="artist_adj@10 $\\downarrow$ (eagerness)")
    ax2.plot(x, music, "^:", color=MUTED, lw=1.6, label="music@10 $\\uparrow$")
    ax2.set_ylim(0.05, 0.87)
    ax2.set_ylabel("holisticness metrics", color="#374151")
    ax2.yaxis.set_major_formatter(afmt)
    ax2.grid(False)

    # annotate the two operating points
    ax.annotate("$\\lambda=0.9$ near-free\n(recall TIE, ild +0.030)", xy=(1, 0.2103),
                xytext=(1.15, 0.30), ha="left", fontsize=6.6, color=GOOD,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8))
    ax.annotate("$\\lambda\\approx0.7$ de-eager\n($-2$ recall pts, ild +0.114)",
                xy=(2, 0.1936), xytext=(1.7, 0.09), ha="left", fontsize=6.6,
                color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))

    # merged legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=6.5, ncol=2)
    ax.set_title("MMR $\\lambda$ is the strong anti-eager lever: diversifying trades exact\nrecall for holisticness, and $\\lambda=0.9$ is essentially free")
    save(fig, "nexttrack_mmr_tradeoff.pdf")


# Fig 25: the dual-tower view-pairing matrix. The complete upper triangle of the
# {latent, delta, cummean}^2 view matrix at fusion_layers=1, six cells, one run
# each. EVERY cell sits below the single-GRU h256 baseline (0.12299), and the
# ordering is monotone in "how much raw latent is in the pair": both-latent is
# best, both-delta is worst. The lower triangle was never run (the pairing is
# symmetric by construction) and is masked rather than mirrored. The read: a GRU
# over raw latents already integrates step-to-step movement (delta) and the
# running mean (cummean) internally, so pairing it with an explicit computation of
# either is redundant -- while two impoverished views ARE mutually complementary
# (delta/cummean beats both its parents CI>0) at a hopeless absolute level.
# (2026-07-25-nexttrack-dual-tower-fusion-scan.md, Results + Rule 4.)
def fig_nexttrack_dualtower_matrix():
    views = ["latent", "delta", "cummean"]
    nan = np.nan
    # rows = view_a, cols = view_b; upper triangle only (i <= j).
    recall = np.array([
        [0.10273, 0.09154, 0.09713],  # latent / {latent, delta, cummean}
        [nan, 0.05800, 0.08665],      # delta  / {delta, cummean}
        [nan, nan, 0.06639],          # cummean / cummean
    ])
    baseline = 0.12299  # C0 single GRU h256, the single-model bar
    best = (0, 0)  # latent/latent, the best dual cell -- still only a TIE with C0

    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#e5e7eb")  # never-run cells: grey, not a low-value colour

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    im = ax.pcolormesh(
        np.arange(len(views) + 1), np.arange(len(views) + 1),
        np.ma.masked_invalid(recall), cmap=cmap, shading="flat",
    )
    ax.set_xticks(np.arange(len(views)) + 0.5)
    ax.set_xticklabels(views)
    ax.set_yticks(np.arange(len(views)) + 0.5)
    ax.set_yticklabels(views)
    ax.set_xlabel("tower B causal view")
    ax.set_ylabel("tower A causal view")
    ax.invert_yaxis()
    ax.grid(False)
    for i in range(len(views)):
        for j in range(len(views)):
            v = recall[i, j]
            if np.isnan(v):
                ax.text(j + 0.5, i + 0.5, "not run\n(symmetric)", ha="center",
                        va="center", fontsize=6.4, color="#6b7280")
                continue
            is_best = (i, j) == best
            ax.text(
                j + 0.5, i + 0.5, f"{v:.4f}", ha="center", va="center", fontsize=8.4,
                color="white" if v < 0.088 else "black",
                fontweight="bold" if is_best else "normal",
            )
            if is_best:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor=BAD, lw=2.2))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("recall@10 (P(next-track hit))")
    cbar.formatter = afmt
    cbar.update_ticks()
    ax.set_title(
        "The raw latent view subsumes the others: every dual cell sits below the\n"
        f"single-GRU baseline ({baseline:.3f}), and the more 'latent' a pair, the better"
    )
    save(fig, "nexttrack_dualtower_matrix.pdf")


# Fig 26: holisticness and recall are ANTI-CORRELATED across the dual-tower
# batch, which is why the crown rule needs a hard recall floor. Every arm that
# lifts 3-factor holisticness@10 does so by retreating from exact prediction: the
# H-ranked order runs almost exactly backwards along recall. Five arms clear the
# paired-DeltaH CI>0 leg (green) and ALL FIVE fail the pre-registered +/-0.015
# recall floor, so Rule 6 never fired and the crown was untouched -- the floor
# caught exactly the `mood-session` degenerate mode it was written for (shown as
# the open marker: H 0.2332 at recall 0.0238 = 34 of 1,431). NB all H values here
# are the 3-FACTOR generation; see fig_nexttrack_power for the generation caveat.
# (2026-07-25-nexttrack-dual-tower-fusion-scan.md, Rule 6 + Finding 5.)
def fig_nexttrack_holisticness_antiwin():
    # (label, recall@10, 3-factor H, DeltaH CI>0?)
    # (label, recall, H, DeltaH CI>0, label dx, dy in points, ha)
    arms = [
        ("C0 h256", 0.12299, 0.14172, None, 11, 7, "left"),
        ("C1 h425", 0.12089, 0.13850, False, 11, -6, "left"),
        ("D-LL-f0", 0.11880, 0.13774, False, -8, 7, "right"),
        ("D-LC-f0", 0.11740, 0.13648, False, -6, -14, "right"),
        ("D-LL", 0.10273, 0.14635, True, 0, 8, "center"),
        ("D-LC", 0.09713, 0.15000, True, 0, 8, "center"),
        ("D-LD", 0.09154, 0.14673, True, 0, 8, "center"),
        ("D-DC", 0.08665, 0.16340, True, 0, 8, "center"),
        ("D-CC", 0.06639, 0.19715, True, 0, 8, "center"),
        ("D-DD", 0.05800, 0.13335, False, 0, 8, "center"),
    ]
    floor = 0.12299 - 0.015  # Rule 6 anti-degenerate recall floor

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axvspan(0.015, floor, color=BAD, alpha=0.07)
    ax.axvline(floor, color=BAD, ls="--", lw=1.4,
               label=f"Rule 6 recall floor ({floor:.3f})")
    ax.axvline(0.12299, color=ACCENT, ls=":", lw=1.2, label="C0 baseline recall")

    for label, r, h, win, dx, dy, ha in arms:
        if win is None:
            c, m, s = ACCENT, "*", 190
        elif win:
            c, m, s = GOOD, "o", 62
        else:
            c, m, s = MUTED, "o", 62
        ax.scatter([r], [h], color=c, marker=m, s=s, zorder=5,
                   edgecolor="white", linewidth=0.6)
        ax.annotate(label, xy=(r, h), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=6.6, color="#374151")
    # the degenerate incumbent the floor exists to block
    ax.scatter([0.0238], [0.2332], facecolor="none", edgecolor=BAD, marker="s",
               s=80, lw=1.6, zorder=5)
    ax.annotate("mood-session\n(degenerate: 34 of 1,431)", xy=(0.0238, 0.2332),
                xytext=(0.040, 0.221), fontsize=6.6, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))

    ax.set_xlabel("recall@10 (P(next-track hit))")
    ax.set_ylabel("holisticness@10 (3-factor generation)")
    ax.xaxis.set_major_formatter(afmt)
    ax.yaxis.set_major_formatter(afmt)
    ax.set_xlim(0.015, 0.145)
    ax.set_ylim(0.128, 0.246)
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color=GOOD, marker="o", ls="", label="$\\Delta$H CI$>$0 (clears the H leg)"),
        plt.Line2D([], [], color=MUTED, marker="o", ls="", label="$\\Delta$H straddles / CI$<$0"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.6)
    ax.set_title("Unconstrained holisticness is maximized by predicting WORSE: all five\n"
                 "$\\Delta$H winners fall below the recall floor, so no crown changed hands")
    save(fig, "nexttrack_holisticness_antiwin.pdf")


# Fig 27: the pre-encoder verdict as a forest plot. Paired per-session bootstrap
# Delta recall@10 against the reproduced C0 baseline (2,000 resamples, rng 1337,
# n=1,431), one row per arm. NOT ONE of the six pre-MLP arms reaches CI>0: five
# lose outright and one ties. The decisive row is S1 (MLP256->GRU256, the only arm
# whose result is unambiguously attributable to the pre-encoder), which does not
# merely tie -- it LOSES, 156 hits against 176. Because a LINEAR pre-encoder at
# pre_hidden >= latent_dim is provably expressivity-neutral (W_i(Vx) = (W_iV)x),
# the ReLU is the only expressivity change, so the rectifier itself is the cause.
# (2026-07-26-nexttrack-pre-encoder-scan.md, Results + Rules 1/7.)
def fig_nexttrack_preencoder_forest():
    # (label, delta, lo, hi, kind) sorted by delta descending
    rows = [
        ("C2 h454 (capacity ctrl)", +0.00210, -0.00699, +0.01118, "ctrl"),
        ("T1-f0 dual A0/B256 f0", +0.00070, -0.01188, +0.01328, "tie"),
        ("C1 h297 (capacity ctrl)", -0.00978, -0.01747, -0.00280, "ctrl"),
        ("S1 MLP256$\\to$GRU256", -0.01398, -0.02657, -0.00140, "key"),
        ("C3 bare dual f1", -0.02027, -0.03494, -0.00697, "ctrl"),
        ("T2 dual A0/B128 f1", -0.02306, -0.03704, -0.00908, "pre"),
        ("T3 dual A0/B384 f1", -0.02306, -0.03704, -0.00908, "pre"),
        ("T4 dual A256/B256 f1", -0.02516, -0.03913, -0.01118, "pre"),
        ("T1 dual A0/B256 f1", -0.02586, -0.03985, -0.01258, "pre"),
    ]
    colour = {"ctrl": ACCENT, "tie": MUTED, "pre": BAD, "key": BAD}
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axvspan(-0.015, 0.015, color=MUTED, alpha=0.14,
               label="$\\pm0.015$ practical tie band")
    ax.axvline(0, color="black", lw=1.1)
    for yi, (label, d, lo, hi, kind) in zip(y, rows):
        c = colour[kind]
        lw = 2.6 if kind == "key" else 1.5
        ax.plot([lo, hi], [yi, yi], color=c, lw=lw, solid_capstyle="butt")
        ax.plot([d], [yi], "o", color=c, ms=7 if kind == "key" else 5, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    ax.set_xlabel("paired $\\Delta$ recall@10 vs C0 (single GRU h256) [95% CI]")
    ax.xaxis.set_major_formatter(afmt)
    ax.set_xlim(-0.050, 0.028)
    ax.set_ylim(-0.85, len(rows) - 0.35)
    ax.grid(axis="y", visible=False)
    ax.text(0.0018, y[3], "the decisive ablation:\nLOSES, 156 vs 176 hits",
            ha="left", va="center", fontsize=6.8, color=BAD)
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color=BAD, marker="o", lw=1.5, label="pre-encoder arm"),
        plt.Line2D([], [], color=ACCENT, marker="o", lw=1.5, label="control (no pre-encoder)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.6)
    ax.set_title("No pre-encoder arm wins, and the single-tower ablation loses outright:\n"
                 "the ReLU is the only expressivity change, so the rectifier is the cause")
    save(fig, "nexttrack_preencoder_forest.pdf")


# Fig 28: the rectifier mechanism MEASURED, not inferred. The B4 follow-up read
# next-item genre decodability off the taps of the pre-encoded models, plus one
# counterfactual tap the SAE engine cannot express -- the model's OWN affine map
# with the ReLU deleted (tools/rectifier_control.py). Input side: the affine map
# is NEUTRAL (~0.8 sigma, exactly as W_i(Vx) = (W_iV)x predicts) and the ReLU is
# the WHOLE loss (~5.9 sigma). frac_exact_zero on the rectified tap is 0.5327,
# matching pre_linear's frac_negative_coords 0.5327 to four digits: the rectifier
# deletes precisely the negative half, leaving ~120 live units for 192 signed
# directions. The damage SURVIVES the recurrence (tower_b fed rectified vs its
# identical twin tower_a fed raw, ~3.9 sigma), and tower_a reproduces the
# standalone plain GRU -- which is what "the bare tower RESCUES" requires.
# 11,184 probe test rows, 25 classes, paired SE <= 0.0065.
# (2026-07-26-nexttrack-pre-encoder-scan.md, Phase B4 / PROJECT-FACTS.md.)
def fig_nexttrack_rectifier():
    labels = ["raw\nlatent", "pre_linear\n(ReLU deleted)", "pre_relu\n(as trained)",
              "tower_a\n(fed raw)", "tower_b\n(fed rectified)", "plain GRU\n(standalone)"]
    acc = [0.4033, 0.4084, 0.3700, 0.4183, 0.3929, 0.4235]
    auc = [0.7609, 0.7667, 0.7508, 0.8004, 0.7845, 0.8006]
    cols = [MUTED, GOOD, BAD, GOOD, BAD, ACCENT]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axvline(2.5, color="#9ca3af", lw=1.0, ls="-")
    ax.bar(x, acc, 0.62, color=cols, edgecolor="white", linewidth=0.8)
    for xi, (a, u) in enumerate(zip(acc, auc)):
        ax.text(xi, a + 0.0030, f"{a:.4f}", ha="center", fontsize=7.2)
        ax.text(xi, 0.3555, f"AUC\n{u:.4f}", ha="center", va="center",
                fontsize=6.5, color="white")
    ax.set_ylim(0.345, 0.4530)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.9)
    ax.set_ylabel("next-item genre decodability (accuracy)")
    ax.set_xlabel("representation tap")
    ax.text(1.0, 0.4505, "input side (before the recurrence)", ha="center",
            fontsize=7.0, color="#374151", style="italic")
    ax.text(4.0, 0.4505, "after the recurrence", ha="center",
            fontsize=7.0, color="#374151", style="italic")
    # the two decisive contrasts
    ax.annotate("", xy=(0, 0.4140), xytext=(1, 0.4140),
                arrowprops=dict(arrowstyle="<->", color=GOOD, lw=1.1))
    ax.text(0.5, 0.4152, "affine map NEUTRAL\n$\\sim$0.8$\\sigma$", ha="center",
            va="bottom", fontsize=6.6, color=GOOD)
    ax.annotate("", xy=(1, 0.4290), xytext=(2, 0.4290),
                arrowprops=dict(arrowstyle="<->", color=BAD, lw=1.4))
    ax.text(1.5, 0.4302, "the ReLU is the WHOLE loss\n$\\sim$5.9$\\sigma$", ha="center",
            va="bottom", fontsize=6.9, color=BAD, fontweight="bold")
    ax.annotate("", xy=(3, 0.4252), xytext=(4, 0.4252),
                arrowprops=dict(arrowstyle="<->", color=BAD, lw=1.1))
    ax.text(3.5, 0.4264, "damage survives\nthe recurrence $\\sim$3.9$\\sigma$",
            ha="center", va="bottom", fontsize=6.6, color=BAD)
    ax.grid(axis="x", visible=False)
    ax.set_title("The rectifier mechanism, measured: the affine map is free and the ReLU\n"
                 "is the entire cost -- half-wave rectification of a zero-centred latent")
    save(fig, "nexttrack_rectifier.pdf")


# Fig 29: the measured statistical POWER of this test split, which reframes the
# architecture nulls. The paired-bootstrap CI half-width scales as 1/sqrt(n): at
# the split's n=1,431 it is ~0.0124 on recall@10 (the mean over the nine
# comparisons of the pre-encoder batch; 0.0128 over the nine of the dual-tower
# batch). An architecture family's ENTIRE spread is ~0.028 recall -- about 2.3
# resolution widths, so of eight adjacent arms only ~2 tiers are separable, and a
# 0.005 difference would need ~8,800 test sessions. The honest reading of the five
# architecture nulls: LARGE HARMS were detected reliably (five arms lose CI<0),
# while MODERATE WINS were never detectable at all. That is a different claim from
# "these architectures do not work".
# (2026-07-25-nexttrack-dual-tower-fusion-scan.md +
#  2026-07-26-nexttrack-pre-encoder-scan.md, paired-Delta tables.)
def fig_nexttrack_power():
    n0, hw0 = 1431, 0.0124  # measured anchor: this split, mean recall@10 half-width
    n = np.logspace(np.log10(500), np.log10(30000), 400)
    hw = hw0 * np.sqrt(n0 / n)
    spread = 0.028   # whole spread of an architecture family on recall@10
    target = 0.005   # the difference worth resolving
    n_needed = n0 * (hw0 / target) ** 2  # ~8,800

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(n, hw, "-", color=ACCENT, lw=2.0,
            label="paired-bootstrap CI half-width $\\propto 1/\\sqrt{n}$")
    ax.axhline(spread, color=MUTED, ls="--", lw=1.3,
               label=f"whole spread of an architecture family ({spread:.3f})")
    ax.axhline(target, color=GOOD, ls=":", lw=1.6,
               label=f"a difference worth resolving ({target:.3f})")

    ax.plot([n0], [hw0], "o", color=BAD, ms=9, zorder=6)
    ax.annotate(f"this split\nn={n0:,}, half-width {hw0:.4f}", xy=(n0, hw0),
                xytext=(1750, 0.0200), fontsize=7.0, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9))
    ax.plot([n_needed], [target], "o", color=GOOD, ms=8, zorder=6)
    ax.annotate(f"n $\\approx$ {round(n_needed, -2):,.0f} sessions needed",
                xy=(n_needed, target), xytext=(5200, 0.0098), fontsize=7.0, color=GOOD,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.9))
    ax.vlines([n0, n_needed], 0, [hw0, target], color=MUTED, lw=0.8, ls=":")

    # the resolvable-tier band: spread / half-width at n0
    ax.text(
        560, 0.0387,
        f"spread / resolution = {spread / hw0:.1f}\n"
        "$\\Rightarrow$ of 8 adjacent arms only $\\sim$2 tiers are\n"
        "separable on recall@10 ($\\sim$1 on holisticness)",
        fontsize=6.9, color="#374151", ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f9fafb", ec="#d1d5db", lw=0.7),
    )
    ax.set_xscale("log")
    ax.set_xlim(500, 30000)
    ax.set_ylim(0, 0.039)
    ax.set_xlabel("test sessions $n$")
    ax.set_ylabel("resolvable $\\Delta$ recall@10 (CI half-width)")
    ax.yaxis.set_major_formatter(afmt)
    ax.legend(loc="center right", fontsize=6.8)
    ax.set_title("The architecture nulls are power-limited: at n=1,431 large harms are\n"
                 "reliably detected but moderate wins were never detectable")
    save(fig, "nexttrack_power.pdf")


# Fig 30: THE CORE ARGUMENT of the walk-surface campaign. The user's OWN
# consecutive-transition cosines in the PCA-192 item space -- 73,632 pairs over the
# 5,723 train sessions of seq-20260715-131139 -- are a broad distribution: median
# 0.261, p10-p90 [-0.192, 0.830], and 27.0% of real steps are NEGATIVE. Real
# listening moves. The engine the showcase actually shipped (single GRU + anchor
# 0.4, gru.onnx) walks with a median step of 0.661, 2.5x tighter than the user
# does: "lots of artists but flat" was many small safe moves, and no one-shot
# metric could see it because holisticness@10 is computed on a top-10 and its
# mood_coh anchors to the GROWING prefix centroid. The blend champion sits at
# 0.317, right on the real median -- but cannot hold a vibe (0.187).
# Bin counts (40 bins of width 0.05 over [-1,1]) transcribed from the
# calibration block of tools/walk_eval.py run on the canonical artifact.
# (PROJECT-FACTS.md roll-up 2026-07-31b; tools/walk_eval.py.)
def fig_nexttrack_walk_transitions():
    edges = np.linspace(-1.0, 1.0, 41)
    counts = np.array([
        0, 0, 0, 0, 0, 0, 1, 0, 13, 39, 120, 337, 679, 1248, 2020, 2461, 2994,
        3074, 3482, 3439, 3491, 3428, 3071, 3084, 3145, 3008, 3013, 2973, 2829,
        2843, 2712, 2509, 2337, 2425, 2343, 2094, 1660, 1357, 1392, 4011,
    ])
    n_real = int(counts.sum())          # 73,632
    real_med = 0.261
    p10, p90 = -0.192, 0.830
    frac_neg = 0.270
    gru_med = 0.661                     # shipped GRU + anchor 0.4, 40 held-out sessions
    champ_med = 0.317                   # blend champion, same 40 sessions
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.axvspan(p10, p90, color=ACCENT, alpha=0.09, lw=0,
               label=f"real p10--p90 [{p10:.3f}, {p90:.3f}]")
    ax.bar(centers, counts, width=0.048, color=ACCENT, alpha=0.55,
           edgecolor="white", linewidth=0.3, label=f"the user's real steps (n={n_real:,})")
    ax.axvline(0.0, color=MUTED, lw=0.9, ls="-")
    ax.axvline(real_med, color=ACCENT, lw=2.0, ls="-",
               label=f"real median step {real_med:.3f}")
    ax.axvline(champ_med, color=GOOD, lw=1.6, ls="--",
               label=f"blend champion walk {champ_med:.3f}")
    ax.axvline(gru_med, color=BAD, lw=2.2, ls="-",
               label=f"SHIPPED GRU walk {gru_med:.3f}")

    ax.annotate(f"{frac_neg:.0%} of real steps\nare NEGATIVE -- the user\nchanges direction",
                xy=(-0.22, 2100), xytext=(-0.73, 1150), fontsize=7.0, color="#374151",
                arrowprops=dict(arrowstyle="->", color="#6b7280", lw=0.9))
    ax.annotate("shipped engine:\n2.5$\\times$ TIGHTER\nthan the user",
                xy=(gru_med, 2450), xytext=(0.705, 2750), fontsize=7.2, color=BAD,
                ha="left", arrowprops=dict(arrowstyle="->", color=BAD, lw=1.1))
    ax.set_xlim(-0.75, 1.0)
    ax.set_ylim(0, 4400)
    ax.set_xlabel("cosine between consecutive tracks (PCA-192 item space)")
    ax.set_ylabel("real consecutive pairs per 0.05 bin")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.set_title("Calibrated against the user's own listening: real sessions MOVE, and the\n"
                 "engine the showcase shipped was walking far too tight to be one of them")
    save(fig, "nexttrack_walk_transitions.pdf")


# Fig 31: the anchor x stride frontier. Two orthogonal, DETERMINISTIC retrieval-time
# controls -- anchor (pull toward the seed-core centroid, holds the geist) and stride
# (subtract s*cos(candidate, previous track), enlarges each step) -- sweep 16 GRU cells
# and 8 champion cells on 20 held-out sessions x 20 steps, cap 1. Sampling was
# deliberately NOT used: it would loosen the stride too but break the permalink's
# same-recipe-same-journey guarantee. The shaded box is the pre-registered target
# window: median step inside the real band's middle (0.20-0.38) AND vibe >= 0.50.
# Neither shipped configuration is in it (GRU a0.4 s0.0 holds the vibe but strides
# 0.709; champion a0.0 s0.0 strides 0.318 but has vibe 0.185); the conjunction is
# reachable only once BOTH controls are on. Lines join a fixed anchor as stride grows.
# (tools/walk_frontier.py, /tmp/walk_frontier.json; PROJECT-FACTS.md 2026-07-31b.)
def fig_nexttrack_walk_frontier():
    # (anchor, stride, med_step, vibe)
    gru = [
        (0.0, 0.0, 0.607, 0.286), (0.0, 0.3, -0.018, 0.200),
        (0.0, 0.6, -0.374, 0.123), (0.0, 1.0, -0.557, 0.084),
        (0.2, 0.0, 0.663, 0.563), (0.2, 0.3, 0.094, 0.426),
        (0.2, 0.6, -0.251, 0.330), (0.2, 1.0, -0.522, 0.185),
        (0.4, 0.0, 0.709, 0.639), (0.4, 0.3, 0.297, 0.588),
        (0.4, 0.6, -0.127, 0.437), (0.4, 1.0, -0.405, 0.302),
        (0.8, 0.0, 0.713, 0.678), (0.8, 0.3, 0.506, 0.672),
        (0.8, 0.6, 0.176, 0.605), (0.8, 1.0, -0.188, 0.460),
    ]
    champ = [
        (0.0, 0.0, 0.318, 0.185), (0.0, 0.5, -0.032, 0.104),
        (1.0, 0.0, 0.577, 0.551), (1.0, 0.5, 0.282, 0.503),
        (2.0, 0.0, 0.596, 0.625), (2.0, 0.5, 0.472, 0.615),
        (4.0, 0.0, 0.645, 0.669), (4.0, 0.5, 0.544, 0.664),
    ]
    tgt_x, tgt_vibe = (0.20, 0.38), 0.50
    real_med = 0.261

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.add_patch(plt.Rectangle((tgt_x[0], tgt_vibe), tgt_x[1] - tgt_x[0], 1.0 - tgt_vibe,
                               fc=GOOD, ec=GOOD, alpha=0.13, lw=1.0, zorder=0))
    ax.text(0.29, 0.955, "target window\nreal stride AND held vibe", ha="center",
            va="top", fontsize=6.8, color=GOOD)
    ax.axvline(real_med, color=ACCENT, lw=1.4, ls=":",
               label=f"the user's real median step {real_med:.3f}")
    ax.axhline(tgt_vibe, color=MUTED, lw=0.8, ls=":")

    for pts, col, mk, lbl in ((gru, ACCENT, "o", "single GRU (gru.onnx)"),
                              (champ, BAD, "s", "blend champion")):
        for a in sorted({p[0] for p in pts}):
            leg = sorted([p for p in pts if p[0] == a], key=lambda p: p[1])
            ax.plot([p[2] for p in leg], [p[3] for p in leg], "-", color=col,
                    lw=0.8, alpha=0.45, zorder=1)
        ax.plot([p[2] for p in pts], [p[3] for p in pts], mk, color=col, ms=5.0,
                mec="white", mew=0.6, label=lbl, zorder=3)

    def tag(p, text, dx, dy, col, weight="normal", fs=6.6):
        ax.annotate(text, xy=(p[2], p[3]), xytext=(p[2] + dx, p[3] + dy), fontsize=fs,
                    color=col, fontweight=weight,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.8))

    tag(gru[8], "SHIPPED GRU\na0.4 s0.0", -0.21, 0.22, BAD, "bold", 7.0)
    tag(gru[9], "GRU a0.4 s0.3\nIN THE WINDOW", -0.42, 0.18, GOOD, "bold")
    tag(champ[0], "champion as-is\na0.0 s0.0", -0.62, -0.14, BAD)
    tag(champ[3], "champion a1.0 s0.5\nIN THE WINDOW", 0.06, -0.23, GOOD, "bold")
    ax.annotate("a0.8 s0.6 just misses\n(step 0.176)", xy=gru[14][2:], fontsize=6.2,
                color=MUTED, xytext=(gru[14][2] - 0.44, gru[14][3] + 0.09),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))

    ax.set_xlim(-0.65, 0.86)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("median step cosine of the generated journey  (stride grows $\\rightarrow$ leftward)")
    ax.set_ylabel("vibe: mean cosine to the seed-core centroid")
    ax.legend(loc="lower right", fontsize=6.9)
    ax.set_title("Anchor and stride are orthogonal, and only their CONJUNCTION reaches the\n"
                 "target window -- neither shipped configuration was inside it")
    save(fig, "nexttrack_walk_frontier.pdf")


# Fig 32: the matched-sample head-to-head. Every arm walked the SAME 40 held-out
# sessions from the same 5-track seeds, so the differences are PAIRED (2000
# bootstrap resamples, rng 1337) against the shipped GRU (anchor 0.4, stride 0).
# One panel per metric, each on its OWN scale -- a dual axis would imply a
# comparability these five quantities do not have. Reading: pushing the single GRU
# to a real stride COSTS vibe and the champion costs much more, while both
# dual-tower arms buy the stride at no vibe cost and add genres; `ground` straddles
# for EVERY arm, so no arm is established as better at predicting the real
# continuation -- only at journey SHAPE.
# (tools/walk_headtohead.py, /tmp/walk_headtohead.json; PROJECT-FACTS.md 2026-07-31b.)
def fig_nexttrack_walk_headtohead():
    arms = ["GRU tuned\na0.4 s0.3", "dual l/l f0\na0.8 s0.4",
            "dual l/cummean f0\na0.8 s0.5", "champion\na1.0 s0.5"]
    # metric -> (better-direction sign, [(delta, lo, hi) per arm])
    data = {
        "$\\Delta$vibe": (+1, [(-0.0409, -0.0602, -0.0221), (+0.0297, +0.0146, +0.0468),
                               (+0.0013, -0.0151, +0.0180), (-0.1327, -0.1629, -0.1041)]),
        "$\\Delta$stride err": (-1, [(-0.2695, -0.3058, -0.2358), (-0.2423, -0.2790, -0.2053),
                                       (-0.2545, -0.2962, -0.2127), (-0.2638, -0.3120, -0.2155)]),
        "$\\Delta$drift": (+1, [(+0.0089, -0.0255, +0.0420), (+0.0259, -0.0011, +0.0521),
                                (+0.0351, +0.0071, +0.0627), (-0.0422, -0.0983, +0.0093)]),
        "$\\Delta$genres": (+1, [(+0.525, -0.475, +1.500), (+1.050, +0.250, +1.876),
                                 (+1.500, +0.500, +2.500), (-0.325, -1.500, +0.876)]),
        "$\\Delta$ground": (+1, [(+0.0116, -0.0143, +0.0408), (+0.0060, -0.0172, +0.0314),
                                 (+0.0059, -0.0163, +0.0285), (-0.0115, -0.0467, +0.0236)]),
    }
    y = np.arange(len(arms))[::-1]

    fig, axes = plt.subplots(1, 5, figsize=(7.4, 3.1))
    for ax, (name, (sign, rows)) in zip(axes, data.items()):
        ax.axvline(0.0, color="#374151", lw=1.0)
        for yi, (d, lo, hi) in zip(y, rows):
            better = (lo > 0) if sign > 0 else (hi < 0)
            worse = (hi < 0) if sign > 0 else (lo > 0)
            col = GOOD if better else (BAD if worse else MUTED)
            ax.errorbar([d], [yi], xerr=[[d - lo], [hi - d]], fmt="o", ms=4.5,
                        color=col, ecolor=col, elinewidth=1.4, capsize=2.4)
        ax.set_yticks(y)
        ax.set_yticklabels(arms if ax is axes[0] else [], fontsize=6.2)
        ax.set_ylim(-0.7, len(arms) - 0.3)
        ax.set_title(name + ("  ($\\downarrow$)" if sign < 0 else "  ($\\uparrow$)"),
                     fontsize=7.6)
        ax.tick_params(axis="x", labelsize=6.2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(axis="y", visible=False)
    axes[0].set_ylabel("vs the SHIPPED GRU", fontsize=7.0)
    fig.suptitle("Matched-sample walk head-to-head: 40 identical held-out sessions, paired bootstrap.\n"
                 "Green = CI on the better side, red = CI on the worse side, grey = straddles zero",
                 fontsize=8.4)
    save(fig, "nexttrack_walk_headtohead.pdf")


# Fig 33: the tower-bank campaign in one glance -- three panels, three findings.
# LEFT: the N-axis at MATCHED capacity (~3.00x the h256 baseline). C1 is a single
# h512 GRU (N=1), B2/B3/B4 are banks of 2/3/4 parallel bare-GRU towers on the raw
# `latent` view. The line is flat, and B4 (four h211 towers) returns the IDENTICAL
# 186 hits as C1 (one h512 GRU) -- paired Delta 0.000000. Topology and width are
# interchangeable at this budget. Error bars are the paired bootstrap CI against
# C0 (2,000 resamples, rng 1337) mapped from recall@10 onto the hit scale; the grey
# band is the split's measured resolution (+-0.0124 recall@10 = +-17.7 hits, Fig 29),
# which every arm sits inside. MIDDLE: the two arms that earned a walk TRADE at the
# primary cell both FLIP at the deployed cell -- the drift gain that bought the
# TRADE evaporates while vibe, the non-negotiable line, goes CI<0. The
# secondary-cell clause is what closed the axis. RIGHT: the campaign's one positive
# finding, and it is not about topology -- at BYTE-IDENTICAL parameters (789,696
# both, model.pt 3,162,277 bytes both) dropout 0.1 beats dropout 0.0 by +0.0049
# recall@10, CI>0. Because seq_dualgru constructs zero dropout modules at
# fusion_layers=0, every f0 arm on record -- including the deployed engine --
# trained under-regularized.
# (2026-08-02-tower-bank-parallel-towers.md, S1 + S2 + secondary-cell tables.)
def fig_nexttrack_tower_bank():
    n_test = 1431
    hits_c0 = 176  # C0 = h256 anchor, bit-identical 10th reproduction

    # -- LEFT: N at matched capacity. (label, N, hits, dlo, dhi) with the paired
    # CI against C0 in recall@10 units, converted to hits below.
    narms = [
        ("C1\nh512", 1, 186, -0.00210, +0.01607),
        ("B2\nh334", 2, 170, -0.01328, +0.00349),
        ("B3\nh256", 3, 172, -0.01048, +0.00559),
        ("B4\nh211", 4, 186, -0.00210, +0.01677),
    ]
    resolution = 0.0124 * n_test  # the split's measured half-width, in hits

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.2, 3.6))

    ax1.axhspan(hits_c0 - resolution, hits_c0 + resolution, color=MUTED, alpha=0.18,
                zorder=0, label="measured resolution ($\\pm$0.0124 R@10)")
    ax1.axhline(hits_c0, color=MUTED, ls="--", lw=1.3,
                label=f"C0 h256 anchor ({hits_c0}/{n_test})")
    xs = [a[1] for a in narms]
    ys = [a[2] for a in narms]
    lo = [a[2] - (hits_c0 + a[3] * n_test) for a in narms]
    hi = [(hits_c0 + a[4] * n_test) - a[2] for a in narms]
    ax1.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=ACCENT, ms=6.0,
                 ecolor=ACCENT, elinewidth=1.3, capsize=3.0, zorder=4,
                 label="$\\sim$3.00$\\times$ params")
    ax1.annotate("B4 $=$ C1: 186 hits both,\n$\\Delta = 0.000000$",
                 xy=(2.5, 191), fontsize=6.9, color=BAD, ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec=BAD, lw=0.7))
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels([f"{a[1]}\n{a[0].replace(chr(10), ' ')}" for a in narms],
                        fontsize=7.0)
    ax1.set_xlabel("$N$ parallel towers (matched capacity)")
    ax1.set_ylabel(f"hits of {n_test} (recall@10)")
    ax1.set_ylim(150, 205)
    ax1.set_title("Topology is flat: four towers\nbuy exactly what width buys", fontsize=8.6)
    ax1.legend(loc="lower left", fontsize=6.0)

    # -- MIDDLE: both walk TRADEs are cell-local. (metric, marker, per-cell
    # (delta, lo, hi) or (delta, None, None) where the report gives a point
    # estimate only because the CI straddles.)
    cells = [0, 1]
    flip = {
        "B2 $\\Delta$vibe": ("o", "-", [(+0.0017, None, None), (-0.0064, -0.0117, -0.0011)]),
        "V3 $\\Delta$vibe": ("s", "-", [(-0.0094, None, None), (-0.0112, -0.0167, -0.0064)]),
        "B2 $\\Delta$drift": ("o", ":", [(+0.0339, +0.0066, +0.0629), (-0.0052, None, None)]),
        "V3 $\\Delta$drift": ("s", ":", [(+0.0286, +0.0009, +0.0592), (+0.0043, None, None)]),
    }
    ax2.axhline(0.0, color="#374151", lw=1.0)
    for name, (marker, ls, rows) in flip.items():
        ax2.plot(cells, [r[0] for r in rows], ls, color=MUTED, lw=1.0, zorder=2)
        for c, (d, dlo, dhi) in zip(cells, rows):
            if dlo is None:
                col, err = MUTED, None
            else:
                col = GOOD if dlo > 0 else BAD
                err = [[d - dlo], [dhi - d]]
            ax2.errorbar([c], [d], yerr=err, fmt=marker, ms=5.5, color=col,
                         ecolor=col, elinewidth=1.3, capsize=2.6, zorder=4)
        nudge = {"B2 $\\Delta$drift": +7.0, "B2 $\\Delta$vibe": -7.0,
                 "V3 $\\Delta$vibe": -5.0}.get(name, 0.0)
        ax2.annotate(name, xy=(cells[-1], rows[-1][0]), xytext=(5, nudge),
                     textcoords="offset points", fontsize=6.4, va="center",
                     color="#374151")
    ax2.set_xticks(cells)
    ax2.set_xticklabels(["a0.4 / s0.3\n(primary)", "a0.8 / s0.5\n(deployed)"], fontsize=6.8)
    ax2.set_xlim(-0.25, 1.95)
    ax2.set_ylabel("paired $\\Delta$ vs in-batch C0")
    ax2.set_title("Both walk TRADEs are cell-local:\nthe drift gain goes, vibe goes CI$<$0", fontsize=8.6)
    ax2.grid(axis="x", visible=False)

    # -- RIGHT: the dropout pair at byte-identical parameters.
    drop = [("R2\ndropout 0.0", 170, MUTED), ("D2\ndropout 0.1", 177, GOOD)]
    x = np.arange(len(drop))
    ax3.bar(x, [d[1] for d in drop], color=[d[2] for d in drop], width=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels([d[0] for d in drop], fontsize=7.0)
    for xi, (_, h, _) in zip(x, drop):
        ax3.text(xi, h + 1.2, str(h), ha="center", fontsize=7.4)
    ax3.set_ylim(150, 190)
    ax3.set_ylabel(f"hits of {n_test} (recall@10)")
    ax3.annotate("$\\Delta = +0.004892$\n[$+$0.000699, $+$0.009783]\nCI$>$0",
                 xy=(0.5, 182), fontsize=6.9, color=GOOD, ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec=GOOD, lw=0.7))
    ax3.set_title("Regularization, not topology:\nsame 789,696 params", fontsize=8.6)
    ax3.grid(axis="x", visible=False)

    fig.suptitle("The parallel-tower-bank axis, closed on both surfaces: no arm wins, "
                 "and the only CI$>$0 result in the batch is dropout", fontsize=9.0)
    save(fig, "nexttrack_tower_bank.pdf")


# Fig 34: the validation-loss dissociation, 4th confirmation and its sharpest
# instance yet. Ten arms of the tower-bank batch, best validation InfoNCE against
# retrieval hits: Spearman = -0.2954, p = 0.407 -- no usable predictive power, and
# the sign is the WRONG way round if anything. B4 (6.5657) and B2 (6.5660) differ by
# 0.0003 in val loss and by 16 HITS; V1 holds the 4th-best val loss in the batch and
# the worst hit count; C0 holds the WORST val loss and beats six of the eight bank
# arms. R2 (dropout 0.0) reaches a BETTER val loss than D2 (dropout 0.1) while
# retrieving 7 fewer hits -- the dissociation and the dropout finding are the same
# coin. This is why the campaign's decision rule never read validation loss.
# (2026-08-02-tower-bank-parallel-towers.md, finding 6.)
def fig_nexttrack_bank_valloss():
    # (arm, best val InfoNCE, hits, early-stop epoch)
    arms = [
        ("C0", 6.5868, 176, 14), ("C1", 6.5653, 186, 14),
        ("R2", 6.5671, 170, 14), ("D2", 6.5695, 177, 14),
        ("B2", 6.5660, 170, 14), ("B3", 6.5694, 172, 14),
        ("B4", 6.5657, 186, 14), ("V1", 6.5664, 162, 21),
        ("V2", 6.5728, 166, 24), ("V3", 6.5813, 170, 24),
    ]
    off = {"C1": (-15, 3), "B4": (6, 2), "B2": (-6, -12), "R2": (5, -4)}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, vl, h, _ in arms:
        col = ACCENT if name.startswith(("B", "V")) else BAD
        ax.plot([vl], [h], "o", ms=6.5, color=col, zorder=4)
        ax.annotate(name, xy=(vl, h), xytext=off.get(name, (5, 3)),
                    textcoords="offset points", fontsize=7.0, color="#374151")
    ax.axhline(176, color=MUTED, ls="--", lw=1.2, label="C0 h256 anchor (176 hits)")

    # the sharpest pair: B4 and B2, 0.0003 apart in val loss and 16 hits apart
    ax.annotate("", xy=(6.5657, 186), xytext=(6.5660, 170),
                arrowprops=dict(arrowstyle="<->", color=BAD, lw=1.1))
    ax.annotate("B4 vs B2: $\\Delta$val loss 0.0003,\n$\\Delta$ 16 hits",
                xy=(6.5660, 177.5), xytext=(6.5668, 190.5), fontsize=6.9, color=BAD,
                ha="left", arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
    ax.text(0.97, 0.06,
            "Spearman(val loss, hits) $= -0.2954$,  $p = 0.407$",
            transform=ax.transAxes, fontsize=7.4, ha="right", color="#374151",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f9fafb", ec="#d1d5db", lw=0.7))
    ax.set_xlabel("best validation InfoNCE (lower is a better fit)")
    ax.set_ylabel("hits of 1431 (recall@10)")
    ax.set_ylim(155, 197)
    ax.legend(loc="upper right", fontsize=7.0)
    ax.set_title("The training objective does not rank retrieval: the 4th confirmation.\n"
                 "Red = single-GRU references, blue = tower banks", fontsize=9.0)
    save(fig, "nexttrack_bank_valloss.pdf")


# Fig 35: the MMR lambda frontier under the live 4-factor crown. LEFT: the
# recall/crown plane. Each engine traces a path as lambda falls from 1.0 (off);
# three arms of the single GRU land above the all-time crown bar H_bar = 0.024872
# at a recall that is a statistical TIE with their own control (TIER-1 "FREE"),
# while every champion arm that reaches a comparable H has already paid recall the
# CI can see. RIGHT: the decisive mechanism question. H1 (sonic diversification
# de-concentrates artists) required |dartist_conc| >= |dartist_adj|; H2 (sticky)
# required the ratio BELOW 0.5. Measured 1.59-3.06x in 8 of 8 arms, and the ratio
# shrinks monotonically as lambda falls -- the easy de-concentration is bought
# first and the mechanism saturates.
# (2026-07-26-nexttrack-mmr-lambda-frontier.md, Results + Findings 2/3/4.)
def fig_nexttrack_mmr_frontier():
    H_BAR = 0.024872  # prior all-time crown bar (h425 capacity control)
    FLOOR = 0.108  # mandatory absolute recall floor

    # (label, recall@10, H@10, tier, label offset) -- tier drives the colour
    champ = [
        ("A0 $\\lambda$1.0", 0.21174, 0.008603, "ctrl", (6, -3)),
        ("A1 0.9", 0.21034, 0.010435, "free", (6, 1)),
        ("A2 0.7", 0.19357, 0.015289, "priced", (6, -2)),
        ("A3 0.5", 0.16422, 0.023576, "rejected", (6, -3)),
        ("A4 0.3", 0.10971, 0.031626, "rejected", (5, 4)),
    ]
    gru = [
        ("B0 $\\lambda$1.0", 0.12299, 0.024539, "ctrl", (5, -9)),
        ("B1 0.9", 0.12089, 0.026761, "free", (5, -1)),
        ("B2 0.8", 0.11950, 0.029458, "free", (-38, -2)),
        ("B4 0.7/p50", 0.12020, 0.030355, "free", (5, 3)),
        ("B3 0.7", 0.11461, 0.034171, "priced", (5, -1)),
    ]
    col = {"ctrl": MUTED, "free": GOOD, "priced": ACCENT, "rejected": BAD}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.9))

    ax1.axhline(H_BAR, color="#374151", ls="--", lw=1.2,
                label=f"prior all-time crown bar {H_BAR}")
    ax1.axvline(FLOOR, color=BAD, ls=":", lw=1.2, label="mandatory recall floor 0.108")
    for arms in (champ, gru):
        ax1.plot([a[1] for a in arms], [a[2] for a in arms], "-", color=MUTED,
                 lw=1.0, zorder=2)
        for lab, r, h, tier, o in arms:
            ax1.plot([r], [h], "o", ms=6.2, color=col[tier], zorder=5)
            ax1.annotate(lab, xy=(r, h), xytext=o, textcoords="offset points",
                         fontsize=6.4, color="#374151")
    ax1.text(0.163, 0.0362, "single GRU: three TIER-1\n\"FREE\" arms clear the bar",
             fontsize=6.6, color=GOOD, ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec=GOOD, lw=0.7))
    ax1.text(0.205, 0.0195, "champion path:\nmechanism only\n(2.85$\\times$ behind\nat $\\lambda$1.0)",
             fontsize=6.4, color="#6b7280", ha="center")
    ax1.set_xlabel("recall@10 (P(next-track hit))")
    ax1.set_ylabel("holisticness@10 (live 4-factor)")
    ax1.set_xlim(0.095, 0.245)
    ax1.set_ylim(0.004, 0.041)
    ax1.xaxis.set_major_formatter(afmt)
    ax1.set_title("The crown is reachable at a recall tie ---\nbut only on the single GRU",
                  fontsize=8.8)
    ax1.legend(loc="lower left", fontsize=6.2)

    # -- RIGHT: the H1/H2 mechanism ratio, per arm, against the H2 threshold.
    ratios_a = [(0.9, 3.06), (0.7, 2.17), (0.5, 1.86), (0.3, 1.59)]
    ratios_b = [(0.9, 2.92), (0.8, 2.78), (0.7, 2.46)]
    b4 = (0.7, 2.64)
    ax2.axhline(1.0, color=MUTED, ls="--", lw=1.1, label="equal movement (H1 boundary)")
    ax2.axhspan(0.0, 0.5, color=BAD, alpha=0.10, zorder=0,
                label="H2 (sticky) required this band")
    ax2.plot([r[0] for r in ratios_a], [r[1] for r in ratios_a], "o-", color=ACCENT,
             lw=1.8, ms=6.0, label="champion R$'$+M+C$'$")
    ax2.plot([r[0] for r in ratios_b], [r[1] for r in ratios_b], "s-", color=GOOD,
             lw=1.8, ms=6.0, label="single GRU h256")
    ax2.plot([b4[0]], [b4[1]], "s", color=GOOD, ms=6.0, mfc="white", mew=1.4)
    ax2.annotate("B4 pool 50", xy=b4, xytext=(6, 2), textcoords="offset points",
                 fontsize=6.4, color=GOOD)
    ax2.annotate("3.06$\\times$", xy=(0.9, 3.06), xytext=(-2, 7),
                 textcoords="offset points", fontsize=6.6, color=ACCENT)
    ax2.annotate("1.59$\\times$", xy=(0.3, 1.59), xytext=(2, -12),
                 textcoords="offset points", fontsize=6.6, color=ACCENT)
    ax2.invert_xaxis()
    ax2.set_xlabel("MMR $\\lambda$ (falling to the right $=$ more diversification)")
    ax2.set_ylabel("$|\\Delta$artist_conc$|$ / $|\\Delta$artist_adj$|$")
    ax2.set_ylim(0.0, 3.6)
    ax2.set_title("H1 confirmed 8/8, H2 refuted by 3--6$\\times$:\nsonic MMR de-concentrates artists",
                  fontsize=8.8)
    ax2.legend(loc="lower left", fontsize=6.2)

    fig.suptitle("MMR $\\lambda$ is the first genuine crown lever on record, and its "
                 "mechanism resolves unanimously", fontsize=9.0)
    save(fig, "nexttrack_mmr_frontier.pdf")


# Fig 36: markov_gate. LEFT: the bucket decomposition that is the whole argument.
# Bucketed by the train-level bigram support n_u of the query's last prefix item,
# the entire effect sits in the n_u = 0 bucket while the warm buckets are
# BIT-IDENTICAL (same hits, and the top-k lists byte-for-byte) on all three
# control/arm pairs -- which is why no global static leg weight could ever find it
# (dropping the Markov leg globally is a -0.0070 tie: the two buckets cancel).
# RIGHT: what the gate does on the recall/crown plane. MMR alone on the champion
# is priced (recall CI<0); the gate alone buys recall AND crown; the two compose,
# and gate+MMR is the first arm on record that is recall-NEUTRAL against the
# champion at 3.6x its holisticness.
# (2026-07-27c-nexttrack-markov-gate-confirm.md, bucket tables + Findings 3/4/5.)
def fig_nexttrack_markov_gate():
    # bucket -> per-pair (control, gated); the warm buckets are one value because
    # the two arms' top-k lists are byte-for-byte identical there.
    buckets = [
        ("$n_u = 0$", [(0.1660, 0.2025), (0.1244, 0.1558), (0.1472, 0.1824)]),
        ("$n_u$ 1--10", [(0.3116, 0.3116), (0.2578, 0.2578), (0.2945, 0.2945)]),
        ("$n_u > 10$", [(0.2326, 0.2326), (0.2304, 0.2304), (0.2151, 0.2151)]),
    ]
    pair_names = ["ds1", "ds2", "ds1 $\\lambda$0.7"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.9))

    positions, ticklabels = [], []
    for bi, (bname, cells) in enumerate(buckets):
        for pi, (ctrl, gated) in enumerate(cells):
            pos = bi * 1.25 + pi * 0.33
            positions.append(pos)
            lab = pair_names[pi]
            ticklabels.append(f"{lab}\n{bname}" if pi == 1 else lab)
            ax1.bar(pos - 0.075, ctrl, 0.14, color=MUTED,
                    label="ungated control" if (bi == 0 and pi == 0) else None)
            ax1.bar(pos + 0.075, gated, 0.14, color=GOOD,
                    label="markov_gate = true" if (bi == 0 and pi == 0) else None)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(ticklabels, fontsize=6.2)
    ax1.annotate("$+$0.0365 / $+$0.0314 / $+$0.0352\n(43 misses $\\to$ hits, 14 hits lost)",
                 xy=(0.41, 0.207), xytext=(0.72, 0.318), fontsize=6.5, color=GOOD,
                 arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec=GOOD, lw=0.7))
    ax1.annotate("warm buckets: top-$k$ lists\nBYTE-FOR-BYTE identical,\n$\\Delta = 0.0000$, $n_{disc} = 0$",
                 xy=(2.83, 0.245), xytext=(2.55, 0.352), fontsize=6.5, color="#374151",
                 ha="center", arrowprops=dict(arrowstyle="->", color="#374151", lw=0.8),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f9fafb", ec="#d1d5db", lw=0.7))
    ax1.set_ylabel("recall@10 within bucket")
    ax1.set_xlabel("bucketed by train bigram support of the last prefix item\n"
                   "(55.6--62.3% of test queries have $n_u = 0$)")
    ax1.set_ylim(0.0, 0.40)
    ax1.set_title("A surgically pure $n_u = 0$ intervention:\nthe warm side does not move at all",
                  fontsize=8.8)
    ax1.legend(loc="upper left", fontsize=6.4)
    ax1.grid(axis="x", visible=False)

    # -- RIGHT: recall x crown plane.
    pts = [
        ("champion C1", 0.21174, 0.008603, MUTED, (6, -8)),
        ("$+$MMR $\\lambda$0.7 (C3)", 0.19357, 0.015289, ACCENT, (-14, -15)),
        ("$+$gate (A1)", 0.23201, 0.018699, GOOD, (-26, 7)),
        ("$+$gate$+$MMR (A3)", 0.21314, 0.030980, GOOD, (-16, 8)),
        ("GRU $\\lambda$0.7/p200 (B3)", 0.11461, 0.034171, MUTED, (6, -3)),
    ]
    for lab, r, h, c, o in pts:
        ax2.plot([r], [h], "o", ms=7.0, color=c, zorder=5)
        ax2.annotate(lab, xy=(r, h), xytext=o, textcoords="offset points",
                     fontsize=6.5, color="#374151")
    ax2.annotate("", xy=(0.23201, 0.018699), xytext=(0.21174, 0.008603),
                 arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.3))
    ax2.annotate("", xy=(0.19357, 0.015289), xytext=(0.21174, 0.008603),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3))
    ax2.annotate("", xy=(0.21314, 0.030980), xytext=(0.19357, 0.015289),
                 arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.3))
    ax2.axvline(0.21174, color=MUTED, ls=":", lw=1.0)
    ax2.text(0.2095, 0.0225, "champion recall", fontsize=6.0, color="#6b7280",
             rotation=90, va="center", ha="right")
    ax2.text(0.150, 0.0275, "gate$+$MMR: recall TIE\n($+$0.00140 straddles) at\n3.6$\\times$ the champion's H",
             fontsize=6.5, color=GOOD, ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec=GOOD, lw=0.7))
    ax2.set_xlabel("recall@10 (P(next-track hit))")
    ax2.set_ylabel("holisticness@10 (live 4-factor)")
    ax2.set_xlim(0.10, 0.252)
    ax2.set_ylim(0.003, 0.038)
    ax2.xaxis.set_major_formatter(afmt)
    ax2.set_title("The gate changes MMR's tier on the champion\nfrom PRICED to FREE",
                  fontsize=8.8)

    fig.suptitle("markov_gate: making one leg abstain where it has no evidence buys "
                 "$+$0.0203 recall and a 2.2$\\times$ crown", fontsize=9.0)
    save(fig, "nexttrack_markov_gate.pdf")


# Fig 37: the eagerness regularizer at a calibrated margin, refuted with a
# mechanism. LEFT: the calibration measured the WRONG distribution. The margins
# were drawn from the item->item transition cosine cos(x_t, x_{t+1}) (median
# 0.262 / p75 0.584 / p90 0.830), but the hinge acts on cos(pred_t, x_t), whose
# ceiling over training sits between 0.5843 and 0.70 -- so p75 and p90 lie
# OUTSIDE the support of the penalized quantity and the arms are provably inert
# (bit-identical loss traces at beta up to 92). RIGHT: where the knob does fire,
# eagerness and relevance are the same mechanism -- there is no separating window
# between de-eagering and breaching the relevance floor.
# (2026-07-30-eager-margin-calibrated-refutation.md, Findings 1 and 2.)
def fig_nexttrack_eager_margin():
    FLOOR = 0.4300  # pre-registered absolute music@10 relevance floor

    # (margin, beta, mean relu(cos - m) reconstructed from epoch-1 loss inflation,
    #  trace bit-identical to the control?)
    arms = [
        (0.0, 3, 0.135478, False), (0.0, 12, 0.058340, False),
        (0.40, 9, 0.000918, False), (0.5843, 19, 0.000000, False),
        (0.70, 35, 0.000000, True), (0.8304, 92, 0.000000, True),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.9))

    xs = np.arange(len(arms))
    cols = [BAD if a[3] else (MUTED if a[2] == 0.0 else ACCENT) for a in arms]
    ax1.bar(xs, [a[2] for a in arms], width=0.62, color=cols)
    for xi, a in zip(xs, arms):
        ax1.text(xi, a[2] + 0.004, f"{a[2]:.6f}", ha="center", fontsize=6.2,
                 color="#374151")
    ax1.axvspan(2.5, 5.5, color=BAD, alpha=0.07, zorder=0)
    ax1.text(4.0, 0.062, "INERT: at $m \\geq 0.5843$ the loss trace is\n"
             "bit-identical to the control at all 14 epochs,\n"
             "at $\\beta$ up to the schema cap of 100",
             ha="center", fontsize=6.4, color=BAD,
             bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec=BAD, lw=0.7))
    ax1.text(0.5, 0.100, "DEGENERATE:\nfires, and destroys\nrelevance", ha="center",
             fontsize=6.4, color=ACCENT,
             bbox=dict(boxstyle="round,pad=0.3", fc="#eff6ff", ec=ACCENT, lw=0.7))
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"$m${a[0]:g}\n$\\beta${a[1]:g}" for a in arms], fontsize=6.8)
    ax1.set_ylabel("mean relu$(\\cos(\\mathrm{pred}_t, x_t) - m)$ at init")
    ax1.set_ylim(0.0, 0.168)
    ax1.set_title("The margins were drawn from the item$\\to$item\ntransition tail; the hinge "
                  "never reaches it", fontsize=8.8)
    ax1.grid(axis="x", visible=False)
    ax1.text(0.975, 0.975,
             "margins drawn from cos$(x_t, x_{t+1})$:\nmedian 0.262 / p75 0.584 / p90 0.830\n"
             "ceiling of the PENALIZED quantity\ncos$(\\mathrm{pred}_t, x_t)$: 0.5843--0.70",
             transform=ax1.transAxes, ha="right", va="top", fontsize=6.2,
             color="#374151",
             bbox=dict(boxstyle="round,pad=0.35", fc="#f9fafb", ec="#d1d5db", lw=0.7))

    # -- RIGHT: no separating window.
    pts = [
        ("C0 control\n170 hits", 0.370588, 0.455347, MUTED, (8, 0)),
        ("B1 $m$0.40 $\\beta$9\n164 hits", 0.367606, 0.447386, ACCENT, (-18, -24)),
        ("A1 $m$0 $\\beta$3\n77 hits", 0.240717, 0.399997, BAD, (6, 2)),
        ("A2 $m$0 $\\beta$12\n20 hits", 0.133535, 0.358132, BAD, (7, -2)),
    ]
    ax2.axhline(FLOOR, color=BAD, ls="--", lw=1.2,
                label="pre-registered relevance floor 0.4300")
    ax2.plot([p[1] for p in pts], [p[2] for p in pts], "-", color=MUTED, lw=1.0, zorder=2)
    for lab, conc, mus, c, o in pts:
        ax2.plot([conc], [mus], "o", ms=7.0, color=c, zorder=5)
        ax2.annotate(lab, xy=(conc, mus), xytext=o, textcoords="offset points",
                     fontsize=6.4, color="#374151")
    ax2.axvline(0.319559, color=GOOD, ls="-.", lw=1.2,
                label="free on-record comparator: fusion_layers$=1$")
    ax2.annotate("$\\Delta$conc $-$0.051 for $-$0.016 recall,\nwithout touching this axis",
                 xy=(0.319559, 0.372), xytext=(0.175, 0.379), fontsize=6.3, color=GOOD,
                 arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8))
    ax2.set_xlabel("artist_conc@10 (lower $=$ less eager)")
    ax2.set_ylabel("music@10 (graded relevance)")
    ax2.set_ylim(0.345, 0.475)
    ax2.set_xlim(0.105, 0.415)
    ax2.xaxis.set_major_formatter(afmt)
    ax2.set_title("No separating window: every point of real\nde-eagering is past the "
                  "relevance floor", fontsize=8.8)
    ax2.legend(loc="lower right", fontsize=6.0)

    fig.suptitle("eager_beta is not a weak knob --- it is mis-specified and mis-scaled",
                 fontsize=9.0)
    save(fig, "nexttrack_eager_margin.pdf")


# Fig 38: the hard per-artist cap, the first de-eagering lever that works, and
# the result nobody predicted -- music@10 goes UP, CI>0, at every cap on both
# models. LEFT: artist_conc collapses. MIDDLE: graded relevance RISES while it
# does (the cap trades LITERAL for GRADED relevance: the items promoted from rank
# 75+ are more musically apt to the true continuation than the same-artist tracks
# they displace). RIGHT: the price, on the crown plane -- gate+cap3 would top the
# crown board by ~40% at 1.5x rank 1's recall, and every arm clears the 155-hit
# recall floor, so none is an instance of the "maximize H by predicting worse"
# pathology.
# (2026-07-30c-artist-cap-measurement.md, both split tables + Findings 1-4.)
def fig_nexttrack_artist_cap():
    caps = ["off", "5", "3", "2"]
    x = np.arange(len(caps))
    # deployed champion, ungated / champion + markov_gate
    conc_u = [0.691653, 0.252613, 0.122975, 0.061946]
    conc_g = [0.598214, 0.231338, 0.113440, 0.057365]
    mus_u = [0.453790, 0.474291, 0.476787, 0.474317]
    mus_g = [0.493189, 0.505703, 0.505762, 0.503767]
    hits_u = [303, 256, 230, 198]
    hits_g = [332, 284, 251, 216]
    H_u = [0.008603, 0.027258, 0.033862, 0.036331]
    H_g = [0.018699, 0.040056, 0.048063, 0.051175]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.6, 3.7))

    ax1.plot(x, conc_u, "o-", color=MUTED, lw=1.8, ms=5.6, label="champion (ungated)")
    ax1.plot(x, conc_g, "s-", color=ACCENT, lw=1.8, ms=5.6, label="$+$ markov_gate")
    ax1.annotate("0.113", xy=(2, conc_g[2]), xytext=(6, 6), textcoords="offset points",
                 fontsize=6.6, color=ACCENT)
    ax1.annotate("0.692", xy=(0, conc_u[0]), xytext=(6, 2), textcoords="offset points",
                 fontsize=6.6, color="#374151")
    ax1.set_xticks(x)
    ax1.set_xticklabels(caps)
    ax1.set_xlabel("per-artist cap on the top-10")
    ax1.set_ylabel("artist_conc@10 (lower better)")
    ax1.set_ylim(0.0, 0.80)
    ax1.yaxis.set_major_formatter(afmt)
    ax1.set_title("Eagerness collapses:\n0.692 $\\to$ 0.113 at cap 3", fontsize=8.6)
    ax1.legend(loc="upper right", fontsize=6.2)

    ax2.axhline(0.43, color=BAD, ls="--", lw=1.1)
    ax2.text(0.05, 0.4325, "canonical relevance floor 0.43", fontsize=6.0, color=BAD)
    ax2.plot(x, mus_u, "o-", color=MUTED, lw=1.8, ms=5.6)
    ax2.plot(x, mus_g, "s-", color=GOOD, lw=1.8, ms=5.6)
    ax2.text(0.53, 0.24, "$\\Delta$music@10 CI$>$0 at EVERY cap on\nBOTH models, and it "
             "replicates on the\n0.64-cold split ($+$0.026 there)",
             transform=ax2.transAxes, ha="center", fontsize=6.3, color=GOOD,
             bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec=GOOD, lw=0.7))
    ax2.set_xticks(x)
    ax2.set_xticklabels(caps)
    ax2.set_xlabel("per-artist cap on the top-10")
    ax2.set_ylabel("music@10 (graded relevance)")
    ax2.set_ylim(0.418, 0.525)
    ax2.set_title("The unpredicted result: graded\nrelevance RISES as it de-eagers",
                  fontsize=8.6)

    ax3.axvline(155, color=BAD, ls=":", lw=1.2)
    ax3.text(158, 0.0015, "recall floor (155 hits)", fontsize=6.0, color=BAD, rotation=90)
    ax3.axhline(0.034171, color="#374151", ls="--", lw=1.1)
    ax3.text(352, 0.0352, "incumbent crown 0.034171", fontsize=6.0, color="#374151",
             ha="right")
    ax3.plot(hits_u, H_u, "o-", color=MUTED, lw=1.4, ms=5.6)
    ax3.plot(hits_g, H_g, "s-", color=GOOD, lw=1.8, ms=5.6)
    for xi, yi, lab, o in ((hits_g[2], H_g[2], "gate $+$ cap 3", (-4, 7)),
                           (hits_g[0], H_g[0], "gate, cap off", (-46, -2)),
                           (hits_u[0], H_u[0], "deployed", (-40, -2))):
        ax3.annotate(lab, xy=(xi, yi), xytext=o, textcoords="offset points",
                     fontsize=6.3, color="#374151")
    ax3.set_xlabel("hits of 1431 (recall@10)")
    ax3.set_ylabel("holisticness@10 (live 4-factor)")
    ax3.set_xlim(150, 360)
    ax3.set_ylim(0.0, 0.062)
    ax3.set_title("Priced, not free: $-$52 hits vs the\ndeployed status quo at cap 3",
                  fontsize=8.6)

    fig.suptitle("The hard per-artist cap works because it cannot be outscored --- and it "
                 "trades LITERAL relevance for GRADED relevance", fontsize=9.0)
    save(fig, "nexttrack_artist_cap.pdf")


# Fig 39: the stage-stack topology campaign, refuted on both surfaces. LEFT: every
# 4-stage arm falls THROUGH the mandatory 0.108 absolute recall floor (155 of
# 1431) while the 3.24x capacity control ties the anchor -- the damage is depth,
# not capacity. MIDDLE: the campaign's most interesting number is that arm B
# reaches the LOWEST validation InfoNCE in the batch while retrieving 55 fewer
# hits than the control: the objective the family trains on and top-10 retrieval
# come apart at depth. RIGHT: the two-surface closure -- the walk read at the
# matched primary cell a0.4/s0.3 refutes the same five arms, so the axis does not
# merely fail to close on one surface, it closes on two.
# (2026-08-01-stage-stack-topology.md, Phase 1 + Phase 2 + epoch diagnostics.)
def fig_nexttrack_stage_stack():
    FLOOR_HITS = 155  # 0.108 x 1431, the mandatory absolute recall floor
    n_test = 1431
    # (arm, stages, hits, dlo, dhi vs in-batch C0 (recall@10), val loss, dvibe,
    #  dvibe lo, dvibe hi, drecall)
    arms = [
        ("C0", "h256", 178, None, None, 6.5832, None, None, None, None),
        ("C1", "h537", 174, -0.01188, +0.00559, 6.5723, -0.0100, None, None, -0.00280),
        ("B", "agga", 123, -0.05381, -0.02306, 6.5378, -0.0258, -0.0421, -0.0108, -0.03843),
        ("E", "gagg", 118, -0.05660, -0.02795, 6.5795, -0.0316, -0.0494, -0.0144, -0.04193),
        ("C", "aggg", 114, -0.06010, -0.03005, 6.5842, -0.0231, -0.0381, -0.0084, -0.04472),
        ("D", "gggg", 110, -0.06219, -0.03284, 6.5977, -0.0140, -0.0255, -0.0022, -0.04752),
        ("A", "agag", 101, -0.06848, -0.03913, 6.5960, -0.0375, -0.0559, -0.0214, -0.05381),
    ]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.8, 3.8))

    xs = np.arange(len(arms))
    cols = [MUTED if a[0] in ("C0", "C1") else BAD for a in arms]
    ax1.bar(xs, [a[2] for a in arms], width=0.62, color=cols)
    for xi, a in zip(xs, arms):
        if a[3] is None:
            continue
        lo = a[2] - (arms[0][2] + a[3] * n_test)
        hi = (arms[0][2] + a[4] * n_test) - a[2]
        ax1.errorbar([xi], [a[2]], yerr=[[lo], [hi]], fmt="none", ecolor="#374151",
                     elinewidth=1.1, capsize=2.6, zorder=5)
    ax1.axhline(FLOOR_HITS, color=BAD, ls="--", lw=1.3,
                label=f"absolute recall floor ({FLOOR_HITS} hits)")
    ax1.axhline(arms[0][2], color=MUTED, ls=":", lw=1.2, label="in-batch control C0 (178)")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{a[0]}\n{a[1]}" for a in arms], fontsize=6.6)
    ax1.set_ylabel(f"hits of {n_test} (recall@10)")
    ax1.set_ylim(80, 212)
    ax1.set_title("All five 4-stage arms fall THROUGH\nthe floor; 3.24$\\times$ capacity ties",
                  fontsize=8.5)
    ax1.legend(loc="upper right", fontsize=6.0)
    ax1.grid(axis="x", visible=False)

    for a in arms:
        col = MUTED if a[0] in ("C0", "C1") else BAD
        ax2.plot([a[5]], [a[2]], "o", ms=6.5, color=col, zorder=4)
        ax2.annotate(a[0], xy=(a[5], a[2]), xytext=(5, 3), textcoords="offset points",
                     fontsize=6.8, color="#374151")
    ax2.axhline(178, color=MUTED, ls=":", lw=1.2)
    ax2.annotate("B: the BEST validation loss in the\nbatch, and 55 FEWER hits than C0",
                 xy=(6.5378, 126), xytext=(6.5455, 152), fontsize=6.4, color=BAD,
                 arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec=BAD, lw=0.7))
    ax2.set_xlabel("best validation InfoNCE (lower $=$ better fit)")
    ax2.set_ylabel(f"hits of {n_test} (recall@10)")
    ax2.set_ylim(90, 200)
    ax2.set_title("Objective and retrieval dissociate\nat depth", fontsize=8.5)

    ax3.axhline(0.0, color="#374151", lw=1.0)
    ax3.axvline(0.0, color="#374151", lw=1.0)
    ax3.fill_between([-0.062, -0.018], -0.048, 0.0, color=BAD, alpha=0.07, zorder=0)
    for a in arms[1:]:
        col = MUTED if a[0] == "C1" else BAD
        err = None
        if a[7] is not None:
            err = [[a[6] - a[7]], [a[8] - a[6]]]
        ax3.errorbar([a[9]], [a[6]], yerr=err, fmt="o", ms=6.5, color=col,
                     ecolor=col, elinewidth=1.1, capsize=2.6, zorder=4)
        ax3.annotate(a[0], xy=(a[9], a[6]), xytext=(5, 2), textcoords="offset points",
                     fontsize=6.8, color="#374151")
    ax3.text(-0.040, -0.0455, "refuted on BOTH surfaces", ha="center", fontsize=6.5,
             color=BAD)
    ax3.text(-0.0045, -0.0165, "C1 ties on both", ha="right", fontsize=6.3, color=MUTED)
    ax3.set_xlabel("$\\Delta$recall@10 vs C0 (one-shot)")
    ax3.set_ylabel("$\\Delta$vibe vs C0 (walk, $a$0.4 / $s$0.3)")
    ax3.set_xlim(-0.062, 0.014)
    ax3.set_ylim(-0.048, 0.010)
    ax3.set_title("Two surfaces, one verdict", fontsize=8.5)

    fig.suptitle("Stage-stack topology: depth is a cost, not a lever --- and the training "
                 "objective would have selected the worst arm", fontsize=9.0)
    save(fig, "nexttrack_stage_stack.pdf")


if __name__ == "__main__":
    fig_ae_latent_knn()
    fig_svm_mae_vs_r2()
    fig_auc_bakeoff()
    fig_auc_noise_band()
    fig_bakeoff_p64()
    fig_xgb_depth_lr_heatmap()
    fig_mlp_calibration_signature()
    fig_mlp_arch_scan()
    fig_burn_topology()
    fig_burn_highvocab()
    fig_leaf_head_ceiling()
    fig_vocab_cap_scan()
    fig_burn_ae_capacity()
    fig_sae_pyramid_rect()
    fig_highvocab_capacity()
    fig_taste_drift()
    fig_nexttrack_compression()
    fig_nexttrack_dim_curve()
    fig_nexttrack_phase2_families()
    fig_nexttrack_music_scatter()
    fig_nexttrack_projection_hardening()
    fig_nexttrack_evr_budget()
    fig_nexttrack_representation()
    fig_nexttrack_mmr_tradeoff()
    fig_nexttrack_dualtower_matrix()
    fig_nexttrack_holisticness_antiwin()
    fig_nexttrack_preencoder_forest()
    fig_nexttrack_rectifier()
    fig_nexttrack_power()
    fig_nexttrack_walk_transitions()
    fig_nexttrack_walk_frontier()
    fig_nexttrack_walk_headtohead()
    fig_nexttrack_tower_bank()
    fig_nexttrack_bank_valloss()
    fig_nexttrack_mmr_frontier()
    fig_nexttrack_markov_gate()
    fig_nexttrack_eager_margin()
    fig_nexttrack_artist_cap()
    fig_nexttrack_stage_stack()
