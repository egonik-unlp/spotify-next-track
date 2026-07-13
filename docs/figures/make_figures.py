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
