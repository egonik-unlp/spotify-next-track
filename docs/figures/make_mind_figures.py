#!/usr/bin/env python3
"""Vector-PDF figures for docs/next-track-mind.tex.

Data transcribed from the per-model SAE analysis
`interp-run-20260723-204848-4e89f-msae` of the next-track GRU
`best-seq-nexttrack-20260718-230948-d3d4a` (dataset seq-20260718-222807;
91,827 session-step rows). One fig_<name>() per figure; outputs land next to
this script. Style mirrors make_taste_figures.py.
"""
from pathlib import Path
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "figure.figsize": (6.0, 3.3),
    "figure.constrained_layout.use": True,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
INK = "#2b2b2b"
ACC = "#c0392b"   # genre concepts
BLU = "#2c6fbb"   # artist concepts


def fig_concepts():
    """Top SAE atoms by next-item concept separation, with their GPT labels."""
    # (separation σ, GPT label, field) — the 10 strongest surfaced atoms.
    rows = [
        (3.81, "DIIV — Brooklyn indie", "artist"),
        (3.51, "Rammstein — live", "artist"),
        (3.39, "Rammstein — alt-metal", "artist"),
        (3.31, "Rammstein — alt-metal", "artist"),
        (3.01, "Pink Floyd — album rock", "genre"),
        (2.82, "Thievery Corp. — downtempo", "artist"),
        (2.82, "Indie-rock favourites", "genre"),
        (2.81, "DIIV — indie soundscape", "artist"),
        (2.71, "Aphex Twin — ambient", "artist"),
        (2.70, "Four Tet — electronica", "artist"),
    ]
    seps = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    colors = [BLU if r[2] == "artist" else ACC for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh(y, seps, color=colors, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("concept separation  (σ, atom fires vs. rest)")
    ax.set_title("The concepts the next-track model formed (GPT-labeled)")
    for yi, s in zip(y, seps):
        ax.text(s + 0.03, yi, f"{s:.1f}σ", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 4.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLU, label="artist concept"),
                       Patch(color=ACC, label="genre concept")],
              loc="lower right", frameon=False, fontsize=8)
    fig.savefig(OUT / "mind_concepts.pdf")
    plt.close(fig)


def fig_genre_repr():
    """Genres the recurrent state separates most strongly (best-atom σ)."""
    fams = ["alternative metal", "brooklyn indie", "album rock", "indie rock",
            "düsseldorf electronic", "ambient", "argentine rock", "electronica"]
    sep = [3.31, 3.03, 3.01, 2.82, 2.53, 2.28, 2.08, 1.95]
    y = np.arange(len(fams))[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.barh(y, sep, color=BLU, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(fams)
    ax.set_xlabel("best atom separation for this next-item genre  (σ)")
    ax.set_title("Every genre is represented — the strongest 8 of 25")
    ax.set_xlim(0, 3.7)
    fig.savefig(OUT / "mind_genre_repr.pdf")
    plt.close(fig)


def fig_decodability():
    """Next-item genre linearly decodable from the recurrent state vs chance."""
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    bars = ["chance\n(majority)", "GRU state\n(linear probe)"]
    acc = [0.0074, 0.417]
    ax.bar(bars, acc, color=[ACC, BLU], width=0.55)
    ax.set_ylabel("next-item genre top-1 accuracy")
    ax.set_title("Genre of the next track is decodable  (AUC 0.79)")
    for i, a in enumerate(acc):
        ax.text(i, a + 0.008, f"{a*100:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 0.48)
    fig.savefig(OUT / "mind_decodability.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_concepts()
    fig_genre_repr()
    fig_decodability()
    print("wrote mind_concepts.pdf, mind_genre_repr.pdf, mind_decodability.pdf")
