#!/usr/bin/env python3
"""Vector-PDF figures for docs/music-taste.tex.

Data transcribed from the corpus aggregate over the rotation dataset
ds-20260612-143014-p64-s42 (23,529 tracks; 11,127 replayed / rotation>=1).
One fig_<name>() per figure; outputs land next to this script.
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
ACC = "#c0392b"
BLU = "#2c6fbb"


def fig_genre_lift():
    fams = ["electronic /\ndance", "rock /\nindie", "pop", "latin /\nargentine",
            "hip-hop /\nr&b", "jazz /\nsoul"]
    lift = [1.15, 1.03, 0.96, 0.73, 0.67, 0.60]
    y = np.arange(len(fams))[::-1]
    colors = [BLU if v >= 1 else ACC for v in lift]
    fig, ax = plt.subplots()
    ax.barh(y, lift, color=colors, height=0.62)
    ax.axvline(1.0, color=INK, lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(fams)
    ax.set_xlabel("replay lift  (share of replays / share of library)")
    ax.set_title("What gets replayed, relative to its share of the library")
    for yi, v in zip(y, lift):
        ax.text(v + 0.01, yi, f"{v:.2f}×", va="center", fontsize=8)
    ax.set_xlim(0, 1.35)
    fig.savefig(OUT / "taste_genre_lift.pdf")
    plt.close(fig)


def fig_acoustic():
    feats = ["instrumentalness", "energy", "danceability", "liveness",
             "speechiness", "valence", "acousticness"]
    delta = [0.057, 0.012, 0.009, 0.003, -0.007, -0.011, -0.021]
    order = np.argsort(delta)
    feats = [feats[i] for i in order]
    delta = [delta[i] for i in order]
    y = np.arange(len(feats))
    colors = [BLU if v >= 0 else ACC for v in delta]
    fig, ax = plt.subplots()
    ax.barh(y, delta, color=colors, height=0.62)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("mean(replayed) − mean(one-and-done)")
    ax.set_title("Acoustic fingerprint of a replayed track")
    fig.savefig(OUT / "taste_acoustic.pdf")
    plt.close(fig)


def fig_era():
    decades = ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
    counts = [357, 389, 1423, 2036, 4417, 2456]
    x = np.arange(len(decades))
    fig, ax = plt.subplots()
    ax.bar(x, counts, color=BLU, width=0.66)
    ax.set_xticks(x)
    ax.set_xticklabels(decades)
    ax.set_ylabel("replayed tracks")
    ax.set_title("When the music you replay was released (median 2014)")
    fig.savefig(OUT / "taste_era.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_genre_lift()
    fig_acoustic()
    fig_era()
    print("wrote taste_genre_lift.pdf, taste_acoustic.pdf, taste_era.pdf")
