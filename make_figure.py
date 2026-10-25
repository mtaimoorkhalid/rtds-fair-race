"""Poster/report figure: C-index and IBS per model per dataset.

Reads every results CSV (classic + deep), plots the 5 seeds as small dots
(just right of each bar) and their mean +/- 1 std as a large marker with an
error bar, one panel per metric.
Saves results/fig_results.png (300 dpi; place it across the full width of an
A2 poster so the text stays readable from ~1 m) and a
PDF version. Rerun it whenever the result CSVs change.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

DATASETS = [("gbsg", "GBSG"), ("metabric", "METABRIC"), ("support", "SUPPORT")]
# fixed order + colour + shape per model (classic = round shapes, deep = pointed shapes)
MODELS = [
    ("coxph", "Cox", "#2a78d6", "o"),
    ("rsf", "RSF", "#eb6834", "s"),
    ("deepsurv", "DeepSurv", "#1baf7a", "^"),
    ("deephit", "DeepHit", "#eda100", "D"),
]
PANELS = [("c_index", "C-index (higher is better)"), ("ibs", "IBS (lower is better)")]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_results() -> pd.DataFrame:
    frames = [pd.read_csv(p) for side in ("classic", "deep")
              for p in sorted((RESULTS / side).glob("*.csv"))]
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_results()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    rng = np.random.default_rng(0)  # only for the small horizontal jitter of seed dots
    offsets = np.linspace(-0.3, 0.3, len(MODELS))

    for ax, (metric, title) in zip(axes, PANELS):
        for d_i, (dkey, _) in enumerate(DATASETS):
            for (mkey, mname, color, marker), off in zip(MODELS, offsets):
                vals = df[(df["model"] == mkey) & (df["dataset"] == dkey)][metric].to_numpy()
                if len(vals) == 0:
                    continue
                x = d_i + off
                jitter = rng.uniform(0.05, 0.1, len(vals))  # seed dots just right of the bar
                ax.scatter(x + jitter, vals, s=40, color=color, alpha=0.45,
                           marker=marker, linewidths=0, zorder=2)
                ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), fmt=marker,
                            ms=15, color=color, mec="white", mew=1.5,
                            elinewidth=2.5, capsize=0, zorder=3)
        ax.set_title(title, fontsize=24, color=INK, loc="left", pad=10)
        ax.set_xticks(range(len(DATASETS)), [name for _, name in DATASETS])
        ax.set_xlim(-0.6, len(DATASETS) - 0.4)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", length=0, labelsize=21)
        ax.tick_params(axis="y", labelsize=18)
        for d_i in range(1, len(DATASETS)):
            ax.axvline(d_i - 0.5, color=GRID, linewidth=1, zorder=0)

    handles = [Line2D([], [], marker=marker, color=color, linestyle="none", ms=16,
                      mec="white", mew=1.5, label=mname)
               for _, mname, color, marker in MODELS]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(MODELS),
               frameon=False, fontsize=21, handletextpad=0.3, columnspacing=1.8)

    out = RESULTS / "fig_results"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {out}.png and {out}.pdf")


if __name__ == "__main__":
    main()