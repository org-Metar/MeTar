"""Render the selected manuscript panels from numerical source tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.colors import BoundaryNorm, ListedColormap

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent / "rendered"

FIG = OUT

SOURCE = Path(__file__).resolve().parent / "external_inputs/legacy"

MM = 1 / 25.4

INK = "#263238"

BLUE = "#2D6F9F"

BLUE_LIGHT = "#91B9D8"

TEAL = "#4A927F"

GRAY = "#6F7882"

LIGHT = "#D9E0E7"

GRID = "#E6EBEF"

PALE = "#F5F7F9"

def style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

def panel(ax: plt.Axes, label: str, x: float = -0.15, y: float = 1.04) -> None:
    ax.text(x, min(y, 0.995), label, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", ha="left", va="bottom", color=INK)

def save_vector(fig: plt.Figure, number: int) -> None:
    stem = FIG / f"Supplementary_Figure{number}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)

def read_legacy(number: int, record: str) -> pd.DataFrame:
    return pd.read_csv(SOURCE / f"supplementary_fig_{number:02d}" / f"{record}.tsv", sep="\t")

def figure4() -> None:
    comp = read_legacy(4, "S4-L021").set_index("component")
    null = read_legacy(4, "S4-L022")
    nsum = read_legacy(4, "S4-L023").set_index("quantity")["value"]
    rnull = read_legacy(4, "S4-L025")
    rsum = read_legacy(4, "S4-L026")
    dc = read_legacy(4, "S4-L027")
    fig = plt.figure(figsize=(178 * MM, 172 * MM))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.25], wspace=0.30,
                             left=0.06, right=0.99, bottom=0.09, top=0.90)
    # The left evidence panels have their own natural heights.  They are not
    # stretched to fill the 30-row table at right; the final row is deliberate
    # whitespace that keeps the two statistical panels proportionate.
    left_column = outer[0, 0].subgridspec(4, 1, height_ratios=[0.24, 0.18, 0.18, 0.40], hspace=0.0)
    a_block = left_column[0, 0].subgridspec(
        2, 2, height_ratios=[0.25, 0.75], width_ratios=[1.15, 0.85],
        wspace=0.56, hspace=0.05
    )
    ax = fig.add_subplot(a_block[1, 0])
    panel(ax, "a", -0.27, 1.04)
    order = ["Drug identity", "Residual", "Cell context"]
    colors = [BLUE, LIGHT, TEAL]
    left = 0.0
    for name, color in zip(order, colors):
        val = float(comp.loc[name, "variance_fraction"])
        ax.barh(0, val, left=left, height=0.42, color=color, edgecolor="white", lw=0.5)
        if val > 0.06:
            ax.text(left + val / 2, 0, f"{val:.1%}", ha="center", va="center", fontsize=7.5,
                    color="white" if color in {BLUE, TEAL} else INK)
        left += val
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of variance")
    ax.set_box_aspect(51.8287 / 69.2152)
    legend_slot = fig.add_subplot(a_block[0, 0])
    legend_slot.set_axis_off()
    legend_slot.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors, order)],
                       loc="upper left", bbox_to_anchor=(0.0, 1.12), ncol=1,
                       handlelength=1.1, columnspacing=1.0, labelspacing=0.25,
                       borderaxespad=0.0)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax2 = fig.add_subplot(a_block[1, 1])
    ax2.hist(null.drug_identity_fraction, bins=38, density=True, color=LIGHT, edgecolor="white", lw=0.25)
    ax2.axvspan(float(nsum.null_drug_identity_q025), float(nsum.null_drug_identity_q975), color=GRAY, alpha=0.12)
    ax2.axvline(float(nsum.null_drug_identity_median), color=GRAY, lw=1.1, label="Null median")
    ax2.axvline(float(nsum.observed_drug_identity_fraction), color=BLUE, lw=1.5, label="Observed")
    ax2.set_xlabel("Drug-identity\nvariance fraction")
    ax2.set_ylabel("Density")
    ax2.set_box_aspect(51.8287 / 51.2705)
    hist_legend_slot = fig.add_subplot(a_block[0, 1])
    hist_legend_slot.set_axis_off()
    hist_legend_slot.legend(handles=[Line2D([0], [0], color=GRAY, lw=1.1, label="Null median"),
                                    Line2D([0], [0], color=BLUE, lw=1.5, label="Observed")],
                            loc="upper left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0,
                            fontsize=7.5, handlelength=1.7, labelspacing=0.25)

    bottom = left_column[2, 0].subgridspec(1, 2, wspace=0.38)
    axes = [fig.add_subplot(bottom[0, i]) for i in range(2)]
    axes[0].text(-0.18, 1.10, "b", transform=axes[0].transAxes, fontsize=10.5,
                 fontweight="bold", ha="left", va="bottom", color=INK)
    for ax, threshold in zip(axes, ["top5%", "top10%"]):
        sub = rsum.loc[rsum.threshold.eq(threshold)].sort_values("n_recovered_contexts")
        xs = np.arange(4)
        means = sub.null_mean_drugs.to_numpy()
        lo = sub.null_q025_drugs.to_numpy()
        hi = sub.null_q975_drugs.to_numpy()
        obs = sub.observed_drugs.to_numpy()
        violin = [rnull.loc[(rnull.threshold.eq(threshold)) & (rnull.n_recovered_contexts.eq(i)), "n_drugs"].to_numpy() for i in xs]
        parts = ax.violinplot(violin, positions=xs, widths=0.62, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(LIGHT); body.set_edgecolor("#B4BEC8"); body.set_alpha(0.85)
        ax.errorbar(xs, means, yerr=[means - lo, hi - means], fmt="_", color=GRAY, capsize=2, lw=0.8)
        ax.bar(xs, obs, width=0.34, color=BLUE, zorder=3)
        for x, y in zip(xs, obs): ax.text(x, y + 0.6, str(int(y)), ha="center", fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xlabel("Recovered cell lines")
        ax.set_box_aspect(59.6519 / 62.992)
        ax.set_ylim(0, 30)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.text(0.5, 1.01, "Top-5%" if threshold == "top5%" else "Top-10%", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Number of drugs")

    hero = outer[0, 1].subgridspec(2, 3, height_ratios=[0.14, 0.86],
                                   width_ratios=[1.75, 1.0, 0.48], wspace=0.04, hspace=0.01)
    head_names = fig.add_subplot(hero[0, 0]); head_heat = fig.add_subplot(hero[0, 1]); head_breadth = fig.add_subplot(hero[0, 2])
    for header in (head_names, head_heat, head_breadth): header.set_axis_off()
    head_names.text(-0.08, 0.98, "c", transform=head_names.transAxes, fontsize=10.5,
                    fontweight="bold", ha="left", va="top")
    head_heat.set_xlim(-0.5, 2.5); head_heat.set_ylim(0, 1)
    for xh, label in enumerate(["A549", "SW480", "NCI-H460"]):
        head_heat.text(xh, 0.36, label, ha="center", va="center", fontsize=7.5, rotation=90)
    head_breadth.set_xlim(0, 3); head_breadth.set_ylim(0, 1)
    head_breadth.text(1.5, 0.92, "Top-5%\ncell lines", ha="center", va="top", fontsize=7.5)
    for xh, label in zip([0.5, 1.5, 2.5], ["1", "2", "3"]):
        head_breadth.text(xh, 0.10, label, ha="center", va="bottom", fontsize=7.5)
    names = fig.add_subplot(hero[1, 0])
    heat = fig.add_subplot(hero[1, 1])
    breadth = fig.add_subplot(hero[1, 2], sharey=heat)
    class_order = {"top5_all_three": 0, "context_specific_top5": 1, "top10_only": 2, "not_recovered_top10": 3}
    dc["class_order"] = dc.recovery_class.map(class_order)
    dc = dc.sort_values(["class_order", "n_contexts_top5", "n_contexts_top10", "mean_best_percentile"],
                        ascending=[True, False, False, True]).reset_index(drop=True)
    n = len(dc)
    names.set_xlim(0, 1); names.set_ylim(n - 0.5, -0.5); names.axis("off")
    for i, row in dc.iterrows():
        if i % 2: names.axhspan(i - 0.5, i + 0.5, color=PALE, zorder=0)
        names.text(0.01, i, str(row.drug), va="center", fontsize=7.5)
    mat = dc[["A549", "SW480", "NCI-H460"]].to_numpy() * 100
    bounds = [0, 1, 5, 10, 50, 101]
    cmap = ListedColormap(["#0E4C70", BLUE, BLUE_LIGHT, "#DCE9F2", "#E7C99A"])
    norm = BoundaryNorm(bounds, cmap.N)
    im = heat.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    heat.set_xticks([]); heat.set_yticks([]); heat.tick_params(length=0)
    for s in heat.spines.values(): s.set_visible(False)
    for i in range(n):
        for j in range(3):
            v = mat[i, j]
            heat.text(j, i, ">50" if v > 50 else f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                      color="white" if v <= 5 else INK)
    breadth.set_xlim(0, 3); breadth.set_ylim(n - 0.5, -0.5)
    breadth.set_xticks([]); breadth.set_yticks([])
    breadth.spines[:].set_visible(False)
    for i, count in enumerate(dc.n_contexts_top5.astype(int)):
        for k in range(3): breadth.scatter(k + 0.5, i, s=14, color=BLUE if k < count else LIGHT, edgecolors="none")
    cax = fig.add_axes([0.61, 0.055, 0.22, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[1, 5, 10, 50])
    cb.set_label("Best-target percentile (%)", fontsize=8.5, labelpad=2); cb.outline.set_visible(False)
    save_vector(fig, 4)
