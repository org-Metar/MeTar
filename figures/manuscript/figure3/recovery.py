"""Render the selected manuscript panels from numerical source tables."""

from __future__ import annotations

from collections import Counter

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib import font_manager

from matplotlib.patches import Patch

import numpy as np

import pandas as pd

OUT = Path(__file__).resolve().parents[1]

WIDTH_MM, HEIGHT_MM = 178, 74

GROUPS = ["2", "3", "≥4"]

STATES = ["all", "some", "none"]

COLORS = {"all": "#2D6F9F", "some": "#C8753D", "none": "#D9E0E7"}

DARK, GRID = "#263238", "#D9E2EC"

BASENAME = "Adapted_cross_cell_recovery_panel"

def classify(hits: int, contexts: int) -> str:
    if hits == contexts:
        return "all"
    return "none" if hits == 0 else "some"

def summarize(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Count every eligible compound exactly once for each cutoff."""
    assert not data.duplicated(["drug", "cell_line"]).any()
    assert not data[["drug", "cell_line", "Hit@10", "Hit@30"]].isna().any().any()
    assert data["Hit@10"].isin([0, 1]).all()
    assert data["Hit@30"].isin([0, 1]).all()
    assert (data["Hit@10"] <= data["Hit@30"]).all()
    coverage = data.groupby("drug")["cell_line"].nunique()
    excluded = coverage[coverage < 2].rename("n_cell_lines").reset_index()
    eligible = data[data.drug.isin(coverage[coverage >= 2].index)].copy()
    compounds = eligible.groupby("drug").agg(
        n_cell_lines=("cell_line", "nunique"),
        recovered_top10=("Hit@10", "sum"),
        recovered_top30=("Hit@30", "sum"),
    ).reset_index()
    compounds["coverage_group"] = compounds.n_cell_lines.map(
        lambda n: str(n) if n < 4 else "≥4"
    )
    records = []
    for k in [10, 30]:
        compounds[f"state_top{k}"] = [
            classify(int(h), int(n))
            for h, n in zip(compounds[f"recovered_top{k}"], compounds.n_cell_lines)
        ]
        independently_counted = Counter()
        for _, rows in eligible.groupby("drug"):
            n = len(rows)
            group = str(n) if n < 4 else "≥4"
            independently_counted[(group, classify(int(rows[f"Hit@{k}"].sum()), n))] += 1
        for group in GROUPS:
            subset = compounds[compounds.coverage_group.eq(group)]
            for state in STATES:
                count = int(subset[f"state_top{k}"].eq(state).sum())
                assert count == independently_counted[(group, state)]
                records.append({
                    "cutoff": k,
                    "measured_cell_lines": group,
                    "state": state,
                    "n_compounds": count,
                    "group_n": len(subset),
                    "percentage": count / len(subset) * 100,
                })
    summary = pd.DataFrame(records)
    assert len(compounds) == 415
    assert compounds.groupby("coverage_group").size().reindex(GROUPS).tolist() == [108, 198, 109]
    assert np.allclose(summary.groupby(["cutoff", "measured_cell_lines"]).percentage.sum(), 100)
    assert len(data) == 1452 and len(eligible) == 1450 and len(excluded) == 2
    return compounds, summary, excluded

def setup_style() -> None:
    assert font_manager.findfont("Arial", fallback_to_default=False)
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.65,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": DARK, "axes.labelcolor": DARK, "axes.edgecolor": DARK,
        "xtick.color": DARK, "ytick.color": DARK,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "savefig.transparent": False,
    })

def draw(summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    axes = []
    for left, k in [(18, 10), (104, 30)]:
        ax = fig.add_axes([left / WIDTH_MM, 21 / HEIGHT_MM, 66 / WIDTH_MM, 42 / HEIGHT_MM])
        axes.append(ax)
        table = summary[summary.cutoff.eq(k)].pivot(
            index="measured_cell_lines", columns="state", values="percentage"
        ).reindex(index=GROUPS, columns=STATES)
        x = np.arange(3)
        bottom = np.zeros(3)
        for state in STATES:
            heights = table[state].to_numpy()
            ax.bar(x, heights, width=0.62, bottom=bottom, color=COLORS[state],
                   edgecolor="white", linewidth=0.6, zorder=3)
            for pos, value, base in zip(x, heights, bottom):
                ax.text(pos, base + value / 2, f"{value:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if state == "all" else DARK, zorder=4)
            bottom += heights
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(0, 100)
        ax.set_xticks(x, GROUPS)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.tick_params(direction="out", length=2.5, width=0.65, pad=3)
        ax.set_xlabel("Measured cell lines per compound", labelpad=5)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linewidth=0.45, color=GRID)
        sizes = summary[summary.cutoff.eq(k)].groupby("measured_cell_lines").group_n.first()
        for pos, group in zip(x, GROUPS):
            ax.text(pos, 103, f"n = {sizes[group]}", ha="center", va="bottom",
                    fontsize=7.5, clip_on=False)
        fig.text((left + 33) / WIDTH_MM, 69 / HEIGHT_MM, f"Top-{k}",
                 ha="center", va="center", fontsize=8.5)
    axes[0].set_ylabel("Compounds (%)", labelpad=5)
    fig.text(3 / WIDTH_MM, 69 / HEIGHT_MM, "d", fontsize=10.5, fontweight="bold",
             ha="left", va="center")
    names = ["All measured cell lines", "Some measured cell lines", "No measured cell lines"]
    handles = [Patch(facecolor=COLORS[state], edgecolor="none", label=label)
               for state, label in zip(STATES, names)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.3 / HEIGHT_MM),
               ncol=3, handlelength=1.25, handleheight=0.8, handletextpad=0.5,
               columnspacing=1.6, borderaxespad=0, frameon=False)
    fig.canvas.draw()
    for fmt in ["pdf", "svg"]:
        fig.savefig(OUT / f"{BASENAME}.{fmt}", metadata={"Creator": "MeTar revision"})
    plt.close(fig)
