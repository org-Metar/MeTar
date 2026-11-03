"""Render the selected manuscript panels from numerical source tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

DATA = Path(__file__).resolve().parent / "external_inputs/biology"

OUT = Path(__file__).resolve().parent / "rendered"

INK = "#263238"

GRAY = "#7C858B"

LIGHT_GRAY = "#A4ABB1"

GRID = "#DDE2E4"

WHITE = "#FFFFFF"

CMAP = LinearSegmentedColormap.from_list("balance", ["#496FA6", "#FAFAF7", "#B9672A"])

DRUGS_GR = (
    "Triamcinolone",
    "Clobetasol propionate",
    "Methylprednisolone succinate",
    "Budesonide",
)

CELL_LINES = ("A549", "SW480", "NCI-H460")

GR_MARKERS = ("FKBP5", "TSC22D3", "DUSP1", "SGK1")

def configure_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

def panel_letter(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.03) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", ha="left")

def export_single(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}_preview.png", dpi=300)
    fig.savefig(OUT / f"{stem}_detail.png", dpi=600)
    plt.close(fig)

def render_figure9() -> None:
    s12 = DATA / "s12"
    s10 = DATA / "s10"
    all_30 = pd.read_csv(s12 / "GR_30drug_canonical_marker_distribution.tsv", sep="\t")
    per_cell = pd.read_csv(s12 / "GR_four_drugs_per_cell_rank_percentile.tsv", sep="\t")
    raw = pd.read_csv(s12 / "GR_allcell_marker_log2fc.tsv", sep="\t")
    rank = pd.read_csv(s10 / "Figure7_tahoe_rank_pair_source.tsv", sep="\t")

    raw = raw.loc[raw.dose_um.eq(5.0)].copy()
    gene_mean = raw.groupby(["drug", "cell_line", "gene"], as_index=False)["log2fc"].mean()
    heat = gene_mean[gene_mean.drug.isin(DRUGS_GR)].pivot_table(
        index=["drug", "cell_line"], columns="gene", values="log2fc", aggfunc="first"
    ).reindex(pd.MultiIndex.from_product([DRUGS_GR, CELL_LINES], names=["drug", "cell_line"]))
    heat = heat.loc[:, GR_MARKERS]
    assert heat.shape == (12, 4) and np.isfinite(heat.to_numpy()).all()

    # Extra vertical room is reserved for the shared cell-line legend and c's drug legend.
    fig = plt.figure(figsize=(178 / 25.4, 185 / 25.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0])
    # Reserve a real right-side lane for the full colorbar quantity label.
    fig.subplots_adjust(left=0.225, right=0.88, top=0.965, bottom=0.145, hspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    lower = gs[1, 0].subgridspec(2, 2, height_ratios=[0.20, 0.80], width_ratios=[0.96, 1.04], hspace=0.16, wspace=0.68)
    legend_ax = fig.add_subplot(lower[0, :])
    bx = fig.add_subplot(lower[1, 0])
    cx = fig.add_subplot(lower[1, 1])

    im = ax.imshow(heat.to_numpy(), cmap=CMAP, norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=4), aspect="auto")
    ax.set_xticks(np.arange(4), GR_MARKERS)
    hero_short = {
        "Clobetasol propionate": "Clobetasol p.",
        "Methylprednisolone succinate": "Methylpred. s.",
    }
    ax.set_yticks(np.arange(12), [f"{hero_short.get(drug, drug)} | {line}" for drug, line in heat.index])
    ax.tick_params(length=0)
    for boundary in (2.5, 5.5, 8.5):
        ax.axhline(boundary, color=WHITE, lw=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025)
    cbar.set_label("log2 fold-change vs plate-matched DMSO")
    cbar.ax.tick_params(labelsize=8)
    ax.text(0, 1.025, "Common 5 µM condition", transform=ax.transAxes, fontsize=8.5, va="bottom")
    panel_letter(ax, "a", x=-0.17, y=1.025)

    line_colors = {"A549": "#D55E00", "SW480": "#009E73", "NCI-H460": "#0072B2"}
    offsets = {"A549": -0.10, "SW480": 0.0, "NCI-H460": 0.10}
    ys = np.arange(4)
    per_cell["drug_label"] = per_cell["drug_label"].str.replace("Clobetasol propionate", "Clobetasol p.", regex=False)
    display_drugs = ("Triamcinolone", "Clobetasol p.", "Methylprednisolone succinate", "Budesonide")
    source_drugs = ("Triamcinolone", "Clobetasol propionate", "Methylprednisolone succinate", "Budesonide")
    for line in CELL_LINES:
        q = per_cell[per_cell.cell_line.eq(line)].set_index("drug_label").reindex(display_drugs)
        bx.scatter(q.rank_percentile_from_top, ys + offsets[line], s=25, color=line_colors[line], label=line, zorder=3)
    q = rank[rank.drug.isin([d.lower() for d in source_drugs])].copy()
    q["drug_label"] = q.drug.map({d.lower(): d for d in source_drugs}).replace({"Clobetasol propionate": "Clobetasol p."})
    q = q.set_index("drug_label").reindex(display_drugs)
    bx.scatter(q.consensus_best_percentile_from_top, ys, s=24, marker="D", color=INK, label="Consensus", zorder=4)
    bx.set_yticks(ys, ["Triamcinolone", "Clobetasol p.", "Methylpred. s.", "Budesonide"])
    bx.invert_yaxis()
    bx.set_xlim(-0.12, 3.35)
    bx.set_xlabel("NR3C1 rank percentile (%)\n(lower is better)")
    bx.grid(axis="x", color=GRID, lw=0.55)
    handles, labels = bx.get_legend_handles_labels()
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", ncol=4, handletextpad=0.4, columnspacing=1.3, borderaxespad=0.2)
    panel_letter(bx, "b", x=-0.18, y=1.04)

    other = all_30[~all_30.nr3c1_annotated]
    colors = {"Triamcinolone": "#CC6677", "Clobetasol propionate": "#44AA99", "Methylprednisolone succinate": "#117733", "Budesonide": "#332288"}
    cx.axvline(0, color=GRAY, lw=0.7, zorder=0)
    for y, line in enumerate(CELL_LINES):
        q = other[other.cell_line.eq(line)]
        cx.scatter(q.four_gene_descriptive_mean_log2fc, np.full(len(q), y), s=12, alpha=0.68, color=LIGHT_GRAY, zorder=1)
    for drug in DRUGS_GR:
        q = all_30[all_30.drug.eq(drug)].set_index("cell_line").reindex(CELL_LINES)
        label = {"Clobetasol propionate": "Clobetasol p.", "Methylprednisolone succinate": "Methylpred. s."}.get(drug, drug)
        cx.scatter(q.four_gene_descriptive_mean_log2fc, np.arange(3), s=32, color=colors[drug], edgecolor=WHITE, linewidth=0.45, label=label, zorder=3)
    cx.set_yticks(np.arange(3), CELL_LINES)
    cx.invert_yaxis()
    cx.set_xlim(-1.8, 2.85)
    cx.set_ylim(2.45, -0.45)
    cx.set_xlabel("Four-gene mean log2 fold-change\n(FKBP5, TSC22D3, DUSP1, SGK1)")
    cx.grid(axis="x", color=GRID, lw=0.55)
    cx.legend(loc="upper left", bbox_to_anchor=(0.00, -0.36), ncol=2, columnspacing=0.9, handletextpad=0.4)
    panel_letter(cx, "c", x=-0.24, y=1.04)
    export_single(fig, "Supplementary_Figure9")
