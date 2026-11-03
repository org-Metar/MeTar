"""Render the selected manuscript panels from numerical source tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

DATA = Path(__file__).resolve().parent / "external_inputs/biology"

OUT = Path(__file__).resolve().parent / "rendered"

INK = "#263238"

TEAL = "#197C78"

AMBER = "#B66B2D"

WHITE = "#FFFFFF"

CMAP = LinearSegmentedColormap.from_list("balance", ["#496FA6", "#FAFAF7", "#B9672A"])

CELL_LINES = ("A549", "SW480", "NCI-H460")

MARKER_GR_DRUGS = ("budesonide", "clobetasol propionate", "methylprednisolone succinate", "triamcinolone")

MARKER_CDK_DRUGS = ("abemaciclib", "ribociclib")

DOSES = (0.05, 0.5, 5.0)

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

def marker_heatmap(ax: plt.Axes, cbar_ax: plt.Axes, data: pd.DataFrame, drugs: tuple[str, ...], marker: str, letter: str, condition: str) -> None:
    order = [(drug, line) for drug in drugs for line in CELL_LINES]
    matrix = np.full((len(order), len(DOSES)), np.nan)
    multiple = np.zeros_like(matrix, dtype=bool)
    for i, (drug, line) in enumerate(order):
        for j, dose in enumerate(DOSES):
            q = data[(data.drug == drug) & (data.cell_context == line) & data.dose_um.eq(dose) & data.marker.eq(marker)]
            if len(q) == 1:
                matrix[i, j] = q.iloc[0].effect
                multiple[i, j] = q.iloc[0].n_samples > 1
    bound = max(float(np.nanmax(np.abs(matrix))), 0.05)
    im = ax.imshow(matrix, cmap=CMAP, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
    short = {"clobetasol propionate": "Clobetasol p.", "methylprednisolone succinate": "Methylpred. s."}
    ax.set_yticks(np.arange(len(order)), [f"{short.get(drug, drug.capitalize())} | {line}" for drug, line in order])
    ax.set_xticks(np.arange(3), ["0.05", "0.5", "5"])
    ax.set_xlabel("Dose (µM)")
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "NA" if not np.isfinite(value) else f"{value:+.2f}{'*' if multiple[i, j] else ''}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.0, color=WHITE if np.isfinite(value) and abs(value) > 0.65 * bound else INK)
    for boundary in range(3, len(order), 3):
        ax.axhline(boundary - 0.5, color=WHITE, lw=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, cax=cbar_ax)
    cbar.set_label("Δ mean log1p(CP10K)")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(pad=2)
    cbar.ax.tick_params(labelsize=8)
    panel_letter(ax, letter, x=-0.36 if len(order) <= 6 else -0.30)
    ax.text(0, 1.03, condition, transform=ax.transAxes, fontsize=8.5, va="bottom")

def build_figure12_pages() -> tuple[plt.Figure, plt.Figure]:
    s12 = DATA / "s12"
    heat = pd.read_csv(s12 / "pilot_heatmap_source.tsv", sep="\t")
    curves = pd.read_csv(s12 / "pilot_sampling_curve_source.tsv", sep="\t")
    strata = pd.read_csv(s12 / "same_n_rank_marker_strata.tsv", sep="\t")

    page1 = plt.figure(figsize=(178 / 25.4, 246 / 25.4))
    gs = page1.add_gridspec(3, 2, height_ratios=[2.0, 1.15, 1.15], width_ratios=[0.965, 0.035], wspace=0.45)
    page1.subplots_adjust(left=0.305, right=0.965, top=0.97, bottom=0.045, hspace=0.38)
    a = page1.add_subplot(gs[0, 0]); ac = page1.add_subplot(gs[0, 1])
    b = page1.add_subplot(gs[1, 0]); bc = page1.add_subplot(gs[1, 1])
    c = page1.add_subplot(gs[2, 0]); cc = page1.add_subplot(gs[2, 1])
    marker_heatmap(a, ac, heat, MARKER_GR_DRUGS, "GR", "a", "Glucocorticoid-associated markers")
    marker_heatmap(b, bc, heat, MARKER_CDK_DRUGS, "S", "b", "S-phase markers")
    marker_heatmap(c, cc, heat, MARKER_CDK_DRUGS, "G2M", "c", "G2M-phase markers")

    page2 = plt.figure(figsize=(178 / 25.4, 173 / 25.4))
    # Restore the original standalone-panel aspect ratios: d=3.61495, e=0.67023.
    # The extra page2 whitespace is deliberate, preserving geometry and type size.
    d = page2.add_axes([0.23, 0.74791, 0.71, 0.20209])
    e = page2.add_axes([0.23, 0.12, 0.26488, 0.4066666667])
    labels = {"GR": "GR markers (12 pools)", "S": "S markers (6 pools)", "G2M": "G2M markers (6 pools)"}
    colors = {"GR": TEAL, "S": AMBER, "G2M": "#7E6C9F"}
    for marker in ("GR", "S", "G2M"):
        q = curves[curves.marker.eq(marker)].sort_values("cell_count")
        d.plot(q.cell_count, q.median_root_mean_squared_deviation, marker="o", ms=3.2, lw=1.3, color=colors[marker], label=labels[marker])
    d.set_xscale("log", base=2)
    d.set_xticks([1, 4, 16, 32, 64, 128, 192, 300], ["1", "4", "16", "32", "64", "128", "192", "300"])
    d.set_xlabel("Cells sampled from each fixed 300-cell pool")
    d.set_ylabel("Median sampling RMSE\n(mean log1p(CP10K))")
    d.legend(loc="upper right", ncol=1)
    panel_letter(d, "d", x=-0.16)

    order = [(drug, line) for drug in (*MARKER_GR_DRUGS, *MARKER_CDK_DRUGS) for line in CELL_LINES]
    counts = (96, 160, 192, 224)
    matrix = np.full((len(order), len(counts)), np.nan)
    for i, (drug, line) in enumerate(order):
        for j, n in enumerate(counts):
            q = strata[(strata.drug == drug) & (strata.cell_context == line) & strata.cell_count.eq(n)]
            if len(q) == 1:
                matrix[i, j] = q.iloc[0].adjusted_pearson_r
    im = e.imshow(matrix, cmap=CMAP, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
    short = {"clobetasol propionate": "Clobetasol p.", "methylprednisolone succinate": "Methylpred. s."}
    e.set_yticks(np.arange(len(order)), [f"{short.get(drug, drug.capitalize())} | {line}" for drug, line in order])
    e.set_xticks(np.arange(4), counts)
    e.set_xlabel("Fixed input cells (30 overlapping draws per stratum)")
    e.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            e.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8.0, color=WHITE if abs(matrix[i, j]) > 0.65 else INK)
    for boundary in (2.5, 5.5, 8.5, 11.5, 14.5):
        e.axhline(boundary, color=INK if boundary == 11.5 else WHITE, lw=1.0)
    for spine in e.spines.values():
        spine.set_visible(False)
    # Independent horizontal lanes: heatmap -> left-side ticks -> bar -> title.
    cbar_ax = page2.add_axes([0.625, 0.12, 0.0213, 0.4066666667])
    cbar = page2.colorbar(im, cax=cbar_ax)
    cbar.set_label("Conditional partial Pearson r")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(pad=2)
    cbar.ax.yaxis.labelpad = 6
    cbar.ax.tick_params(labelsize=8)
    panel_letter(e, "e", x=-0.22)
    e.text(0, 1.03, "Marker–target-rank association (72 strata)", transform=e.transAxes, fontsize=8.5, va="bottom")
    return page1, page2

def render_figure12() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page1, page2 = build_figure12_pages()
    with PdfPages(OUT / "Supplementary_Figure12.pdf") as pdf:
        pdf.savefig(page1)
        pdf.savefig(page2)
    for idx, fig in enumerate((page1, page2), start=1):
        fig.savefig(OUT / f"Supplementary_Figure12_page{idx}.svg")
        fig.savefig(OUT / f"Supplementary_Figure12_page{idx}_preview.png", dpi=300)
        fig.savefig(OUT / f"Supplementary_Figure12_page{idx}_detail.png", dpi=600)
        plt.close(fig)
