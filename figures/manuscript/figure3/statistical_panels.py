"""Render the selected manuscript panels from numerical source tables."""

from __future__ import annotations

import csv

import json

import math

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pymupdf

from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"

PANELS = ROOT / "previews" / "panels"

CONTRACT_PATH = Path(__file__).resolve().parent / "rendering_style.json"

CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

PALETTE = CONTRACT["palette"]

TYPOGRAPHY = CONTRACT["typography"]

PAGE_W_MM = float(CONTRACT["render"]["width_mm"])

PANEL_B_H_MM = 65.0

BOTTOM_H_MM = 51.5

BOTTOM_WIDTHS_MM = {"c": 68.0, "d": 49.0, "e": 49.0}

INK = PALETTE["text"]

GRID = PALETTE["grid"]

METHODS = ["MeTar direct", "SSGCN", "PertKGE"]

METHOD_STYLE = {
    "MeTar direct": {"color": PALETTE["MeTar"], "marker": "o"},
    "SSGCN": {"color": PALETTE["SSGCN"], "marker": "s"},
    "PertKGE": {"color": PALETTE["PertKGE"], "marker": "^"},
}

CELL_ORDER = ["HUES3", "XC.P910", "XC.P909", "NCIH841", "HELA", "SNGM", "RMUGS", "HT29", "LOVO", "HT115", "HCC515", "VCAP", "NCIH1694", "SKLU1"]

METRIC_ORDER = ["Hit@10", "NDCG@10", "MeanAP", "Hit@50", "AUROC"]

SUPPORT_ORDER = [8, 16, 32, 64, 128]

def read_tsv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))

def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)

def near(actual: float, expected: float, message: str, tol: float = 1e-8) -> None:
    require(math.isclose(actual, expected, rel_tol=tol, abs_tol=tol), message)

def mm(value: float) -> float:
    return value / 25.4

def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": TYPOGRAPHY["tick_category_legend_heatmap_value"]["pt"],
        "axes.labelsize": TYPOGRAPHY["axis_label"]["pt"],
        "axes.labelcolor": INK,
        "xtick.labelsize": TYPOGRAPHY["tick_category_legend_heatmap_value"]["pt"],
        "ytick.labelsize": TYPOGRAPHY["tick_category_legend_heatmap_value"]["pt"],
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.65,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
        "savefig.pad_inches": 0,
        "axes.unicode_minus": False,
    })

def add_panel_label(fig: plt.Figure, label: str, width_mm: float, height_mm: float) -> None:
    fig.text(mm(3.0) / mm(width_mm), 1.0 - mm(2.8) / mm(height_mm), label, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK, family="Arial")

def method_handles() -> list[Line2D]:
    return [Line2D([0], [0], color=METHOD_STYLE[m]["color"], marker=METHOD_STYLE[m]["marker"], linestyle="-", markersize=4.2, linewidth=1.0, markeredgewidth=0.3, label=m) for m in METHODS]

def save_panel(fig: plt.Figure, label: str, width_mm: float, height_mm: float) -> Path:
    path = PANELS / f"Figure3{label}.pdf"
    fig.savefig(path, format="pdf", facecolor="white", metadata={"Title": f"Figure 3, panel {label}"})
    plt.close(fig)
    with pymupdf.open(path) as doc:
        require(len(doc) == 1, f"Panel {label} is not one page")
        near(doc[0].rect.width, mm(width_mm) * 72.0, f"Panel {label} width changed", 1e-5)
        near(doc[0].rect.height, mm(height_mm) * 72.0, f"Panel {label} height changed", 1e-5)
    return path

def build_panel_b(rows: list[dict[str, str]]) -> Path:
    values = {(r["method_label"], r["cell_line"]): float(r["Hit@10"]) * 100.0 for r in rows}
    fig = plt.figure(figsize=(mm(PAGE_W_MM), mm(PANEL_B_H_MM)))
    ax = fig.add_axes([0.125, 0.155, 0.855, 0.755])
    y = np.arange(len(CELL_ORDER), dtype=float)
    offsets = {"MeTar direct": -0.37, "SSGCN": 0.0, "PertKGE": 0.37}
    for i, cell in enumerate(CELL_ORDER):
        vals = [values[(m, cell)] for m in METHODS]
        ax.hlines(y[i], min(vals), max(vals), color="#BCC8D2", linewidth=1.0, zorder=1)
    for method in METHODS:
        style = METHOD_STYLE[method]
        ax.scatter([values[(method, cell)] for cell in CELL_ORDER], y + offsets[method], s=21, c=style["color"], marker=style["marker"], edgecolors="white", linewidths=0.45, clip_on=False, zorder=3)
    ax.set_yticks(y, CELL_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(-1.5, 70)
    ax.set_xticks(np.arange(0, 71, 10))
    ax.axvline(0, color=GRID, linewidth=0.75, zorder=0)
    ax.set_xlabel("Hit@10 (%)", labelpad=2.4)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=method_handles(), ncol=3, frameon=False, fontsize=8, handlelength=1.35, handletextpad=0.45, columnspacing=1.25, loc="upper right", bbox_to_anchor=(0.99, 1.045), borderaxespad=0)
    add_panel_label(fig, "b", PAGE_W_MM, PANEL_B_H_MM)
    path = save_panel(fig, "b", PAGE_W_MM, PANEL_B_H_MM)
    return path

def build_panel_c(rows: list[dict[str, str]]) -> Path:
    lookup = {(r["method"], r["metric"]): r for r in rows}
    width = BOTTOM_WIDTHS_MM["c"]
    fig = plt.figure(figsize=(mm(width), mm(BOTTOM_H_MM)))
    ax = fig.add_axes([17.2 / width, 10.2 / BOTTOM_H_MM, (width - 19.4) / width, (BOTTOM_H_MM - 14.0) / BOTTOM_H_MM])
    y = np.arange(len(METRIC_ORDER), dtype=float)
    offsets = {"MeTar direct": 0.22, "SSGCN": 0.0, "PertKGE": -0.22}
    for i, metric in enumerate(METRIC_ORDER):
        for method in METHODS:
            r = lookup[(method, metric)]
            mean, low, high = float(r["mean_pct"]), float(r["ci95_low_pct"]), float(r["ci95_high_pct"])
            style = METHOD_STYLE[method]
            ax.errorbar(mean, y[i] + offsets[method], xerr=[[mean - low], [high - mean]], fmt=style["marker"], color=style["color"], ecolor=style["color"], elinewidth=0.9, capsize=1.4, capthick=0.85, markersize=4.0, markeredgecolor="white", markeredgewidth=0.35, zorder=3)
    ax.set_yticks(y, METRIC_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Cell-line mean (%)", labelpad=2.3)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=2.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(fig, "c", width, BOTTOM_H_MM)
    return save_panel(fig, "c", width, BOTTOM_H_MM)

def build_panel_d(support: list[dict[str, str]], summaries: list[dict[str, str]]) -> Path:
    bins = ["<=64", "65-256", ">256"]
    bin_n = {"<=64": 4, "65-256": 6, ">256": 4}
    lookup = {(r["support_bin"], r["method"]): r for r in summaries}
    width = BOTTOM_WIDTHS_MM["d"]
    fig = plt.figure(figsize=(mm(width), mm(BOTTOM_H_MM)))
    ax = fig.add_axes([10.2 / width, 10.2 / BOTTOM_H_MM, (width - 12.5) / width, (BOTTOM_H_MM - 13.5) / BOTTOM_H_MM])
    base_x = np.arange(1, 4, dtype=float)
    offsets = {"MeTar direct": -0.19, "SSGCN": 0.0, "PertKGE": 0.19}
    bin_index = {"<=64": 0, "65-256": 1, ">256": 2}
    for method in METHODS:
        style = METHOD_STYLE[method]
        for support_bin in bins:
            rows = [r for r in support if r["method_label"] == method and (("<=64" if int(r["support_groups"]) <= 64 else "65-256" if int(r["support_groups"]) <= 256 else ">256") == support_bin)]
            center = base_x[bin_index[support_bin]] + offsets[method]
            ax.scatter([center + (((i % 5) - 2) * 0.035) for i, _ in enumerate(rows)], [float(r["Hit@10_pct"]) for r in rows], s=16, c=style["color"], marker=style["marker"], alpha=0.52, edgecolors="none", clip_on=False, zorder=2)
            summary = lookup[(support_bin, method)]
            ax.errorbar(center, float(summary["mean_hit10_pct"]), yerr=float(summary["sem_hit10_pct"]), fmt=style["marker"], color=style["color"], ecolor=style["color"], elinewidth=0.95, capsize=1.6, capthick=0.9, markersize=4.0, markeredgecolor="white", markeredgewidth=0.35, zorder=4)
    ax.set_xlim(0.56, 3.44)
    ax.set_ylim(-2.5, 72)
    ax.set_xticks(base_x, [f"{b}\n(n={bin_n[b]})" for b in bins])
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Hit@10 (%)", labelpad=1.5)
    ax.set_xlabel("Support groups per cell", labelpad=2.0)
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.axhline(0, color=PALETTE["Random"], linewidth=0.85, zorder=1)
    ax.tick_params(axis="x", length=0, pad=2.2)
    ax.tick_params(axis="y", pad=1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(fig, "d", width, BOTTOM_H_MM)
    path = save_panel(fig, "d", width, BOTTOM_H_MM)
    return path

def build_panel_e(delta: list[dict[str, str]], summaries: list[dict[str, str]]) -> Path:
    width = BOTTOM_WIDTHS_MM["e"]
    fig = plt.figure(figsize=(mm(width), mm(BOTTOM_H_MM)))
    ax = fig.add_axes([12.8 / width, 10.2 / BOTTOM_H_MM, (width - 14.2) / width, (BOTTOM_H_MM - 13.5) / BOTTOM_H_MM])
    jitter = {cell: float(offset) for cell, offset in zip(CELL_ORDER, np.linspace(-0.17, 0.17, len(CELL_ORDER)))}
    lookup = {int(r["support_size"]): r for r in summaries}
    base_x = np.arange(1, len(SUPPORT_ORDER) + 1, dtype=float)
    for i, size in enumerate(SUPPORT_ORDER):
        rows = [r for r in delta if int(r["support_size"]) == size]
        ax.scatter([base_x[i] + jitter[r["cell_line"]] for r in rows], [float(r["delta_hit10_pp"]) for r in rows], s=15, c=PALETTE["Random"], marker="o", alpha=0.43, edgecolors="none", zorder=2)
        summary = lookup[size]
        ax.errorbar(base_x[i], float(summary["mean_delta_hit10_pp"]), yerr=float(summary["sem_delta_hit10_pp"]), fmt="o", color=PALETTE["MeTar"], ecolor=PALETTE["MeTar"], elinewidth=1.0, capsize=1.8, capthick=0.9, markersize=4.6, markeredgecolor="white", markeredgewidth=0.4, zorder=4)
    ax.axhline(0, color=PALETTE["Random"], linewidth=0.8, linestyle=(0, (2.5, 2.0)), zorder=1)
    ax.set_xlim(0.58, 5.42)
    ax.set_ylim(-4.0, 9.0)
    ax.set_xticks(base_x, [str(size) for size in SUPPORT_ORDER])
    ax.set_yticks([-4, -2, 0, 2, 4, 6, 8])
    ax.set_ylabel("ΔHit@10 (pp)", labelpad=1.6)
    ax.set_xlabel("Support groups per cell", labelpad=2.0)
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=2.2)
    ax.tick_params(axis="y", pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(fig, "e", width, BOTTOM_H_MM)
    return save_panel(fig, "e", width, BOTTOM_H_MM)
