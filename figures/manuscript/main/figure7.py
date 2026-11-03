"""Render the selected manuscript panels from numerical source tables."""

from __future__ import annotations

import json

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.patches import Patch, Rectangle

import numpy as np

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"

PREVIEWS = ROOT / "previews"

CONTRACT_PATH = Path(__file__).resolve().parent / "visual_contract_20261010_v3.json"

WIDTH_MM = 178.0

FULL_HEIGHT_MM = 144.0

FULL_TOP_CROP_MM = 0.0

DETAIL_DPI = 600

CANDIDATE_LABELS_LINCS = 668

MATCHED_PERCENT = 10 / CANDIDATE_LABELS_LINCS * 100.0

SCATTER_CLIP_FLOOR = 0.15

SCATTER_JITTER_LOG_SD = 0.015

SCATTER_JITTER_SEEDS = {"LINCS": 4815, "Tahoe": 4816}

SOURCE_FILES = [
    "Figure7_lincs_hitk_curve_source.tsv",
    "Figure7_lincs_recallk_curve_source.tsv",
    "Figure7_lincs_rank_pair_source.tsv",
    "Figure7_lincs_state_summary_source.tsv",
    "Figure7_tahoe_hitk_curve_source.tsv",
    "Figure7_tahoe_recallk_curve_source.tsv",
    "Figure7_tahoe_rank_pair_source.tsv",
    "Figure7_tahoe_state_summary_source.tsv",
]

def load_contract() -> dict[str, object]:
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Missing v3 visual contract: {CONTRACT_PATH}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return contract

def configure_style(contract: dict[str, object]) -> dict[str, str]:
    palette = contract["palette"]
    colors = {
        "consensus": palette["MeTar"],
        "mean": palette["SSGCN"],
        "random": palette["Random"],
        "top5": palette["MeTar"],
        "top10": "#A8BFD5",
        "missed": "#D5DBE2",
        "ink": palette["text"],
        "muted": "#697887",
        "grid": palette["grid"],
        "threshold": "#8794A1",
        "diagonal": "#A6B5C4",
    }
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "text.color": colors["ink"],
            "axes.linewidth": 0.65,
            "axes.edgecolor": colors["ink"],
            "axes.labelcolor": colors["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.color": colors["ink"],
            "ytick.color": colors["ink"],
            "legend.frameon": False,
            "legend.labelcolor": colors["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return colors

def read_sources() -> dict[str, pd.DataFrame]:
    return {
        "lincs_hit": pd.read_csv(DATA / SOURCE_FILES[0], sep="\t"),
        "lincs_recall": pd.read_csv(DATA / SOURCE_FILES[1], sep="\t"),
        "lincs_pairs": pd.read_csv(DATA / SOURCE_FILES[2], sep="\t"),
        "lincs_states": pd.read_csv(DATA / SOURCE_FILES[3], sep="\t"),
        "tahoe_hit": pd.read_csv(DATA / SOURCE_FILES[4], sep="\t"),
        "tahoe_recall": pd.read_csv(DATA / SOURCE_FILES[5], sep="\t"),
        "tahoe_pairs": pd.read_csv(DATA / SOURCE_FILES[6], sep="\t"),
        "tahoe_states": pd.read_csv(DATA / SOURCE_FILES[7], sep="\t"),
    }

def style_axes(ax: plt.Axes, colors: dict[str, str]) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in (ax.spines["left"], ax.spines["bottom"]):
        spine.set_linewidth(0.65)
        spine.set_color(colors["ink"])

def plot_curve(ax: plt.Axes, data: pd.DataFrame, ylabel: str, ylim: tuple[float, float], top_label: str, colors: dict[str, str]) -> None:
    methods = ["random", "single_cell_mean", "multi_cell_consensus"]
    color_map = {"random": colors["random"], "single_cell_mean": colors["mean"], "multi_cell_consensus": colors["consensus"]}
    marker_map = {"random": None, "single_cell_mean": "o", "multi_cell_consensus": "s"}
    linestyle_map = {"random": (0, (3, 2)), "single_cell_mean": "solid", "multi_cell_consensus": "solid"}
    for method in methods:
        sub = data[data["method"].eq(method)].sort_values("threshold_percent")
        if sub.empty:
            continue
        x = sub["threshold_percent"].to_numpy(float)
        y = sub["value"].to_numpy(float) * 100.0
        yerr = sub["sd_across_seeds"].fillna(0).to_numpy(float) * 100.0 if "sd_across_seeds" in sub else None
        kwargs = {
            "color": color_map[method], "linestyle": linestyle_map[method], "marker": marker_map[method],
            "markersize": 3.5, "linewidth": 1.1, "zorder": 3 if method != "random" else 2,
        }
        if yerr is not None and np.any(yerr > 0):
            ax.errorbar(x, y, yerr=yerr, elinewidth=0.6, capsize=1.4, capthick=0.6, **kwargs)
        else:
            ax.plot(x, y, **kwargs)
    ax.axvline(MATCHED_PERCENT, color=colors["threshold"], linewidth=0.6, ymin=0.0, ymax=1.0, clip_on=True, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.12, 11.8)
    ax.set_ylim(*ylim)
    ax.set_xticks([0.15, 0.75, 1.5, 3.0, 7.5])
    ax.set_xticklabels(["0.15", "0.75", "1.5", "3.0", "7.5"])
    ax.set_yticks([0, 20, 40, 60, 80] if ylim[1] > 70 else [0, 15, 30, 45, 60])
    ax.set_xlabel("Rank fraction (%)", fontsize=8.5, fontweight="normal", labelpad=2.0)
    ax.set_ylabel(ylabel, fontsize=8.5, fontweight="normal", labelpad=1.8)
    ax.tick_params(axis="both", labelsize=8.0, pad=1.0, width=0.65, length=2.8)
    for text in ax.get_xticklabels():
        text.set_rotation(30)
        text.set_rotation_mode("anchor")
        text.set_ha("right")
        text.set_fontweight("normal")
    for text in ax.get_yticklabels():
        text.set_fontweight("normal")
    ax.grid(color=colors["grid"], linewidth=0.5, which="major")
    ax.set_axisbelow(True)
    style_axes(ax, colors)

def plot_rank_scatter(ax: plt.Axes, paired: pd.DataFrame, dataset: str, colors: dict[str, str]) -> None:
    if dataset == "LINCS":
        x_col, y_col, alpha, size = "cell_line_mean_percentile_from_top", "consensus_percentile_from_top", 0.16, 10.5
    else:
        x_col, y_col, alpha, size = "single_cell_mean_best_percentile_from_top", "consensus_best_percentile_from_top", 0.65, 27.0
    x = paired[x_col].astype(float).clip(lower=SCATTER_CLIP_FLOOR)
    y = paired[y_col].astype(float).clip(lower=SCATTER_CLIP_FLOOR)
    rng = np.random.default_rng(SCATTER_JITTER_SEEDS[dataset])
    x = x * np.exp(rng.normal(0.0, SCATTER_JITTER_LOG_SD, len(x)))
    y = y * np.exp(rng.normal(0.0, SCATTER_JITTER_LOG_SD, len(y)))
    ax.scatter(x, y, s=size, color=colors["consensus"], alpha=alpha, linewidth=0, zorder=3)
    lims = (0.13, 120.0)
    ax.plot(lims, lims, color=colors["diagonal"], linewidth=0.7, linestyle="--", zorder=1)
    ax.axvline(MATCHED_PERCENT, color=colors["threshold"], linewidth=0.6, clip_on=True, zorder=2)
    ax.axhline(MATCHED_PERCENT, color=colors["threshold"], linewidth=0.6, clip_on=True, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_box_aspect(1.0)
    ticks = [0.3, MATCHED_PERCENT, 3, 10, 30, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.3", "1.5", "3", "10", "30", "100"])
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0.3", "1.5", "3", "10", "30", "100"])
    ax.set_xlabel("Cell-line mean best-target percentile (%)", fontsize=8.5, fontweight="normal", labelpad=2.0)
    ax.set_ylabel("Consensus percentile (%)", fontsize=8.5, fontweight="normal", labelpad=1.8)
    ax.tick_params(axis="both", labelsize=8.0, pad=1.0, width=0.65, length=2.8)
    for text in ax.get_xticklabels():
        text.set_rotation(30)
        text.set_rotation_mode("anchor")
        text.set_ha("right")
        text.set_fontweight("normal")
    for text in ax.get_yticklabels():
        text.set_fontweight("normal")
    ax.grid(color=colors["grid"], linewidth=0.5, which="major")
    ax.set_axisbelow(True)
    style_axes(ax, colors)

def state_values(frame: pd.DataFrame) -> list[int]:
    if "group" in frame.columns:
        frame = frame[frame["group"].eq("consensus_rank")]
    values = frame.set_index("state")["count"].to_dict()
    return [int(values[s]) for s in ("top5", "top10_only", "missed_top10")]

def plot_state_bar(ax: plt.Axes, frame: pd.DataFrame, colors: dict[str, str]) -> dict[str, object]:
    values = state_values(frame)
    total = float(sum(values))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    bar_y, bar_h = 0.38, 0.40
    left = 0.0
    bar_colors = [colors["top5"], colors["top10"], colors["missed"]]
    count_geometry = []
    for value, color in zip(values, bar_colors):
        width = value / total
        ax.add_patch(Rectangle((left, bar_y), width, bar_h, transform=ax.transAxes, facecolor=color, edgecolor="white", linewidth=0.8, clip_on=False))
        center = left + width / 2.0
        ax.plot([center, center], [bar_y, 0.22], transform=ax.transAxes, color=colors["muted"], linewidth=0.45, clip_on=False)
        ax.text(center, 0.08, str(value), transform=ax.transAxes, ha="center", va="center", fontsize=8.0, fontweight="normal", color=colors["ink"])
        count_geometry.append({"value": value, "segment_width_fraction": width, "center_fraction": center, "placement": "external_below_with_short_leader", "font_size_pt": 8.0})
        left += width
    width_mm = ax.get_position().width * ax.figure.get_size_inches()[0] * 25.4
    height_mm = ax.get_position().height * ax.figure.get_size_inches()[1] * 25.4 * bar_h
    return {"values": values, "total": int(total), "container_width_mm": width_mm, "visible_height_mm": height_mm, "counts": count_geometry}

def add_method_legend(fig: plt.Figure, rect: list[float], colors: dict[str, str]) -> None:
    ax = fig.add_axes(rect, frameon=False)
    ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=colors["consensus"], marker="s", linewidth=1.1, markersize=3.7, label="Consensus"),
        Line2D([0], [0], color=colors["mean"], marker="o", linewidth=1.1, markersize=3.7, label="Cell-line mean"),
        Line2D([0], [0], color=colors["random"], linewidth=1.0, linestyle=(0, (3, 2)), label="Random"),
    ]
    legend = ax.legend(handles=handles, loc="center", ncol=3, fontsize=8.0, handlelength=1.35, handletextpad=0.4, columnspacing=1.0)
    for text in legend.get_texts():
        text.set_fontweight("normal")

def add_state_legend(fig: plt.Figure, rect: list[float], colors: dict[str, str]) -> None:
    ax = fig.add_axes(rect, frameon=False)
    ax.set_axis_off()
    handles = [Patch(facecolor=colors["top5"], edgecolor="none", label="Top 5%"), Patch(facecolor=colors["top10"], edgecolor="none", label="Top 5–10%"), Patch(facecolor=colors["missed"], edgecolor="none", label="Missed >10%")]
    legend = ax.legend(handles=handles, loc="center", ncol=3, fontsize=8.0, handlelength=0.9, handletextpad=0.3, columnspacing=0.8)
    for text in legend.get_texts():
        text.set_fontweight("normal")

def add_tag(fig: plt.Figure, x: float, y: float, letter: str, key: str, colors: dict[str, str]) -> None:
    fig.text(x, y, letter, ha="left", va="center", fontsize=10.5, fontweight="bold", color=colors["ink"])

def build_full_layout(sources: dict[str, pd.DataFrame], colors: dict[str, str]) -> tuple[plt.Figure, dict[str, object]]:
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, FULL_HEIGHT_MM / 25.4), dpi=DETAIL_DPI, facecolor="white")
    def mm_rect(x, bottom, width, height):
        return [x / WIDTH_MM, bottom / FULL_HEIGHT_MM, width / WIDTH_MM, height / FULL_HEIGHT_MM]
    axes = {
        "7a_hit": fig.add_axes(mm_rect(13.35, 90.57, 39.16, 38.25)),
        "7a_recall": fig.add_axes(mm_rect(58.74, 90.57, 39.16, 38.25)),
        "7b_hit": fig.add_axes(mm_rect(13.35, 24.52, 39.16, 38.25)),
        "7b_recall": fig.add_axes(mm_rect(58.74, 24.52, 39.16, 38.25)),
        "7c": fig.add_axes(mm_rect(102.35, 88.32, 65.86, 40.5)),
        "7c_state": fig.add_axes(mm_rect(102.35, 68.82, 65.86, 9.75)),
        "7d": fig.add_axes(mm_rect(102.35, 22.27, 65.86, 40.5)),
        "7d_state": fig.add_axes(mm_rect(102.35, 3.52, 65.86, 9.75)),
    }
    add_method_legend(fig, mm_rect(16.91, 130.43, 72.98, 5.472), colors)
    add_state_legend(fig, mm_rect(100.57, 130.43, 71.2, 5.472), colors)
    add_tag(fig, 0.047, 133.82 / FULL_HEIGHT_MM, "a", "", colors)
    add_tag(fig, 0.047, 67.77 / FULL_HEIGHT_MM, "b", "", colors)
    add_tag(fig, 0.530, 133.82 / FULL_HEIGHT_MM, "c", "", colors)
    add_tag(fig, 0.530, 67.77 / FULL_HEIGHT_MM, "d", "", colors)
    plot_curve(axes["7a_hit"], sources["lincs_hit"], "Hit@K (%)", (0, 82), "K", colors)
    plot_curve(axes["7a_recall"], sources["lincs_recall"], "Recall@K (%)", (0, 62), "K", colors)
    plot_curve(axes["7b_hit"], sources["tahoe_hit"], "Hit@K eq. (%)", (0, 82), "", colors)
    plot_curve(axes["7b_recall"], sources["tahoe_recall"], "Recall@K eq. (%)", (0, 82), "", colors)
    plot_rank_scatter(axes["7c"], sources["lincs_pairs"], "LINCS", colors)
    plot_rank_scatter(axes["7d"], sources["tahoe_pairs"], "Tahoe", colors)
    c_state = plot_state_bar(axes["7c_state"], sources["lincs_states"], colors)
    d_state = plot_state_bar(axes["7d_state"], sources["tahoe_states"], colors)
    export_height_mm = FULL_HEIGHT_MM - FULL_TOP_CROP_MM
    export_axes = {}
    for name, ax in axes.items():
        left, bottom, width, height = ax.get_position().bounds
        export_axes[name] = [float(left), float(bottom * FULL_HEIGHT_MM / export_height_mm), float(width), float(height * FULL_HEIGHT_MM / export_height_mm)]
    geometry = {"width_mm": WIDTH_MM, "height_mm": export_height_mm, "layout_height_mm": FULL_HEIGHT_MM, "top_crop_mm": FULL_TOP_CROP_MM, "scope": ["7a", "7b", "7c", "7d"], "axes": export_axes, "state_bars": {"7c": c_state, "7d": d_state}}
    return fig, geometry

def main() -> None:
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    colors = configure_style(load_contract())
    fig, _geometry = build_full_layout(read_sources(), colors)
    import pymupdf as fitz
    doc = fitz.open(stream=figure_pdf_bytes(fig), filetype="pdf")
    page = doc[0]
    spans = [s for b in page.get_text("dict")["blocks"] for line in b.get("lines", []) for s in line["spans"]]
    drawings = page.get_drawings()
    top = min([s["bbox"][1] for s in spans] + [d["rect"].y0 for d in drawings if not d["rect"].is_empty and d["rect"].width < page.rect.width * .99])
    crop = max(0, top - 1.75 * 72 / 25.4)
    with fitz.open() as out:
        final = out.new_page(width=page.rect.width, height=page.rect.height-crop)
        final.show_pdf_page(final.rect, doc, 0, clip=fitz.Rect(0, crop, page.rect.width, page.rect.height))
        out.save(ROOT / "Figure7.pdf")
        for dpi, name in [(300, "Figure7_178mm.png"), (600, "Figure7_600dpi.png")]:
            final.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False).save(PREVIEWS / name)
    doc.close()
    plt.close(fig)

def figure_pdf_bytes(fig) -> bytes:
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf")
    return buffer.getvalue()
