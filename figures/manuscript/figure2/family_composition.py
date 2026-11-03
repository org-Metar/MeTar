from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from statistical_panel_style import FAMILIES, FAMILY_LABELS, MOAS, MOA_LABELS
from panel_layout import ROW_BOTTOMS, TEXT, axis_mm, fig_text, panel_letter


def draw_g(fig: Figure, ratios: np.ndarray, q_values: np.ndarray) -> dict:
    bottom = ROW_BOTTOMS["g"] - 1.1
    grid = {"x": 33.0, "y": bottom + 13.8, "width": 132.0, "height": 48.0}
    colorbar_box = {"x": 166.3, "y": bottom + 16.0, "width": 1.9, "height": 43.6}
    cmap = LinearSegmentedColormap.from_list(
        "fold_ratio_v3", ["#496FA6", "#FAFAF7", "#B9672A"])
    norm = Normalize(vmin=-3, vmax=3, clip=False)
    log2_values = np.log2(ratios)
    displayed = np.clip(log2_values, -3, 3)
    ax = axis_mm(fig, grid["x"], grid["y"], grid["width"], grid["height"])
    image = ax.imshow(displayed.T, cmap=cmap, norm=norm, aspect="auto",
                      interpolation="nearest", origin="upper")
    ax.set_xlim(-0.5, len(MOAS) - 0.5)
    ax.set_ylim(len(FAMILIES) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(MOAS)))
    ax.set_yticks(np.arange(len(FAMILIES)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(-0.5, len(MOAS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FAMILIES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="both", bottom=False, left=False, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cell_texts = []
    significance = []
    for family_index in range(len(FAMILIES)):
        for moa_index in range(len(MOAS)):
            ratio = float(ratios[moa_index, family_index])
            rgb = cmap(norm(float(np.clip(np.log2(ratio), -3, 3))))[:3]
            luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            color = TEXT if luminance > 0.59 else "#FFFFFF"
            cell_texts.append(ax.text(
                moa_index, family_index, f"{ratio:.2f}", ha="center", va="center",
                fontsize=8.0, fontweight="normal", color=color, fontfamily="Arial"))
            if float(q_values[moa_index, family_index]) < 0.05:
                significance.append(ax.scatter(
                    [moa_index + 0.34], [family_index - 0.32], s=12, marker="o",
                    color=TEXT, edgecolors="white", linewidths=0.25, zorder=5))

    family_texts = []
    for index, family in enumerate(FAMILIES):
        center_y = grid["y"] + grid["height"] * (
            1 - (index + 0.5) / len(FAMILIES))
        family_texts.append(fig_text(
            fig, 30.7, center_y, FAMILY_LABELS[family], ha="right", va="center",
            fontsize=8.0, fontweight="normal", linespacing=1.0))
    moa_texts = []
    cell_width = grid["width"] / len(MOAS)
    for index, moa in enumerate(MOAS):
        center_x = grid["x"] + cell_width * (index + 0.5)
        moa_texts.append(fig_text(
            fig, center_x, bottom + 12.9, MOA_LABELS[moa], ha="center", va="top",
            fontsize=8.0, fontweight="normal", linespacing=1.0))
    family_axis_label = fig_text(
        fig, 3.5, bottom + 65.0, "Target family", ha="left", va="center",
        fontsize=8.5, fontweight="normal")
    moa_axis_label = fig_text(
        fig, grid["x"] + grid["width"] / 2, bottom + 65.0, "MOA class",
        ha="center", va="center", fontsize=8.5, fontweight="normal")
    letter = panel_letter(fig, "g", 3.5, bottom + 69.6)

    cax = axis_mm(fig, colorbar_box["x"], colorbar_box["y"],
                  colorbar_box["width"], colorbar_box["height"])
    colorbar = fig.colorbar(image, cax=cax, ticks=[-3, 0, 3])
    colorbar.ax.set_yticklabels(["1/8×", "1×", "8×"])
    colorbar.ax.tick_params(labelsize=8.0, length=2.0, width=0.6,
                            pad=1.0, colors=TEXT)
    colorbar.outline.set_visible(False)
    for spine in cax.spines.values():
        spine.set_visible(False)
    for label in cax.get_yticklabels():
        label.set_fontfamily("Arial")
        label.set_fontweight("normal")
        label.set_color(TEXT)
    return {
        "axes": [ax, cax],
        "artists": [letter, family_axis_label, moa_axis_label,
                    *family_texts, *moa_texts],
        "cell_texts": cell_texts,
        "significance_artists": significance,
        "geometry_mm": {
            "grid": grid, "colorbar": colorbar_box, "rows": 8, "columns": 6,
            "cell_width": cell_width, "cell_height": grid["height"] / len(FAMILIES),
            "display_transpose_only": True, "value_font_pt": 8.0,
            "significant_value_font_weight": "normal",
            "significance_markers": len(significance), "log2_range": [-3, 3],
        },
        "displayed_log2_values": displayed,
    }
