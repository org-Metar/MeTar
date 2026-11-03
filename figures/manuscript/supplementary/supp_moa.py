"""Render the selected manuscript panels from numerical source tables."""

from __future__ import annotations

import json

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent

SOURCE = Path(__file__).resolve().parent / "external_inputs/legacy"

WIDTH_MM = 178

TEXT = "#263238"

GRID = "#D9E0E3"

BLUE = "#356C86"

BLUE_LIGHT = "#8DB0BF"

TEAL = "#1F7A75"

GOLD = "#D8A15D"

GREY = "#A7AFB4"

ZERO = "#7C878D"

FOLD_CMAP = LinearSegmentedColormap.from_list("fold", ["#496FA6", "#FAFAF7", "#B9672A"])

PRIMARY = [
    "neurotransmitter_signaling",
    "kinase_signaling",
    "metabolic_or_inflammatory_enzyme",
    "nuclear_hormone_signaling",
    "dna_rna_chromatin",
    "ion_channel",
]

FAMILIES = [
    "kinase_signaling",
    "nuclear_hormone_receptor",
    "neurotransmitter_receptor",
    "ion_channel",
    "transporter",
    "dna_rna_chromatin",
    "cytoskeleton",
    "proteostasis",
]

LABELS = {
    "neurotransmitter_signaling": "Neurotransmitter\nsignaling",
    "kinase_signaling": "Kinase\nsignaling",
    "metabolic_or_inflammatory_enzyme": "Metabolic /\ninflammatory enzyme",
    "nuclear_hormone_signaling": "Nuclear-hormone\nsignaling",
    "dna_rna_chromatin": "DNA / RNA /\nchromatin",
    "ion_channel": "Ion channel",
    "multiple_classes": "Multiple MOA\nclasses",
    "unclassified_moa": "Unclassified\nMOA",
    "nuclear_hormone_receptor": "Nuclear hormone receptor",
    "neurotransmitter_receptor": "Neurotransmitter receptor",
    "transporter": "Transporter",
    "cytoskeleton": "Cytoskeleton",
    "proteostasis": "Proteostasis",
}

SHORT_MOA = {
    "neurotransmitter_signaling": "Neurotransmitter\nsignaling",
    "kinase_signaling": "Kinase\nsignaling",
    "metabolic_or_inflammatory_enzyme": "Metabolic /\ninflammatory",
    "nuclear_hormone_signaling": "Nuclear\nhormone",
    "dna_rna_chromatin": "DNA / RNA /\nchromatin",
    "ion_channel": "Ion\nchannel",
}

FAMILY_LABELS = {
    "kinase_signaling": "Kinase signaling",
    "nuclear_hormone_receptor": "Nuclear hormone receptor",
    "neurotransmitter_receptor": "Neurotransmitter receptor",
    "ion_channel": "Ion channel",
    "transporter": "Transporter",
    "dna_rna_chromatin": "DNA / RNA / chromatin",
    "cytoskeleton": "Cytoskeleton",
    "proteostasis": "Proteostasis",
}

INPUTS = {
    "BIO-053": SOURCE / "s11/fig2_category_descriptive.tsv",
    "BIO-014": SOURCE / "s11/moa_recovery_counts.tsv",
    "BIO-012": SOURCE / "s11/moa_recovery_adjusted_probabilities.tsv",
    "BIO-004": SOURCE / "s11/boundary_summary.json",
    "BIO-040": SOURCE / "s10/lincs_moa_gain_summary.tsv",
    "BIO-047": SOURCE / "s10/tahoe_moa_recovery_summary.tsv",
    "BIO-038": SOURCE / "s10/lincs_moa_adjusted_sensitivity.tsv",
    "BIO-039": SOURCE / "s10/lincs_moa_adjusted_sensitivity_summary.tsv",
    "BIO-003": SOURCE / "s11/annotated_target_family_pure_moa.tsv",
    "BIO-011": SOURCE / "s11/moa_family_heatmap.tsv",
}

def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.titlepad": 0,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.edgecolor": TEXT,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

def panel_header(ax: plt.Axes, letter: str, identity: str) -> None:
    ax.text(-0.02, 1.13, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10.5, fontweight="bold", color=TEXT, clip_on=False)
    if identity:
        ax.text(0.01, 1.13, identity, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.5, color=TEXT, clip_on=False)

def open_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
    ax.tick_params(length=2.5, width=0.65)

def export(fig: plt.Figure, number: int) -> dict:
    stem = HERE / f"Figure{number}"
    fig.canvas.draw()
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    return {"figure": number}


def figure8() -> dict:
    desc = pd.read_csv(INPUTS["BIO-053"], sep="\t")
    desc = desc[(desc.dimension == "moa_class") & (desc.scope == "all_drugs")].copy()
    if len(desc) != 10 or int(desc.n_drugs.sum()) != 494:
        raise ValueError("Figure 8a source must contain 10 exhaustive categories and 494 drugs")
    order_a = [
        "nuclear_hormone_signaling", "dna_rna_chromatin", "neurotransmitter_signaling",
        "kinase_signaling", "metabolic_or_inflammatory_enzyme", "ion_channel",
        "multiple_classes", "unclassified_moa",
    ]
    a = desc[desc.n_drugs >= 20].set_index("category").loc[order_a]
    if int(a.n_drugs.sum()) != 481:
        raise ValueError("Figure 8a plotted cohort must contain 481 drugs")
    overall = float(np.average(desc.MeTar_mean_hit10, weights=desc.n_drugs))

    counts = pd.read_csv(INPUTS["BIO-014"], sep="\t").set_index("moa_class").loc[PRIMARY]
    boundary = json.loads(INPUTS["BIO-004"].read_text(encoding="utf-8"))
    overall_clusters = int(boundary["omnibus"][0]["n_target_clusters"])
    if int(counts.n_drugs.sum()) != 358 or overall_clusters != 196:
        raise ValueError("Figure 8b cohort must contain 358 drugs in 196 clusters")
    adjusted = pd.read_csv(INPUTS["BIO-012"], sep="\t")

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 150 / 25.4))
    gs = fig.add_gridspec(2, 2, left=0.225, right=0.985, bottom=0.105, top=0.925,
                          wspace=0.78, hspace=0.48)
    axa, axb, axc, axd = [fig.add_subplot(gs[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]

    y = np.arange(len(a))
    means = 100 * a.MeTar_mean_hit10.to_numpy()
    lo = 100 * a.MeTar_ci95_low.to_numpy()
    hi = 100 * a.MeTar_ci95_high.to_numpy()
    axa.errorbar(means, y, xerr=np.vstack([means - lo, hi - means]), fmt="o", color=BLUE,
                 ecolor=BLUE_LIGHT, markersize=4.2, elinewidth=1.2, capsize=2.0, zorder=3)
    axa.axvline(overall * 100, color=ZERO, linestyle=(0, (3, 2)), linewidth=0.85)
    axa.set_yticks(y, [LABELS[c] for c in order_a])
    axa.invert_yaxis()
    axa.set_xlim(0, 100)
    axa.set_xlabel("Drug-equal annotated-target Hit@10 (%)")
    for pos, n in zip(y, a.n_drugs):
        axa.text(101.5, pos, str(int(n)), va="center", ha="left", fontsize=8, clip_on=False)
    axa.text(101.5, -0.8, "n", va="bottom", ha="left", fontsize=7.5, clip_on=False)
    axa.text(overall * 100 + 1.0, 1.01, f"all 494: {overall * 100:.1f}%", transform=axa.get_xaxis_transform(),
             fontsize=7.5, va="bottom", ha="left", color=ZERO)
    open_axis(axa)
    panel_header(axa, "a", "")

    yb = np.arange(len(PRIMARY))
    left = np.zeros(len(PRIMARY))
    states = [
        ("no_hit_fraction", GREY, "No recovery"),
        ("minority_hit_fraction", GOLD, "<50% contexts"),
        ("majority_hit_fraction", TEAL, "≥50% contexts"),
    ]
    for col, color, label in states:
        vals = counts[col].to_numpy()
        axb.barh(yb, vals, left=left, height=0.62, color=color, label=label)
        left += vals
    axb.set_yticks(yb, [LABELS[c] for c in PRIMARY])
    axb.invert_yaxis()
    axb.set_xlim(0, 1.12)
    axb.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    axb.set_xlabel("Compounds (%)")
    for pos, n in zip(yb, counts.n_drugs):
        axb.text(1.035, pos, str(int(n)), va="center", ha="left", fontsize=8)
    axb.text(1.035, -0.8, "n", va="bottom", ha="left", fontsize=7.5)
    legend_labels = ["None", "<50%", "≥50%"]
    handles = [Patch(facecolor=color, edgecolor="none", label=short)
               for (_, color, _), short in zip(states, legend_labels)]
    axb.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.005), ncol=3,
               handlelength=0.9, columnspacing=0.65, borderaxespad=0, fontsize=7.5)
    open_axis(axb)
    panel_header(axb, "b", "")

    for ax, endpoint, letter, identity in [
        (axc, "no_hit", "c", "No recovery"),
        (axd, "majority_hit", "d", "Recovery in ≥50% contexts"),
    ]:
        part = adjusted[(adjusted.scope == "all_drugs") & (adjusted.endpoint == endpoint)].set_index("moa_class").loc[PRIMARY]
        val = part.adjusted_probability.to_numpy()
        low = part.ci95_low.to_numpy()
        high = part.ci95_high.to_numpy()
        ax.errorbar(val, yb, xerr=np.vstack([val - low, high - val]), fmt="o", color=BLUE,
                    ecolor=BLUE_LIGHT, markersize=4.2, elinewidth=1.2, capsize=2.0, zorder=3)
        ax.set_yticks(yb, [LABELS[c] for c in PRIMARY])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1.0"])
        ax.set_xlabel("Model-standardized probability")
        open_axis(ax)
        panel_header(ax, letter, identity)

    info = export(fig, 8)
    info["scientific_checks"] = {
        "panel_a": {"plotted_categories": 8, "plotted_drugs": 481, "all_drugs": 494,
                    "overall_percent": round(overall * 100, 6), "bootstrap_draws": 10000},
        "panels_b_d": {"drugs": 358, "target_set_clusters": 196, "categories": 6,
                       "adjusted_rows_used": 12},
    }
    return info

def figure10() -> dict:
    gain = pd.read_csv(INPUTS["BIO-040"], sep="\t")
    gain_plot = gain[gain.n_drugs >= 10].copy()
    if int(gain.n_drugs.sum()) != 415 or int(gain_plot.n_drugs.sum()) != 405 or len(gain_plot) != 8:
        raise ValueError("Figure 10a must preserve 415 total and 405 plotted drugs")
    tahoe = pd.read_csv(INPUTS["BIO-047"], sep="\t")
    if int(tahoe.n_total.sum()) != 30:
        raise ValueError("Figure 10b must preserve 30 Tahoe compounds")
    coef = pd.read_csv(INPUTS["BIO-038"], sep="\t")
    coef = coef[coef.term.str.startswith("C(moa_class)")].copy()
    summary = pd.read_csv(INPUTS["BIO-039"], sep="\t").iloc[0]
    if int(summary.n_drugs) != 405 or len(coef) != 7:
        raise ValueError("Figure 10c must preserve seven contrasts from 405 drugs")

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 148 / 25.4))
    gs = fig.add_gridspec(2, 2, left=0.225, right=0.90, bottom=0.095, top=0.925,
                          width_ratios=[1.03, 1.0], height_ratios=[1.22, 0.83],
                          wspace=0.75, hspace=0.46)
    axa = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, :])

    y = np.arange(len(gain_plot))
    mean = gain_plot.mean_consensus_gain_pp.to_numpy()
    low = gain_plot.bootstrap95_low.to_numpy()
    high = gain_plot.bootstrap95_high.to_numpy()
    axa.errorbar(mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="o", color=BLUE,
                 ecolor=BLUE_LIGHT, markersize=4.2, elinewidth=1.2, capsize=2.0, zorder=3)
    axa.axvline(0, color=ZERO, linewidth=0.8)
    axa.set_yticks(y, [f"{LABELS[c]} ({int(n)})" for c, n in zip(gain_plot.moa_class, gain_plot.n_drugs)])
    axa.invert_yaxis()
    axa.set_xlabel("Consensus gain (percentage points)")
    open_axis(axa)
    panel_header(axa, "a", "")

    coef = coef.sort_values("coefficient_pp")
    yc = np.arange(len(coef))
    cv = coef.coefficient_pp.to_numpy()
    cl = coef.ci95_low.to_numpy()
    ch = coef.ci95_high.to_numpy()
    axc.errorbar(cv, yc, xerr=np.vstack([cv - cl, ch - cv]), fmt="o", color=TEAL,
                 ecolor="#8CB8B4", markersize=4.2, elinewidth=1.2, capsize=2.0, zorder=3)
    axc.axvline(0, color=ZERO, linewidth=0.8)
    axc.set_yticks(yc, [LABELS[c] for c in coef.label])
    axc.set_xlabel("Adjusted coefficient (percentage points)\nreference: DNA / RNA / chromatin")
    open_axis(axc)
    panel_header(axc, "c", "")

    columns = ["shared_all_three", "partial", "missed_all_three"]
    matrix = tahoe[columns].to_numpy(dtype=float)
    props = matrix / tahoe.n_total.to_numpy()[:, None]
    cmap = LinearSegmentedColormap.from_list("tahoe", ["#FFFFFF", "#D8E8EC", "#2F7087"])
    im = axb.imshow(props, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="none")
    axb.set_yticks(range(len(tahoe)), [LABELS[c] for c in tahoe.moa_class])
    axb.set_xticks(range(3), ["Recovered in all\nthree cell lines", "Recovered in one\nor two cell lines", "No recovery in\nany cell line"])
    axb.tick_params(length=0)
    axb.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    axb.set_yticks(np.arange(-0.5, len(tahoe), 1), minor=True)
    axb.grid(which="minor", color="white", linewidth=1.1)
    axb.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axb.text(j, i, f"{int(matrix[i, j])}  ({props[i, j] * 100:.0f}%)", ha="center", va="center",
                     fontsize=8, color="white" if props[i, j] > 0.58 else TEXT)
    cax = fig.add_axes([0.925, 0.095, 0.012, 0.255])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.set_label("Within-category proportion", fontsize=8.5)
    cb.ax.tick_params(labelsize=8, length=2)
    panel_header(axb, "b", "Tahoe recovery state")

    info = export(fig, 10)
    info["scientific_checks"] = {
        "panel_a": {"all_drugs": 415, "plotted_drugs": 405, "categories": 8,
                    "bootstrap_draws": 5000},
        "panel_b": {"compounds": 30, "state_totals": {"all_three": 17, "one_or_two": 8, "none": 5}},
        "panel_c": {"drugs": 405, "moa_contrasts": 7, "interval": "HC3 95%"},
    }
    return info

def family_matrix(frame: pd.DataFrame, value: str) -> np.ndarray:
    return (
        frame.pivot(index="family", columns="moa_class", values=value)
        .reindex(index=FAMILIES, columns=PRIMARY)
        .to_numpy(dtype=float)
    )

def style_heatmap(ax: plt.Axes) -> None:
    ax.set_yticks(range(len(FAMILIES)), [FAMILY_LABELS[f] for f in FAMILIES])
    ax.set_xticks(range(len(PRIMARY)), [SHORT_MOA[c] for c in PRIMARY])
    ax.tick_params(length=0, pad=4)
    ax.set_xticks(np.arange(-0.5, len(PRIMARY), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FAMILIES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

def figure13() -> dict:
    ref = pd.read_csv(INPUTS["BIO-003"], sep="\t")
    heat = pd.read_csv(INPUTS["BIO-011"], sep="\t")
    ref = ref[ref.moa_class.isin(PRIMARY)]
    retained = heat[(heat.moa_class.isin(PRIMARY)) & (heat.condition == "targets_retained")]
    removed = heat[(heat.moa_class.isin(PRIMARY)) & (heat.condition == "targets_removed")]
    if not (len(ref) == len(retained) == len(removed) == 48):
        raise ValueError("Figure 13 must contain 48 cells per panel")
    if retained.pure_fold_over_random_candidates.isna().any() or removed.pure_fold_over_random_candidates.isna().any():
        raise ValueError("Figure 13 enrichment values must be complete")
    counts = retained.drop_duplicates("moa_class").set_index("moa_class")
    if int(counts.loc[PRIMARY, "n_pure_drugs"].sum()) != 300 or int(counts.loc[PRIMARY, "n_pure_target_clusters"].sum()) != 181:
        raise ValueError("Figure 13 cohort must contain 300 drugs in 181 clusters")
    if int(retained.q_bh_all_family_cells.notna().sum() + removed.q_bh_all_family_cells.notna().sum()) != 96:
        raise ValueError("Figure 13 must preserve 96 corrected tests")

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 184 / 25.4))
    gs = fig.add_gridspec(3, 1, left=0.285, right=0.885, bottom=0.105, top=0.91, hspace=0.50)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    m0 = family_matrix(ref, "annotated_target_fraction")
    m1 = np.log2(family_matrix(retained, "pure_fold_over_random_candidates"))
    m2 = np.log2(family_matrix(removed, "pure_fold_over_random_candidates"))
    im0 = axes[0].imshow(m0, vmin=0, vmax=1, cmap="Greys", aspect="auto", interpolation="none")
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    im1 = axes[1].imshow(np.clip(m1, -3, 3), norm=norm, cmap=FOLD_CMAP, aspect="auto", interpolation="none")
    axes[2].imshow(np.clip(m2, -3, 3), norm=norm, cmap=FOLD_CMAP, aspect="auto", interpolation="none")
    for ax in axes:
        style_heatmap(ax)

    for ax, frame, matrix in [(axes[1], retained, m1), (axes[2], removed, m2)]:
        indexed = frame.set_index(["family", "moa_class"])
        for i, family in enumerate(FAMILIES):
            for j, category in enumerate(PRIMARY):
                q = indexed.loc[(family, category), "q_bh_all_family_cells"]
                if pd.notna(q) and q < 0.05:
                    color = "white" if abs(matrix[i, j]) > 1.35 else TEXT
                    ax.text(j, i, "•", ha="center", va="center", fontsize=11, color=color)

    panel_header(axes[0], "a", "")
    panel_header(axes[1], "b", "Annotated targets retained")
    panel_header(axes[2], "c", "Annotated targets removed")
    for j, category in enumerate(PRIMARY):
        axes[0].text(j, -0.86, f"{int(counts.loc[category, 'n_pure_drugs'])} / {int(counts.loc[category, 'n_pure_target_clusters'])}",
                     ha="center", va="bottom", fontsize=7.5, color=TEXT, clip_on=False)
    axes[0].text(-1.15, -0.86, "drugs /\nclusters", ha="right", va="bottom", fontsize=7.5, clip_on=False)

    cax0 = fig.add_axes([0.915, 0.68, 0.015, 0.205])
    cb0 = fig.colorbar(im0, cax=cax0, ticks=[0, 0.5, 1])
    cb0.set_label("Annotated-target fraction", fontsize=8.5)
    cb0.ax.tick_params(labelsize=8, length=2)
    cax1 = fig.add_axes([0.915, 0.13, 0.015, 0.43])
    cb1 = fig.colorbar(im1, cax=cax1, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb1.ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    cb1.set_label("Top-ten fraction / eligible-candidate fraction", fontsize=8.5)
    cb1.ax.tick_params(labelsize=8, length=2)

    info = export(fig, 13)
    info["scientific_checks"] = {
        "matrix_shape_each_panel": [8, 6],
        "values": {"annotated": 48, "retained": 48, "removed": 48},
        "tests_retained_plus_removed": 96,
        "permutations": 1999,
        "drugs": 300,
        "target_set_clusters": 181,
        "enrichment_clip_log2": [-3, 3],
    }
    return info
