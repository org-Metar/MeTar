"""Render the selected manuscript panels from numerical source tables."""

from __future__ import annotations

import json

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

DATA = Path(__file__).resolve().parent / "external_inputs/biology"

OUT = Path(__file__).resolve().parent / "rendered"

INK = "#263238"

TEAL = "#197C78"

BLUE = "#4776A8"

AMBER = "#B66B2D"

GRAY = "#7C858B"

LIGHT_GRAY = "#A4ABB1"

WHITE = "#FFFFFF"

PRIMARY = (
    "neurotransmitter_signaling",
    "kinase_signaling",
    "metabolic_or_inflammatory_enzyme",
    "nuclear_hormone_signaling",
    "dna_rna_chromatin",
    "ion_channel",
)

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

def add_interval(ax: plt.Axes, y: float, center: float, lo: float, hi: float, color: str, marker: str = "o") -> None:
    ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round", zorder=2)
    ax.scatter(center, y, s=25, color=color, marker=marker, edgecolor=WHITE, linewidth=0.45, zorder=3)

def render_figure11() -> None:
    s11 = DATA / "s11"
    summary = json.loads((s11 / "functional_concordance_summary.json").read_text(encoding="utf-8"))
    boot_all = pd.read_csv(s11 / "cluster_bootstrap.tsv", sep="\t")["cluster_bootstrap_mean_auc"]
    boot_low = pd.read_csv(s11 / "missed_cluster_bootstrap.tsv", sep="\t")["missed_cluster_bootstrap_mean_auc"]
    nulls = pd.read_csv(s11 / "label_permutation_null.tsv", sep="\t")
    gene_nulls = pd.read_csv(s11 / "matched_gene_null.tsv", sep="\t")
    strata = pd.read_csv(s11 / "moa_recovery_reactome_concordance.tsv", sep="\t")

    fig, axes = plt.subplots(2, 2, figsize=(178 / 25.4, 175 / 25.4), gridspec_kw={"width_ratios": [0.92, 1.08]})
    fig.subplots_adjust(left=0.22, right=0.965, top=0.95, bottom=0.10, hspace=0.50, wspace=1.05)
    a, b, c, d = axes.flat

    a_items = [
        ("Low recovery\n283 drugs / 196 clusters", summary["missed_partial_cluster_balanced_mean_auc"], boot_low.quantile(0.025), boot_low.quantile(0.975), TEAL),
        ("All eligible\n443 drugs / 255 clusters", summary["cluster_balanced_mean_auc"], boot_all.quantile(0.025), boot_all.quantile(0.975), INK),
    ]
    for y, (_, center, lo, hi, color) in enumerate(a_items):
        add_interval(a, y, center, lo, hi, color)
    a.axvline(0.5, color=LIGHT_GRAY, ls=(0, (3, 2)), lw=0.8)
    a.set_yticks([0, 1], ["Low recovery\n283 drugs /\n196 clusters", "All eligible\n443 drugs /\n255 clusters"])
    a.set_xlim(0.49, 0.56)
    a.set_ylim(-0.55, 1.55)
    a.set_xlabel("Functional rank AUC")
    panel_letter(a, "a", x=-0.45)
    a.text(0, 1.03, "Known targets removed", transform=a.transAxes, fontsize=8.5, va="bottom")

    def interval(series: pd.Series) -> tuple[float, float, float]:
        return float(series.mean()), float(series.quantile(0.025)), float(series.quantile(0.975))

    b_items = [
        ("Observed low recovery", summary["missed_partial_cluster_balanced_mean_auc"], float(boot_low.quantile(0.025)), float(boot_low.quantile(0.975)), TEAL, "o"),
        ("Target-set label permutation\n1,999 accepted permutations", *interval(nulls["missed_cluster_label_permutation_mean_auc"]), BLUE, "s"),
        ("Matched genes: annotation + PPI\n499 draws", *interval(gene_nulls["missed_cluster_matched_gene_null_mean_auc"]), AMBER, "s"),
        ("Matched genes + global rank\n499 draws", *interval(gene_nulls["missed_rank_aware_cluster_matched_gene_null_mean_auc"]), GRAY, "s"),
    ]
    for y, item in enumerate(b_items):
        _, center, lo, hi, color, marker = item
        add_interval(b, y, center, lo, hi, color, marker)
    b.axvline(0.5, color=LIGHT_GRAY, ls=(0, (3, 2)), lw=0.8)
    b.axvline(summary["missed_partial_cluster_balanced_mean_auc"], color=TEAL, ls=(0, (2, 2)), lw=0.7)
    b.set_yticks(np.arange(4), [
        "Observed low\nrecovery",
        "Target-set label\npermutation\n1,999 accepted permutations",
        "Matched genes:\nannotation + PPI\n499 draws",
        "Matched genes + global\nrank\n499 draws",
    ])
    b.invert_yaxis()
    b.set_xlim(0.49, 0.56)
    b.set_xlabel("Functional rank AUC")
    panel_letter(b, "b", x=-0.06)
    b.text(0, 1.03, "Low recovery (Hit@10 < 0.50)", transform=b.transAxes, fontsize=8.5, va="bottom")

    pooled = strata[strata.moa_class.eq("all_moa")].set_index("recovery_group")
    groups = ("no_hit", "minority_hit", "majority_hit")
    names = {"no_hit": "No recovery", "minority_hit": "Recovery in <50%", "majority_hit": "Recovery in ≥50%"}
    for y, group in enumerate(groups):
        row = pooled.loc[group]
        add_interval(c, y, row.mean_functional_auc, row.ci95_low, row.ci95_high, TEAL)
    c.axvline(0.5, color=LIGHT_GRAY, ls=(0, (3, 2)), lw=0.8)
    c.set_yticks(np.arange(3), [
        f"No recovery\n{int(pooled.loc['no_hit'].n_drugs)} drugs /\n{int(pooled.loc['no_hit'].n_target_clusters)} clusters",
        f"Recovery in <50%\n{int(pooled.loc['minority_hit'].n_drugs)} drugs /\n{int(pooled.loc['minority_hit'].n_target_clusters)} clusters",
        f"Recovery in ≥50%\n{int(pooled.loc['majority_hit'].n_drugs)} drugs /\n{int(pooled.loc['majority_hit'].n_target_clusters)} clusters",
    ])
    c.invert_yaxis()
    c.set_xlim(0.49, 0.59)
    c.set_xlabel("Functional rank AUC")
    panel_letter(c, "c", x=-0.45)

    subset = strata[strata.moa_class.isin(PRIMARY) & strata.recovery_group.eq("no_hit")].set_index("moa_class")
    for y, category in enumerate(PRIMARY):
        row = subset.loc[category]
        color = TEAL if row.q_bh_moa_recovery_tests < 0.05 else GRAY
        add_interval(d, y, row.mean_functional_auc, row.ci95_low, row.ci95_high, color)
    d.axvline(0.5, color=LIGHT_GRAY, ls=(0, (3, 2)), lw=0.8)
    d.set_yticks(np.arange(6), [
        f"Neurotransmitter\nsignaling\n{int(subset.loc['neurotransmitter_signaling'].n_drugs)} drugs / {int(subset.loc['neurotransmitter_signaling'].n_target_clusters)} clusters",
        f"Kinase signaling\n{int(subset.loc['kinase_signaling'].n_drugs)} drugs / {int(subset.loc['kinase_signaling'].n_target_clusters)} clusters",
        f"Metabolic/inflammatory\nenzymes\n{int(subset.loc['metabolic_or_inflammatory_enzyme'].n_drugs)} drugs / {int(subset.loc['metabolic_or_inflammatory_enzyme'].n_target_clusters)} clusters",
        f"Nuclear-hormone\nsignaling\n{int(subset.loc['nuclear_hormone_signaling'].n_drugs)} drugs / {int(subset.loc['nuclear_hormone_signaling'].n_target_clusters)} clusters",
        f"DNA/RNA/chromatin\n{int(subset.loc['dna_rna_chromatin'].n_drugs)} drugs / {int(subset.loc['dna_rna_chromatin'].n_target_clusters)} clusters",
        f"Ion channels\n{int(subset.loc['ion_channel'].n_drugs)} drugs / {int(subset.loc['ion_channel'].n_target_clusters)} clusters",
    ])
    d.invert_yaxis()
    d.set_xlim(0.40, 0.65)
    d.set_xlabel("Functional rank AUC")
    panel_letter(d, "d", x=-0.06)
    d.text(0, 1.03, "Strict no-hit (184 drugs / 141 clusters)", transform=d.transAxes, fontsize=8.5, va="bottom")
    export_single(fig, "Supplementary_Figure11")
