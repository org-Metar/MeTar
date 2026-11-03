"""Render the selected manuscript panels from numerical source tables."""

from __future__ import annotations

import json

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent / "rendered"

FIG = OUT / "figures"

SOURCE = Path(__file__).resolve().parent / "external_inputs/legacy"

METAR = Path(__file__).resolve().parent

FIG7_GENE = SOURCE / "recovered_originals/fig2_single_target_gene_scores.tsv"

FIG7_TERM = SOURCE / "recovered_originals/fig2_pathway_continuous_rank.tsv"

FIG3_BOOTSTRAP = SOURCE / "recovered_originals/Figure3_cluster_bootstrap_curve.tsv"

FIG3_BOOTSTRAP_TOP5 = SOURCE / "recovered_originals/Figure3_cluster_bootstrap_top5.tsv"

MM = 1 / 25.4

INK = "#263238"

BLUE = "#2D6F9F"

BLUE_LIGHT = "#91B9D8"

ORANGE = "#C8753D"

RANDOM = "#949CA6"

LIGHT = "#D9E0E7"

GRID = "#E6EBEF"

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

def figure1() -> None:
    enrich = read_legacy(1, "S4-L002")
    summary = read_legacy(1, "S4-L003").set_index("target_family")
    pairs = read_legacy(1, "S4-L004")
    selected = read_legacy(1, "S4-L005")
    order = selected.groupby("primary_target", sort=False)["plot_order"].min().sort_values().index.tolist()
    fig = plt.figure(figsize=(178 * MM, 128 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.72, 1.48], wspace=0.34,
                          left=0.075, right=0.985, bottom=0.12, top=0.89)
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", -0.20, 1.02)
    x = np.arange(len(enrich))
    obs = enrich["observed_any_shared_target_neighbor_rate"].to_numpy() * 100
    se = enrich["observed_se"].to_numpy() * 100
    rnd = enrich["permutation_random_rate_mean"].to_numpy() * 100
    ax.fill_between(x, obs - 1.96 * se, obs + 1.96 * se, color=BLUE_LIGHT, alpha=0.35, lw=0)
    ax.plot(x, obs, marker="o", ms=4, lw=1.5, color=BLUE, label="MeTar nearest neighbors")
    ax.plot(x, rnd, marker="o", ms=3.5, lw=1.2, color=RANDOM, label="Random neighbors")
    ax.set_xticks(x, enrich["k"].astype(str))
    ax.set_xlabel("Nearest neighbors (k)")
    ax.set_ylabel("Drugs with a shared-target neighbor (%)")
    ax.set_ylim(0, max(60, (obs + 1.96 * se).max() + 3))
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", -0.10, 1.02)
    rng = np.random.default_rng(20260612)
    positions = np.arange(len(order))
    for yi, fam in enumerate(order):
        involves = pairs.target_a.eq(fam) | pairs.target_b.eq(fam)
        same = pairs.loc[involves & pairs.same_target_family, "cosine_similarity"].to_numpy()
        diff = pairs.loc[involves & (~pairs.same_target_family), "cosine_similarity"].to_numpy()
        for vals, off, color in [(diff, -0.14, "#C9D1D9"), (same, 0.14, BLUE)]:
            q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
            lo, hi = np.quantile(vals, [0.05, 0.95])
            ax.plot([lo, hi], [yi + off] * 2, color=color, lw=0.8)
            ax.add_patch(plt.Rectangle((q1, yi + off - 0.09), q3 - q1, 0.18,
                                       facecolor=color, edgecolor=INK, lw=0.35))
            ax.plot([med, med], [yi + off - 0.09, yi + off + 0.09], color=INK, lw=0.55)
            if len(vals) <= 40:
                jitter = rng.uniform(-0.05, 0.05, len(vals))
                ax.scatter(vals, yi + off + jitter, s=3.5, color=color, alpha=0.55, edgecolors="none")
        n_same = int(summary.loc[fam, "n_within_pairs"])
        n_diff = int(summary.loc[fam, "n_between_pairs"])
        ax.text(1.025, yi, f"{n_same}/{n_diff}", ha="left", va="center", fontsize=7.5)
    ax.set_yticks(positions, order)
    ax.invert_yaxis()
    ax.set_xlim(-0.3, 1.38)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Single annotated target")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.text(1.025, -0.85, "Same/different\npairs", ha="left", va="bottom", fontsize=7.5)
    fig.legend(handles=[Patch(facecolor=BLUE, label="Same target"),
                        Patch(facecolor="#C9D1D9", label="Different target")],
               loc="upper right", bbox_to_anchor=(0.98, 0.995), ncol=2,
               handlelength=1.2, columnspacing=1.2)
    save_vector(fig, 1)

def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values[np.isfinite(values)])
    return x, np.arange(1, len(x) + 1) / len(x)

def figure2() -> None:
    same = read_legacy(2, "S4-L011")
    ctrl = read_legacy(2, "S4-L010")
    overlap = read_legacy(2, "S4-L013")
    states = read_legacy(2, "S4-L014")
    fig = plt.figure(figsize=(178 * MM, 105 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], wspace=0.40, hspace=0.55,
                          left=0.09, right=0.985, bottom=0.19, top=0.88)
    ax = fig.add_subplot(gs[0, :])
    panel(ax, "a", -0.07, 1.04)
    specs = [
        (same.score_cosine.to_numpy(), BLUE, "-", "Direct score, same drug"),
        (ctrl.score_cosine.to_numpy(), BLUE_LIGHT, "-", "Direct score, control"),
        (same.standardized_expression_cosine.to_numpy(), "#506B86", "--", "Input, same drug"),
        (ctrl.standardized_expression_cosine.to_numpy(), "#BCC8D4", "--", "Input, control"),
    ]
    for vals, color, ls, label in specs:
        xx, yy = ecdf(vals)
        ax.plot(xx, yy, color=color, ls=ls, lw=1.35, label=label)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, lw=0.55)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.54, 0.985),
               handlelength=2.1, columnspacing=1.4)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "b", -0.18, 1.06)
    k = sorted(overlap.k.unique())
    for comp, color, label in [("same_drug_cross_cell", BLUE, "Same drug"),
                               ("different_drug_matched_control", RANDOM, "Matched control")]:
        sub = overlap.loc[overlap.comparison.eq(comp)].set_index("k").loc[k]
        ax.errorbar(k, sub.mean_jaccard, yerr=sub.sem_jaccard, marker="s", ms=4,
                    lw=1.25, capsize=2, color=color, label=label)
    ax.set_xticks(k)
    ax.set_xlabel("Top-k targets")
    ax.set_ylabel("Jaccard overlap")
    ax.grid(axis="y", color=GRID, lw=0.55)
    legend_b = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.075),
                         ncol=2, borderaxespad=0.0, handlelength=1.7,
                         columnspacing=1.2)
    fig.canvas.draw()
    axis_box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    legend_box = legend_b.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "c", -0.15, 1.06)
    state_order = ["both recovered", "one context only", "neither recovered"]
    colors = [BLUE, ORANGE, LIGHT]
    for yi, threshold in enumerate(["Hit@10", "Hit@30"]):
        sub = states.loc[states.threshold.eq(threshold)].set_index("state")
        left = 0.0
        for s, color in zip(state_order, colors):
            val = float(sub.loc[s, "fraction"])
            ax.barh(yi, val, left=left, height=0.48, color=color, edgecolor="white", lw=0.5)
            ax.text(left + val / 2, yi, f"{val:.1%}", ha="center", va="center",
                    fontsize=7.5, color="white" if color != LIGHT else INK)
            left += val
    ax.set_yticks([0, 1], ["Hit@10", "Hit@30"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of same-drug cross-cell pairs")
    fig.legend(handles=[Patch(color=c, label=s) for c, s in zip(colors, state_order)],
               ncol=3, loc="lower center", bbox_to_anchor=(0.73, 0.025), fontsize=7.5,
               handlelength=1.2, columnspacing=1.4)
    save_vector(fig, 2)

def figure3() -> None:
    data = pd.read_csv(FIG3_BOOTSTRAP, sep="\t")
    endpoint = pd.read_csv(FIG3_BOOTSTRAP_TOP5, sep="\t")
    assert set(data.bootstrap_replicates) == {20000}
    assert set(data.bootstrap_seed) == {20260713}
    assert set(data.n_drug_clusters) == {177}
    fig = plt.figure(figsize=(178 * MM, 76 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 0.72], wspace=0.40,
                          left=0.09, right=0.98, bottom=0.19, top=0.93)
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", -0.16, 1.04)
    for analysis, color, label in [("full_universe", "#4D5968", "Full 16,583-gene universe"),
                                   ("closed_labels_masked", BLUE, "668 closed labels masked")]:
        sub = data.loc[data.analysis.eq(analysis)].sort_values("threshold_percent")
        x = sub.threshold_percent.to_numpy() * 100
        y = sub.recovery.to_numpy() * 100
        lo = sub.drug_cluster_bootstrap95_low.to_numpy() * 100
        hi = sub.drug_cluster_bootstrap95_high.to_numpy() * 100
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0)
        ax.plot(x, y, marker="o", ms=4, lw=1.5, color=color, label=label)
    ax.set_xticks([1, 5, 10, 20], ["1%", "5%", "10%", "20%"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Top percentile of ranked gene list")
    ax.set_ylabel("Profiles with a withheld target recovered (%)")
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.legend(loc="lower right")

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", -0.23, 1.04)
    endpoint = endpoint.set_index("analysis").loc[["full_universe", "closed_labels_masked"]]
    vals = endpoint.recovery.to_numpy() * 100
    lows = endpoint.drug_cluster_bootstrap95_low.to_numpy() * 100
    highs = endpoint.drug_cluster_bootstrap95_high.to_numpy() * 100
    xx = np.arange(2)
    ax.bar(xx, vals, color=["#4D5968", BLUE], width=0.58)
    ax.errorbar(xx, vals, yerr=[vals - lows, highs - vals], fmt="none", ecolor=INK, capsize=2.5, lw=0.8)
    for x, row in zip(xx, endpoint.itertuples()):
        label_y = row.drug_cluster_bootstrap95_high * 100 + 1.5
        ax.text(x, label_y, f"{row.recovery:.1%}\n({row.n_success}/{row.n_profiles})",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xx, ["Full\nuniverse", "Closed labels\nmasked"])
    ax.set_ylim(0, 80)
    ax.set_ylabel("Top-5% recovery (%)")
    ax.grid(axis="y", color=GRID, lw=0.55)
    save_vector(fig, 3)

def figure7() -> dict[str, object]:
    scores = pd.read_csv(FIG7_GENE, sep="\t")
    terms = pd.read_csv(FIG7_TERM, sep="\t")
    term = terms.loc[terms.term_id.eq("GO:0010628") & terms.library.eq("GO_BP")].iloc[0]
    members = set(str(term.term_genes).split("|"))
    data = scores[["canonical_gene", "n_drugs", "metar_hit10"]].copy()
    data["member"] = data.canonical_gene.isin(members)
    assert len(data) == 179 and data.member.sum() == 38 and len(terms) == 1526
    fig, ax = plt.subplots(figsize=(89 * MM, 70 * MM))
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.27, top=0.91)
    rng = np.random.default_rng(20260928)
    for x, member, color in [(0, True, BLUE), (1, False, "#A7B1B8")]:
        vals = data.loc[data.member.eq(member), "metar_hit10"].to_numpy()
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(vals)), vals, s=13, color=color, alpha=0.78,
                   edgecolor="white", linewidth=0.2)
        ax.plot([x - 0.22, x + 0.22], [vals.mean(), vals.mean()], color=INK, lw=1.8)
    ax.set_xlim(-0.45, 1.45); ax.set_ylim(-0.025, 1.025)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks([0, 1], [f"GO:0010628\nn = 38; mean = {data.loc[data.member,'metar_hit10'].mean():.3f}",
                           f"Other genes\nn = 141; mean = {data.loc[~data.member,'metar_hit10'].mean():.3f}"])
    ax.set_ylabel("Gene-level mean Hit@10")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", color=GRID, lw=0.55)
    save_vector(fig, 7)
    data.to_csv(OUT / "Supplementary_Figure7_source_data.tsv", sep="\t", index=False)
    return {"genes": 179, "term_genes": 38, "other_genes": 141, "joint_BH_tests": 1526,
            "rank_q_global": float(term.rank_q_global), "mean_in": float(term.mean_hit10_in),
            "mean_out": float(term.mean_hit10_out)}
