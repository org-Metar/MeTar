#!/usr/bin/env python
"""Build a diagnostic macro Recall@K preview for Fig. 7."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

ROUND430_SCORES = (
    PROJECT
    / "claude_round430_multicell_consensus_prediction_20260624"
    / "archive"
    / "second_direction_multicell_consensus_20260624"
    / "scores"
    / "drug_level"
    / "adapted"
)
ROUND430_PER_DRUG = (
    PROJECT
    / "claude_round430_multicell_consensus_prediction_20260624"
    / "server_synced_outputs"
    / "metrics"
    / "multicell_consensus_per_drug.tsv"
)
ROUND492_SOURCE = PROJECT / "claude_round492_fig7_explicit_hitk_axis_20260706" / "source_data"
SC_METRICS = ROUND492_SOURCE / "round260_singlecell_score_metrics.tsv"
SC_SCORES = ROUND492_SOURCE / "round260_singlecell_fullspace_scores.tsv.gz"
SC_LABELS = ROUND492_SOURCE / "round260_singlecell_target_labels.tsv"

OUT = ROOT / "outputs"
SOURCE_OUT = ROOT / "source_data"
QA_OUT = ROOT / "qa"
FIG_STEM = "Figure7_R493_macro_recall_preview"

CANDIDATE_LABELS_LINCS = 668
RECOVERY_KS = [1, 5, 10, 20, 30, 50]
MATCHED_PERCENT = 10 / CANDIDATE_LABELS_LINCS * 100.0

COLORS = {
    "random": "#B8C1CB",
    "single": "#C9856B",
    "consensus": "#2F5D8C",
    "raw": "#7A8795",
    "ink": "#202833",
    "muted": "#6E7884",
    "grid": "#E8ECEF",
    "threshold": "#202833",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.linewidth": 0.72,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def split_targets(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.replace(";", "|").split("|") if part.strip()]


def recall_at_threshold(scores: np.ndarray, pos_idx: list[int], k: int) -> float:
    if not pos_idx:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")[:k]
    recovered = len(set(order.tolist()).intersection(pos_idx))
    return recovered / float(len(pos_idx))


def random_recall(k: int, n_labels: int) -> float:
    return min(float(k) / float(n_labels), 1.0)


def compute_lincs_recall() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    display = {
        "cell_zscore_mean": ("Consensus", "multi_cell_consensus", COLORS["consensus"]),
        "raw_score_mean": ("Raw score mean", "raw_score_mean", COLORS["raw"]),
    }
    for seed in [1, 2, 3]:
        for file_method, (label, method, _) in display.items():
            path = ROUND430_SCORES / f"seed_{seed}" / f"{file_method}_drug_scores.tsv.gz"
            df = pd.read_csv(path, sep="\t")
            labels = [c for c in df.columns if c not in {"drug", "n_cell_lines", "cell_lines", "n_profiles_total", "n_drug_cell_units", "true_targets"}]
            if len(labels) != CANDIDATE_LABELS_LINCS:
                raise RuntimeError(f"{path.name}: expected 668 labels, found {len(labels)}")
            label_to_idx = {label_name: i for i, label_name in enumerate(labels)}
            for _, row in df.iterrows():
                targets = split_targets(row["true_targets"])
                pos_idx = [label_to_idx[target] for target in targets if target in label_to_idx]
                scores = row[labels].to_numpy(dtype=np.float64)
                for k in RECOVERY_KS:
                    rows.append(
                        {
                            "dataset": "LINCS",
                            "method": method,
                            "display_label": label,
                            "seed": seed,
                            "drug": row["drug"],
                            "rank_cutoff": f"Recall@{k}",
                            "k": k,
                            "threshold_percent": k / len(labels) * 100.0,
                            "macro_recall": recall_at_threshold(scores, pos_idx, k),
                            "n_positive": len(pos_idx),
                            "candidate_labels": len(labels),
                            "denominator_type": "same_415_drugs_x_3_seeds",
                        }
                    )
        for k in RECOVERY_KS:
            rows.append(
                {
                    "dataset": "LINCS",
                    "method": "random",
                    "display_label": "Random",
                    "seed": seed,
                    "drug": "__random_expectation__",
                    "rank_cutoff": f"Recall@{k}",
                    "k": k,
                    "threshold_percent": k / CANDIDATE_LABELS_LINCS * 100.0,
                    "macro_recall": random_recall(k, CANDIDATE_LABELS_LINCS),
                    "n_positive": np.nan,
                    "candidate_labels": CANDIDATE_LABELS_LINCS,
                    "denominator_type": "random_expectation_same_label_space",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_OUT / "Figure7_R493_lincs_macro_recall_source.tsv", sep="\t", index=False)
    return out


def zscore_row(row: np.ndarray) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float64)
    sd = float(arr.std())
    if not math.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / sd


def compute_tahoe_recall() -> pd.DataFrame:
    metrics = pd.read_csv(SC_METRICS, sep="\t")
    scores = pd.read_csv(SC_SCORES, sep="\t")
    labels = pd.read_csv(SC_LABELS, sep="\t")["target_label"].astype(str).tolist()
    label_to_idx = {label: i for i, label in enumerate(labels)}
    merged = scores.merge(
        metrics[["cell_line", "drug", "positive_targets", "n_holdout_positive"]],
        on=["cell_line", "drug"],
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for drug, grp in merged.groupby("drug", sort=True):
        if grp["cell_line"].nunique() != 3:
            continue
        if grp["positive_targets"].nunique() != 1:
            raise RuntimeError(f"Positive targets differ across cell lines for {drug}")
        targets = split_targets(grp["positive_targets"].iloc[0])
        pos_idx = [label_to_idx[target] for target in targets if target in label_to_idx]
        score_matrix = grp[labels].to_numpy(dtype=np.float64)
        consensus_scores = np.vstack([zscore_row(row) for row in score_matrix]).mean(axis=0)
        for k in RECOVERY_KS:
            cutoff = max(1, int(round(k / CANDIDATE_LABELS_LINCS * len(labels))))
            cell_recalls = [recall_at_threshold(row, pos_idx, cutoff) for row in score_matrix]
            rows.extend(
                [
                    {
                        "dataset": "Tahoe",
                        "method": "single_cell_mean",
                        "display_label": "Cell-line mean",
                        "drug": drug,
                        "rank_cutoff": f"Recall@{k} eq.",
                        "k": k,
                        "top_n_genes": cutoff,
                        "threshold_percent": cutoff / len(labels) * 100.0,
                        "macro_recall": float(np.mean(cell_recalls)),
                        "n_positive": len(pos_idx),
                        "candidate_labels": len(labels),
                        "denominator_type": "same_30_shared_drugs",
                    },
                    {
                        "dataset": "Tahoe",
                        "method": "multi_cell_consensus",
                        "display_label": "Consensus",
                        "drug": drug,
                        "rank_cutoff": f"Recall@{k} eq.",
                        "k": k,
                        "top_n_genes": cutoff,
                        "threshold_percent": cutoff / len(labels) * 100.0,
                        "macro_recall": recall_at_threshold(consensus_scores, pos_idx, cutoff),
                        "n_positive": len(pos_idx),
                        "candidate_labels": len(labels),
                        "denominator_type": "same_30_shared_drugs",
                    },
                    {
                        "dataset": "Tahoe",
                        "method": "random",
                        "display_label": "Random",
                        "drug": drug,
                        "rank_cutoff": f"Recall@{k} eq.",
                        "k": k,
                        "top_n_genes": cutoff,
                        "threshold_percent": cutoff / len(labels) * 100.0,
                        "macro_recall": random_recall(cutoff, len(labels)),
                        "n_positive": len(pos_idx),
                        "candidate_labels": len(labels),
                        "denominator_type": "random_expectation_same_label_space",
                    },
                ]
            )
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_OUT / "Figure7_R493_tahoe_macro_recall_source.tsv", sep="\t", index=False)
    return out


def summarize_recall(df: pd.DataFrame) -> pd.DataFrame:
    if df["dataset"].iloc[0] == "LINCS":
        nonrandom = df[df["method"].ne("random")]
        seed_means = (
            nonrandom.groupby(["dataset", "method", "display_label", "k", "seed", "threshold_percent"], as_index=False)["macro_recall"].mean()
        )
        summary = (
            seed_means.groupby(["dataset", "method", "display_label", "k"], as_index=False)
            .agg(
                threshold_percent=("threshold_percent", "mean"),
                value=("macro_recall", "mean"),
                sd_across_seeds=("macro_recall", "std"),
            )
        )
        random_rows = (
            df[df["method"].eq("random")]
            .groupby(["dataset", "method", "display_label", "k"], as_index=False)
            .agg(threshold_percent=("threshold_percent", "mean"), value=("macro_recall", "mean"))
        )
        random_rows["sd_across_seeds"] = 0.0
        return pd.concat([summary, random_rows], ignore_index=True)
    return (
        df.groupby(["dataset", "method", "display_label", "k"], as_index=False)
        .agg(threshold_percent=("threshold_percent", "mean"), value=("macro_recall", "mean"))
        .assign(sd_across_seeds=0.0)
    )


def add_hitk_axis(ax: plt.Axes) -> None:
    positions = [k / CANDIDATE_LABELS_LINCS * 100.0 for k in RECOVERY_KS]
    top = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    top.set_xticks(positions)
    top.set_xticklabels([str(k) for k in RECOVERY_KS])
    top.set_xlabel("Recall@K equivalent", labelpad=1.4, fontsize=7.0, color=COLORS["ink"])
    top.tick_params(axis="x", length=2.0, width=0.55, pad=0.8, labelsize=6.3, colors=COLORS["muted"])
    top.spines["top"].set_color("#B8C1CB")
    top.spines["top"].set_linewidth(0.55)


def plot_panel(ax: plt.Axes, summary: pd.DataFrame, dataset: str) -> None:
    if dataset == "LINCS":
        order = ["random", "multi_cell_consensus"]
    else:
        order = ["random", "single_cell_mean", "multi_cell_consensus"]
    color_map = {
        "random": COLORS["random"],
        "raw_score_mean": COLORS["raw"],
        "single_cell_mean": COLORS["single"],
        "multi_cell_consensus": COLORS["consensus"],
    }
    marker_map = {
        "random": "o",
        "raw_score_mean": "^",
        "single_cell_mean": "o",
        "multi_cell_consensus": "s",
    }
    style_map = {"random": (0, (3, 2))}
    end_points: list[tuple[str, float, float, str]] = []
    for method in order:
        sub = summary[summary["method"].eq(method)].sort_values("threshold_percent")
        if sub.empty:
            continue
        y = sub["value"].to_numpy(dtype=float) * 100.0
        x = sub["threshold_percent"].to_numpy(dtype=float)
        yerr = sub["sd_across_seeds"].fillna(0).to_numpy(dtype=float) * 100.0
        ax.errorbar(
            x,
            y,
            yerr=yerr if np.any(yerr > 0) else None,
            color=color_map[method],
            marker=marker_map[method],
            markersize=3.4,
            lw=1.35 if method != "random" else 1.1,
            linestyle=style_map.get(method, "-"),
            capsize=2.0 if np.any(yerr > 0) else 0,
            elinewidth=0.65,
            label=sub["display_label"].iloc[0],
            zorder=3 if method != "random" else 2,
        )
        end_points.append((sub["display_label"].iloc[0], float(x[-1]), float(y[-1]), color_map[method]))
    ax.axvline(MATCHED_PERCENT, color=COLORS["threshold"], lw=0.72)
    ax.set_xscale("log")
    ax.set_xlim(0.12, 11.5)
    ax.set_ylim(0, 82)
    ax.set_xticks([0.15, 0.75, 1.5, 3.0, 7.5])
    ax.set_xticklabels(["0.15", "0.75", "1.5", "3.0", "7.5"])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.grid(color=COLORS["grid"], linewidth=0.7, which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel("Rank-fraction threshold (%)", labelpad=2.0)
    ax.set_ylabel("Macro Recall@K (%)")
    add_hitk_axis(ax)
    for label, x, y, color in end_points:
        if label == "Consensus" and dataset == "Tahoe":
            y += 2.2
        if label == "Cell-line mean":
            y -= 1.8
        if label == "Random":
            y += 2.2
        ax.text(x * 1.06, y, label, color=color, fontsize=6.7, va="center", ha="left")


def build_figure(lincs_summary: pd.DataFrame, tahoe_summary: pd.DataFrame) -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.935, bottom=0.20, top=0.76, wspace=0.42)
    plot_panel(axes[0], lincs_summary, "LINCS")
    plot_panel(axes[1], tahoe_summary, "Tahoe")
    axes[0].text(-0.16, 1.10, "a", transform=axes[0].transAxes, fontsize=10, fontweight="bold", va="top")
    axes[1].text(-0.16, 1.10, "b", transform=axes[1].transAxes, fontsize=10, fontweight="bold", va="top")
    fig.text(0.30, 0.955, "LINCS drug-level macro recall", ha="center", va="center", fontsize=9.2, fontweight="bold", color=COLORS["ink"])
    fig.text(0.745, 0.955, "Tahoe drug-level macro recall", ha="center", va="center", fontsize=9.2, fontweight="bold", color=COLORS["ink"])
    fig.text(
        0.105,
        0.055,
        "Diagnostic only: LINCS exact cell-line Recall@K requires per-cell score export.",
        ha="left",
        va="center",
        fontsize=6.2,
        color=COLORS["muted"],
    )
    for ext in ["svg", "pdf", "png", "tiff"]:
        path = OUT / f"{FIG_STEM}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=450)
        elif ext == "tiff":
            fig.savefig(path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path)
    plt.close(fig)


def main() -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    lincs = compute_lincs_recall()
    tahoe = compute_tahoe_recall()
    lincs_summary = summarize_recall(lincs)
    tahoe_summary = summarize_recall(tahoe)
    lincs_summary.to_csv(SOURCE_OUT / "Figure7_R493_lincs_macro_recall_summary.tsv", sep="\t", index=False)
    tahoe_summary.to_csv(SOURCE_OUT / "Figure7_R493_tahoe_macro_recall_summary.tsv", sep="\t", index=False)
    build_figure(lincs_summary, tahoe_summary)
    matched = {
        "lincs": lincs_summary[lincs_summary["k"].eq(10)].sort_values("value", ascending=False).to_dict(orient="records"),
        "tahoe": tahoe_summary[tahoe_summary["k"].eq(10)].sort_values("value", ascending=False).to_dict(orient="records"),
        "note": "Recall is macro-averaged over true targets per drug; LINCS cell-line mean exact recall is not computed because per-cell score rows are not in the local Round430 archive.",
        "outputs": [str(OUT / f"{FIG_STEM}.{ext}") for ext in ["svg", "pdf", "png", "tiff"]],
    }
    (QA_OUT / "Figure7_R493_recall_preview_summary.json").write_text(json.dumps(matched, indent=2), encoding="utf-8")
    print(json.dumps(matched, indent=2))


if __name__ == "__main__":
    main()
