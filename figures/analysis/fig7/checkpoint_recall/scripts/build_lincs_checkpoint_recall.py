#!/usr/bin/env python
"""Compute checkpoint-provenance LINCS macro Recall@K for Fig. 7 preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

ROUND429_ROOT_DEFAULT = Path(
    "/NFS2_home/NFS2_home_1/lwn/claude/manuscript/write/"
    "claude_round429_lincs_layer_sensitivity_execution_20260623"
)
LABEL_ORDER_DEFAULT = Path(
    "/home/lwn/claude/manuscript/write/"
    "metar_companion_data_package_round360_20260608/data/labels/seen_668_genes.tsv"
)

LOCAL_TAHOE_SUMMARY_DEFAULT = (
    PROJECT
    / "claude_round493_fig7_recall_preview_20260706"
    / "source_data"
    / "Figure7_R493_tahoe_macro_recall_summary.tsv"
)

SOURCE_OUT = ROOT / "source_data"
OUT = ROOT / "outputs"
QA_OUT = ROOT / "qa"
FIG_STEM = "Figure7_R494_lincs_checkpoint_macro_recall_preview"

RECOVERY_KS = [1, 5, 10, 20, 30, 50]
META_COLS = [
    "sig_id",
    "drug",
    "cell_line",
    "pert_dose",
    "pert_time",
    "split",
    "n_profiles",
    "true_targets",
]
CANDIDATE_LABELS_LINCS = 668
MATCHED_PERCENT = 10 / CANDIDATE_LABELS_LINCS * 100.0

COLORS = {
    "random": "#B8C1CB",
    "single": "#C9856B",
    "consensus": "#2F5D8C",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch_checkpoint(path: Path) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment check
        raise RuntimeError("torch is required to validate checkpoint loading") from exc

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return {
            "type": "dict",
            "keys": [str(key) for key in list(obj.keys())[:25]],
        }
    return {"type": type(obj).__name__, "keys": []}


def read_label_order(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    if "gene" in df.columns:
        labels = df["gene"].astype(str).tolist()
    elif "target_label" in df.columns:
        labels = df["target_label"].astype(str).tolist()
    else:
        labels = df.iloc[:, 0].astype(str).tolist()
    if len(labels) != CANDIDATE_LABELS_LINCS:
        raise ValueError(f"Expected 668 labels, found {len(labels)} in {path}")
    return labels


def split_targets(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    for sep in [";", ","]:
        text = text.replace(sep, "|")
    return [part.strip() for part in text.split("|") if part.strip()]


def canonical_targets(value: object) -> str:
    return "|".join(sorted(split_targets(value)))


def zscore_row(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sd = float(arr.std())
    if not math.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - float(arr.mean())) / sd


def collapse_drug_cell(df: pd.DataFrame, labels: list[str], source: Path) -> pd.DataFrame:
    missing_meta = [col for col in META_COLS if col not in df.columns]
    if missing_meta:
        raise ValueError(f"{source}: missing metadata columns {missing_meta}")
    missing_labels = [label for label in labels if label not in df.columns]
    if missing_labels:
        raise ValueError(f"{source}: missing label columns, e.g. {missing_labels[:5]}")

    work = df.assign(_target=df["true_targets"].map(canonical_targets))
    inconsistent = work.groupby(["drug", "cell_line"], dropna=False)["_target"].nunique()
    bad = inconsistent[inconsistent.gt(1)]
    if len(bad):
        raise ValueError(f"{source}: target labels differ within drug-cell units: {bad.index[:5].tolist()}")

    scores = work.groupby(["drug", "cell_line"], dropna=False)[labels].mean().reset_index()
    meta = (
        work.groupby(["drug", "cell_line"], dropna=False)
        .agg(
            n_profiles=("n_profiles", "sum"),
            input_rows=("sig_id", "size"),
            true_targets=("_target", "first"),
        )
        .reset_index()
    )
    unit = scores.merge(meta, on=["drug", "cell_line"], how="left", validate="one_to_one")
    target_n = unit.groupby("drug", dropna=False)["true_targets"].nunique()
    bad_drugs = target_n[target_n.gt(1)]
    if len(bad_drugs):
        raise ValueError(f"{source}: target labels differ across cell lines: {bad_drugs.index[:10].tolist()}")
    return unit


def topk_metrics(scores: np.ndarray, pos_idx: list[int], k: int) -> tuple[float, float, int]:
    if not pos_idx:
        return float("nan"), float("nan"), 0
    order = np.argsort(-scores, kind="mergesort")[:k]
    recovered = len(set(order.tolist()).intersection(pos_idx))
    return recovered / float(len(pos_idx)), 1.0 if recovered > 0 else 0.0, recovered


def random_hit_expectation(k: int, n_labels: int, n_positive: int) -> float:
    if n_positive <= 0:
        return float("nan")
    if k >= n_labels:
        return 1.0
    miss = 1.0
    for i in range(k):
        miss *= (n_labels - n_positive - i) / float(n_labels - i)
    return 1.0 - miss


def seed_paths(round429_root: Path, seed: int) -> dict[str, Path]:
    base = round429_root / "outputs" / "lincs_layer_sensitivity"
    return {
        "checkpoint": base / "runs" / "L5_current_signature" / f"seed_{seed}" / "checkpoint.pt",
        "adapter": base
        / "support_adapter"
        / "L5_current_signature"
        / f"seed_{seed}"
        / "support_adapter.pt",
        "score": base
        / "scores"
        / "L5_current_signature"
        / f"seed_{seed}"
        / "support_adapted_scores.tsv.gz",
    }


def checkpoint_audit(round429_root: Path, seed: int) -> dict[str, object]:
    paths = seed_paths(round429_root, seed)
    rec: dict[str, object] = {"seed": seed}
    for key, path in paths.items():
        rec[f"{key}_path"] = str(path)
        rec[f"{key}_exists"] = path.exists()
        rec[f"{key}_size_bytes"] = int(path.stat().st_size) if path.exists() else 0
        rec[f"{key}_sha256"] = sha256_file(path) if path.exists() else ""
    for key in ["checkpoint", "adapter"]:
        path = paths[key]
        if not path.exists():
            rec[f"{key}_load_ok"] = False
            rec[f"{key}_load_type"] = ""
            rec[f"{key}_load_keys"] = ""
            continue
        info = load_torch_checkpoint(path)
        rec[f"{key}_load_ok"] = True
        rec[f"{key}_load_type"] = info["type"]
        rec[f"{key}_load_keys"] = "|".join(info["keys"])
    return rec


def compute_lincs(args: argparse.Namespace) -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)

    round429_root = Path(args.round429_root)
    label_order = Path(args.label_order)
    labels = read_label_order(label_order)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    seeds = [int(seed) for seed in args.seeds]

    seed_units: dict[int, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    denominators: list[set[str]] = []

    for seed in seeds:
        audit = checkpoint_audit(round429_root, seed)
        score_path = seed_paths(round429_root, seed)["score"]
        df = pd.read_csv(score_path, sep="\t")
        unit = collapse_drug_cell(df, labels, score_path)
        cell_counts = unit.groupby("drug", dropna=False)["cell_line"].nunique()
        multi_drugs = set(cell_counts[cell_counts.ge(args.min_cell_lines)].index.astype(str))
        denominators.append(multi_drugs)
        seed_units[seed] = unit
        audit.update(
            {
                "input_profile_rows": int(len(df)),
                "drug_cell_units": int(len(unit)),
                "all_drugs": int(cell_counts.shape[0]),
                "multi_cell_drugs": int(len(multi_drugs)),
                "min_cell_lines_required": int(args.min_cell_lines),
                "candidate_labels": int(len(labels)),
            }
        )
        audit_rows.append(audit)

    common_drugs = set.intersection(*denominators)
    rows: list[dict[str, object]] = []
    for seed, unit in seed_units.items():
        unit = unit[unit["drug"].astype(str).isin(common_drugs)].copy()
        for drug, grp in unit.groupby("drug", sort=True, dropna=False):
            targets = split_targets(grp["true_targets"].iloc[0])
            pos_idx = [label_to_idx[target] for target in targets if target in label_to_idx]
            if not pos_idx:
                continue
            n_cell_lines = int(grp["cell_line"].nunique())
            cell_lines = "|".join(sorted(map(str, grp["cell_line"].unique())))
            score_matrix = grp[labels].to_numpy(dtype=np.float64)
            consensus = np.vstack([zscore_row(row) for row in score_matrix]).mean(axis=0)

            for k in RECOVERY_KS:
                cell_recalls: list[float] = []
                cell_hits: list[float] = []
                cell_recovered: list[int] = []
                for vector in score_matrix:
                    recall, hit, recovered = topk_metrics(vector, pos_idx, k)
                    cell_recalls.append(recall)
                    cell_hits.append(hit)
                    cell_recovered.append(recovered)

                recall, hit, recovered = topk_metrics(consensus, pos_idx, k)
                rows.extend(
                    [
                        {
                            "dataset": "LINCS",
                            "method": "cell_line_mean",
                            "display_label": "Cell-line mean",
                            "seed": seed,
                            "drug": drug,
                            "k": k,
                            "threshold_percent": k / len(labels) * 100.0,
                            "macro_recall": float(np.mean(cell_recalls)),
                            "hitk": float(np.mean(cell_hits)),
                            "recovered_targets_mean": float(np.mean(cell_recovered)),
                            "n_positive": len(pos_idx),
                            "n_cell_lines": n_cell_lines,
                            "cell_lines": cell_lines,
                            "candidate_labels": len(labels),
                            "denominator_type": "same_415_multicell_drugs_x_3_seeds",
                            "score_source": "Round429_L5_support_adapted_checkpoint_export",
                        },
                        {
                            "dataset": "LINCS",
                            "method": "multi_cell_consensus",
                            "display_label": "Consensus",
                            "seed": seed,
                            "drug": drug,
                            "k": k,
                            "threshold_percent": k / len(labels) * 100.0,
                            "macro_recall": recall,
                            "hitk": hit,
                            "recovered_targets_mean": float(recovered),
                            "n_positive": len(pos_idx),
                            "n_cell_lines": n_cell_lines,
                            "cell_lines": cell_lines,
                            "candidate_labels": len(labels),
                            "denominator_type": "same_415_multicell_drugs_x_3_seeds",
                            "score_source": "Round429_L5_support_adapted_checkpoint_export",
                        },
                        {
                            "dataset": "LINCS",
                            "method": "random",
                            "display_label": "Random",
                            "seed": seed,
                            "drug": drug,
                            "k": k,
                            "threshold_percent": k / len(labels) * 100.0,
                            "macro_recall": k / float(len(labels)),
                            "hitk": random_hit_expectation(k, len(labels), len(pos_idx)),
                            "recovered_targets_mean": k * len(pos_idx) / float(len(labels)),
                            "n_positive": len(pos_idx),
                            "n_cell_lines": n_cell_lines,
                            "cell_lines": cell_lines,
                            "candidate_labels": len(labels),
                            "denominator_type": "matched_random_expectation_same_positive_counts",
                            "score_source": "analytic_random_expectation",
                        },
                    ]
                )

    source = pd.DataFrame(rows)
    if source.empty:
        raise RuntimeError("No LINCS recall rows were produced")

    seed_means = (
        source.groupby(["dataset", "method", "display_label", "k", "seed", "threshold_percent"], as_index=False)
        .agg(
            macro_recall=("macro_recall", "mean"),
            hitk=("hitk", "mean"),
            n_drugs=("drug", "nunique"),
        )
    )
    summary = (
        seed_means.groupby(["dataset", "method", "display_label", "k"], as_index=False)
        .agg(
            threshold_percent=("threshold_percent", "mean"),
            value=("macro_recall", "mean"),
            sd_across_seeds=("macro_recall", "std"),
            hitk_value=("hitk", "mean"),
            hitk_sd_across_seeds=("hitk", "std"),
            n_drugs=("n_drugs", "min"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "k", "method"])
    )
    summary[["sd_across_seeds", "hitk_sd_across_seeds"]] = summary[
        ["sd_across_seeds", "hitk_sd_across_seeds"]
    ].fillna(0.0)

    audit = pd.DataFrame(audit_rows)
    audit["strict_common_multicell_drugs"] = len(common_drugs)
    audit["label_order_path"] = str(label_order)
    audit["label_order_sha256"] = sha256_file(label_order)
    audit["round429_root"] = str(round429_root)
    audit["checkpoint_policy"] = (
        "checkpoint_and_support_adapter_loaded_for_provenance; "
        "metrics computed from the matching Round429 support_adapted_scores.tsv.gz checkpoint export"
    )

    source_path = SOURCE_OUT / "Figure7_R494_lincs_checkpoint_macro_recall_source.tsv"
    summary_path = SOURCE_OUT / "Figure7_R494_lincs_checkpoint_macro_recall_summary.tsv"
    audit_path = SOURCE_OUT / "Figure7_R494_lincs_checkpoint_recall_audit.tsv"
    source.to_csv(source_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    audit.to_csv(audit_path, sep="\t", index=False)

    k10 = summary[summary["k"].eq(10)].sort_values("value", ascending=False)
    qa = {
        "round": "Round494",
        "article_version_lincs": {
            "source_round": "Round429",
            "layer": "L5_current_signature",
            "mode": "support_adapted",
            "seeds": seeds,
            "candidate_labels": len(labels),
            "strict_common_multicell_drugs": len(common_drugs),
        },
        "outputs": {
            "source": str(source_path),
            "summary": str(summary_path),
            "audit": str(audit_path),
        },
        "recall_at_10": k10.to_dict(orient="records"),
        "hitk_at_10_audit": k10[
            ["method", "display_label", "hitk_value", "hitk_sd_across_seeds", "n_drugs", "n_seeds"]
        ].to_dict(orient="records"),
    }
    (QA_OUT / "Figure7_R494_lincs_checkpoint_recall_summary.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, indent=2))


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
    order = ["random", "cell_line_mean", "single_cell_mean", "multi_cell_consensus"]
    color_map = {
        "random": COLORS["random"],
        "cell_line_mean": COLORS["single"],
        "single_cell_mean": COLORS["single"],
        "multi_cell_consensus": COLORS["consensus"],
    }
    marker_map = {
        "random": "o",
        "cell_line_mean": "o",
        "single_cell_mean": "o",
        "multi_cell_consensus": "s",
    }
    style_map = {"random": (0, (3, 2))}
    end_points: list[tuple[str, float, float, str]] = []
    for method in order:
        sub = summary[summary["method"].eq(method)].sort_values("threshold_percent")
        if sub.empty:
            continue
        x = sub["threshold_percent"].to_numpy(dtype=float)
        y = sub["value"].to_numpy(dtype=float) * 100.0
        yerr = sub["sd_across_seeds"].fillna(0.0).to_numpy(dtype=float) * 100.0
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
            label=str(sub["display_label"].iloc[0]),
            zorder=3 if method != "random" else 2,
        )
        end_points.append((str(sub["display_label"].iloc[0]), float(x[-1]), float(y[-1]), color_map[method]))
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
        y_offset = 0.0
        if label in {"Consensus", "Three-cell consensus"}:
            y_offset = 1.9 if dataset == "Tahoe" else 1.2
        elif "Cell-line" in label:
            y_offset = -1.6
        elif label == "Random":
            y_offset = 2.0
        ax.text(x * 1.06, y + y_offset, label, color=color, fontsize=6.7, va="center", ha="left")


def plot_recall(args: argparse.Namespace) -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)

    lincs_summary = pd.read_csv(args.lincs_summary, sep="\t")
    tahoe_summary = pd.read_csv(args.tahoe_summary, sep="\t")
    tahoe_copy = SOURCE_OUT / "Figure7_R494_tahoe_macro_recall_summary_from_R493.tsv"
    tahoe_summary.to_csv(tahoe_copy, sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.935, bottom=0.20, top=0.76, wspace=0.42)
    plot_panel(axes[0], lincs_summary, "LINCS")
    plot_panel(axes[1], tahoe_summary, "Tahoe")
    axes[0].text(-0.16, 1.10, "a", transform=axes[0].transAxes, fontsize=10, fontweight="bold", va="top")
    axes[1].text(-0.16, 1.10, "b", transform=axes[1].transAxes, fontsize=10, fontweight="bold", va="top")
    fig.text(
        0.30,
        0.955,
        "LINCS checkpoint-derived macro recall",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.745,
        0.955,
        "Tahoe drug-level macro recall",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.105,
        0.055,
        "LINCS: Round429 L5 support-adapted checkpoint exports; same 415 drugs x 3 seeds.",
        ha="left",
        va="center",
        fontsize=6.2,
        color=COLORS["muted"],
    )
    paths = []
    for ext in ["svg", "pdf", "png", "tiff"]:
        path = OUT / f"{FIG_STEM}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=450)
        elif ext == "tiff":
            fig.savefig(path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path)
        paths.append(str(path))
    plt.close(fig)

    matched = {
        "figure": paths,
        "lincs_recall_at_10": lincs_summary[lincs_summary["k"].eq(10)]
        .sort_values("value", ascending=False)
        .to_dict(orient="records"),
        "tahoe_recall_at_10": tahoe_summary[tahoe_summary["k"].eq(10)]
        .sort_values("value", ascending=False)
        .to_dict(orient="records"),
    }
    (QA_OUT / "Figure7_R494_macro_recall_preview_summary.json").write_text(
        json.dumps(matched, indent=2), encoding="utf-8"
    )
    print(json.dumps(matched, indent=2))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    comp = sub.add_parser("compute-lincs", help="Compute exact LINCS macro Recall@K")
    comp.add_argument("--round429-root", default=str(ROUND429_ROOT_DEFAULT))
    comp.add_argument("--label-order", default=str(LABEL_ORDER_DEFAULT))
    comp.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
    comp.add_argument("--min-cell-lines", type=int, default=2)

    plot = sub.add_parser("plot", help="Plot LINCS/Tahoe macro Recall@K preview")
    plot.add_argument(
        "--lincs-summary",
        default=str(SOURCE_OUT / "Figure7_R494_lincs_checkpoint_macro_recall_summary.tsv"),
    )
    plot.add_argument("--tahoe-summary", default=str(LOCAL_TAHOE_SUMMARY_DEFAULT))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "compute-lincs":
        compute_lincs(args)
    elif args.command == "plot":
        plot_recall(args)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
