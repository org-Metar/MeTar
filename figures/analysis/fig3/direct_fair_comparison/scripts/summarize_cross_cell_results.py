"""Summarize fair cross-cell comparison and within-MeTar adaptation results.

This module computes numeric source tables. Final figure rendering is handled by
the maintained manuscript figure scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[5]
MAIN_METHODS = ["MeTar direct", "SSGCN", "PertKGE"]
METRICS = ["Hit@10", "NDCG@10", "MeanAP", "Hit@50", "AUROC"]
T_CRIT_95_DF13 = 2.160368656


def input_paths(package_root: Path, direct_source: Path | None = None) -> dict[str, Path]:
    """Find maintained comparator inputs and downloaded direct-model results."""
    direct = direct_source or package_root / "figures/reproduction/fig3/round309/source_data"
    comparison = REPOSITORY / "data/plot_sources/figure3/comparison_inputs"
    inputs = {
        "comparator_per_cell": comparison / "Figure3_adapted_comparison_per_cell_hit10.tsv",
        "comparator_endpoints": comparison / "Figure3_adapted_comparison_endpoint_dot_matrix.tsv",
        "delta_summary": REPOSITORY / "data/plot_sources/figure3/Figure3b_paired_delta_summary.tsv",
        "direct_per_cell": direct / "Figure4_R309_per_cell_hit10.tsv",
        "direct_overall": direct / "Figure4_R309_overall_metric_table.tsv",
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    return inputs


def load_data(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Use direct MeTar for external comparison, retaining matched comparator units."""
    baseline_per_cell = pd.read_csv(paths["comparator_per_cell"], sep="\t")
    direct_per_cell = pd.read_csv(paths["direct_per_cell"], sep="\t")
    direct_overall = pd.read_csv(paths["direct_overall"], sep="\t")
    direct = direct_per_cell[
        direct_per_cell["method_label"].eq("MeTar direct")
        & direct_per_cell["comparison_mode"].eq("score_mean")
    ].copy()
    direct["method"] = "Round246 no-adapt"
    direct["method_label"] = "MeTar direct"
    direct["comparison_role"] = "main_fair_no_support_adaptation"
    external = baseline_per_cell[baseline_per_cell["method_label"].isin(["SSGCN", "PertKGE"])].copy()
    external["comparison_role"] = "main_external_comparator"
    return {
        "fair_per_cell": pd.concat([direct, external], ignore_index=True, sort=False),
        "direct_overall": direct_overall[
            direct_overall["method_label"].eq("MeTar direct")
            & direct_overall["comparison_mode"].eq("score_mean")
        ].copy(),
        "baseline_overall": pd.read_csv(paths["comparator_endpoints"], sep="\t"),
        "delta_summary": pd.read_csv(paths["delta_summary"], sep="\t"),
    }


def endpoint_ci_source(per_cell: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for method in methods:
        method_df = per_cell[per_cell["method_label"] == method]
        for metric in METRICS:
            values = method_df[metric].astype(float) * 100.0
            n = int(values.notna().sum())
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if n > 1 else 0.0
            sem = sd / np.sqrt(n) if n > 0 else np.nan
            ci95 = T_CRIT_95_DF13 * sem if n == 14 else 1.96 * sem
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_cells": n,
                    "mean_pct": mean,
                    "sd_pct": sd,
                    "sem_pct": sem,
                    "ci95_pct": ci95,
                    "ci95_low_pct": mean - ci95,
                    "ci95_high_pct": mean + ci95,
                }
            )
    return pd.DataFrame(rows)

def support_continuous_source(per_cell: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    columns = [
        "cell_line",
        "sample_bin",
        "support_groups",
        "n_groups",
        "n_profiles",
        "method_label",
        "Hit@10",
    ]
    source = per_cell.loc[per_cell["method_label"].isin(methods), columns].copy()
    source["Hit@10_pct"] = source["Hit@10"].astype(float) * 100.0
    source = source.drop(columns=["Hit@10"])
    return source.sort_values(["support_groups", "cell_line", "method_label"]).reset_index(drop=True)

def endpoint_source(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    direct = data["direct_overall"].iloc[0]
    for metric in METRICS:
        rows.append({"method": "MeTar direct", "raw_method": direct["method"],
                     "metric": metric, "value_pct": float(direct[metric]) * 100.0,
                     "comparison_role": "main_fair_no_support_adaptation"})
    for method in ["SSGCN", "PertKGE"]:
        for _, row in data["baseline_overall"][data["baseline_overall"]["method"].eq(method)].iterrows():
            if row["metric"] in METRICS:
                rows.append({"method": method, "raw_method": row["raw_method"],
                             "metric": row["metric"], "value_pct": float(row["value_pct"]),
                             "comparison_role": "main_external_comparator"})
    return pd.DataFrame(rows)


def support_strata_source(per_cell: pd.DataFrame) -> pd.DataFrame:
    source = support_continuous_source(per_cell, MAIN_METHODS)
    labels = {"small_<=64_support_groups": "<=64",
              "medium_65_256_support_groups": "65-256",
              "large_>256_support_groups": ">256"}
    rows = []
    for group, label in labels.items():
        for method in MAIN_METHODS:
            vals = source[source["sample_bin"].eq(group) & source["method_label"].eq(method)]["Hit@10_pct"].astype(float).to_numpy()
            if vals.size:
                rows.append({"support_bin": label, "method": method,
                             "n_cells": int(vals.size), "mean_hit10_pct": float(vals.mean()),
                             "sem_hit10_pct": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0})
    return pd.DataFrame(rows)


def generate_sources(package_root: Path, output: Path, direct_source: Path | None = None) -> list[Path]:
    data = load_data(input_paths(package_root, direct_source))
    if data["fair_per_cell"]["cell_line"].nunique() != 14:
        raise ValueError("Expected all 14 held-out cell lines in the fair comparison")
    if not set(MAIN_METHODS).issubset(data["fair_per_cell"]["method_label"]):
        raise ValueError("A comparator is missing from the fair comparison")
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        "Figure3_direct_comparison_fair_per_cell_hit10_source.tsv": data["fair_per_cell"],
        "Figure3_direct_comparison_fair_endpoint_dot_matrix_source.tsv": endpoint_source(data),
        "Figure3_direct_comparison_endpoint_estimates_95ci_source.tsv": endpoint_ci_source(data["fair_per_cell"], MAIN_METHODS),
        "Figure3_direct_comparison_support_continuous_hit10_source.tsv": support_continuous_source(data["fair_per_cell"], MAIN_METHODS),
        "Figure3_direct_comparison_adaptation_delta_summary_source.tsv": data["delta_summary"],
        "Figure3_direct_comparison_support_strata_summary_source.tsv": support_strata_source(data["fair_per_cell"]),
    }
    for name, frame in sources.items():
        frame.to_csv(output / name, sep="\t", index=False)
    return [output / name for name in sources]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--direct-source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = generate_sources(args.package_root.resolve(), args.output_dir.resolve(), args.direct_source)
    print(f"Wrote {len(outputs)} cross-cell source tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
