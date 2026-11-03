"""Compute manuscript result tables without historical figure layouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_MAP={"Round259":"MeTar","PertKGE_original_inductive":"PertKGE","Random":"Random"}
UNIT_LABELS={"profile_paired":"Profile","drug_cell_mean_paired":"Drug-cell","drug_mean_across_cells_paired":"Drug mean"}
THRESHOLDS=[0.01,0.05,0.10,0.20]


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def build_threshold_aggregation_source(thr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = thr[
        thr["denominator_scope"].eq("paired_round259_pertkge_native_percentile")
        & thr["threshold"].isin(THRESHOLDS)
        & thr["evaluation_unit"].isin(list(UNIT_LABELS))
    ].copy()
    for _, row in sub.iterrows():
        rows.append(
            {
                "aggregation": UNIT_LABELS[str(row["evaluation_unit"])],
                "evaluation_unit": row["evaluation_unit"],
                "threshold": float(row["threshold"]),
                "threshold_percent": float(row["threshold"]) * 100,
                "method": METHOD_MAP.get(str(row["method"]), str(row["method"])),
                "hits": row.get("hits", np.nan),
                "n": int(row["n"]),
                "recovery_fraction": float(row["rate"]),
                "recovery_percent": float(row["rate"]) * 100,
                "wilson95_low": row.get("wilson95_low", np.nan),
                "wilson95_high": row.get("wilson95_high", np.nan),
                "rank_cutoff": row.get("rank_cutoff", np.nan),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / "Figure4_threshold_aggregation_source.tsv", sep="\t", index=False)
    return out

def build_profile_threshold_source(threshold_source: pd.DataFrame) -> pd.DataFrame:
    out = threshold_source[threshold_source["aggregation"].eq("Profile")].copy()
    out.to_csv(OUTPUT / "Figure4_profile_threshold_bar_source.tsv", sep="\t", index=False)
    return out

def build_aggregation_top10_source(threshold_source: pd.DataFrame) -> pd.DataFrame:
    out = threshold_source[threshold_source["threshold"].eq(0.10)].copy()
    out.to_csv(OUTPUT / "Figure4_aggregation_top10_bar_source.tsv", sep="\t", index=False)
    return out

def build_per_cell_source(drug: pd.DataFrame) -> pd.DataFrame:
    dc = drug[drug["group_type"].eq("drug_cell_mean")].copy()
    rows = []
    for cell, group in dc.groupby("cell_line", sort=False):
        rows.append(
            {
                "cell_line": cell,
                "n_drug_cell_units": int(len(group)),
                "n_drugs": int(group["drug"].nunique()),
                "metar_top5": float((group["round259_best_percentile"] <= 0.05).mean()),
                "pertkge_top5": float((group["pertkge_best_percentile"] <= 0.05).mean()),
                "metar_top10": float((group["round259_best_percentile"] <= 0.10).mean()),
                "pertkge_top10": float((group["pertkge_best_percentile"] <= 0.10).mean()),
                "metar_better_fraction": float((group["round259_best_percentile"] < group["pertkge_best_percentile"]).mean()),
                "metar_median_percentile": float(group["round259_best_percentile"].median()),
                "pertkge_median_percentile": float(group["pertkge_best_percentile"].median()),
            }
        )
    out = pd.DataFrame(rows).sort_values("n_drug_cell_units", ascending=False)
    out.to_csv(OUTPUT / "Figure4_per_cell_top10_source.tsv", sep="\t", index=False)
    return out

def build_drug_distribution_source(drug: pd.DataFrame) -> pd.DataFrame:
    d = drug[drug["group_type"].eq("drug_mean_across_cells")].copy()
    out = pd.DataFrame(
        {
            "drug": d["drug"],
            "metar_best_percentile": d["round259_best_percentile"],
            "pertkge_best_percentile": d["pertkge_best_percentile"],
            "metar_best_percentile_percent": d["round259_best_percentile"] * 100,
            "pertkge_best_percentile_percent": d["pertkge_best_percentile"] * 100,
            "pertkge_minus_metar_percentile_pp": (d["pertkge_best_percentile"] - d["round259_best_percentile"]) * 100,
            "metar_better": d["round259_best_percentile"] < d["pertkge_best_percentile"],
        }
    )
    out.to_csv(OUTPUT / "Figure4_drug_level_percentile_distribution_source.tsv", sep="\t", index=False)
    return out

def build_drugcell_pair_source(drug: pd.DataFrame) -> pd.DataFrame:
    d = drug[drug["group_type"].eq("drug_cell_mean")].copy()
    out = pd.DataFrame(
        {
            "drug": d["drug"],
            "cell_line": d["cell_line"],
            "n_profiles_in_group_round259": d["n_profiles_in_group_round259"],
            "n_profiles_in_group_pertkge": d["n_profiles_in_group_pertkge"],
            "metar_best_percentile": d["round259_best_percentile"],
            "pertkge_best_percentile": d["pertkge_best_percentile"],
            "metar_best_percentile_percent": d["round259_best_percentile"] * 100,
            "pertkge_best_percentile_percent": d["pertkge_best_percentile"] * 100,
            "metar_earlier": d["round259_best_percentile"] < d["pertkge_best_percentile"],
            "metar_top10": d["round259_best_percentile"] <= 0.10,
            "pertkge_top10": d["pertkge_best_percentile"] <= 0.10,
        }
    )
    out.to_csv(OUTPUT / "Figure4_drugcell_paired_percentile_map_source.tsv", sep="\t", index=False)
    return out



def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-summary",type=Path,required=True)
    parser.add_argument("--paired-percentiles",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    global OUTPUT
    OUTPUT=args.output_dir
    OUTPUT.mkdir(parents=True,exist_ok=False)
    threshold=pd.read_csv(args.threshold_summary,sep="\t")
    paired=pd.read_csv(args.paired_percentiles,sep="\t")
    aggregation=build_threshold_aggregation_source(threshold)
    build_profile_threshold_source(aggregation)
    build_aggregation_top10_source(aggregation)
    build_per_cell_source(paired)
    build_drug_distribution_source(paired)
    build_drugcell_pair_source(paired)


if __name__ == "__main__":
    main()
