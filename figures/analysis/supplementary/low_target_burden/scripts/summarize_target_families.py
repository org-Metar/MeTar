"""Summarize response similarities within and between single-target drug families."""
from __future__ import annotations

import argparse
import math
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

def parse_targets(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [
        part.strip()
        for part in text.replace(";", "|").replace(",", "|").split("|")
        if part.strip()
    ]

def load_inputs() -> tuple[pd.DataFrame, np.ndarray, dict[str, int], pd.DataFrame]:
    sim_df = pd.read_csv(SIM_PATH, sep="\t")
    drugs = sim_df["drug"].tolist()
    sim = sim_df.drop(columns=["drug"]).to_numpy(dtype=float)
    drug_to_idx = {drug: i for i, drug in enumerate(drugs)}
    burden = pd.read_csv(BURDEN_PATH, sep="\t")
    burden["target_list"] = burden["targets_in_668"].map(parse_targets)
    return sim_df, sim, drug_to_idx, burden

def select_single_target_family_drugs(
    burden: pd.DataFrame, drug_to_idx: dict[str, int]
) -> pd.DataFrame:
    eligible = burden[
        burden["drug"].isin(drug_to_idx)
        & burden["target_list"].map(lambda values: len(values) == 1)
    ].copy()
    eligible["primary_target"] = eligible["target_list"].map(lambda values: values[0])
    family_sizes = eligible["primary_target"].value_counts()
    retained = family_sizes[family_sizes >= MIN_SINGLE_TARGET_FAMILY_SIZE]
    retained_families = retained.sort_values(ascending=False).index.tolist()

    rows = []
    for family in retained_families:
        subset = eligible[eligible["primary_target"] == family].copy()
        subset = subset.sort_values(["n_profiles", "drug"], ascending=[False, True])
        for rank, row in enumerate(subset.itertuples(index=False), start=1):
            rows.append(
                {
                    "drug": row.drug,
                    "primary_target": family,
                    "family_rank": rank,
                    "n_profiles": int(row.n_profiles),
                    "n_cell_lines": int(row.n_cell_lines),
                    "n_targets_in_668": int(row.n_targets_in_668),
                    "n_targets_all_annotations": int(row.n_targets_all_annotations),
                    "targets_in_668": row.targets_in_668,
                }
            )
    selected = pd.DataFrame(rows)
    selected["plot_order"] = np.arange(len(selected)) + 1
    return selected

def long_pair_table(
    selected: pd.DataFrame, sim: np.ndarray, drug_to_idx: dict[str, int]
) -> pd.DataFrame:
    rows = []
    records = selected.set_index("drug").to_dict(orient="index")
    for drug_a, drug_b in combinations(selected["drug"], 2):
        target_a = records[drug_a]["primary_target"]
        target_b = records[drug_b]["primary_target"]
        rows.append(
            {
                "drug_a": drug_a,
                "drug_b": drug_b,
                "target_a": target_a,
                "target_b": target_b,
                "same_target_family": target_a == target_b,
                "cosine_similarity": sim[drug_to_idx[drug_a], drug_to_idx[drug_b]],
            }
        )
    return pd.DataFrame(rows)

def family_pair_summary(
    selected: pd.DataFrame, sim: np.ndarray, drug_to_idx: dict[str, int]
) -> pd.DataFrame:
    single_target_families = selected["primary_target"].unique().tolist()
    rows = []
    for family in single_target_families:
        family_drugs = selected.loc[selected["primary_target"] == family, "drug"].tolist()
        other_drugs = selected.loc[selected["primary_target"] != family, "drug"].tolist()
        within = [
            sim[drug_to_idx[a], drug_to_idx[b]]
            for a, b in combinations(family_drugs, 2)
        ]
        between = [
            sim[drug_to_idx[a], drug_to_idx[b]]
            for a in family_drugs
            for b in other_drugs
        ]
        rows.append(
            {
                "target_family": family,
                "n_drugs": len(family_drugs),
                "n_within_pairs": len(within),
                "n_between_pairs": len(between),
                "within_mean": float(np.mean(within)),
                "within_median": float(np.median(within)),
                "within_q25": float(np.quantile(within, 0.25)),
                "within_q75": float(np.quantile(within, 0.75)),
                "between_mean": float(np.mean(between)),
                "between_median": float(np.median(between)),
                "between_q25": float(np.quantile(between, 0.25)),
                "between_q75": float(np.quantile(between, 0.75)),
                "median_delta": float(np.median(within) - np.median(between)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["median_delta", "n_drugs"], ascending=[False, False])

def target_level_matrix(
    selected: pd.DataFrame, sim: np.ndarray, drug_to_idx: dict[str, int]
) -> tuple[pd.DataFrame, list[str]]:
    families = selected["primary_target"].unique().tolist()
    rows = []
    for family_a in families:
        drugs_a = selected.loc[selected["primary_target"] == family_a, "drug"].tolist()
        row = {"target_family": family_a}
        for family_b in families:
            drugs_b = selected.loc[selected["primary_target"] == family_b, "drug"].tolist()
            values = []
            if family_a == family_b:
                values = [
                    sim[drug_to_idx[a], drug_to_idx[b]]
                    for a, b in combinations(drugs_a, 2)
                ]
            else:
                values = [
                    sim[drug_to_idx[a], drug_to_idx[b]]
                    for a in drugs_a
                    for b in drugs_b
                ]
            row[family_b] = float(np.median(values)) if values else np.nan
        rows.append(row)
    return pd.DataFrame(rows), families

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--similarity",type=Path,required=True)
    parser.add_argument("--target-annotations",type=Path,required=True)
    parser.add_argument("--min-family-size",type=int,default=4)
    args=parser.parse_args()
    global SIM_PATH,BURDEN_PATH,SOURCE_DIR,MIN_SINGLE_TARGET_FAMILY_SIZE
    SIM_PATH=args.similarity
    BURDEN_PATH=args.target_annotations
    SOURCE_DIR=args.output_dir
    MIN_SINGLE_TARGET_FAMILY_SIZE=args.min_family_size
    args.output_dir.mkdir(parents=True,exist_ok=False)
    _, sim, drug_to_idx, burden = load_inputs()
    selected = select_single_target_family_drugs(burden, drug_to_idx)
    selected = selected.reset_index(drop=True)
    selected["plot_order"] = np.arange(len(selected)) + 1
    selected.to_csv(SOURCE_DIR / "suppfig1b_low_target_burden_selected_drugs.tsv", sep="\t", index=False)

    selected_drugs = selected["drug"].tolist()
    idx = [drug_to_idx[drug] for drug in selected_drugs]
    heat = sim[np.ix_(idx, idx)]
    heat_df = pd.DataFrame(heat, columns=selected_drugs)
    heat_df.insert(0, "drug", selected_drugs)
    heat_df.to_csv(SOURCE_DIR / "suppfig1b_low_target_burden_similarity_matrix.tsv", sep="\t", index=False)

    pair_df = long_pair_table(selected, sim, drug_to_idx)
    pair_df.to_csv(SOURCE_DIR / "suppfig1b_low_target_burden_pairwise_similarity.tsv", sep="\t", index=False)
    summary = family_pair_summary(selected, sim, drug_to_idx)
    summary.to_csv(SOURCE_DIR / "suppfig1b_low_target_burden_familywise_summary.tsv", sep="\t", index=False)
    matrix_df, families = target_level_matrix(selected, sim, drug_to_idx)
    matrix_df.to_csv(SOURCE_DIR / "suppfig1b_low_target_burden_target_family_matrix.tsv", sep="\t", index=False)



if __name__ == "__main__":
    main()
