"""Compute manuscript result tables without historical figure layouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE_LABELS_LINCS=668
LINCS_MATCHED_K=10
MATCHED_FRACTION=LINCS_MATCHED_K/CANDIDATE_LABELS_LINCS
MATCHED_PERCENT=MATCHED_FRACTION*100.0
RECOVERY_KS=[1,5,10,20,30,50]


def zscore_row(row: np.ndarray) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float64)
    sd = float(arr.std())
    if not math.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / sd

def rank_diagnostics(scores: np.ndarray, pos_idx: list[int]) -> dict[str, object]:
    order = np.argsort(-scores, kind="mergesort")
    ranks = [int(np.where(order == p)[0][0]) + 1 for p in pos_idx]
    best_rank = min(ranks)
    best_fraction = best_rank / float(len(scores))
    return {
        "best_rank": best_rank,
        "best_percentile_from_top": best_fraction * 100.0,
        "hit_matched_toppercent": int(best_fraction <= MATCHED_FRACTION),
        "hit_top5pct": int(best_fraction <= 0.05),
        "hit_top10pct": int(best_fraction <= 0.10),
    }

def random_any_hit(percentile: float, n_positive: pd.Series | np.ndarray) -> float:
    counts = np.asarray(n_positive, dtype=np.float64)
    return float(np.mean(1.0 - np.power(1.0 - percentile, counts)))

def compute_lincs_recovery_curve() -> pd.DataFrame:
    summary = pd.read_csv(ROUND430_SUMMARY, sep="\t")
    per_drug = pd.read_csv(ROUND430_PER_DRUG, sep="\t")
    positives = (
        per_drug[
            per_drug["mode"].eq("adapted")
            & per_drug["method"].eq("cell_zscore_mean")
            & per_drug["seed"].eq(1)
        ]["n_true_targets"]
        .astype(float)
        .to_numpy()
    )
    rows: list[dict[str, object]] = []
    method_map = {
        "single_cell_mean": "Cell-line mean",
        "cell_zscore_mean": "Consensus",
    }
    sub = summary[summary["mode"].eq("adapted") & summary["method"].isin(method_map)].copy()
    for k in RECOVERY_KS:
        fraction = k / CANDIDATE_LABELS_LINCS
        rows.append(
            {
                "dataset": "LINCS",
                "method": "random",
                "display_label": "Random",
                "rank_cutoff": f"Hit@{k}",
                "threshold_percent": fraction * 100.0,
                "value": random_any_hit(fraction, positives),
                "sd_across_seeds": 0.0,
                "denominator": 415,
                "denominator_type": "same_415_drugs",
                "candidate_labels": CANDIDATE_LABELS_LINCS,
                "source_metric": f"multi_label_random_at_{k}_of_668",
            }
        )
        for method, label in method_map.items():
            row = sub[sub["method"].eq(method)].iloc[0]
            rows.append(
                {
                    "dataset": "LINCS",
                    "method": "multi_cell_consensus" if method == "cell_zscore_mean" else method,
                    "display_label": label,
                    "rank_cutoff": f"Hit@{k}",
                    "threshold_percent": fraction * 100.0,
                    "value": float(row[f"Hit@{k}_mean"]),
                    "sd_across_seeds": float(row[f"Hit@{k}_std"]),
                    "denominator": int(row["n_drugs"]),
                    "denominator_type": "same_415_drugs",
                    "candidate_labels": CANDIDATE_LABELS_LINCS,
                    "source_metric": f"Hit@{k}_in_668_label_space",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_OUT / "Figure7_lincs_recovery_curve_source.tsv", sep="\t", index=False)
    return out

def compute_lincs_rank_pairs() -> pd.DataFrame:
    per_drug = pd.read_csv(ROUND430_PER_DRUG, sep="\t")
    adapted = per_drug[per_drug["mode"].eq("adapted")].copy()
    consensus = adapted[adapted["method"].eq("cell_zscore_mean")][
        ["drug", "seed", "Hit@10", "best_true_target_percentile"]
    ].copy()
    single = adapted[adapted["method"].eq("single_cell_mean")][
        ["drug", "seed", "Hit@10", "best_true_rank_mean_across_cells"]
    ].copy()
    consensus["consensus_percentile_from_top"] = 100.0 - consensus["best_true_target_percentile"].astype(float)
    single["cell_line_mean_percentile_from_top"] = (
        single["best_true_rank_mean_across_cells"].astype(float) / CANDIDATE_LABELS_LINCS * 100.0
    )
    paired = single.merge(
        consensus[["drug", "seed", "Hit@10", "consensus_percentile_from_top"]],
        on=["drug", "seed"],
        suffixes=("_cell_line_mean", "_consensus"),
        validate="one_to_one",
    )
    paired["matched_percent"] = MATCHED_PERCENT
    paired["denominator_type"] = "same_415_drugs_x_3_seeds"
    paired["candidate_labels"] = CANDIDATE_LABELS_LINCS
    paired.to_csv(SOURCE_OUT / "Figure7_lincs_rank_pair_source.tsv", sep="\t", index=False)
    return paired

def compute_lincs_state_summary() -> pd.DataFrame:
    per_drug = pd.read_csv(ROUND430_PER_DRUG, sep="\t")
    consensus = per_drug[(per_drug["mode"].eq("adapted")) & (per_drug["method"].eq("cell_zscore_mean"))].copy()
    consensus["percentile_from_top"] = 100.0 - consensus["best_true_target_percentile"].astype(float)
    top5 = int(consensus["percentile_from_top"].le(5.0).sum())
    top10 = int(consensus["percentile_from_top"].le(10.0).sum())
    rows = [
        {
            "dataset": "LINCS",
            "state": "top5",
            "display_label": "top 5%",
            "count": top5,
            "denominator": int(len(consensus)),
            "denominator_type": "drug_seed_evaluations",
        },
        {
            "dataset": "LINCS",
            "state": "top10_only",
            "display_label": "top 5-10%",
            "count": top10 - top5,
            "denominator": int(len(consensus)),
            "denominator_type": "drug_seed_evaluations",
        },
        {
            "dataset": "LINCS",
            "state": "missed_top10",
            "display_label": "missed top 10%",
            "count": int(len(consensus) - top10),
            "denominator": int(len(consensus)),
            "denominator_type": "drug_seed_evaluations",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_OUT / "Figure7_lincs_state_summary_source.tsv", sep="\t", index=False)
    return out

def compute_tahoe_consensus() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(SC_METRICS, sep="\t")
    scores = pd.read_csv(SC_SCORES, sep="\t")
    labels = pd.read_csv(SC_LABELS, sep="\t")["target_label"].astype(str).tolist()
    if len(labels) != 16583:
        raise RuntimeError(f"Expected 16,583 target labels, found {len(labels)}")
    if set(metrics["cell_line"]) != {"A549", "SW480", "NCI-H460"}:
        raise RuntimeError("Unexpected Tahoe cell-line set")
    missing = [label for label in labels if label not in scores.columns]
    if missing:
        raise RuntimeError(f"Score table missing labels, e.g. {missing[:5]}")

    merged = scores.merge(
        metrics[
            [
                "cell_line",
                "drug",
                "drug_category",
                "positive_targets",
                "n_holdout_positive",
                "best_rank",
                "best_percentile",
                "hit_top5pct",
                "hit_top10pct",
            ]
        ],
        on=["cell_line", "drug"],
        how="inner",
        validate="one_to_one",
    )
    label_to_idx = {label: i for i, label in enumerate(labels)}
    per_drug_rows: list[dict[str, object]] = []
    per_cell_rows: list[dict[str, object]] = []

    for drug, grp in merged.groupby("drug", sort=True):
        if grp["cell_line"].nunique() != 3:
            continue
        if grp["positive_targets"].nunique() != 1:
            raise RuntimeError(f"Positive targets differ across cell lines for {drug}")
        target_text = str(grp["positive_targets"].iloc[0])
        targets = [target for target in target_text.split("|") if target]
        pos_idx = [label_to_idx[target] for target in targets if target in label_to_idx]
        if len(pos_idx) != len(targets):
            raise RuntimeError(f"{drug}: not all positive targets are in the target label table")

        score_matrix = grp[labels].to_numpy(dtype=np.float64)
        consensus_scores = np.vstack([zscore_row(row) for row in score_matrix]).mean(axis=0)
        diag = rank_diagnostics(consensus_scores, pos_idx)
        matched_single = grp["best_percentile"].astype(float).le(MATCHED_FRACTION).astype(float)

        per_drug_rows.append(
            {
                "drug": drug,
                "cell_lines": "|".join(sorted(grp["cell_line"].astype(str))),
                "positive_targets": target_text,
                "n_holdout_positive": int(grp["n_holdout_positive"].iloc[0]),
                "candidate_labels": len(labels),
                "single_cell_mean_matched_toppercent": float(matched_single.mean()),
                "single_cell_mean_best_percentile_from_top": float(grp["best_percentile"].astype(float).mean() * 100.0),
                "single_cell_mean_top5pct": float(grp["hit_top5pct"].mean()),
                "single_cell_mean_top10pct": float(grp["hit_top10pct"].mean()),
                "single_cell_top5_count": int(grp["hit_top5pct"].sum()),
                "single_cell_top10_count": int(grp["hit_top10pct"].sum()),
                **{f"consensus_{key}": value for key, value in diag.items()},
            }
        )
        for _, cell_row in grp.sort_values("cell_line").iterrows():
            per_cell_rows.append(
                {
                    "drug": drug,
                    "cell_line": cell_row["cell_line"],
                    "positive_targets": target_text,
                    "best_rank": int(cell_row["best_rank"]),
                    "rank_percentile_from_top": float(cell_row["best_percentile"] * 100.0),
                    "hit_matched_toppercent": int(float(cell_row["best_percentile"]) <= MATCHED_FRACTION),
                    "hit_top5pct": int(cell_row["hit_top5pct"]),
                    "hit_top10pct": int(cell_row["hit_top10pct"]),
                }
            )

    per_drug = pd.DataFrame(per_drug_rows).sort_values(["consensus_best_percentile_from_top", "drug"], kind="mergesort").reset_index(drop=True)
    per_cell = pd.DataFrame(per_cell_rows)
    if len(per_drug) != 30:
        raise RuntimeError(f"Expected 30 shared Tahoe drugs, found {len(per_drug)}")

    summary = pd.DataFrame(
        [
            {
                "dataset": "Tahoe",
                "method": "random_expectation",
                "display_label": "Random",
                "matched_cutoff": f"Top {MATCHED_PERCENT:.2f}%",
                "matched_percent": MATCHED_PERCENT,
                "value": random_any_hit(MATCHED_FRACTION, per_drug["n_holdout_positive"]),
                "numerator_equivalent_drugs": random_any_hit(MATCHED_FRACTION, per_drug["n_holdout_positive"]) * len(per_drug),
                "denominator": len(per_drug),
                "denominator_type": "same_30_shared_drugs",
                "candidate_labels": len(labels),
                "aggregation": "multi_label_random_expectation_per_drug",
            },
            {
                "dataset": "Tahoe",
                "method": "single_cell_mean",
                "display_label": "Cell-line mean",
                "matched_cutoff": f"Top {MATCHED_PERCENT:.2f}%",
                "matched_percent": MATCHED_PERCENT,
                "value": float(per_drug["single_cell_mean_matched_toppercent"].mean()),
                "numerator_equivalent_drugs": float(per_drug["single_cell_mean_matched_toppercent"].sum()),
                "denominator": len(per_drug),
                "denominator_type": "same_30_shared_drugs",
                "candidate_labels": len(labels),
                "aggregation": "mean_of_three_cell_line_binary_recovery_values_per_drug",
            },
            {
                "dataset": "Tahoe",
                "method": "three_cell_consensus",
                "display_label": "Three-cell consensus",
                "matched_cutoff": f"Top {MATCHED_PERCENT:.2f}%",
                "matched_percent": MATCHED_PERCENT,
                "value": float(per_drug["consensus_hit_matched_toppercent"].mean()),
                "numerator_equivalent_drugs": float(per_drug["consensus_hit_matched_toppercent"].sum()),
                "denominator": len(per_drug),
                "denominator_type": "same_30_shared_drugs",
                "candidate_labels": len(labels),
                "aggregation": "binary_recovery_after_three_score_vectors_are_zscored_and_averaged",
            },
        ]
    )

    state_rows = []
    for _, row in per_drug.iterrows():
        if int(row["single_cell_top5_count"]) == 3:
            single_state = "top5_in_all_three"
        elif int(row["single_cell_top10_count"]) == 3:
            single_state = "top10_in_all_three"
        elif int(row["single_cell_top10_count"]) > 0:
            single_state = "context_specific_top10"
        else:
            single_state = "missed_top10"

        if int(row["consensus_hit_top5pct"]) == 1:
            consensus_state = "top5"
        elif int(row["consensus_hit_top10pct"]) == 1:
            consensus_state = "top10_only"
        else:
            consensus_state = "missed_top10"
        state_rows.append({"drug": row["drug"], "single_cell_pattern": single_state, "consensus_state": consensus_state})
    states = pd.DataFrame(state_rows)
    state_summary = pd.concat(
        [
            states["single_cell_pattern"].value_counts().rename_axis("state").reset_index(name="count").assign(group="single_cell_pattern"),
            states["consensus_state"].value_counts().rename_axis("state").reset_index(name="count").assign(group="consensus_rank"),
        ],
        ignore_index=True,
    )

    audit = pd.DataFrame(
        [
            {
                "analysis": "Fig7 Round492 high-contrast method colors and non-overlapping shared legend",
                "matched_percent": MATCHED_PERCENT,
                "lincs_matched_cutoff": "Hit@10 among 668 labels",
                "tahoe_matched_cutoff": f"top {MATCHED_PERCENT:.4f}% among 16,583 genes",
                "recovery_k_values": "|".join(str(k) for k in RECOVERY_KS),
                "tahoe_cell_lines": "A549|SW480|NCI-H460",
                "tahoe_n_shared_drugs": int(len(per_drug)),
                "tahoe_n_drug_cell_units_input": int(len(per_cell)),
                "tahoe_candidate_labels": int(len(labels)),
                "denominator_rule": "A/B panels use matched LINCS rank-fraction thresholds; Tahoe single-cell mean is averaged over the same 30 shared drugs.",
            }
        ]
    )

    rank_pairs = per_drug[
        [
            "drug",
            "positive_targets",
            "single_cell_mean_best_percentile_from_top",
            "consensus_best_percentile_from_top",
            "single_cell_mean_matched_toppercent",
            "consensus_hit_matched_toppercent",
            "candidate_labels",
        ]
    ].copy()
    rank_pairs["dataset"] = "Tahoe"
    rank_pairs["matched_percent"] = MATCHED_PERCENT
    rank_pairs["denominator_type"] = "same_30_shared_drugs"

    curve_rows: list[dict[str, object]] = []
    for k in RECOVERY_KS:
        fraction = k / CANDIDATE_LABELS_LINCS
        percent = fraction * 100.0
        per_drug_single = (
            per_cell.assign(recovered=per_cell["rank_percentile_from_top"].astype(float).le(percent).astype(float))
            .groupby("drug", as_index=False)["recovered"]
            .mean()
        )
        single_value = float(per_drug_single["recovered"].mean())
        consensus_value = float(per_drug["consensus_best_percentile_from_top"].astype(float).le(percent).mean())
        random_value = random_any_hit(fraction, per_drug["n_holdout_positive"])
        curve_rows.extend(
            [
                {
                    "dataset": "Tahoe",
                    "method": "random",
                    "display_label": "Random",
                    "rank_cutoff": f"Top {percent:.2f}%",
                    "threshold_percent": percent,
                    "value": random_value,
                    "numerator_equivalent_drugs": random_value * len(per_drug),
                    "denominator": len(per_drug),
                    "denominator_type": "same_30_shared_drugs",
                    "candidate_labels": len(labels),
                    "aggregation": "multi_label_random_expectation_per_drug",
                },
                {
                    "dataset": "Tahoe",
                    "method": "single_cell_mean",
                    "display_label": "Cell-line mean",
                    "rank_cutoff": f"Top {percent:.2f}%",
                    "threshold_percent": percent,
                    "value": single_value,
                    "numerator_equivalent_drugs": single_value * len(per_drug),
                    "denominator": len(per_drug),
                    "denominator_type": "same_30_shared_drugs",
                    "candidate_labels": len(labels),
                    "aggregation": "mean_of_three_cell_line_binary_recovery_values_per_drug",
                },
                {
                    "dataset": "Tahoe",
                    "method": "multi_cell_consensus",
                    "display_label": "Consensus",
                    "rank_cutoff": f"Top {percent:.2f}%",
                    "threshold_percent": percent,
                    "value": consensus_value,
                    "numerator_equivalent_drugs": consensus_value * len(per_drug),
                    "denominator": len(per_drug),
                    "denominator_type": "same_30_shared_drugs",
                    "candidate_labels": len(labels),
                    "aggregation": "binary_recovery_after_three_score_vectors_are_zscored_and_averaged",
                },
            ]
        )
    recovery_curve = pd.DataFrame(curve_rows)

    per_drug.to_csv(SOURCE_OUT / "Figure7_tahoe_per_drug_consensus_source.tsv", sep="\t", index=False)
    per_cell.to_csv(SOURCE_OUT / "Figure7_tahoe_per_cell_percentile_source.tsv", sep="\t", index=False)
    summary.to_csv(SOURCE_OUT / "Figure7_tahoe_matched_toppercent_source.tsv", sep="\t", index=False)
    state_summary.to_csv(SOURCE_OUT / "Figure7_tahoe_state_summary_source.tsv", sep="\t", index=False)
    rank_pairs.to_csv(SOURCE_OUT / "Figure7_tahoe_rank_pair_source.tsv", sep="\t", index=False)
    audit.to_csv(SOURCE_OUT / "Figure7_protocol_audit_source.tsv", sep="\t", index=False)
    recovery_curve.to_csv(SOURCE_OUT / "Figure7_tahoe_recovery_curve_source.tsv", sep="\t", index=False)
    return per_drug, per_cell, summary, state_summary, rank_pairs, recovery_curve



def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lincs-per-drug",type=Path,required=True)
    parser.add_argument("--lincs-summary",type=Path,required=True)
    parser.add_argument("--single-cell-metrics",type=Path,required=True)
    parser.add_argument("--single-cell-scores",type=Path,required=True)
    parser.add_argument("--single-cell-labels",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    global ROUND430_PER_DRUG,ROUND430_SUMMARY,SC_METRICS,SC_SCORES,SC_LABELS,SOURCE_OUT
    ROUND430_PER_DRUG=args.lincs_per_drug
    ROUND430_SUMMARY=args.lincs_summary
    SC_METRICS=args.single_cell_metrics
    SC_SCORES=args.single_cell_scores
    SC_LABELS=args.single_cell_labels
    SOURCE_OUT=args.output_dir
    SOURCE_OUT.mkdir(parents=True,exist_ok=False)
    compute_lincs_recovery_curve()
    compute_lincs_rank_pairs()
    compute_lincs_state_summary()
    compute_tahoe_consensus()


if __name__ == "__main__":
    main()
